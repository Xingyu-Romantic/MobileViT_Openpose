from paddle import nn
import paddle
from paddle.fluid.layers.nn import pad
from paddle.nn import Conv2D, BatchNorm2D
from einops import rearrange
from modules.conv import conv, conv_dw, conv_dw_no_bn

class Cpm(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Layer):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Layer):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]



def conv_bn(inp,oup,kernel_size=3,stride=1):
    return nn.Sequential(
        nn.Conv2D(inp,oup,kernel_size=kernel_size,stride=stride,padding=kernel_size//2),
        nn.BatchNorm2D(oup),
        nn.Silu()
    )

class PreNorm(nn.Layer):
    def __init__(self,dim,fn):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.ln(x),**kwargs)

class FeedForward(nn.Layer):
    def __init__(self,dim,mlp_dim,dropout) :
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Layer):
    def __init__(self,dim,heads,head_dim,dropout):
        super().__init__()
        inner_dim=heads*head_dim
        project_out=not(heads==1 and head_dim==dim)

        self.heads=heads
        self.scale=head_dim**-0.5

        self.attend=nn.Softmax(axis=-1)
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias_attr=False)
        
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        qkv=self.to_qkv(x).chunk(3,axis=-1)
        q,k,v=map(lambda t:rearrange(t.numpy(),'b p n (h d) -> b p h n d',h=self.heads),qkv)
        q, k, v = paddle.to_tensor(q), paddle.to_tensor(k), paddle.to_tensor(v)
        print(type(k))
        print(k.shape)
        dots=paddle.matmul(q,k.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        out=paddle.matmul(attn,v)
        #out = out.transpose((0, 2, 1, 3)).reshape((B_, N, C))
        out=out.numpy()
        out=rearrange(out,'b p h n d -> b p n (h d)')
        out=paddle.to_tensor(out)
        return self.to_out(out)





class Transformer(nn.Layer):
    def __init__(self,dim,depth,heads,head_dim,mlp_dim,dropout=0.):
        super().__init__()
        self.layers=nn.LayerList([])
        for _ in range(depth):
            self.layers.append(nn.LayerList([
                PreNorm(dim,Attention(dim,heads,head_dim,dropout)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
            ]))


    def forward(self,x):
        out=x
        for att,ffn in self.layers:
            out=out+att(out)
            out=out+ffn(out)
        return out

class MobileViTAttention(nn.Layer):
    def __init__(self,in_channel=3,dim=512,kernel_size=3,patch_size=7,depth=3,mlp_dim=1024):
        super().__init__()
        self.ph,self.pw=patch_size,patch_size
        self.conv1=nn.Conv2D(in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)
        self.conv2=nn.Conv2D(in_channel,dim,kernel_size=1)

        self.trans=Transformer(dim=dim,depth=depth,heads=8,head_dim=64,mlp_dim=mlp_dim)

        self.conv3=nn.Conv2D(dim,in_channel,kernel_size=1)
        self.conv4=nn.Conv2D(2*in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)

    def forward(self,x):
        y=x.clone() #bs,c,h,w

        ## Local Representation
        y=self.conv2(self.conv1(x)) #bs,dim,h,w

        ## Global Representation
        _,_,h,w=y.shape
        y = y.numpy()
        y=rearrange(y,'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim',ph=self.ph,pw=self.pw) #bs,h,w,dim
        y = paddle.to_tensor(y)
        y=self.trans(y)
        y = y.numpy()
        y=rearrange(y,'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)',ph=self.ph,pw=self.pw,nh=h//self.ph,nw=w//self.pw) #bs,dim,h,w
        y = paddle.to_tensor(y)
        ## Fusion
        y=self.conv3(y) #bs,dim,h,w
        y=paddle.concat([x,y],1) #bs,2*dim,h,w
        y=self.conv4(y) #bs,c,h,w

        return y


class MV2Block(nn.Layer):
    def __init__(self,inp,out,stride=1,expansion=4):
        super().__init__()
        self.stride=stride
        hidden_dim=inp*expansion
        self.use_res_connection=stride==1 and inp==out

        if expansion==1:
            self.conv=nn.Sequential(
                nn.Conv2D(hidden_dim,hidden_dim,kernel_size=3,stride=self.stride,padding=1,groups=hidden_dim,bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.Silu(),
                nn.Conv2D(hidden_dim,out,kernel_size=1,stride=1,bias_attr=False),
                nn.BatchNorm2D(out)
            )
        else:
            self.conv=nn.Sequential(
                nn.Conv2D(inp,hidden_dim,kernel_size=1,stride=1,bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.Silu(),
                nn.Conv2D(hidden_dim,hidden_dim,kernel_size=3,stride=1,padding=1,groups=hidden_dim,bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                nn.Silu(),
                nn.Conv2D(hidden_dim,out,kernel_size=1,stride=1,bias_attr=False),
                nn.Silu(),
                nn.BatchNorm2D(out)
            )
    def forward(self,x):
        if(self.use_res_connection):
            out=x+self.conv(x)
        else:
            out=self.conv(x)
        return out

class MobileViT(nn.Layer):
    def __init__(self,image_size,dims,channels,num_classes,depths=[2,4,3],expansion=4,kernel_size=3,patch_size=2):
        super().__init__()
        ih,iw=image_size,image_size
        ph,pw=patch_size,patch_size
        assert iw%pw==0 and ih%ph==0

        self.conv1=conv_bn(3,channels[0],kernel_size=3,stride=patch_size)
        self.mv2=nn.LayerList([])
        self.m_vits=nn.LayerList([])


        self.mv2.append(MV2Block(channels[0],channels[1],1))
        self.mv2.append(MV2Block(channels[1],channels[2],2))
        self.mv2.append(MV2Block(channels[2],channels[3],1))
        self.mv2.append(MV2Block(channels[2],channels[3],1)) # x2
        self.mv2.append(MV2Block(channels[3],channels[4],2))
        self.m_vits.append(MobileViTAttention(channels[4],dim=dims[0],kernel_size=kernel_size,patch_size=patch_size,depth=depths[0],mlp_dim=int(2*dims[0])))
        self.mv2.append(MV2Block(channels[4],channels[5],2))
        self.m_vits.append(MobileViTAttention(channels[5],dim=dims[1],kernel_size=kernel_size,patch_size=patch_size,depth=depths[1],mlp_dim=int(4*dims[1])))
        self.mv2.append(MV2Block(channels[5],channels[6],2))
        self.m_vits.append(MobileViTAttention(channels[6],dim=dims[2],kernel_size=kernel_size,patch_size=patch_size,depth=depths[2],mlp_dim=int(4*dims[2])))

        
        self.conv2=conv_bn(channels[-2],channels[-1],kernel_size=1)
        self.pool=nn.AvgPool2D(image_size//32,1)
        self.fc=nn.Linear(channels[-1],num_classes,bias_attr=False)

    def forward(self,x):
        y=self.conv1(x) #
        y=self.mv2[0](y)
        y=self.mv2[1](y) #
        y=self.mv2[2](y)
        y=self.mv2[3](y)
        y=self.mv2[4](y) #
        y=self.m_vits[0](y)

        y=self.mv2[5](y) #
        y=self.m_vits[1](y)

        y=self.mv2[6](y) #
        y=self.m_vits[2](y)

        y=self.conv2(y)
        y=self.pool(y).view(y.shape[0],-1) 
        y=self.fc(y)
        return y

class PoseEstimationWithMobileViT(nn.Layer):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = MobileViT(224, [60, 80, 96], [32, 64, 128, 128, 256, 256, 512, 512], num_classes=512)
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.LayerList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(paddle.concat([backbone_features, stages_output[-2], stages_output[-1]], axis=1)))

        return stages_output

def mobilevit_xxs():
    dims=[60,80,96]
    channels= [16, 16, 24, 24, 48, 64, 80, 320]
    return MobileViT(224,dims,channels,num_classes=1000)

def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 80, 96, 384]
    return MobileViT(224, dims, channels, num_classes=1000)

def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    return MobileViT(224, dims, channels, num_classes=1000)


def count_paratermeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    input=paddle.randn((1,3,224,224))

    ### mobilevit_xxs
    mvit_xxs=mobilevit_xxs()
    out=mvit_xxs(input)
    print(out.shape)

    ### mobilevit_xs
    mvit_xs=mobilevit_xs()
    out=mvit_xs(input)
    print(out.shape)


    ### mobilevit_s
    mvit_s=mobilevit_s()
    out=mvit_s(input)
    print(out.shape)
