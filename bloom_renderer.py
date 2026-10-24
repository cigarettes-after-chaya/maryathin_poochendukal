#!/usr/bin/env python3
"""Bloom: procedural botanical animation renderer.

Uses NumPy + OpenCV to render vertical 9:16 MP4 animations for the Bloom website.
The flower meshes are parametric surfaces: petals transition from closed buds to
open corollas while stems, leaves, veins, stamens and pollen particles are revealed.

Usage:
    python bloom_renderer.py                 # render all five flowers
    python bloom_renderer.py --flower lily   # render one flower
    python bloom_renderer.py --preview       # render a still preview for each flower

Requirements: pip install numpy opencv-python; ffmpeg installed on PATH.
"""
import argparse, math, os, subprocess
import cv2
import numpy as np

W, H, FPS, DURATION = 540, 960, 15, 6.5
TOTAL = int(FPS * DURATION)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')

FLOWERS = {
    "lily": {"title":"LILIUM ORIENTALIS", "petals":6, "length":205, "width":58,
             "colors":[(255,244,244),(235,157,188),(196,91,139)], "center":(74,160,221), "kind":"lily"},
    "sunflower": {"title":"HELIANTHUS ANNUUS", "petals":28, "length":168, "width":20,
             "colors":[(92,210,255),(55,174,246),(35,135,225)], "center":(45,49,61), "kind":"sunflower"},
    "rose": {"title":"ROSA · GARDEN ROSE", "petals":22, "length":132, "width":45,
             "colors":[(93,39,183),(116,57,220),(151,91,240),(178,130,247)], "center":(54,39,130), "kind":"rose"},
    "poppy": {"title":"PAPAVER · FIELD POPPY", "petals":4, "length":184, "width":76,
             "colors":[(48,53,228),(63,89,246),(93,134,255)], "center":(37,39,42), "kind":"poppy"},
    "passiflora": {"title":"PASSIFLORA CAERULEA", "petals":10, "length":174, "width":37,
             "colors":[(232,239,248),(220,225,238),(202,215,231)], "center":(156,125,184), "kind":"passiflora"},
}

def smooth(a,b,x):
    t=np.clip((x-a)/max(1e-8,b-a),0,1)
    return t*t*(3-2*t)
def smoother(a,b,x):
    t=np.clip((x-a)/max(1e-8,b-a),0,1)
    return t*t*t*(t*(t*6-15)+10)
def mix(a,b,t): return np.asarray(a)*(1-t)+np.asarray(b)*t

def backdrop(t):
    yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
    x=(xx-W*.5)/(W*.72); y=(yy-H*.47)/(H*.68)
    glow=np.exp(-2.3*(x*x+y*y))
    warm=np.exp(-((x-.2)**2+(y+.28)**2)*4)
    img=np.zeros((H,W,3),np.float32)
    # BGR cinematic deep botanical studio
    img[:,:,0]=14+30*glow+8*warm
    img[:,:,1]=13+27*glow+8*warm
    img[:,:,2]=20+31*glow+17*warm
    img += (np.sin(xx*.018+yy*.009+t)*.55+0.5)[...,None]*1.0
    return np.clip(img,0,255).astype(np.uint8)

def project(x,y,z,t):
    # very gentle camera orbit/parallax for a 3D presentation
    az=-.12+.22*smoother(0,1,t/DURATION)
    xx=x*math.cos(az)-z*math.sin(az)
    zz=x*math.sin(az)+z*math.cos(az)
    perspective=1.0/(1.0+zz/1700)
    return int(W*.5+xx*perspective), int(425-y*perspective+zz*.10)

def draw_gradient_poly(img, pts, color, shade=1.0, outline=None, thickness=1):
    p=np.asarray(pts,np.int32)
    if len(p)<3:return
    # soft dark edge then body highlight
    c=np.clip(np.asarray(color,dtype=float)*shade,0,255).astype(np.uint8).tolist()
    cv2.fillPoly(img,[p],c,lineType=cv2.LINE_AA)
    if outline is not None: cv2.polylines(img,[p],True,outline,thickness,cv2.LINE_AA)

def petal_point(u,v,az,theta,length,width,kind,depth=0):
    # u follows the midrib from base to tip, v spans the blade.
    profile=max(0,math.sin(math.pi*(u**.72)))**.72
    if kind=='rose': profile=max(0,math.sin(math.pi*(u**.78)))**.62*(.86+.14*math.sin(u*math.pi*2))
    if kind=='sunflower': profile=max(0,math.sin(math.pi*(u**.82)))**.58
    if kind=='poppy': profile=max(0,math.sin(math.pi*(u**.7)))**.5
    w=width*profile*(1-.16*u)+width*.06*(1-u)**2
    # curved blade, central keel and subtle rim wave
    r=length*u*math.sin(theta)
    y=length*u*math.cos(theta)
    if kind=='lily': y += 15*u*u + 9*(v*v)*u
    if kind=='rose': y += 17*u*u + 8*v*v
    if kind=='poppy': y += 7*math.sin(u*math.pi)*abs(v)
    r += 10*math.sin(u*math.pi)*v*v
    r += 3.4*math.sin(u*math.pi*5+az)*abs(v)**2
    side=w*v
    x=r*math.cos(az)-side*math.sin(az)
    z=r*math.sin(az)+side*math.cos(az)+depth
    return x,y,z

def draw_petal_mesh(img,t,idx,total,cfg,center=(0,0,0),layer=0):
    bloom=smoother(1.15+layer*.22,5.5+layer*.25,t)
    # starts as upright closed bud, then recurves outward; layers open in sequence
    theta0=.10 + .05*(idx%3)
    theta_open={"lily":0.92,"sunflower":0.98,"rose":1.05,"poppy":0.90,"passiflora":0.98}[cfg['kind']]
    theta=theta0+(theta_open-theta0)*bloom
    az=2*math.pi*idx/total + .11*math.sin(idx*3.7)
    if cfg['kind']=='rose':
        az += layer*.31
        theta=min(1.58,theta+layer*.055)
    if cfg['kind']=='sunflower': theta*=.90
    length=cfg['length']*(1-.11*layer)
    width=cfg['width']*(1.0 if layer==0 else .91)
    # petal reveal moves from base to tip, like a materialization wave
    reveal=smoother(1.25+layer*.15,4.7+layer*.18,t)
    if reveal<=.001:return
    max_u=max(.08,reveal)
    rows=9; cols=5
    quads=[]
    for i in range(rows-1):
        u0=max_u*i/(rows-1);u1=max_u*(i+1)/(rows-1)
        for j in range(cols-1):
            v0=-1+2*j/(cols-1);v1=-1+2*(j+1)/(cols-1)
            xyz=[petal_point(u0,v0,az,theta,length,width,cfg['kind'],layer*4),
                 petal_point(u1,v0,az,theta,length,width,cfg['kind'],layer*4),
                 petal_point(u1,v1,az,theta,length,width,cfg['kind'],layer*4),
                 petal_point(u0,v1,az,theta,length,width,cfg['kind'],layer*4)]
            zmean=sum(p[2] for p in xyz)/4
            pts=[project(p[0]+center[0],p[1]+center[1],p[2]+center[2],t) for p in xyz]
            umean=(u0+u1)/2; vmean=(v0+v1)/2
            base=cfg['colors'][(idx+layer)%len(cfg['colors'])]
            # base throat colour and tip illumination
            shade=.66+.34*(1-abs(vmean))+.12*math.sin(umean*math.pi)
            if cfg['kind']=='lily':
                throat=smooth(.04,.35,umean)*(1-smooth(.38,.77,umean))*math.exp(-(vmean/.34)**2)
                base=mix(base,(93,38,153),throat*.48)
            elif cfg['kind']=='rose':
                shade*=.83+.24*umean
            elif cfg['kind']=='sunflower':
                shade*=.91+.18*(1-umean)
            elif cfg['kind']=='poppy': shade*=.90+.22*(1-umean)
            elif cfg['kind']=='passiflora': shade*=.91+.14*(1-abs(vmean))
            quads.append((zmean,pts,base,shade,umean,vmean))
    quads.sort(key=lambda q:q[0])
    for z,pts,base,shade,u,v in quads:
        draw_gradient_poly(img,pts,base,shade,outline=None)
        if abs(v)<.10 and int(u*100)%10==0:
            cv2.line(img,pts[0],pts[2],tuple(int(min(255,x*1.16)) for x in base),1,cv2.LINE_AA)

def draw_stem_leaf(img,t):
    grow=smoother(0,2.5,t)
    if grow<=0:return
    # tapered curved stem grows upward from the bottom into the flower head
    end_y=int(865-425*grow)
    pts=[]
    for i in range(40):
        u=i/39
        if u>grow:break
        x=W*.5 + 19*math.sin(u*2.5)+12*u
        y=865-(865-end_y)*(u/max(grow,1e-6))
        pts.append((int(x),int(y)))
    if len(pts)>1:
        cv2.polylines(img,[np.array(pts,np.int32)],False,(34,91,43),11,cv2.LINE_AA)
        cv2.polylines(img,[np.array(pts,np.int32)],False,(68,133,65),5,cv2.LINE_AA)
        cv2.polylines(img,[np.array(pts,np.int32)],False,(112,164,91),1,cv2.LINE_AA)
    leaf_specs=[(.24,-1,1.25,.95),(.42,1,1.0,.84),(.61,-1,.8,.75),(.73,1,.7,.7)]
    for k,(u,side,length,scale) in enumerate(leaf_specs):
        if t < .65+k*.27:continue
        origin=(W*.5+int(19*math.sin(u*2.5)+12*u),int(865-425*u))
        direction=side
        L=length*145*smoother(.65+k*.27,1.8+k*.27,t)
        base=np.array(origin,float)
        tip=base+np.array([direction*L, -L*.37])
        perp=np.array([-direction*.24*L,-.38*L])
        poly=[base,base+perp*.34,tip,base-perp*.28]
        draw_gradient_poly(img,poly,(41,106,48),.92,outline=(26,75,35),thickness=1)
        cv2.line(img,tuple(base.astype(int)),tuple(tip.astype(int)),(105,155,78),1,cv2.LINE_AA)
        for v in (.3,.52,.7):
            p=base+(tip-base)*v
            cv2.line(img,tuple(p.astype(int)),tuple((p+perp*.18).astype(int)),(73,132,66),1,cv2.LINE_AA)

def draw_center(img,t,cfg):
    reveal=smoother(5.1,6.5,t)
    if reveal<=.001:return
    cx,cy=W//2,425
    kind=cfg['kind']
    if kind=='sunflower':
        cv2.circle(img,(cx,cy),int(51*reveal),(39,40,46),-1,cv2.LINE_AA)
        cv2.circle(img,(cx,cy),int(46*reveal),(55,62,76),-1,cv2.LINE_AA)
        for i in range(500):
            a=i*2.399963; r=math.sqrt(i/500)*43*reveal
            x=int(cx+math.cos(a)*r);y=int(cy+math.sin(a)*r)
            cv2.circle(img,(x,y),1,(95+(i%3)*25,125+(i%4)*18,160+(i%3)*12),-1,cv2.LINE_AA)
    elif kind=='poppy':
        cv2.circle(img,(cx,cy),int(25*reveal),(30,31,32),-1,cv2.LINE_AA)
        cv2.circle(img,(cx,cy),int(18*reveal),(46,47,49),-1,cv2.LINE_AA)
        for i in range(34):
            a=2*math.pi*i/34
            p1=(int(cx+math.cos(a)*19*reveal),int(cy+math.sin(a)*19*reveal))
            p2=(int(cx+math.cos(a)*29*reveal),int(cy+math.sin(a)*29*reveal))
            cv2.line(img,p1,p2,(176,174,128),1,cv2.LINE_AA)
    elif kind=='passiflora':
        for i in range(76):
            a=2*math.pi*i/76; inner=25*reveal; outer=(48+(i%4)*2)*reveal
            col=(245,241,235) if i%4==0 else ((176,142,91) if i%3==0 else (219,194,143))
            p1=(int(cx+math.cos(a)*inner),int(cy+math.sin(a)*inner))
            pm=(int(cx+math.cos(a+.06)*((inner+outer)/2)),int(cy+math.sin(a+.06)*((inner+outer)/2)))
            p2=(int(cx+math.cos(a)*outer),int(cy+math.sin(a)*outer))
            cv2.line(img,p1,pm,col,2,cv2.LINE_AA);cv2.line(img,pm,p2,col,1,cv2.LINE_AA)
        cv2.circle(img,(cx,cy),int(17*reveal),(124,97,158),-1,cv2.LINE_AA)
        for i in range(5):
            a=2*math.pi*i/5-math.pi/2
            x=int(cx+math.cos(a)*18*reveal);y=int(cy+math.sin(a)*18*reveal)
            cv2.line(img,(cx,cy),(x,y-8),(185,155,209),4,cv2.LINE_AA)
            cv2.ellipse(img,(x,y-8),(5,7),0,0,360,(230,213,239),-1,cv2.LINE_AA)
    elif kind=='lily':
        for i in range(6):
            a=2*math.pi*i/6+.15
            p1=(cx,cy);p2=(int(cx+math.cos(a)*37*reveal),int(cy+math.sin(a)*32*reveal))
            cv2.line(img,p1,p2,(91,175,224),2,cv2.LINE_AA)
            cv2.ellipse(img,p2,(4,8),math.degrees(a),0,360,(54,132,205),-1,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx+4,int(cy-36*reveal)),(94,182,113),3,cv2.LINE_AA)
        cv2.circle(img,(cx+4,int(cy-37*reveal)),4,(128,198,126),-1,cv2.LINE_AA)
    else: # rose's curled central whorl
        for i in range(5):
            a=i*2.1
            r=(4+i*3)*reveal
            x=int(cx+math.cos(a)*r);y=int(cy+math.sin(a)*r)
            cv2.ellipse(img,(x,y),(9+i,13+i),i*29,0,310,(94+i*10,43+i*5,177+i*9),-1,cv2.LINE_AA)
        cv2.circle(img,(cx,cy),int(10*reveal),(75,32,143),-1,cv2.LINE_AA)

def render_frame(idx,cfg):
    t=idx/FPS
    img=backdrop(t)
    draw_stem_leaf(img,t)
    kind=cfg['kind']
    if kind=='rose':
        # layered spiral: outer petals spread first; inner petals unfurl last
        specs=[]
        for layer,count in [(0,8),(1,8),(2,6)]:
            for i in range(count): specs.append((layer,i,count))
        # back-to-front approximate depth order
        for layer,i,count in specs:
            draw_petal_mesh(img,t,i,count,{**cfg,'length':cfg['length']*(1-.13*layer),'width':cfg['width']*(1-.15*layer)},layer=layer)
    else:
        count=cfg['petals']
        # render petals from back to front with slight angle variation
        for i in sorted(range(count),key=lambda k:math.sin(2*math.pi*k/count)):
            draw_petal_mesh(img,t,i,count,cfg)
    draw_center(img,t,cfg)
    # floating pollen / construction motes
    rng=np.random.RandomState(42)
    for i in range(48):
        phase=i*1.73
        x=int(W*.5+math.sin(phase+t*.45)*(95+(i%5)*24))
        y=int(420+math.cos(phase*1.4+t*.38)*(145+(i%4)*22))
        alpha=.25+.45*(.5+.5*math.sin(t*2+phase))
        col=(int(90*alpha),int(170*alpha),int(228*alpha))
        cv2.circle(img,(x,y),1+(i%3==0),col,-1,cv2.LINE_AA)
    # warm bloom/glow polish
    bright=np.maximum(img.astype(np.float32)-185,0).astype(np.uint8)
    glow=cv2.GaussianBlur(bright,(0,0),sigmaX=11)
    img=cv2.addWeighted(img,1.0,glow,.48,0)
    # restrained specimen telemetry, like the supplied renderer
    hud=smoother(.2,.8,t)*(1-.25*smoother(5.5,6.4,t))
    overlay=img.copy()
    cv2.putText(overlay,cfg['title'],(28,48),cv2.FONT_HERSHEY_SIMPLEX,.60,(225,230,238),1,cv2.LINE_AA)
    cv2.putText(overlay,'3D PARAMETRIC BOTANICAL STUDY',(28,72),cv2.FONT_HERSHEY_SIMPLEX,.34,(145,164,180),1,cv2.LINE_AA)
    if t<2.4: stage='STAGE 01 // STEMS & LANCEOLATE FOLIAGE'
    elif t<4.8: stage='STAGE 02 // PETAL GEOMETRY UNFURLING'
    elif t<6.6: stage='STAGE 03 // PIGMENT & SURFACE DETAIL'
    else: stage='STAGE 04 // COMPLETE BLOOM REVEAL'
    cv2.putText(overlay,stage,(28,H-62),cv2.FONT_HERSHEY_SIMPLEX,.35,(194,207,226),1,cv2.LINE_AA)
    cv2.rectangle(overlay,(28,H-43),(W-28,H-39),(52,62,73),-1)
    cv2.rectangle(overlay,(28,H-43),(28+int((W-56)*t/DURATION),H-39),(160,190,231),-1)
    return cv2.addWeighted(overlay,hud*.86,img,1-hud*.86,0)

def render_video(name,cfg):
    path=os.path.join(OUT,f'{name}.mp4')
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{W}x{H}','-pix_fmt','bgr24','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',path]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(TOTAL):
        frame=render_frame(i,cfg)
        proc.stdin.write(frame.tobytes())
        if (i+1)%48==0: print(f'  {name}: {i+1}/{TOTAL} frames',flush=True)
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed while rendering '+name)
    print('Saved',path,os.path.getsize(path),'bytes')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--flower',choices=list(FLOWERS),help='render just one flower')
    parser.add_argument('--preview',action='store_true',help='save preview PNGs rather than videos')
    args=parser.parse_args()
    chosen={args.flower:FLOWERS[args.flower]} if args.flower else FLOWERS
    os.makedirs(OUT, exist_ok=True)
    for name,cfg in chosen.items():
        if args.preview:
            frame=render_frame(int(6.2*FPS),cfg)
            path=os.path.join(OUT,f'{name}_preview.png')
            cv2.imwrite(path,frame);print('Saved',path)
        else: render_video(name,cfg)
if __name__=='__main__': main()
