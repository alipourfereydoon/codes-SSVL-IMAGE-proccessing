#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ssvl_image_processor import ImageDataset, EdgeDetectionKernel, QuantumDotSimulator, SSVLVerifier, MetricsEvaluator


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', help='optional input image; omitted means geometric phantom')
    ap.add_argument('--size',type=int,default=256,help='phantom width/height (default: 256)')
    ap.add_argument('--output',default='results',help='output directory (default: results)')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--jitter',type=float,default=.10,help='charge trajectory jitter (0..1)')
    ap.add_argument('--parameter-variance',type=float,default=.04)
    ap.add_argument('--bit-flip',type=float,default=.015)
    ap.add_argument('--delta-t-over-t',type=float,default=.15,help='nominal normalized trajectory deviation')
    ap.add_argument('--spin-noise',type=float,default=.015)
    ap.add_argument('--edge-threshold',type=float,default=None,help='absolute grayscale gradient threshold')
    args=ap.parse_args()
    if not 0<=args.jitter<=1 or not 0<=args.bit_flip<=1 or args.parameter_variance<0:
        ap.error('jitter and bit-flip must be in [0,1]; parameter-variance must be nonnegative')
    rng=np.random.default_rng(args.seed)
    image,truth=ImageDataset.load(args.image,args.size,args.edge_threshold)
    clean=EdgeDetectionKernel.detect(image,args.edge_threshold)

    classical=clean.astype(float)
    qd,qscore=QuantumDotSimulator(args.jitter,args.parameter_variance,args.bit_flip).process(clean,rng)
    mitigated,sscore,fs=SSVLVerifier().verify(qd,qscore,rng,args.delta_t_over_t,args.spin_noise)
    methods={'Classical CMOS':(clean,classical),'Charge-Only Quantum Dot':(qd,qscore),'SSVL-Enhanced Quantum Dot':(mitigated,sscore)}
    results={name:MetricsEvaluator.evaluate(truth,pred,score) for name,(pred,score) in methods.items()}
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    with open(out/'metrics.json','w',encoding='utf-8') as f: json.dump(results,f,indent=2,ensure_ascii=False,allow_nan=True)
    with open(out/'metrics.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['method']+list(next(iter(results.values())).keys())); writer.writeheader()
        for name,row in results.items(): writer.writerow({'method':name,**row})
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    axes[0].imshow(image,cmap='gray',vmin=0,vmax=1); axes[0].set_title('Input')
    axes[1].imshow(truth,cmap='gray'); axes[1].set_title('Reference edges')
    for ax,(name,(pred,_)) in zip(axes[2:],list(methods.items())[1:]):
        ax.imshow(pred,cmap='gray'); ax.set_title(name.replace(' ','\n'))
    for ax in axes: ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    # Fidelity summary for inspecting the auxiliary-spin model.
    with open(out/'ssvl_summary.json','w',encoding='utf-8') as f:
        json.dump({'mean_spin_fidelity':float(np.mean(fs)),'delta_t_over_t_nominal':args.delta_t_over_t,
                   'formula':'F_S = clip(1 - 0.33*(deltaT/T)^2 + Gaussian noise, 0, 1)',
                   'seed':args.seed},f,indent=2)
    print(json.dumps(results,indent=2,allow_nan=True)); print('Saved results to',out.resolve())
if __name__=='__main__': main()
