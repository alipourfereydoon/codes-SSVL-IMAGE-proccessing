
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from scipy.stats import rankdata


class ImageDataset:
    @staticmethod
    def phantom(size=256):
        if size < 32: raise ValueError("size must be at least 32")
        y, x = np.mgrid[:size, :size]
        im = np.zeros((size, size), dtype=float) + 0.08
        im[((x-size*.32)**2 + (y-size*.36)**2) < (size*.15)**2] = .88
        im[(x>size*.56)&(x<size*.86)&(y>size*.19)&(y<size*.43)] = .58
        tri = (y>size*.55)&(y<size*.84)&(x>size*.16)&(x<size*.49)&(y > -0.75*(x-size*.16)+size*.84)
        im[tri] = .98
        im[(np.abs(y-(size*.70 + .10*(x-size*.55))) < max(1,size*.012))&(x>size*.53)&(x<size*.88)] = .42
        return np.clip(im,0,1)

    @staticmethod
    def edges(image, threshold=None):
        return EdgeDetectionKernel.detect(image, threshold=threshold)

    @classmethod
    def load(cls, path=None, size=256, threshold=None):
        if path is None:
            image=cls.phantom(size)
        else:
            image=np.asarray(Image.open(path).convert('L'), dtype=np.float64)/255.0
            if image.shape[0] < 3 or image.shape[1] < 3: raise ValueError("image dimensions must be >= 3")
        truth=cls.edges(image, threshold)
        return image, truth


class EdgeDetectionKernel:
    @staticmethod
    def detect(image, threshold=None):
        a=np.asarray(image,dtype=float)
        gx=np.zeros_like(a); gy=np.zeros_like(a)
        gx[:,1:]=np.abs(a[:,1:]-a[:,:-1])
        gy[1:,:]=np.abs(a[1:,:]-a[:-1,:])
        if threshold is None:
            positive=np.r_[gx[gx>0],gy[gy>0]]
            t=max(0.06,float(np.quantile(positive,.65)) if positive.size else .06)
        else: t=float(threshold)
        ex=gx>=t; ey=gy>=t
        return np.logical_or(np.logical_xor(ex,ey), np.logical_and(ex,ey))


@dataclass
class QuantumDotSimulator:
    jitter: float = 0.10
    parameter_variance: float = 0.04
    bit_flip: float = 0.015

    def process(self, clean_edges, rng):
        """Stochastic charge readout: local threshold variation, missed/extra transitions."""
        truth=np.asarray(clean_edges,dtype=bool)
        h,w=truth.shape
        field=gaussian_filter(rng.normal(0,self.parameter_variance,(h,w)),sigma=2)
        p_fp=np.clip(self.bit_flip + self.jitter*.16 + np.maximum(field,0)*.2,0,.45)
        p_fn=np.clip(self.bit_flip + self.jitter*.55 + np.maximum(-field,0)*.2,0,.60)
        u=rng.random((h,w))
        return np.where(truth, u>=p_fn, u<p_fp), np.where(truth,1-p_fn,p_fp)


class SSVLVerifier:
    """Phenomenological auxiliary-spin verification and consensus error mitigation."""
    @staticmethod
    def fidelity(delta_t_over_t, rng, noise=0.015):
        d=np.asarray(delta_t_over_t,dtype=float)
        return np.clip(1.0-0.33*d*d+rng.normal(0,noise,size=d.shape),0,1)

    def verify(self, charge, confidence, rng, delta_t_over_t=.15, noise=.015):
        charge=np.asarray(charge,dtype=bool)
        delta=np.clip(rng.normal(delta_t_over_t,.045,charge.shape),0,1.5)
        fs=self.fidelity(delta,rng,noise)
        local=np.quantile(fs,.28)
        spin_vote=fs >= local
        support=gaussian_filter(spin_vote.astype(float),sigma=1.0)
        score=.58*np.asarray(confidence)+.42*support
        threshold=np.clip(.50 + .10*(1-fs),.48,.62)
        mitigated=(charge & (score>=threshold)) | ((~charge)&(support>.68)&(score>threshold+.08))
        return mitigated, np.clip(.55*confidence+.45*fs,0,1), fs


class MetricsEvaluator:
    @staticmethod
    def evaluate(truth, pred, score=None):
        t=np.asarray(truth,dtype=bool); p=np.asarray(pred,dtype=bool)
        tp=np.count_nonzero(t&p); tn=np.count_nonzero(~t&~p)
        fp=np.count_nonzero(~t&p); fn=np.count_nonzero(t&~p)
        fpr=fp/max(fp+tn,1); fnr=fn/max(fn+tp,1)
        mse=float(np.mean((t.astype(float)-p.astype(float))**2))
        psnr=float('inf') if mse==0 else float(10*np.log10(1/mse))
        ssim=float(structural_similarity(t.astype(float),p.astype(float),data_range=1.0))
        if score is None: score=p.astype(float)
        try:
            flat=t.ravel(); values=np.asarray(score).ravel(); npos=int(flat.sum()); nneg=len(flat)-npos
            auc=float((rankdata(values)[flat].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else float('nan')
        except Exception: auc=float('nan')
        return {'PSNR_dB':psnr,'SSIM':ssim,'FPR':float(fpr),'FNR':float(fnr),'AUC':auc,
                'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn)}
