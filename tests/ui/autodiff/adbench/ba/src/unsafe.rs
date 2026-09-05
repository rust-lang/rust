//@ ignore-auxiliary lib.rs
use crate::compute_zach_weight_error;
use crate::BA_NCAMPARAMS;
use std::autodiff::autodiff_reverse;
use std::convert::TryInto;

unsafe fn sqsum(x: *const f64, n: usize) -> f64 {
    let mut sum = 0.;
    for i in 0..n {
        let v = unsafe { *x.add(i) };
        sum += v * v;
    }
    sum
}

#[inline]
unsafe fn cross(a: *const f64, b: *const f64, out: *mut f64) {
    *out.add(0) = *a.add(1) * *b.add(2) - *a.add(2) * *b.add(1);
    *out.add(1) = *a.add(2) * *b.add(0) - *a.add(0) * *b.add(2);
    *out.add(2) = *a.add(0) * *b.add(1) - *a.add(1) * *b.add(0);
}

unsafe fn radial_distort(rad_params: *const f64, proj: *mut f64) {
    let rsq = sqsum(proj, 2);
    let l = 1. + *rad_params.add(0) * rsq + *rad_params.add(1) * rsq * rsq;
    *proj.add(0) = *proj.add(0) * l;
    *proj.add(1) = *proj.add(1) * l;
}

unsafe fn rodrigues_rotate_point(rot: *const f64, pt: *const f64, rotated_pt: *mut f64) {
    let sqtheta = sqsum(rot, 3);
    if sqtheta != 0. {
        let theta = sqtheta.sqrt();
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let theta_inverse = 1. / theta;
        let mut w = [0.; 3];
        for i in 0..3 {
            w[i] = *rot.add(i) * theta_inverse;
        }
        let mut w_cross_pt = [0.; 3];
        cross(w.as_ptr(), pt, w_cross_pt.as_mut_ptr());
        let tmp = (w[0] * *pt.add(0) + w[1] * *pt.add(1) + w[2] * *pt.add(2)) * (1. - costheta);
        for i in 0..3 {
            *rotated_pt.add(i) = *pt.add(i) * costheta + w_cross_pt[i] * sintheta + w[i] * tmp;
        }
    } else {
        let mut rot_cross_pt = [0.; 3];
        cross(rot, pt, rot_cross_pt.as_mut_ptr());
        for i in 0..3 {
            *rotated_pt.add(i) = *pt.add(i) + rot_cross_pt[i];
        }
    }
}

unsafe fn project(cam: *const f64, X: *const f64, proj: *mut f64) {
    let C = cam.add(3);
    let mut Xo = [0.; 3];
    let mut Xcam = [0.; 3];

    Xo[0] = *X.add(0) - *C.add(0);
    Xo[1] = *X.add(1) - *C.add(1);
    Xo[2] = *X.add(2) - *C.add(2);

    rodrigues_rotate_point(cam, Xo.as_ptr(), Xcam.as_mut_ptr());

    *proj.add(0) = Xcam[0] / Xcam[2];
    *proj.add(1) = Xcam[1] / Xcam[2];

    radial_distort(cam.add(9), proj);
    *proj.add(0) = *proj.add(0) * *cam.add(6) + *cam.add(7);
    *proj.add(1) = *proj.add(1) * *cam.add(6) + *cam.add(8);
}

#[no_mangle]
pub unsafe extern "C" fn rust_unsafe_dcompute_reproj_error(
    cam: *const f64,
    dcam: *mut f64,
    x: *const f64,
    dx: *mut f64,
    w: *const f64,
    wb: *mut f64,
    feat: *const f64,
    err: *mut f64,
    derr: *mut f64,
) {
    unsafe { dcompute_reproj_error(cam, dcam, x, dx, w, wb, feat, err, derr) };
}

#[autodiff_reverse(
    dcompute_reproj_error,
    Duplicated,
    Duplicated,
    Duplicated,
    Const,
    DuplicatedOnly
)]
pub unsafe fn compute_reproj_error(
    cam: *const f64,
    x: *const f64,
    w: *const f64,
    feat: *const f64,
    err: *mut f64,
) {
    let mut proj = [0.; 2];
    project(cam, x, proj.as_mut_ptr());
    *err.add(0) = *w * (proj[0] - *feat.add(0));
    *err.add(1) = *w * (proj[1] - *feat.add(1));
}

#[no_mangle]
unsafe extern "C" fn rust2_unsafe_ba_objective(
    n: i32,
    m: i32,
    p: i32,
    cams: *const f64,
    x: *const f64,
    w: *const f64,
    obs: *const i32,
    feats: *const f64,
    reproj_err: *mut f64,
    w_err: *mut f64,
) {
    let n = n as usize;
    let m = m as usize;
    let p = p as usize;
    for i in 0..p {
        let cam_idx = *obs.add(i * 2 + 0) as usize;
        let pt_idx = *obs.add(i * 2 + 1) as usize;
        let start = cam_idx * BA_NCAMPARAMS;

        let cam: *const f64 = cams.add(start);
        let x: *const f64 = x.add(pt_idx * 3);
        let w: *const f64 = w.add(i);
        let feat: *const f64 = feats.add(i * 2);
        let reproj_err: *mut f64 = reproj_err.add(i * 2);
        compute_reproj_error(cam, x, w, feat, reproj_err);
    }

    for i in 0..p {
        compute_zach_weight_error(w.add(i), w_err.add(i));
    }
}
