//@ ignore-auxiliary lib.rs
use crate::compute_zach_weight_error;
use crate::BA_NCAMPARAMS;
use std::autodiff::autodiff_reverse;
use std::convert::TryInto;

fn sqsum(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn radial_distort(rad_params: &[f64], proj: &mut [f64]) {
    let rsq = sqsum(proj);
    let l = 1. + rad_params[0] * rsq + rad_params[1] * rsq * rsq;
    proj[0] = proj[0] * l;
    proj[1] = proj[1] * l;
}

fn rodrigues_rotate_point(rot: &[f64; 3], pt: &[f64; 3], rotated_pt: &mut [f64; 3]) {
    let sqtheta = sqsum(rot);
    if sqtheta != 0. {
        let theta = sqtheta.sqrt();
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let theta_inverse = 1. / theta;
        let mut w = [0.; 3];
        for i in 0..3 {
            w[i] = rot[i] * theta_inverse;
        }
        let w_cross_pt = cross(&w, &pt);
        let tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (1. - costheta);
        for i in 0..3 {
            rotated_pt[i] = pt[i] * costheta + w_cross_pt[i] * sintheta + w[i] * tmp;
        }
    } else {
        let rot_cross_pt = cross(&rot, &pt);
        for i in 0..3 {
            rotated_pt[i] = pt[i] + rot_cross_pt[i];
        }
    }
}

fn project(cam: &[f64; 11], X: &[f64; 3], proj: &mut [f64; 2]) {
    let C = &cam[3..6];
    let mut Xo = [0.; 3];
    let mut Xcam = [0.; 3];

    Xo[0] = X[0] - C[0];
    Xo[1] = X[1] - C[1];
    Xo[2] = X[2] - C[2];

    rodrigues_rotate_point(cam.first_chunk::<3>().unwrap(), &Xo, &mut Xcam);

    proj[0] = Xcam[0] / Xcam[2];
    proj[1] = Xcam[1] / Xcam[2];

    radial_distort(&cam[9..], proj);

    proj[0] = proj[0] * cam[6] + cam[7];
    proj[1] = proj[1] * cam[6] + cam[8];
}

#[no_mangle]
pub extern "C" fn rust_dcompute_reproj_error(
    cam: *const [f64; 11],
    dcam: *mut [f64; 11],
    x: *const [f64; 3],
    dx: *mut [f64; 3],
    w: *const [f64; 1],
    wb: *mut [f64; 1],
    feat: *const [f64; 2],
    err: *mut [f64; 2],
    derr: *mut [f64; 2],
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
pub fn compute_reproj_error(
    cam: *const [f64; 11],
    x: *const [f64; 3],
    w: *const [f64; 1],
    feat: *const [f64; 2],
    err: *mut [f64; 2],
) {
    let cam = unsafe { &*cam };
    let w = unsafe { *(*w).get_unchecked(0) };
    let x = unsafe { &*x };
    let feat = unsafe { &*feat };
    let err = unsafe { &mut *err };
    let mut proj = [0.; 2];
    project(cam, x, &mut proj);
    err[0] = w * (proj[0] - feat[0]);
    err[1] = w * (proj[1] - feat[1]);
}

// n number of cameras
// m number of points
// p number of observations
// cams: 11*n cameras in format [r1 r2 r3 C1 C2 C3 f u0 v0 k1 k2]
//            r1, r2, r3 are angle - axis rotation parameters(Rodrigues)
//            [C1 C2 C3]' is the camera center
//            f is the focal length in pixels
//            [u0 v0]' is the principal point
//            k1, k2 are radial distortion parameters
// X: 3*m points
// obs: 2*p observations (pairs cameraIdx, pointIdx)
// feats: 2*p features (x,y coordinates corresponding to observations)
// reproj_err: 2*p errors of observations
// w_err: p weight "error" terms
fn rust_ba_objective(
    n: usize,
    m: usize,
    p: usize,
    cams: &[f64],
    x: &[f64],
    w: &[f64],
    obs: &[i32],
    feats: &[f64],
    reproj_err: &mut [f64],
    w_err: &mut [f64],
) {
    assert_eq!(cams.len(), n * 11);
    assert_eq!(x.len(), m * 3);
    assert_eq!(w.len(), p);
    assert_eq!(obs.len(), p * 2);
    assert_eq!(feats.len(), p * 2);
    assert_eq!(reproj_err.len(), p * 2);
    assert_eq!(w_err.len(), p);

    for i in 0..p {
        let cam_idx = obs[i * 2 + 0] as usize;
        let pt_idx = obs[i * 2 + 1] as usize;
        let start = cam_idx * BA_NCAMPARAMS;
        let cam: &[f64; 11] = unsafe {
            cams[start..]
                .get_unchecked(..11)
                .try_into()
                .unwrap_unchecked()
        };
        let x: &[f64; 3] = unsafe {
            x[pt_idx * 3..]
                .get_unchecked(..3)
                .try_into()
                .unwrap_unchecked()
        };
        let w: &[f64; 1] = unsafe { w[i..].get_unchecked(..1).try_into().unwrap_unchecked() };
        let feat: &[f64; 2] = unsafe {
            feats[i * 2..]
                .get_unchecked(..2)
                .try_into()
                .unwrap_unchecked()
        };
        let reproj_err: &mut [f64; 2] = unsafe {
            reproj_err[i * 2..]
                .get_unchecked_mut(..2)
                .try_into()
                .unwrap_unchecked()
        };
        compute_reproj_error(cam, x, w, feat, reproj_err);
    }

    for i in 0..p {
        let w_err: &mut f64 = unsafe { w_err.get_unchecked_mut(i) };
        compute_zach_weight_error(w[i..].as_ptr(), w_err as *mut f64);
    }
}

#[no_mangle]
extern "C" fn rust2_ba_objective(
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
    let cams = unsafe { std::slice::from_raw_parts(cams, n * 11) };
    let x = unsafe { std::slice::from_raw_parts(x, m * 3) };
    let w = unsafe { std::slice::from_raw_parts(w, p) };
    let obs = unsafe { std::slice::from_raw_parts(obs, p * 2) };
    let feats = unsafe { std::slice::from_raw_parts(feats, p * 2) };
    let reproj_err = unsafe { std::slice::from_raw_parts_mut(reproj_err, p * 2) };
    let w_err = unsafe { std::slice::from_raw_parts_mut(w_err, p) };
    rust_ba_objective(n, m, p, cams, x, w, obs, feats, reproj_err, w_err);
}
