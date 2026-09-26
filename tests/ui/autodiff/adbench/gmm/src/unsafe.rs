//@ ignore-auxiliary lib.rs
use crate::Wishart;
use std::autodiff::autodiff_reverse;
use std::f64::consts::PI;

//#[cfg(feature = "libm")]
//use libm::lgamma;
//
//#[cfg(not(feature = "libm"))]
mod cmath {
    extern "C" {
        pub fn lgamma(x: f64) -> f64;
    }
}
//#[cfg(not(feature = "libm"))]
#[inline]
fn lgamma(x: f64) -> f64 {
    unsafe { cmath::lgamma(x) }
}

#[no_mangle]
pub extern "C" fn rust_unsafe_dgmm_objective(
    d: i32,
    k: i32,
    n: i32,
    alphas: *const f64,
    dalphas: *mut f64,
    means: *const f64,
    dmeans: *mut f64,
    icf: *const f64,
    dicf: *mut f64,
    x: *const f64,
    wishart: *const Wishart,
    err: *mut f64,
    derr: *mut f64,
) {
    let k = k as usize;
    let n = n as usize;
    let d = d as usize;
    unsafe {
        dgmm_objective(
            d, k, n, alphas, dalphas, means, dmeans, icf, dicf, x, wishart, err, derr,
        );
    }
}

#[no_mangle]
pub extern "C" fn rust_unsafe_gmm_objective(
    d: i32,
    k: i32,
    n: i32,
    alphas: *const f64,
    means: *const f64,
    icf: *const f64,
    x: *const f64,
    wishart: *const Wishart,
    err: *mut f64,
) {
    let k = k as usize;
    let n = n as usize;
    let d = d as usize;
    unsafe {
        gmm_objective(d, k, n, alphas, means, icf, x, wishart, err);
    }
}

#[autodiff_reverse(
    dgmm_objective,
    Const,
    Const,
    Const,
    Duplicated,
    Duplicated,
    Duplicated,
    Const,
    Const,
    DuplicatedOnly
)]
pub unsafe fn gmm_objective(
    d: usize,
    k: usize,
    n: usize,
    alphas: *const f64,
    means: *const f64,
    icf: *const f64,
    x: *const f64,
    wishart: *const Wishart,
    err: *mut f64,
) {
    let constant = -(n as f64) * d as f64 * 0.5 * (2.0 * PI).ln();
    let icf_sz = d * (d + 1) / 2;
    let mut qdiags = vec![0.; d * k];
    let mut sum_qs = vec![0.; k];
    let mut xcentered = vec![0.; d];
    let mut qxcentered = vec![0.; d];
    let mut main_term = vec![0.; k];

    preprocess_qs(d, k, icf, sum_qs.as_mut_ptr(), qdiags.as_mut_ptr());

    let mut slse = 0.;
    for ix in 0..n {
        for ik in 0..k {
            subtract(d, x.add(ix * d), means.add(ik * d), xcentered.as_mut_ptr());
            qtimesx(
                d,
                qdiags.as_mut_ptr().add(ik * d),
                icf.add(ik * icf_sz + d),
                xcentered.as_ptr(),
                qxcentered.as_mut_ptr(),
            );
            main_term[ik] = *alphas.add(ik) + sum_qs[ik] - 0.5 * sqnorm(d, qxcentered.as_ptr());
            //main_term[ik] = alphas[ik] + sum_qs[ik] - 0.5 * sqnorm(d, &Qxcentered[0]);
        }

        slse = slse + log_sum_exp(k, main_term.as_ptr());
    }

    let lse_alphas = log_sum_exp(k, alphas);

    *err = constant + slse - n as f64 * lse_alphas
        + log_wishart_prior(d, k, *wishart, sum_qs.as_ptr(), qdiags.as_ptr(), icf);
}

unsafe fn arr_max(n: usize, x: *const f64) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..n {
        if max < *x.add(i) {
            max = *x.add(i);
        }
    }
    max
}

unsafe fn preprocess_qs(d: usize, k: usize, icf: *const f64, sum_qs: *mut f64, qdiags: *mut f64) {
    let icf_sz = d * (d + 1) / 2;
    for ik in 0..k {
        *sum_qs.add(ik) = 0.;
        for id in 0..d {
            let q = *icf.add(ik * icf_sz + id);
            *sum_qs.add(ik) = *sum_qs.add(ik) + q;
            *qdiags.add(ik * d + id) = q.exp();
        }
    }
}

unsafe fn subtract(d: usize, x: *const f64, y: *const f64, out: *mut f64) {
    for i in 0..d {
        *out.add(i) = *x.add(i) - *y.add(i);
    }
}

unsafe fn qtimesx(d: usize, q_diag: *const f64, ltri: *const f64, x: *const f64, out: *mut f64) {
    for i in 0..d {
        *out.add(i) = *q_diag.add(i) * *x.add(i);
    }

    for i in 0..d {
        let mut lparamsidx = i * (2 * d - i - 1) / 2;
        for j in i + 1..d {
            *out.add(j) = *out.add(j) + *ltri.add(lparamsidx) * *x.add(i);
            lparamsidx += 1;
        }
    }
}

unsafe fn log_sum_exp(n: usize, x: *const f64) -> f64 {
    let mx = arr_max(n, x);
    let mut semx: f64 = 0.0;

    for i in 0..n {
        semx = semx + (*x.add(i) - mx).exp();
    }
    semx.ln() + mx
}

fn log_gamma_distrib(a: f64, p: f64) -> f64 {
    0.25 * p * (p - 1.) * PI.ln()
        + (1..=p as usize)
            .map(|j| lgamma(a + 0.5 * (1. - j as f64)))
            .sum::<f64>()
}

unsafe fn log_wishart_prior(
    p: usize,
    k: usize,
    wishart: Wishart,
    sum_qs: *const f64,
    qdiags: *const f64,
    icf: *const f64,
) -> f64 {
    let n = p + wishart.m as usize + 1;
    let icf_sz = p * (p + 1) / 2;

    let c = n as f64 * p as f64 * (wishart.gamma.ln() - 0.5 * 2f64.ln())
        - log_gamma_distrib(0.5 * n as f64, p as f64);

    let mut out = 0.;

    for ik in 0..k {
        let frobenius =
            sqnorm(p, qdiags.add(ik * p)) + sqnorm(icf_sz - p, icf.add(ik * icf_sz + p));
        out = out + 0.5 * wishart.gamma * wishart.gamma * (frobenius)
            - wishart.m as f64 * *sum_qs.add(ik);
    }

    out - k as f64 * c
}

unsafe fn sqnorm(n: usize, x: *const f64) -> f64 {
    let mut sum = 0.;
    for i in 0..n {
        sum += *x.add(i) * *x.add(i);
    }
    sum
}
