//@ ignore-auxiliary lib.rs
use crate::Wishart;
use std::autodiff::autodiff_reverse;
use std::f64::consts::PI;

//#[cfg(feature = "libm")]
//use libm::lgamma;

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
pub extern "C" fn rust_dgmm_objective(
    d: usize,
    k: usize,
    n: usize,
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
    let alphas = unsafe { std::slice::from_raw_parts(alphas, k) };
    let means = unsafe { std::slice::from_raw_parts(means, k * d) };
    let icf = unsafe { std::slice::from_raw_parts(icf, k * d * (d + 1) / 2) };
    let x = unsafe { std::slice::from_raw_parts(x, n * d) };
    let wishart: Wishart = unsafe { *wishart };
    let mut my_err = unsafe { *err };

    let d_alphas = unsafe { std::slice::from_raw_parts_mut(dalphas, k) };
    let d_means = unsafe { std::slice::from_raw_parts_mut(dmeans, k * d) };
    let d_icf = unsafe { std::slice::from_raw_parts_mut(dicf, k * d * (d + 1) / 2) };
    let mut my_derr = unsafe { *derr };
    let (mut qdiags, mut sum_qs, mut xcentered, mut qxcentered, mut main_term) =
        get_workspace(d, k);
    let (mut bqdiags, mut bsum_qs, mut bxcentered, mut bqxcentered, mut bmain_term) =
        get_workspace(d, k);

    unsafe {
        dgmm_objective(
            d,
            k,
            n,
            alphas,
            d_alphas,
            means,
            d_means,
            icf,
            d_icf,
            x,
            wishart.gamma,
            wishart.m,
            &mut my_err,
            &mut my_derr,
            &mut qdiags,
            &mut bqdiags,
            &mut sum_qs,
            &mut bsum_qs,
            &mut xcentered,
            &mut bxcentered,
            &mut qxcentered,
            &mut bqxcentered,
            &mut main_term,
            &mut bmain_term,
        )
    };

    unsafe { *err = my_err };
    unsafe { *derr = my_derr };
}

#[no_mangle]
pub extern "C" fn rust_gmm_objective(
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
    let alphas = unsafe { std::slice::from_raw_parts(alphas, k) };
    let means = unsafe { std::slice::from_raw_parts(means, k * d) };
    let icf = unsafe { std::slice::from_raw_parts(icf, k * d * (d + 1) / 2) };
    let x = unsafe { std::slice::from_raw_parts(x, n * d) };
    let wishart: Wishart = unsafe { *wishart };
    let mut my_err = unsafe { *err };
    let (mut qdiags, mut sum_qs, mut xcentered, mut qxcentered, mut main_term) =
        get_workspace(d, k);
    gmm_objective(
        d,
        k,
        n,
        alphas,
        means,
        icf,
        x,
        wishart.gamma,
        wishart.m,
        &mut my_err,
        &mut qdiags,
        &mut sum_qs,
        &mut xcentered,
        &mut qxcentered,
        &mut main_term,
    );
    unsafe { *err = my_err };
}

fn get_workspace(d: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let qdiags = vec![0.; d * k];
    let sum_qs = vec![0.; k];
    let xcentered = vec![0.; d];
    let qxcentered = vec![0.; d];
    let main_term = vec![0.; k];
    (qdiags, sum_qs, xcentered, qxcentered, main_term)
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
    Const,
    DuplicatedOnly,
    Duplicated,
    Duplicated,
    Duplicated,
    Duplicated,
    Duplicated
)]
pub fn gmm_objective(
    d: usize,
    k: usize,
    n: usize,
    alphas: &[f64],
    means: &[f64],
    icf: &[f64],
    x: &[f64],
    gamma: f64,
    m: i32,
    err: &mut f64,
    qdiags: &mut [f64],
    sum_qs: &mut [f64],
    xcentered: &mut [f64],
    qxcentered: &mut [f64],
    main_term: &mut [f64],
) {
    let wishart: Wishart = Wishart { gamma, m };
    let constant = -(n as f64) * d as f64 * 0.5 * (2.0 * PI).ln();
    let icf_sz = d * (d + 1) / 2;

    // Let the compiler know sizes so it can eliminate bounds checks
    assert_eq!(qdiags.len(), d * k);
    assert_eq!(sum_qs.len(), k);
    assert_eq!(xcentered.len(), d);
    assert_eq!(qxcentered.len(), d);
    assert_eq!(main_term.len(), k);

    preprocess_qs(d, k, icf, sum_qs, qdiags);

    let mut slse = 0.;
    for ix in 0..n {
        for ik in 0..k {
            subtract(
                d,
                &x[ix as usize * d as usize..],
                &means[ik as usize * d as usize..],
                xcentered,
            );
            qtimesx(
                d,
                &qdiags[ik as usize * d as usize..],
                &icf[ik as usize * icf_sz as usize + d as usize..],
                &*xcentered,
                qxcentered,
            );
            main_term[ik as usize] =
                alphas[ik as usize] + sum_qs[ik as usize] - 0.5 * sqnorm(&*qxcentered);
        }

        slse = slse + log_sum_exp(k, &main_term);
    }

    let lse_alphas = log_sum_exp(k, alphas);

    *err = constant + slse - n as f64 * lse_alphas
        + log_wishart_prior(d, k, wishart, &sum_qs, &*qdiags, icf);
}

fn arr_max(n: usize, x: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for i in 0..n {
        if max < x[i] {
            max = x[i];
        }
    }
    max
}

fn preprocess_qs(d: usize, k: usize, icf: &[f64], sum_qs: &mut [f64], qdiags: &mut [f64]) {
    let icf_sz = d * (d + 1) / 2;
    for ik in 0..k {
        sum_qs[ik as usize] = 0.;
        for id in 0..d {
            let q = icf[ik as usize * icf_sz as usize + id as usize];
            sum_qs[ik as usize] = sum_qs[ik as usize] + q;
            qdiags[ik as usize * d as usize + id as usize] = q.exp();
        }
    }
}
fn subtract(d: usize, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert!(x.len() >= d);
    assert!(y.len() >= d);
    assert!(out.len() >= d);
    for i in 0..d {
        out[i] = x[i] - y[i];
    }
}

fn qtimesx(d: usize, q_diag: &[f64], ltri: &[f64], x: &[f64], out: &mut [f64]) {
    assert!(out.len() >= d);
    assert!(q_diag.len() >= d);
    assert!(x.len() >= d);
    for i in 0..d {
        out[i] = q_diag[i] * x[i];
    }

    for i in 0..d {
        let mut lparamsidx = i * (2 * d - i - 1) / 2;
        for j in i + 1..d {
            out[j] = out[j] + ltri[lparamsidx] * x[i];
            lparamsidx += 1;
        }
    }
}

fn log_sum_exp(n: usize, x: &[f64]) -> f64 {
    let mx = arr_max(n, x);
    let semx: f64 = x.iter().map(|x| (x - mx).exp()).sum();
    semx.ln() + mx
}
fn log_gamma_distrib(a: f64, p: f64) -> f64 {
    0.25 * p * (p - 1.) * PI.ln()
        + (1..=p as usize)
            .map(|j| lgamma(a + 0.5 * (1. - j as f64)))
            .sum::<f64>()
}

fn log_wishart_prior(
    p: usize,
    k: usize,
    wishart: Wishart,
    sum_qs: &[f64],
    qdiags: &[f64],
    icf: &[f64],
) -> f64 {
    let n = p + wishart.m as usize + 1;
    let icf_sz = p * (p + 1) / 2;

    let c = n as f64 * p as f64 * (wishart.gamma.ln() - 0.5 * 2f64.ln())
        - log_gamma_distrib(0.5 * n as f64, p as f64);

    let out = (0..k)
        .map(|ik| {
            let frobenius = sqnorm(&qdiags[ik * p as usize..][..p])
                + sqnorm(&icf[ik * icf_sz as usize + p as usize..][..icf_sz - p]);
            0.5 * wishart.gamma * wishart.gamma * (frobenius)
                - (wishart.m as f64) * sum_qs[ik as usize]
        })
        .sum::<f64>();

    out - k as f64 * c
}

fn sqnorm(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}
