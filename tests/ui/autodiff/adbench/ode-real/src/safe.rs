//@ ignore-auxiliary lib.rs
use std::autodiff::autodiff_reverse;
use std::convert::TryInto;

const N: usize = 32;
const xmin: f64 = 0.;
const xmax: f64 = 1.;
const ymin: f64 = 0.;
const ymax: f64 = 1.;

#[inline(always)]
fn range(min: f64, max: f64, i: usize, N_var: usize) -> f64 {
    (max - min) / (N_var as f64 - 1.) * i as f64 + min
}

fn brusselator_f(x: f64, y: f64, t: f64) -> f64 {
    let eq1 = (x - 0.3) * (x - 0.3) + (y - 0.6) * (y - 0.6) <= 0.1 * 0.1;
    let eq2 = t >= 1.1;
    if eq1 && eq2 {
        5.0
    } else {
        0.0
    }
}

#[expect(unused)]
fn init_brusselator(u: &mut [f64], v: &mut [f64]) {
    assert!(u.len() == N * N);
    assert!(v.len() == N * N);
    for i in 0..N {
        for j in 0..N {
            let x = range(xmin, xmax, i, N);
            let y = range(ymin, ymax, j, N);
            u[N * i + j] = 22.0 * (y * (1.0 - y)) * (y * (1.0 - y)).sqrt();
            v[N * i + j] = 27.0 * (x * (1.0 - x)) * (x * (1.0 - x)).sqrt();
        }
    }
}

#[no_mangle]
#[autodiff_reverse(
    dbrusselator_2d_loop,
    Duplicated,
    Duplicated,
    Duplicated,
    Duplicated,
    Duplicated,
    Const
)]
pub fn brusselator_2d_loop(
    d_u: &mut [f64; N * N],
    d_v: &mut [f64; N * N],
    u: &[f64; N * N],
    v: &[f64; N * N],
    p: &[f64; 3],
    t: f64,
) {
    let A = p[0];
    let B = p[1];
    let alpha = p[2];
    let dx = 1. / (N - 1) as f64;
    let alpha = alpha / (dx * dx);
    for i in 0..N {
        for j in 0..N {
            let x = range(xmin, xmax, i, N);
            let y = range(ymin, ymax, j, N);
            let ip1 = if i == N - 1 { i } else { i + 1 };
            let im1 = if i == 0 { i } else { i - 1 };
            let jp1 = if j == N - 1 { j } else { j + 1 };
            let jm1 = if j == 0 { j } else { j - 1 };
            let u2v = u[N * i + j] * u[N * i + j] * v[N * i + j];
            d_u[N * i + j] = alpha
                * (u[N * im1 + j] + u[N * ip1 + j] + u[N * i + jp1] + u[N * i + jm1]
                    - 4. * u[N * i + j])
                + B
                + u2v
                - (A + 1.) * u[N * i + j]
                + brusselator_f(x, y, t);
            d_v[N * i + j] = alpha
                * (v[N * im1 + j] + v[N * ip1 + j] + v[N * i + jp1] + v[N * i + jm1]
                    - 4. * v[N * i + j])
                + A * u[N * i + j]
                - u2v;
        }
    }
}

pub type StateType = [f64; 2 * N * N];

pub fn lorenz(x: &StateType, dxdt: &mut StateType, t: f64) {
    let p = [3.4, 1., 10.];
    let (tmp1, tmp2) = dxdt.split_at_mut(N * N);
    let mut dxdt1: [f64; N * N] = tmp1.try_into().unwrap();
    let mut dxdt2: [f64; N * N] = tmp2.try_into().unwrap();
    let (tmp1, tmp2) = x.split_at(N * N);
    let u: [f64; N * N] = tmp1.try_into().unwrap();
    let v: [f64; N * N] = tmp2.try_into().unwrap();
    brusselator_2d_loop(&mut dxdt1, &mut dxdt2, &u, &v, &p, t);
}
