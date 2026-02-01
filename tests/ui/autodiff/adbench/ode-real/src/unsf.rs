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
unsafe fn init_brusselator(u: *mut f64, v: *mut f64) {
    for i in 0..N {
        for j in 0..N {
            let x = range(xmin, xmax, i, N);
            let y = range(ymin, ymax, j, N);
            *u.add(N * i + j) = 22.0 * (y * (1.0 - y)) * (y * (1.0 - y)).sqrt();
            *v.add(N * i + j) = 27.0 * (x * (1.0 - x)) * (x * (1.0 - x)).sqrt();
        }
    }
}

#[no_mangle]
#[autodiff_reverse(
    dbrusselator_2d_loop_unsf,
    Duplicated,
    Duplicated,
    Duplicated,
    Duplicated,
    Duplicated,
    Const
)]
pub unsafe fn brusselator_2d_loop_unsf(
    d_u: *mut f64,
    d_v: *mut f64,
    u: *const f64,
    v: *const f64,
    p: *const f64,
    t: f64,
) {
    let A = *p.add(0);
    let B = *p.add(1);
    let alpha = *p.add(2);
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
            let u2v = *u.add(N * i + j) * *u.add(N * i + j) * *v.add(N * i + j);
            *d_u.add(N * i + j) = alpha
                * (*u.add(N * im1 + j)
                    + *u.add(N * ip1 + j)
                    + *u.add(N * i + jp1)
                    + *u.add(N * i + jm1)
                    - 4. * *u.add(N * i + j))
                + B
                + u2v
                - (A + 1.) * *u.add(N * i + j)
                + brusselator_f(x, y, t);
            *d_v.add(N * i + j) = alpha
                * (*v.add(N * im1 + j)
                    + *v.add(N * ip1 + j)
                    + *v.add(N * i + jp1)
                    + *v.add(N * i + jm1)
                    - 4. * *v.add(N * i + j))
                + A * *u.add(N * i + j)
                - u2v;
        }
    }
}

type StateType = [f64; 2 * N * N];

pub unsafe fn lorenz(x: *const StateType, dxdt: *mut StateType, t: f64) {
    let p = [3.4, 1., 10.];
    let x = x as *const f64;
    let dxdt = dxdt as *mut f64;
    let dxdt1: *mut f64 = dxdt as *mut f64;
    let dxdt2: *mut f64 = unsafe { dxdt.add(N * N) } as *mut f64;
    //let (tmp1, tmp2) = dxdt.split_at_mut(N * N);
    //let mut dxdt1: [f64; N * N] = tmp1.try_into().unwrap();
    //let mut dxdt2: [f64; N * N] = tmp2.try_into().unwrap();
    let u: *const f64 = x as *const f64;
    let v: *const f64 = unsafe { x.add(N * N) } as *const f64;
    //let (tmp1, tmp2) = x.split_at(N * N);
    //let u: [f64; N * N] = tmp1.try_into().unwrap();
    //let v: [f64; N * N] = tmp2.try_into().unwrap();
    unsafe {
        brusselator_2d_loop_unsf(
            dxdt1 as *mut f64,
            dxdt2 as *mut f64,
            u as *const f64,
            v as *const f64,
            p.as_ptr(),
            t,
        )
    };
}
