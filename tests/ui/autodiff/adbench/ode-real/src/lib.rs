//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat --crate-type=staticlib
//@ build-pass
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]
#![feature(iter_next_chunk)]
#![feature(array_ptr_get)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod safe;
pub mod unsf;

type StateType = [f64; 2 * N * N];

const N: usize = 32;

#[no_mangle]
pub extern "C" fn rust_lorenz_unsf(x: *const StateType, dxdt: *mut StateType, t: f64) {
    let x: &StateType = unsafe { &*x };
    let dxdt: &mut StateType = unsafe { &mut *dxdt };
    unsafe { unsf::lorenz(x, dxdt, t) };
}

#[no_mangle]
pub extern "C" fn rust_lorenz_safe(x: *const StateType, dxdt: *mut StateType, t: f64) {
    let x: &StateType = unsafe { &*x };
    let dxdt: &mut StateType = unsafe { &mut *dxdt };
    safe::lorenz(x, dxdt, t);
}

#[no_mangle]
pub extern "C" fn rust_dbrusselator_2d_loop_unsf(
    adjoint: *mut StateType,
    x: *const StateType,
    dx: *mut StateType,
    p: *const [f64; 3],
    dp: *mut [f64; 3],
    t: f64,
) {
    let mut null1 = [0.; 1 * N * N];
    let mut null2 = [0.; 1 * N * N];
    let dx1: *mut f64 = dx.as_mut_ptr();
    let dx2: *mut f64 = unsafe { dx.as_mut_ptr().add(N * N) };
    let dadj1: *mut f64 = adjoint.as_mut_ptr();
    let dadj2: *mut f64 = unsafe { adjoint.as_mut_ptr().add(N * N) };
    let x1: *const f64 = x.as_ptr();
    let x2: *const f64 = unsafe { x.as_ptr().add(N * N) };

    unsafe {
        unsf::dbrusselator_2d_loop_unsf(
            null1.as_mut_ptr(),
            dadj1,
            null2.as_mut_ptr(),
            dadj2,
            x1,
            dx1,
            x2,
            dx2,
            p as *mut f64,
            dp as *mut f64,
            t,
        )
    };
}

#[no_mangle]
pub extern "C" fn rust_dbrusselator_2d_loop_safe(
    adjoint: *mut StateType,
    x: *const StateType,
    dx: *mut StateType,
    p: *const [f64; 3],
    dp: *mut [f64; 3],
    t: f64,
) {
    let x: &StateType = unsafe { &*x };
    let dx: &mut StateType = unsafe { &mut *dx };
    let adjoint: &mut StateType = unsafe { &mut *adjoint };

    let p: &[f64; 3] = unsafe { &*p };
    let dp: &mut [f64; 3] = unsafe { &mut *dp };

    assert!(p[0] == 3.4);
    assert!(p[1] == 1.);
    assert!(p[2] == 10.);
    assert!(t == 2.1);

    //let mut x1 = [0.; 2 * N * N];
    //let mut dx1 = [0.; 2 *N * N];
    //let (tmp1, tmp2) = x1.split_at_mut(N * N);
    //let mut x1: [f64; N * N] = tmp1.try_into().unwrap();
    //let mut x2: [f64; N * N] = tmp2.try_into().unwrap();
    //init_brusselator(&mut x1, &mut x2);
    //for i in 0..N*N {
    //    let tmp = (x1[i] - x[i]).abs();
    //    if (tmp / x[i] > 1e-5) {
    //        dbg!(tmp);
    //        dbg!(tmp / x[i]);
    //        dbg!(i);
    //        dbg!(x1[i]);
    //        dbg!(x[i]);
    //        println!("x1[{}] = {} != x[{}] = {}", i, x1[i], i, x[i]);
    //        panic!();
    //    }
    //}

    // Alternative ways to split the inputs
    //let [ mut dx1, mut dx2]: [[f64; N*N]; 2] =
    //unsafe { *std::mem::transmute::<*mut StateType, &mut [[f64; N*N]; 2]>(dx) };
    //let [dx1, dx2]: &mut [[f64; N*N];2] =
    //unsafe { dx.cast::<[[f64; N*N]; 2]>().as_mut().unwrap() };

    // https://discord.com/channels/273534239310479360/273541522815713281/1236945105601040446
    let ([dx1, dx2], []): (&mut [[f64; N * N]], &mut [f64]) = dx.as_chunks_mut() else {
        unreachable!()
    };
    let ([dadj1, dadj2], []): (&mut [[f64; N * N]], &mut [f64]) = adjoint.as_chunks_mut() else {
        unreachable!()
    };
    let ([x1, x2], []): (&[[f64; N * N]], &[f64]) = x.as_chunks() else {
        unreachable!()
    };

    let mut null1 = [0.; 1 * N * N];
    let mut null2 = [0.; 1 * N * N];
    safe::dbrusselator_2d_loop(
        &mut null1, dadj1, &mut null2, dadj2, x1, dx1, x2, dx2, p, dp, t,
    );
    return;
}
