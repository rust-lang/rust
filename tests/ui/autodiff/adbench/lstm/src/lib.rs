//@ compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat --crate-type=staticlib
//@ build-pass
//@ dont-check-compiler-stderr
//@ dont-check-compiler-stdout
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]

pub(crate) mod safe;
pub(crate) mod unsf;
use std::slice;

#[no_mangle]
pub extern "C" fn rust_unsafe_lstm_objective(
    l: i32,
    c: i32,
    b: i32,
    main_params: *const f64,
    extra_params: *const f64,
    state: *mut f64,
    sequence: *const f64,
    loss: *mut f64,
) {
    let l = l as usize;
    let c = c as usize;
    let b = b as usize;
    unsafe {
        unsf::lstm_unsafe_objective(l, c, b, main_params, extra_params, state, sequence, loss);
    }
}
#[no_mangle]
pub extern "C" fn rust_safe_lstm_objective(
    l: i32,
    c: i32,
    b: i32,
    main_params: *const f64,
    extra_params: *const f64,
    state: *mut f64,
    sequence: *const f64,
    loss: *mut f64,
) {
    let l = l as usize;
    let c = c as usize;
    let b = b as usize;
    let (main_params, extra_params, state, sequence) = unsafe {
        (
            slice::from_raw_parts(main_params, 2 * l * 4 * b),
            slice::from_raw_parts(extra_params, 3 * b),
            slice::from_raw_parts_mut(state, 2 * l * b),
            slice::from_raw_parts(sequence, c * b),
        )
    };

    unsafe {
        safe::lstm_objective(
            l,
            c,
            b,
            main_params,
            extra_params,
            state,
            sequence,
            &mut *loss,
        );
    }
}

#[no_mangle]
pub extern "C" fn rust_unsafe_dlstm_objective(
    l: i32,
    c: i32,
    b: i32,
    main_params: *const f64,
    d_main_params: *mut f64,
    extra_params: *const f64,
    d_extra_params: *mut f64,
    state: *mut f64,
    sequence: *const f64,
    res: *mut f64,
    d_res: *mut f64,
) {
    let l = l as usize;
    let c = c as usize;
    let b = b as usize;
    unsafe {
        unsf::d_lstm_unsafe_objective(
            l,
            c,
            b,
            main_params,
            d_main_params,
            extra_params,
            d_extra_params,
            state,
            sequence,
            res,
            d_res,
        );
    }
}
#[no_mangle]
pub extern "C" fn rust_safe_dlstm_objective(
    l: i32,
    c: i32,
    b: i32,
    main_params: *const f64,
    d_main_params: *mut f64,
    extra_params: *const f64,
    d_extra_params: *mut f64,
    state: *mut f64,
    sequence: *const f64,
    res: *mut f64,
    d_res: *mut f64,
) {
    let l = l as usize;
    let c = c as usize;
    let b = b as usize;
    let (main_params, d_main_params, extra_params, d_extra_params, state, sequence) = unsafe {
        (
            slice::from_raw_parts(main_params, 2 * l * 4 * b),
            slice::from_raw_parts_mut(d_main_params, 2 * l * 4 * b),
            slice::from_raw_parts(extra_params, 3 * b),
            slice::from_raw_parts_mut(d_extra_params, 3 * b),
            slice::from_raw_parts_mut(state, 2 * l * b),
            slice::from_raw_parts(sequence, c * b),
        )
    };

    unsafe {
        safe::d_lstm_objective(
            l,
            c,
            b,
            main_params,
            d_main_params,
            extra_params,
            d_extra_params,
            state,
            sequence,
            &mut *res,
            &mut *d_res,
        );
    }
}
