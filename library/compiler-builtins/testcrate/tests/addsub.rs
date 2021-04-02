#![allow(unused_macros)]

use testcrate::*;

macro_rules! sum {
    ($($i:ty, $fn_add:ident, $fn_sub:ident);*;) => {
        $(
            fuzz_2(N, |x: $i, y: $i| {
                let add0 = x.wrapping_add(y);
                let sub0 = x.wrapping_sub(y);
                let add1: $i = $fn_add(x, y);
                let sub1: $i = $fn_sub(x, y);
                if add0 != add1 {
                    panic!(
                        "{}({}, {}): std: {}, builtins: {}",
                        stringify!($fn_add), x, y, add0, add1
                    );
                }
                if sub0 != sub1 {
                    panic!(
                        "{}({}, {}): std: {}, builtins: {}",
                        stringify!($fn_sub), x, y, sub0, sub1
                    );
                }
            });
        )*
    };
}

macro_rules! overflowing_sum {
    ($($i:ty, $fn_add:ident, $fn_sub:ident);*;) => {
        $(
            fuzz_2(N, |x: $i, y: $i| {
                let add0 = x.overflowing_add(y);
                let sub0 = x.overflowing_sub(y);
                let add1: ($i, bool) = $fn_add(x, y);
                let sub1: ($i, bool) = $fn_sub(x, y);
                if add0.0 != add1.0 || add0.1 != add1.1 {
                    panic!(
                        "{}({}, {}): std: {:?}, builtins: {:?}",
                        stringify!($fn_add), x, y, add0, add1
                    );
                }
                if sub0.0 != sub1.0 || sub0.1 != sub1.1 {
                    panic!(
                        "{}({}, {}): std: {:?}, builtins: {:?}",
                        stringify!($fn_sub), x, y, sub0, sub1
                    );
                }
            });
        )*
    };
}

#[test]
fn addsub() {
    use compiler_builtins::int::addsub::{
        __rust_i128_add, __rust_i128_addo, __rust_i128_sub, __rust_i128_subo, __rust_u128_add,
        __rust_u128_addo, __rust_u128_sub, __rust_u128_subo,
    };

    // Integer addition and subtraction is very simple, so 100 fuzzing passes should be plenty.
    sum!(
        u128, __rust_u128_add, __rust_u128_sub;
        i128, __rust_i128_add, __rust_i128_sub;
    );
    overflowing_sum!(
        u128, __rust_u128_addo, __rust_u128_subo;
        i128, __rust_i128_addo, __rust_i128_subo;
    );
}

macro_rules! float_sum {
    ($($f:ty, $fn_add:ident, $fn_sub:ident);*;) => {
        $(
            fuzz_float_2(N, |x: $f, y: $f| {
                let add0 = x + y;
                let sub0 = x - y;
                let add1: $f = $fn_add(x, y);
                let sub1: $f = $fn_sub(x, y);
                if !Float::eq_repr(add0, add1) {
                    panic!(
                        "{}({}, {}): std: {}, builtins: {}",
                        stringify!($fn_add), x, y, add0, add1
                    );
                }
                if !Float::eq_repr(sub0, sub1) {
                    panic!(
                        "{}({}, {}): std: {}, builtins: {}",
                        stringify!($fn_sub), x, y, sub0, sub1
                    );
                }
            });
        )*
    };
}

#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
#[test]
fn float_addsub() {
    use compiler_builtins::float::{
        add::{__adddf3, __addsf3},
        sub::{__subdf3, __subsf3},
        Float,
    };

    float_sum!(
        f32, __addsf3, __subsf3;
        f64, __adddf3, __subdf3;
    );
}

#[cfg(target_arch = "arm")]
#[test]
fn float_addsub_arm() {
    use compiler_builtins::float::{
        add::{__adddf3vfp, __addsf3vfp},
        sub::{__subdf3vfp, __subsf3vfp},
        Float,
    };

    float_sum!(
        f32, __addsf3vfp, __subsf3vfp;
        f64, __adddf3vfp, __subdf3vfp;
    );
}
