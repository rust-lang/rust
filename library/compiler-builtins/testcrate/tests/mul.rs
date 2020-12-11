#![allow(unused_macros)]

use testcrate::*;

macro_rules! mul {
    ($($i:ty, $fn:ident);*;) => {
        $(
            fuzz_2(N, |x: $i, y: $i| {
                let mul0 = x.wrapping_mul(y);
                let mul1: $i = $fn(x, y);
                if mul0 != mul1 {
                    panic!(
                        "{}({}, {}): std: {}, builtins: {}",
                        stringify!($fn), x, y, mul0, mul1
                    );
                }
            });
        )*
    };
}

#[test]
fn mul() {
    use compiler_builtins::int::mul::{__muldi3, __multi3};

    mul!(
        u64, __muldi3;
        i128, __multi3;
    );
}

macro_rules! overflowing_mul {
    ($($i:ty, $fn:ident);*;) => {
        $(
            fuzz_2(N, |x: $i, y: $i| {
                let (mul0, o0) = x.overflowing_mul(y);
                let mut o1 = 0i32;
                let mul1: $i = $fn(x, y, &mut o1);
                let o1 = o1 != 0;
                if mul0 != mul1 || o0 != o1 {
                    panic!(
                        "{}({}, {}): std: ({}, {}), builtins: ({}, {})",
                        stringify!($fn), x, y, mul0, o0, mul1, o1
                    );
                }
            });
        )*
    };
}

#[test]
fn overflowing_mul() {
    use compiler_builtins::int::mul::{
        __mulodi4, __mulosi4, __muloti4, __rust_i128_mulo, __rust_u128_mulo,
    };

    overflowing_mul!(
        i32, __mulosi4;
        i64, __mulodi4;
        i128, __muloti4;
    );
    fuzz_2(N, |x: u128, y: u128| {
        let (mul0, o0) = x.overflowing_mul(y);
        let (mul1, o1) = __rust_u128_mulo(x, y);
        if mul0 != mul1 || o0 != o1 {
            panic!(
                "__rust_u128_mulo({}, {}): std: ({}, {}), builtins: ({}, {})",
                x, y, mul0, o0, mul1, o1
            );
        }
        let x = x as i128;
        let y = y as i128;
        let (mul0, o0) = x.overflowing_mul(y);
        let (mul1, o1) = __rust_i128_mulo(x, y);
        if mul0 != mul1 || o0 != o1 {
            panic!(
                "__rust_i128_mulo({}, {}): std: ({}, {}), builtins: ({}, {})",
                x, y, mul0, o0, mul1, o1
            );
        }
    });
}

macro_rules! float_mul {
    ($($f:ty, $fn:ident);*;) => {
        $(
            fuzz_float_2(N, |x: $f, y: $f| {
                let mul0 = x * y;
                let mul1: $f = $fn(x, y);
                // multiplication of subnormals is not currently handled
                if !(Float::is_subnormal(mul0) || Float::is_subnormal(mul1)) {
                    if !Float::eq_repr(mul0, mul1) {
                        panic!(
                            "{}({}, {}): std: {}, builtins: {}",
                            stringify!($fn), x, y, mul0, mul1
                        );
                    }
                }
            });
        )*
    };
}

#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
#[test]
fn float_mul() {
    use compiler_builtins::float::{
        mul::{__muldf3, __mulsf3},
        Float,
    };

    float_mul!(
        f32, __mulsf3;
        f64, __muldf3;
    );
}

#[cfg(target_arch = "arm")]
#[test]
fn float_mul_arm() {
    use compiler_builtins::float::{
        mul::{__muldf3vfp, __mulsf3vfp},
        Float,
    };

    float_mul!(
        f32, __mulsf3vfp;
        f64, __muldf3vfp;
    );
}
