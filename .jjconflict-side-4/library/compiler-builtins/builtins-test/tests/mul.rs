#![cfg_attr(f16_enabled, feature(f16))]
#![cfg_attr(f128_enabled, feature(f128))]
#![allow(unused_macros, unused_features)]

use builtins_test::*;

mod int_mul {
    use super::*;

    macro_rules! mul {
        ($($i:ty, $fn:ident);*;) => {
            $(
                #[test]
                fn $fn() {
                    use compiler_builtins::int::mul::$fn;

                    fuzz_2(N, |x: $i, y: $i| {
                        let mul0 = x.wrapping_mul(y);
                        let mul1: $i = $fn(x, y);
                        if mul0 != mul1 {
                            panic!(
                                "{func}({x}, {y}): std: {mul0}, builtins: {mul1}",
                                func = stringify!($fn),
                            );
                        }
                    });

                }
            )*
        };
    }

    mul! {
        u64, __muldi3;
        i128, __multi3;
    }
}

mod int_overflowing_mul {
    use super::*;

    macro_rules! overflowing_mul {
        ($($i:ty, $fn:ident);*;) => {
            $(
                #[test]
                fn $fn() {
                    use compiler_builtins::int::mul::$fn;

                    fuzz_2(N, |x: $i, y: $i| {
                        let (mul0, o0) = x.overflowing_mul(y);
                        let mut o1 = 0i32;
                        let mul1: $i = $fn(x, y, &mut o1);
                        let o1 = o1 != 0;
                        if mul0 != mul1 || o0 != o1 {
                            panic!(
                                "{func}({x}, {y}): std: ({mul0}, {o0}), builtins: ({mul1}, {o1})",
                                func = stringify!($fn),
                            );
                        }
                    });
                }
            )*
        };
    }

    overflowing_mul! {
        i32, __mulosi4;
        i64, __mulodi4;
        i128, __muloti4;
    }

    #[test]
    fn overflowing_mul_u128() {
        use compiler_builtins::int::mul::{__rust_i128_mulo, __rust_u128_mulo};

        fuzz_2(N, |x: u128, y: u128| {
            let mut o1 = 0;
            let (mul0, o0) = x.overflowing_mul(y);
            let mul1 = __rust_u128_mulo(x, y, &mut o1);
            if mul0 != mul1 || i32::from(o0) != o1 {
                panic!("__rust_u128_mulo({x}, {y}): std: ({mul0}, {o0}), builtins: ({mul1}, {o1})",);
            }
            let x = x as i128;
            let y = y as i128;
            let (mul0, o0) = x.overflowing_mul(y);
            let mul1 = __rust_i128_mulo(x, y, &mut o1);
            if mul0 != mul1 || i32::from(o0) != o1 {
                panic!("__rust_i128_mulo({x}, {y}): std: ({mul0}, {o0}), builtins: ({mul1}, {o1})",);
            }
        });
    }
}

macro_rules! float_mul {
    ($($f:ty, $fn:ident, $apfloat_ty:ident, $sys_available:meta);*;) => {
        $(
            #[test]
            fn $fn() {
                use imp::$fn;
                use compiler_builtins::support::Float;
                use core::ops::Mul;

                fuzz_float_2(N, |x: $f, y: $f| {
                    let mul0 = apfloat_fallback!($f, $apfloat_ty, $sys_available, Mul::mul, x, y);
                    let mul1: $f = $fn(x, y);
                    if !Float::eq_repr(mul0, mul1) {
                        panic!(
                            "{func}({x:?}, {y:?}): std: {mul0:?}, builtins: {mul1:?}",
                            func = stringify!($fn),
                        );
                    }
                });
            }
        )*
    };
}

mod float_mul {
    mod imp {
        #[cfg(f16_enabled)]
        pub use compiler_builtins::float::mul::__mulhf3;
        pub use compiler_builtins::float::mul::{__muldf3, __mulsf3};
        #[cfg(f128_enabled)]
        cfg_select! {
            any(target_arch = "powerpc", target_arch = "powerpc64") => {
                pub use compiler_builtins::float::mul::__mulkf3 as __multf3;
            }
            _ => {
                pub use compiler_builtins::float::mul::__multf3;
            }
        }
    }

    use super::*;

    #[cfg(f16_enabled)]
    float_mul! {
        f16, __mulhf3, Half, all();
    }

    // Hard float is inaccurate on i586 but we can still test against apfloat.
    float_mul! {
        f32, __mulsf3, Single, not(x86_no_sse2);
        f64, __muldf3, Double, not(x86_no_sse2);
    }

    #[cfg(f128_enabled)]
    float_mul! {
        f128, __multf3, Quad, not(no_sys_f128);
    }
}
