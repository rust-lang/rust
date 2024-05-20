#![allow(unused_macros)]
#![feature(f128)]
#![feature(f16)]

use testcrate::*;

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
                                "{}({}, {}): std: {}, builtins: {}",
                                stringify!($fn), x, y, mul0, mul1
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
                                "{}({}, {}): std: ({}, {}), builtins: ({}, {})",
                                stringify!($fn), x, y, mul0, o0, mul1, o1
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
}

macro_rules! float_mul {
    ($($f:ty, $fn:ident, $apfloat_ty:ident, $sys_available:meta);*;) => {
        $(
            #[test]
            fn $fn() {
                use compiler_builtins::float::{mul::$fn, Float};
                use core::ops::Mul;

                fuzz_float_2(N, |x: $f, y: $f| {
                    let mul0 = apfloat_fallback!($f, $apfloat_ty, $sys_available, Mul::mul, x, y);
                    let mul1: $f = $fn(x, y);
                    // multiplication of subnormals is not currently handled
                    if !(Float::is_subnormal(mul0) || Float::is_subnormal(mul1)) {
                        if !Float::eq_repr(mul0, mul1) {
                            panic!(
                                "{}({:?}, {:?}): std: {:?}, builtins: {:?}",
                                stringify!($fn), x, y, mul0, mul1
                            );
                        }
                    }
                });
            }
        )*
    };
}

#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
mod float_mul {
    use super::*;

    float_mul! {
        f32, __mulsf3, Single, all();
        f64, __muldf3, Double, all();
    }
}

#[cfg(not(feature = "no-f16-f128"))]
#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
#[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
mod float_mul_f128 {
    use super::*;

    float_mul! {
        f128, __multf3, Quad,
        // FIXME(llvm): there is a bug in LLVM rt.
        // See <https://github.com/llvm/llvm-project/issues/91840>.
        not(any(feature = "no-sys-f128", all(target_arch = "aarch64", target_os = "linux")));
    }
}

#[cfg(not(feature = "no-f16-f128"))]
#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
mod float_mul_f128_ppc {
    use super::*;

    float_mul! {
        f128, __mulkf3, Quad, not(feature = "no-sys-f128");
    }
}

#[cfg(target_arch = "arm")]
mod float_mul_arm {
    use super::*;

    float_mul! {
        f32, __mulsf3vfp, Single, all();
        f64, __muldf3vfp, Double, all();
    }
}
