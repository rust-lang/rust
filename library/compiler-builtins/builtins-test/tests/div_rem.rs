#![feature(f128)]
#![allow(unused_macros)]

use builtins_test::*;
use compiler_builtins::int::sdiv::{__divmoddi4, __divmodsi4, __divmodti4};
use compiler_builtins::int::udiv::{__udivmoddi4, __udivmodsi4, __udivmodti4, u128_divide_sparc};

// Division algorithms have by far the nastiest and largest number of edge cases, and experience shows
// that sometimes 100_000 iterations of the random fuzzer is needed.

/// Creates intensive test functions for division functions of a certain size
macro_rules! test {
    (
        $n:expr, // the number of bits in a $iX or $uX
        $uX:ident, // unsigned integer that will be shifted
        $iX:ident, // signed version of $uX
        $test_name:ident, // name of the test function
        $unsigned_name:ident, // unsigned division function
        $signed_name:ident // signed division function
    ) => {
        #[test]
        fn $test_name() {
            fuzz_2(N, |lhs, rhs| {
                if rhs == 0 {
                    return;
                }

                let mut rem: $uX = 0;
                let quo: $uX = $unsigned_name(lhs, rhs, Some(&mut rem));
                if rhs <= rem || (lhs != rhs.wrapping_mul(quo).wrapping_add(rem)) {
                    panic!(
                        "unsigned division function failed with lhs:{} rhs:{} \
                        std:({}, {}) builtins:({}, {})",
                        lhs,
                        rhs,
                        lhs.wrapping_div(rhs),
                        lhs.wrapping_rem(rhs),
                        quo,
                        rem
                    );
                }

                // test the signed division function also
                let lhs = lhs as $iX;
                let rhs = rhs as $iX;
                let mut rem: $iX = 0;
                let quo: $iX = $signed_name(lhs, rhs, &mut rem);
                // We cannot just test that
                // `lhs == rhs.wrapping_mul(quo).wrapping_add(rem)`, but also
                // need to make sure the remainder isn't larger than the divisor
                // and has the correct sign.
                let incorrect_rem = if rem == 0 {
                    false
                } else if rhs == $iX::MIN {
                    // `rhs.wrapping_abs()` would overflow, so handle this case
                    // separately.
                    (lhs.is_negative() != rem.is_negative()) || (rem == $iX::MIN)
                } else {
                    (lhs.is_negative() != rem.is_negative())
                        || (rhs.wrapping_abs() <= rem.wrapping_abs())
                };
                if incorrect_rem || lhs != rhs.wrapping_mul(quo).wrapping_add(rem) {
                    panic!(
                        "signed division function failed with lhs:{} rhs:{} \
                        std:({}, {}) builtins:({}, {})",
                        lhs,
                        rhs,
                        lhs.wrapping_div(rhs),
                        lhs.wrapping_rem(rhs),
                        quo,
                        rem
                    );
                }
            });
        }
    };
}

test!(32, u32, i32, div_rem_si4, __udivmodsi4, __divmodsi4);
test!(64, u64, i64, div_rem_di4, __udivmoddi4, __divmoddi4);
test!(128, u128, i128, div_rem_ti4, __udivmodti4, __divmodti4);

#[test]
fn divide_sparc() {
    fuzz_2(N, |lhs, rhs| {
        if rhs == 0 {
            return;
        }

        let mut rem: u128 = 0;
        let quo: u128 = u128_divide_sparc(lhs, rhs, &mut rem);
        if rhs <= rem || (lhs != rhs.wrapping_mul(quo).wrapping_add(rem)) {
            panic!(
                "u128_divide_sparc({}, {}): \
                std:({}, {}), builtins:({}, {})",
                lhs,
                rhs,
                lhs.wrapping_div(rhs),
                lhs.wrapping_rem(rhs),
                quo,
                rem
            );
        }
    });
}

macro_rules! float {
    ($($f:ty, $fn:ident, $apfloat_ty:ident, $sys_available:meta);*;) => {
        $(
            #[test]
            fn $fn() {
                use compiler_builtins::float::{div::$fn, Float};
                use core::ops::Div;

                fuzz_float_2(N, |x: $f, y: $f| {
                    let quo0: $f = apfloat_fallback!($f, $apfloat_ty, $sys_available, Div::div, x, y);
                    let quo1: $f = $fn(x, y);

                    // ARM SIMD instructions always flush subnormals to zero
                    if cfg!(target_arch = "arm") &&
                        ((Float::is_subnormal(quo0)) || Float::is_subnormal(quo1)) {
                        return;
                    }

                    if !Float::eq_repr(quo0, quo1) {
                        panic!(
                            "{}({:?}, {:?}): std: {:?}, builtins: {:?}",
                            stringify!($fn),
                            x,
                            y,
                            quo0,
                            quo1
                        );
                    }
                });
            }
        )*
    };
}

#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
mod float_div {
    use super::*;

    float! {
        f32, __divsf3, Single, all();
        f64, __divdf3, Double, all();
    }

    #[cfg(not(feature = "no-f16-f128"))]
    #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
    float! {
        f128, __divtf3, Quad,
        // FIXME(llvm): there is a bug in LLVM rt.
        // See <https://github.com/llvm/llvm-project/issues/91840>.
        not(any(feature = "no-sys-f128", all(target_arch = "aarch64", target_os = "linux")));
    }

    #[cfg(not(feature = "no-f16-f128"))]
    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    float! {
        f128, __divkf3, Quad, not(feature = "no-sys-f128");
    }
}
