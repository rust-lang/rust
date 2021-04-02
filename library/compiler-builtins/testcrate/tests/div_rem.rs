#![allow(unused_macros)]

use compiler_builtins::int::sdiv::{__divmoddi4, __divmodsi4, __divmodti4};
use compiler_builtins::int::udiv::{__udivmoddi4, __udivmodsi4, __udivmodti4, u128_divide_sparc};
use testcrate::*;

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
    ($($i:ty, $fn:ident);*;) => {
        $(
            fuzz_float_2(N, |x: $i, y: $i| {
                let quo0 = x / y;
                let quo1: $i = $fn(x, y);
                // division of subnormals is not currently handled
                if !(Float::is_subnormal(quo0) || Float::is_subnormal(quo1)) {
                    if !Float::eq_repr(quo0, quo1) {
                        panic!(
                            "{}({}, {}): std: {}, builtins: {}",
                            stringify!($fn), x, y, quo0, quo1
                        );
                    }
                }
            });
        )*
    };
}

#[cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]
#[test]
fn float_div() {
    use compiler_builtins::float::{
        div::{__divdf3, __divsf3},
        Float,
    };

    float!(
        f32, __divsf3;
        f64, __divdf3;
    );
}

#[cfg(target_arch = "arm")]
#[test]
fn float_div_arm() {
    use compiler_builtins::float::{
        div::{__divdf3vfp, __divsf3vfp},
        Float,
    };

    float!(
        f32, __divsf3vfp;
        f64, __divdf3vfp;
    );
}
