macro_rules! int_module {
    ($T:ident, $U:ident) => {
        use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
        use core::$T::*;

        const UMAX: $U = $U::MAX;

        use crate::num;

        #[test]
        fn test_overflows() {
            assert!(MAX > 0);
            assert!(MIN <= 0);
            assert_eq!(MIN + MAX + 1, 0);
        }

        #[test]
        fn test_num() {
            num::test_num(10 as $T, 2 as $T);
        }

        #[test]
        fn test_bitwise_operators() {
            assert_eq!(0b1110 as $T, (0b1100 as $T).bitor(0b1010 as $T));
            assert_eq!(0b1000 as $T, (0b1100 as $T).bitand(0b1010 as $T));
            assert_eq!(0b0110 as $T, (0b1100 as $T).bitxor(0b1010 as $T));
            assert_eq!(0b1110 as $T, (0b0111 as $T).shl(1));
            assert_eq!(0b0111 as $T, (0b1110 as $T).shr(1));
            assert_eq!(-(0b11 as $T) - (1 as $T), (0b11 as $T).not());
        }

        test_runtime_and_compiletime! {

            fn test_rem_euclid() {
                assert_eq_const_safe!((-1 as $T).rem_euclid(MIN), MAX);
            }

            fn test_abs() {
                assert_eq_const_safe!((1 as $T).abs(), 1 as $T);
                assert_eq_const_safe!((0 as $T).abs(), 0 as $T);
                assert_eq_const_safe!((-1 as $T).abs(), 1 as $T);
            }

            fn test_signum() {
                assert_eq_const_safe!((1 as $T).signum(), 1 as $T);
                assert_eq_const_safe!((0 as $T).signum(), 0 as $T);
                assert_eq_const_safe!((-0 as $T).signum(), 0 as $T);
                assert_eq_const_safe!((-1 as $T).signum(), -1 as $T);
            }

            fn test_is_positive() {
                assert!((1 as $T).is_positive());
                assert!(!(0 as $T).is_positive());
                assert!(!(-0 as $T).is_positive());
                assert!(!(-1 as $T).is_positive());
            }

            fn test_is_negative() {
                assert!(!(1 as $T).is_negative());
                assert!(!(0 as $T).is_negative());
                assert!(!(-0 as $T).is_negative());
                assert!((-1 as $T).is_negative());
            }
        }

        const A: $T = 0b0101100;
        const B: $T = 0b0100001;
        const C: $T = 0b1111001;

        const _0: $T = 0;
        const _1: $T = !0;

        test_runtime_and_compiletime! {
            fn test_count_ones() {
                assert_eq_const_safe!(A.count_ones(), 3);
                assert_eq_const_safe!(B.count_ones(), 2);
                assert_eq_const_safe!(C.count_ones(), 5);
            }

            fn test_count_zeros() {
                assert_eq_const_safe!(A.count_zeros(), $T::BITS - 3);
                assert_eq_const_safe!(B.count_zeros(), $T::BITS - 2);
                assert_eq_const_safe!(C.count_zeros(), $T::BITS - 5);
            }

            fn test_leading_trailing_ones() {
                const A: $T = 0b0101_1111;
                assert_eq_const_safe!(A.trailing_ones(), 5);
                assert_eq_const_safe!((!A).leading_ones(), $T::BITS - 7);

                assert_eq_const_safe!(A.reverse_bits().leading_ones(), 5);

                assert_eq_const_safe!(_1.leading_ones(), $T::BITS);
                assert_eq_const_safe!(_1.trailing_ones(), $T::BITS);

                assert_eq_const_safe!((_1 << 1).trailing_ones(), 0);
                assert_eq_const_safe!(MAX.leading_ones(), 0);

                assert_eq_const_safe!((_1 << 1).leading_ones(), $T::BITS - 1);
                assert_eq_const_safe!(MAX.trailing_ones(), $T::BITS - 1);

                assert_eq_const_safe!(_0.leading_ones(), 0);
                assert_eq_const_safe!(_0.trailing_ones(), 0);

                const X: $T = 0b0010_1100;
                assert_eq_const_safe!(X.leading_ones(), 0);
                assert_eq_const_safe!(X.trailing_ones(), 0);
            }

            fn test_rotate() {
                assert_eq_const_safe!(A.rotate_left(6).rotate_right(2).rotate_right(4), A);
                assert_eq_const_safe!(B.rotate_left(3).rotate_left(2).rotate_right(5), B);
                assert_eq_const_safe!(C.rotate_left(6).rotate_right(2).rotate_right(4), C);

                // Rotating these should make no difference
                //
                // We test using 124 bits because to ensure that overlong bit shifts do
                // not cause undefined behavior. See #10183.
                assert_eq_const_safe!(_0.rotate_left(124), _0);
                assert_eq_const_safe!(_1.rotate_left(124), _1);
                assert_eq_const_safe!(_0.rotate_right(124), _0);
                assert_eq_const_safe!(_1.rotate_right(124), _1);

                // Rotating by 0 should have no effect
                assert_eq_const_safe!(A.rotate_left(0), A);
                assert_eq_const_safe!(B.rotate_left(0), B);
                assert_eq_const_safe!(C.rotate_left(0), C);
                // Rotating by a multiple of word size should also have no effect
                assert_eq_const_safe!(A.rotate_left(128), A);
                assert_eq_const_safe!(B.rotate_left(128), B);
                assert_eq_const_safe!(C.rotate_left(128), C);
            }

            fn test_swap_bytes() {
                assert_eq_const_safe!(A.swap_bytes().swap_bytes(), A);
                assert_eq_const_safe!(B.swap_bytes().swap_bytes(), B);
                assert_eq_const_safe!(C.swap_bytes().swap_bytes(), C);

                // Swapping these should make no difference
                assert_eq_const_safe!(_0.swap_bytes(), _0);
                assert_eq_const_safe!(_1.swap_bytes(), _1);
            }

            fn test_le() {
                assert_eq_const_safe!($T::from_le(A.to_le()), A);
                assert_eq_const_safe!($T::from_le(B.to_le()), B);
                assert_eq_const_safe!($T::from_le(C.to_le()), C);
                assert_eq_const_safe!($T::from_le(_0), _0);
                assert_eq_const_safe!($T::from_le(_1), _1);
                assert_eq_const_safe!(_0.to_le(), _0);
                assert_eq_const_safe!(_1.to_le(), _1);
            }

            fn test_be() {
                assert_eq_const_safe!($T::from_be(A.to_be()), A);
                assert_eq_const_safe!($T::from_be(B.to_be()), B);
                assert_eq_const_safe!($T::from_be(C.to_be()), C);
                assert_eq_const_safe!($T::from_be(_0), _0);
                assert_eq_const_safe!($T::from_be(_1), _1);
                assert_eq_const_safe!(_0.to_be(), _0);
                assert_eq_const_safe!(_1.to_be(), _1);
            }

            fn test_signed_checked_div() {
                assert_eq_const_safe!((10 as $T).checked_div(2), Some(5));
                assert_eq_const_safe!((5 as $T).checked_div(0), None);
                assert_eq_const_safe!(isize::MIN.checked_div(-1), None);
            }

            fn test_saturating_abs() {
                assert_eq_const_safe!((0 as $T).saturating_abs(), 0);
                assert_eq_const_safe!((123 as $T).saturating_abs(), 123);
                assert_eq_const_safe!((-123 as $T).saturating_abs(), 123);
                assert_eq_const_safe!((MAX - 2).saturating_abs(), MAX - 2);
                assert_eq_const_safe!((MAX - 1).saturating_abs(), MAX - 1);
                assert_eq_const_safe!(MAX.saturating_abs(), MAX);
                assert_eq_const_safe!((MIN + 2).saturating_abs(), MAX - 1);
                assert_eq_const_safe!((MIN + 1).saturating_abs(), MAX);
                assert_eq_const_safe!(MIN.saturating_abs(), MAX);
            }

            fn test_saturating_neg() {
                assert_eq_const_safe!((0 as $T).saturating_neg(), 0);
                assert_eq_const_safe!((123 as $T).saturating_neg(), -123);
                assert_eq_const_safe!((-123 as $T).saturating_neg(), 123);
                assert_eq_const_safe!((MAX - 2).saturating_neg(), MIN + 3);
                assert_eq_const_safe!((MAX - 1).saturating_neg(), MIN + 2);
                assert_eq_const_safe!(MAX.saturating_neg(), MIN + 1);
                assert_eq_const_safe!((MIN + 2).saturating_neg(), MAX - 1);
                assert_eq_const_safe!((MIN + 1).saturating_neg(), MAX);
                assert_eq_const_safe!(MIN.saturating_neg(), MAX);
            }
        }

        #[test]
        fn test_from_str() {
            fn from_str<T: std::str::FromStr>(t: &str) -> Option<T> {
                std::str::FromStr::from_str(t).ok()
            }
            assert_eq!(from_str::<$T>("0"), Some(0 as $T));
            assert_eq!(from_str::<$T>("3"), Some(3 as $T));
            assert_eq!(from_str::<$T>("10"), Some(10 as $T));
            assert_eq!(from_str::<i32>("123456789"), Some(123456789 as i32));
            assert_eq!(from_str::<$T>("00100"), Some(100 as $T));

            assert_eq!(from_str::<$T>("-1"), Some(-1 as $T));
            assert_eq!(from_str::<$T>("-3"), Some(-3 as $T));
            assert_eq!(from_str::<$T>("-10"), Some(-10 as $T));
            assert_eq!(from_str::<i32>("-123456789"), Some(-123456789 as i32));
            assert_eq!(from_str::<$T>("-00100"), Some(-100 as $T));

            assert_eq!(from_str::<$T>(""), None);
            assert_eq!(from_str::<$T>(" "), None);
            assert_eq!(from_str::<$T>("x"), None);
        }

        test_runtime_and_compiletime! {
            fn test_from_str_radix() {
                assert_eq_const_safe!($T::from_str_radix("123", 10), Ok(123 as $T));
                assert_eq_const_safe!($T::from_str_radix("1001", 2), Ok(9 as $T));
                assert_eq_const_safe!($T::from_str_radix("123", 8), Ok(83 as $T));
                assert_eq_const_safe!(i32::from_str_radix("123", 16), Ok(291 as i32));
                assert_eq_const_safe!(i32::from_str_radix("ffff", 16), Ok(65535 as i32));
                assert_eq_const_safe!(i32::from_str_radix("FFFF", 16), Ok(65535 as i32));
                assert_eq_const_safe!($T::from_str_radix("z", 36), Ok(35 as $T));
                assert_eq_const_safe!($T::from_str_radix("Z", 36), Ok(35 as $T));

                assert_eq_const_safe!($T::from_str_radix("-123", 10), Ok(-123 as $T));
                assert_eq_const_safe!($T::from_str_radix("-1001", 2), Ok(-9 as $T));
                assert_eq_const_safe!($T::from_str_radix("-123", 8), Ok(-83 as $T));
                assert_eq_const_safe!(i32::from_str_radix("-123", 16), Ok(-291 as i32));
                assert_eq_const_safe!(i32::from_str_radix("-ffff", 16), Ok(-65535 as i32));
                assert_eq_const_safe!(i32::from_str_radix("-FFFF", 16), Ok(-65535 as i32));
                assert_eq_const_safe!($T::from_str_radix("-z", 36), Ok(-35 as $T));
                assert_eq_const_safe!($T::from_str_radix("-Z", 36), Ok(-35 as $T));

                assert!($T::from_str_radix("Z", 35).is_err());
                assert!($T::from_str_radix("-9", 2).is_err());
                assert!($T::from_str_radix("10_0", 10).is_err());
                assert!(u32::from_str_radix("-9", 10).is_err());
            }

            fn test_pow() {
                {
                    const R: $T = 2;
                    assert_eq_const_safe!(R.pow(2), 4 as $T);
                    assert_eq_const_safe!(R.pow(0), 1 as $T);
                    assert_eq_const_safe!(R.wrapping_pow(2), 4 as $T);
                    assert_eq_const_safe!(R.wrapping_pow(0), 1 as $T);
                    assert_eq_const_safe!(R.checked_pow(2), Some(4 as $T));
                    assert_eq_const_safe!(R.checked_pow(0), Some(1 as $T));
                    assert_eq_const_safe!(R.overflowing_pow(2), (4 as $T, false));
                    assert_eq_const_safe!(R.overflowing_pow(0), (1 as $T, false));
                    assert_eq_const_safe!(R.saturating_pow(2), 4 as $T);
                    assert_eq_const_safe!(R.saturating_pow(0), 1 as $T);
                }

                {
                    const R: $T = MAX;
                    // use `^` to represent .pow() with no overflow.
                    // if itest::MAX == 2^j-1, then itest is a `j` bit int,
                    // so that `itest::MAX*itest::MAX == 2^(2*j)-2^(j+1)+1`,
                    // thussaturating_pow the overflowing result is exactly 1.
                    assert_eq_const_safe!(R.wrapping_pow(2), 1 as $T);
                    assert_eq_const_safe!(R.checked_pow(2), None);
                    assert_eq_const_safe!(R.overflowing_pow(2), (1 as $T, true));
                    assert_eq_const_safe!(R.saturating_pow(2), MAX);
                }

                {
                    // test for negative exponent.
                    const R: $T = -2;
                    assert_eq_const_safe!(R.pow(2), 4 as $T);
                    assert_eq_const_safe!(R.pow(3), -8 as $T);
                    assert_eq_const_safe!(R.pow(0), 1 as $T);
                    assert_eq_const_safe!(R.wrapping_pow(2), 4 as $T);
                    assert_eq_const_safe!(R.wrapping_pow(3), -8 as $T);
                    assert_eq_const_safe!(R.wrapping_pow(0), 1 as $T);
                    assert_eq_const_safe!(R.checked_pow(2), Some(4 as $T));
                    assert_eq_const_safe!(R.checked_pow(3), Some(-8 as $T));
                    assert_eq_const_safe!(R.checked_pow(0), Some(1 as $T));
                    assert_eq_const_safe!(R.overflowing_pow(2), (4 as $T, false));
                    assert_eq_const_safe!(R.overflowing_pow(3), (-8 as $T, false));
                    assert_eq_const_safe!(R.overflowing_pow(0), (1 as $T, false));
                    assert_eq_const_safe!(R.saturating_pow(2), 4 as $T);
                    assert_eq_const_safe!(R.saturating_pow(3), -8 as $T);
                    assert_eq_const_safe!(R.saturating_pow(0), 1 as $T);
                }
            }

            fn test_div_floor() {
                const A: $T = 8;
                const B: $T = 3;
                assert_eq_const_safe!(A.div_floor(B), 2);
                assert_eq_const_safe!(A.div_floor(-B), -3);
                assert_eq_const_safe!((-A).div_floor(B), -3);
                assert_eq_const_safe!((-A).div_floor(-B), 2);
            }

            fn test_div_ceil() {
                const A: $T = 8;
                const B: $T = 3;
                assert_eq_const_safe!(A.div_ceil(B), 3);
                assert_eq_const_safe!(A.div_ceil(-B), -2);
                assert_eq_const_safe!((-A).div_ceil(B), -2);
                assert_eq_const_safe!((-A).div_ceil(-B), 3);
            }

            fn test_next_multiple_of() {
                assert_eq_const_safe!((16 as $T).next_multiple_of(8), 16);
                assert_eq_const_safe!((23 as $T).next_multiple_of(8), 24);
                assert_eq_const_safe!((16 as $T).next_multiple_of(-8), 16);
                assert_eq_const_safe!((23 as $T).next_multiple_of(-8), 16);
                assert_eq_const_safe!((-16 as $T).next_multiple_of(8), -16);
                assert_eq_const_safe!((-23 as $T).next_multiple_of(8), -16);
                assert_eq_const_safe!((-16 as $T).next_multiple_of(-8), -16);
                assert_eq_const_safe!((-23 as $T).next_multiple_of(-8), -24);
                assert_eq_const_safe!(MIN.next_multiple_of(-1), MIN);
            }

            fn test_checked_next_multiple_of() {
                assert_eq_const_safe!((16 as $T).checked_next_multiple_of(8), Some(16));
                assert_eq_const_safe!((23 as $T).checked_next_multiple_of(8), Some(24));
                assert_eq_const_safe!((16 as $T).checked_next_multiple_of(-8), Some(16));
                assert_eq_const_safe!((23 as $T).checked_next_multiple_of(-8), Some(16));
                assert_eq_const_safe!((-16 as $T).checked_next_multiple_of(8), Some(-16));
                assert_eq_const_safe!((-23 as $T).checked_next_multiple_of(8), Some(-16));
                assert_eq_const_safe!((-16 as $T).checked_next_multiple_of(-8), Some(-16));
                assert_eq_const_safe!((-23 as $T).checked_next_multiple_of(-8), Some(-24));
                assert_eq_const_safe!((1 as $T).checked_next_multiple_of(0), None);
                assert_eq_const_safe!(MAX.checked_next_multiple_of(2), None);
                assert_eq_const_safe!(MIN.checked_next_multiple_of(-3), None);
                assert_eq_const_safe!(MIN.checked_next_multiple_of(-1), Some(MIN));
            }

            fn test_carrying_add() {
                assert_eq_const_safe!(MAX.carrying_add(1, false), (MIN, true));
                assert_eq_const_safe!(MAX.carrying_add(0, true), (MIN, true));
                assert_eq_const_safe!(MAX.carrying_add(1, true), (MIN + 1, true));
                assert_eq_const_safe!(MAX.carrying_add(-1, false), (MAX - 1, false));
                assert_eq_const_safe!(MAX.carrying_add(-1, true), (MAX, false)); // no intermediate overflow
                assert_eq_const_safe!(MIN.carrying_add(-1, false), (MAX, true));
                assert_eq_const_safe!(MIN.carrying_add(-1, true), (MIN, false)); // no intermediate overflow
                assert_eq_const_safe!((0 as $T).carrying_add(MAX, true), (MIN, true));
                assert_eq_const_safe!((0 as $T).carrying_add(MIN, true), (MIN + 1, false));
            }

            fn test_borrowing_sub() {
                assert_eq_const_safe!(MIN.borrowing_sub(1, false), (MAX, true));
                assert_eq_const_safe!(MIN.borrowing_sub(0, true), (MAX, true));
                assert_eq_const_safe!(MIN.borrowing_sub(1, true), (MAX - 1, true));
                assert_eq_const_safe!(MIN.borrowing_sub(-1, false), (MIN + 1, false));
                assert_eq_const_safe!(MIN.borrowing_sub(-1, true), (MIN, false)); // no intermediate overflow
                assert_eq_const_safe!(MAX.borrowing_sub(-1, false), (MIN, true));
                assert_eq_const_safe!(MAX.borrowing_sub(-1, true), (MAX, false)); // no intermediate overflow
                assert_eq_const_safe!((0 as $T).borrowing_sub(MIN, false), (MIN, true));
                assert_eq_const_safe!((0 as $T).borrowing_sub(MIN, true), (MAX, false));
            }

            fn test_widening_mul() {
                assert_eq_const_safe!(MAX.widening_mul(MAX), (1, MAX / 2));
                assert_eq_const_safe!(MIN.widening_mul(MAX), (MIN as $U, MIN / 2));
                assert_eq_const_safe!(MIN.widening_mul(MIN), (0, MAX / 2 + 1));
            }

            fn test_carrying_mul() {
                assert_eq_const_safe!(MAX.carrying_mul(MAX, 0), (1, MAX / 2));
                assert_eq_const_safe!(
                    MAX.carrying_mul(MAX, MAX),
                    (UMAX / 2 + 1, MAX / 2)
                );
                assert_eq_const_safe!(
                    MAX.carrying_mul(MAX, MIN),
                    (UMAX / 2 + 2, MAX / 2 - 1)
                );
                assert_eq_const_safe!(MIN.carrying_mul(MAX, 0), (MIN as $U, MIN / 2));
                assert_eq_const_safe!(MIN.carrying_mul(MAX, MAX), (UMAX, MIN / 2));
                assert_eq_const_safe!(MIN.carrying_mul(MAX, MIN), (0, MIN / 2));
                assert_eq_const_safe!(MIN.carrying_mul(MIN, 0), (0, MAX / 2 + 1));
                assert_eq_const_safe!(
                    MIN.carrying_mul(MIN, MAX),
                    (UMAX / 2, MAX / 2 + 1)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul(MIN, MIN),
                    (UMAX / 2 + 1, MAX / 2)
                );
            }

            fn test_carrying_mul_add() {
                assert_eq_const_safe!(MAX.carrying_mul_add(MAX, 0, 0), (1, MAX / 2));
                assert_eq_const_safe!(
                    MAX.carrying_mul_add(MAX, MAX, 0),
                    (UMAX / 2 + 1, MAX / 2)
                );
                assert_eq_const_safe!(
                    MAX.carrying_mul_add(MAX, MIN, 0),
                    (UMAX / 2 + 2, MAX / 2 - 1)
                );
                assert_eq_const_safe!(
                    MAX.carrying_mul_add(MAX, MAX, MAX),
                    (UMAX, MAX / 2)
                );
                assert_eq_const_safe!(
                    MAX.carrying_mul_add(MAX, MAX, MIN),
                    (0, MAX / 2)
                );
                assert_eq_const_safe!(
                    MAX.carrying_mul_add(MAX, MIN, MIN),
                    (1, MAX / 2 - 1)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MAX, 0, 0),
                    (MIN as $U, MIN / 2)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MAX, MAX, 0),
                    (UMAX, MIN / 2)
                );
                assert_eq_const_safe!(MIN.carrying_mul_add(MAX, MIN, 0), (0, MIN / 2));
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MAX, MAX, MAX),
                    (UMAX / 2 - 1, MIN / 2 + 1)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MAX, MAX, MIN),
                    (UMAX / 2, MIN / 2)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MAX, MIN, MIN),
                    (UMAX / 2 + 1, MIN / 2 - 1)
                );
                assert_eq_const_safe!(MIN.carrying_mul_add(MIN, 0, 0), (0, MAX / 2 + 1));
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MIN, MAX, 0),
                    (UMAX / 2, MAX / 2 + 1)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MIN, MIN, 0),
                    (UMAX / 2 + 1, MAX / 2)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MIN, MAX, MAX),
                    (UMAX - 1, MAX / 2 + 1)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MIN, MAX, MIN),
                    (UMAX, MAX / 2)
                );
                assert_eq_const_safe!(
                    MIN.carrying_mul_add(MIN, MIN, MIN),
                    (0, MAX / 2)
                );
            }

            fn test_midpoint() {
                assert_eq_const_safe!(<$T>::midpoint(1, 3), 2);
                assert_eq_const_safe!(<$T>::midpoint(3, 1), 2);

                assert_eq_const_safe!(<$T>::midpoint(0, 0), 0);
                assert_eq_const_safe!(<$T>::midpoint(0, 2), 1);
                assert_eq_const_safe!(<$T>::midpoint(2, 0), 1);
                assert_eq_const_safe!(<$T>::midpoint(2, 2), 2);

                assert_eq_const_safe!(<$T>::midpoint(1, 4), 2);
                assert_eq_const_safe!(<$T>::midpoint(4, 1), 2);
                assert_eq_const_safe!(<$T>::midpoint(3, 4), 3);
                assert_eq_const_safe!(<$T>::midpoint(4, 3), 3);

                assert_eq_const_safe!(<$T>::midpoint(<$T>::MIN, <$T>::MAX), 0);
                assert_eq_const_safe!(<$T>::midpoint(<$T>::MAX, <$T>::MIN), 0);
                assert_eq_const_safe!(<$T>::midpoint(<$T>::MIN, <$T>::MIN), <$T>::MIN);
                assert_eq_const_safe!(<$T>::midpoint(<$T>::MAX, <$T>::MAX), <$T>::MAX);

                assert_eq_const_safe!(<$T>::midpoint(<$T>::MIN, 6), <$T>::MIN / 2 + 3);
                assert_eq_const_safe!(<$T>::midpoint(6, <$T>::MIN), <$T>::MIN / 2 + 3);
                assert_eq_const_safe!(<$T>::midpoint(<$T>::MAX, 6), <$T>::MAX / 2 + 3);
                assert_eq_const_safe!(<$T>::midpoint(6, <$T>::MAX), <$T>::MAX / 2 + 3);
            }
        }

        // test_unbounded_sh* constants
        const SHIFT_AMOUNT_OVERFLOW: u32 = <$T>::BITS;
        const SHIFT_AMOUNT_OVERFLOW2: u32 = <$T>::BITS + 3;
        const SHIFT_AMOUNT_OVERFLOW3: u32 = <$T>::BITS << 2;

        const SHIFT_AMOUNT_TEST_ONE: u32 = <$T>::BITS >> 1;
        const SHIFT_AMOUNT_TEST_TWO: u32 = <$T>::BITS >> 3;
        const SHIFT_AMOUNT_TEST_THREE: u32 = (<$T>::BITS >> 1) - 1;
        const SHIFT_AMOUNT_TEST_FOUR: u32 = <$T>::BITS - 1;

        test_runtime_and_compiletime! {
            fn test_unbounded_shl() {
                // <$T>::MIN
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_ONE), (<$T>::MIN << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_TWO), (<$T>::MIN << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_THREE), (<$T>::MIN << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MIN << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, 1), (<$T>::MIN << 1));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, 3), (<$T>::MIN << 3));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, 5), (<$T>::MIN << 5));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW3), 0);

                // <$T>::MAX
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_ONE), (<$T>::MAX << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_TWO), (<$T>::MAX << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_THREE), (<$T>::MAX << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MAX << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, 1), (<$T>::MAX << 1));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, 3), (<$T>::MAX << 3));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, 5), (<$T>::MAX << 5));
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 1
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_ONE), (1 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_TWO), (1 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_THREE), (1 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_FOUR), (1 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, 1), (1 << 1));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, 3), (1 << 3));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, 5), (1 << 5));
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW3), 0);

                // -1
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_TEST_ONE), (-1 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_TEST_TWO), (-1 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_TEST_THREE), (-1 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_TEST_FOUR), (-1 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, 1), (-1 << 1));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, 3), (-1 << 3));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, 5), (-1 << 5));
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(-1, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 8
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_ONE), (8 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_TWO), (8 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_THREE), (8 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_FOUR), (8 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, 1), (8 << 1));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, 3), (8 << 3));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, 5), (8 << 5));
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(8, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 17
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_ONE), (17 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_TWO), (17 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_THREE), (17 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_FOUR), (17 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, 1), (17 << 1));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, 3), (17 << 3));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, 5), (17 << 5));
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shl(17, SHIFT_AMOUNT_OVERFLOW3), 0);
            }

            fn test_unbounded_shr() {
                // <$T>::MIN
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_ONE), (<$T>::MIN >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_TWO), (<$T>::MIN >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_THREE), (<$T>::MIN >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MIN >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, 1), (<$T>::MIN >> 1));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, 3), (<$T>::MIN >> 3));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, 5), (<$T>::MIN >> 5));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW), -1);
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW2), -1);
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW3), -1);

                // <$T>::MAX
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_ONE), (<$T>::MAX >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_TWO), (<$T>::MAX >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_THREE), (<$T>::MAX >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MAX >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, 1), (<$T>::MAX >> 1));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, 3), (<$T>::MAX >> 3));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, 5), (<$T>::MAX >> 5));
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 1
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_ONE), (1 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_TWO), (1 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_THREE), (1 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_FOUR), (1 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, 1), (1 >> 1));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, 3), (1 >> 3));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, 5), (1 >> 5));
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW3), 0);

                // -1
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_TEST_ONE), (-1 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_TEST_TWO), (-1 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_TEST_THREE), (-1 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_TEST_FOUR), (-1 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, 1), (-1 >> 1));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, 3), (-1 >> 3));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, 5), (-1 >> 5));
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_OVERFLOW), -1);
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_OVERFLOW), -1);
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_OVERFLOW2), -1);
                assert_eq_const_safe!(<$T>::unbounded_shr(-1, SHIFT_AMOUNT_OVERFLOW3), -1);

                // 8
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_ONE), (8 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_TWO), (8 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_THREE), (8 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_FOUR), (8 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, 1), (8 >> 1));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, 3), (8 >> 3));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, 5), (8 >> 5));
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(8, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 17
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_ONE), (17 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_TWO), (17 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_THREE), (17 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_FOUR), (17 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, 1), (17 >> 1));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, 3), (17 >> 3));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, 5), (17 >> 5));
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!(<$T>::unbounded_shr(17, SHIFT_AMOUNT_OVERFLOW3), 0);
            }
        }
    };
}
