macro_rules! uint_module {
    ($T:ident) => {
        use core::num::ParseIntError;
        use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
        use core::$T::*;

        use crate::num;

        #[test]
        fn test_overflows() {
            assert!(MAX > 0);
            assert!(MIN <= 0);
            assert!((MIN + MAX).wrapping_add(1) == 0);
        }

        #[test]
        fn test_num() {
            num::test_num(10 as $T, 2 as $T);
        }

        #[test]
        fn test_bitwise_operators() {
            assert!(0b1110 as $T == (0b1100 as $T).bitor(0b1010 as $T));
            assert!(0b1000 as $T == (0b1100 as $T).bitand(0b1010 as $T));
            assert!(0b0110 as $T == (0b1100 as $T).bitxor(0b1010 as $T));
            assert!(0b1110 as $T == (0b0111 as $T).shl(1));
            assert!(0b0111 as $T == (0b1110 as $T).shr(1));
            assert!(MAX - (0b1011 as $T) == (0b1011 as $T).not());
        }

        const A: $T = 0b0101100;
        const B: $T = 0b0100001;
        const C: $T = 0b1111001;

        const _0: $T = 0;
        const _1: $T = !0;

        test_runtime_and_compiletime! {
            fn test_count_ones() {
                assert!(A.count_ones() == 3);
                assert!(B.count_ones() == 2);
                assert!(C.count_ones() == 5);
            }

            fn test_count_zeros() {
                assert!(A.count_zeros() == $T::BITS - 3);
                assert!(B.count_zeros() == $T::BITS - 2);
                assert!(C.count_zeros() == $T::BITS - 5);
            }

            fn test_leading_trailing_ones() {
                const A: $T = 0b0101_1111;
                assert_eq_const_safe!(u32: A.trailing_ones(), 5);
                assert_eq_const_safe!(u32: (!A).leading_ones(), $T::BITS - 7);

                assert_eq_const_safe!(u32: A.reverse_bits().leading_ones(), 5);

                assert_eq_const_safe!(u32: _1.leading_ones(), $T::BITS);
                assert_eq_const_safe!(u32: _1.trailing_ones(), $T::BITS);

                assert_eq_const_safe!(u32: (_1 << 1).trailing_ones(), 0);
                assert_eq_const_safe!(u32: (_1 >> 1).leading_ones(), 0);

                assert_eq_const_safe!(u32: (_1 << 1).leading_ones(), $T::BITS - 1);
                assert_eq_const_safe!(u32: (_1 >> 1).trailing_ones(), $T::BITS - 1);

                assert_eq_const_safe!(u32: _0.leading_ones(), 0);
                assert_eq_const_safe!(u32: _0.trailing_ones(), 0);

                const X: $T = 0b0010_1100;
                assert_eq_const_safe!(u32: X.leading_ones(), 0);
                assert_eq_const_safe!(u32: X.trailing_ones(), 0);
            }

            fn test_bit_width() {
                assert_eq_const_safe!(u32: A.bit_width(), 6);
                assert_eq_const_safe!(u32: B.bit_width(), 6);
                assert_eq_const_safe!(u32: C.bit_width(), 7);
                assert_eq_const_safe!(u32: _0.bit_width(), 0);
                assert_eq_const_safe!(u32: _1.bit_width(), $T::BITS);
            }

            fn test_rotate() {
                assert_eq_const_safe!($T: A.rotate_left(6).rotate_right(2).rotate_right(4), A);
                assert_eq_const_safe!($T: B.rotate_left(3).rotate_left(2).rotate_right(5), B);
                assert_eq_const_safe!($T: C.rotate_left(6).rotate_right(2).rotate_right(4), C);

                // Rotating these should make no difference
                //
                // We test using 124 bits because to ensure that overlong bit shifts do
                // not cause undefined behavior. See #10183.
                assert_eq_const_safe!($T: _0.rotate_left(124), _0);
                assert_eq_const_safe!($T: _1.rotate_left(124), _1);
                assert_eq_const_safe!($T: _0.rotate_right(124), _0);
                assert_eq_const_safe!($T: _1.rotate_right(124), _1);

                // Rotating by 0 should have no effect
                assert_eq_const_safe!($T: A.rotate_left(0), A);
                assert_eq_const_safe!($T: B.rotate_left(0), B);
                assert_eq_const_safe!($T: C.rotate_left(0), C);
                // Rotating by a multiple of word size should also have no effect
                assert_eq_const_safe!($T: A.rotate_left(128), A);
                assert_eq_const_safe!($T: B.rotate_left(128), B);
                assert_eq_const_safe!($T: C.rotate_left(128), C);
            }

            fn test_funnel_shift() {
                // Shifting by 0 should have no effect
                assert_eq_const_safe!($T: <$T>::funnel_shl(A, B, 0), A);
                assert_eq_const_safe!($T: <$T>::funnel_shr(A, B, 0), B);

                assert_eq_const_safe!($T: <$T>::funnel_shl(_0, _1, 4), 0b1111);
                assert_eq_const_safe!($T: <$T>::funnel_shr(_0, _1, 4), _1 >> 4);
                assert_eq_const_safe!($T: <$T>::funnel_shl(_1, _0, 4), _1 << 4);

                assert_eq_const_safe!($T: <$T>::funnel_shl(_1, _1, 4), <$T>::rotate_left(_1, 4));
                assert_eq_const_safe!($T: <$T>::funnel_shr(_1, _1, 4), <$T>::rotate_right(_1, 4));
            }

            fn test_swap_bytes() {
                assert_eq_const_safe!($T: A.swap_bytes().swap_bytes(), A);
                assert_eq_const_safe!($T: B.swap_bytes().swap_bytes(), B);
                assert_eq_const_safe!($T: C.swap_bytes().swap_bytes(), C);

                // Swapping these should make no difference
                assert_eq_const_safe!($T: _0.swap_bytes(), _0);
                assert_eq_const_safe!($T: _1.swap_bytes(), _1);
            }

            fn test_reverse_bits() {
                assert_eq_const_safe!($T: A.reverse_bits().reverse_bits(), A);
                assert_eq_const_safe!($T: B.reverse_bits().reverse_bits(), B);
                assert_eq_const_safe!($T: C.reverse_bits().reverse_bits(), C);

                // Swapping these should make no difference
                assert_eq_const_safe!($T: _0.reverse_bits(), _0);
                assert_eq_const_safe!($T: _1.reverse_bits(), _1);
            }

            fn test_le() {
                assert_eq_const_safe!($T: $T::from_le(A.to_le()), A);
                assert_eq_const_safe!($T: $T::from_le(B.to_le()), B);
                assert_eq_const_safe!($T: $T::from_le(C.to_le()), C);
                assert_eq_const_safe!($T: $T::from_le(_0), _0);
                assert_eq_const_safe!($T: $T::from_le(_1), _1);
                assert_eq_const_safe!($T: _0.to_le(), _0);
                assert_eq_const_safe!($T: _1.to_le(), _1);
            }

            fn test_be() {
                assert_eq_const_safe!($T: $T::from_be(A.to_be()), A);
                assert_eq_const_safe!($T: $T::from_be(B.to_be()), B);
                assert_eq_const_safe!($T: $T::from_be(C.to_be()), C);
                assert_eq_const_safe!($T: $T::from_be(_0), _0);
                assert_eq_const_safe!($T: $T::from_be(_1), _1);
                assert_eq_const_safe!($T: _0.to_be(), _0);
                assert_eq_const_safe!($T: _1.to_be(), _1);
            }

            fn test_unsigned_checked_div() {
                assert_eq_const_safe!(Option<$T>: (10 as $T).checked_div(2), Some(5));
                assert_eq_const_safe!(Option<$T>: (5 as $T).checked_div(0), None);
            }
        }

        #[test]
        #[should_panic = "attempt to funnel shift left with overflow"]
        fn test_funnel_shl_overflow() {
            let _ = <$T>::funnel_shl(A, B, $T::BITS);
        }

        #[test]
        #[should_panic = "attempt to funnel shift right with overflow"]
        fn test_funnel_shr_overflow() {
            let _ = <$T>::funnel_shr(A, B, $T::BITS);
        }

        #[test]
        fn test_funnel_shifts_runtime() {
            for i in 0..$T::BITS - 1 {
                assert_eq!(<$T>::funnel_shl(A, 0, i), A << i);
                assert_eq!(<$T>::funnel_shl(A, A, i), A.rotate_left(i));

                assert_eq!(<$T>::funnel_shr(0, A, i), A >> i);
                assert_eq!(<$T>::funnel_shr(A, A, i), A.rotate_right(i));
            }
        }

        #[test]
        fn test_isolate_highest_one() {
            const BITS: $T = <$T>::MAX;
            const MOST_SIG_ONE: $T = 1 << (<$T>::BITS - 1);

            // Right shift the most significant one through each
            // bit position, starting with all bits set
            let mut i = 0;
            while i < <$T>::BITS {
                assert_eq!(
                    (BITS >> i).isolate_highest_one(),
                    (MOST_SIG_ONE >> i).isolate_highest_one(),
                );
                i += 1;
            }
        }

        #[test]
        fn test_isolate_lowest_one() {
            const BITS: $T = <$T>::MAX;
            const LEAST_SIG_ONE: $T = 1;

            // Left shift the least significant one through each
            // bit position, starting with all bits set
            let mut i = 0;
            while i < <$T>::BITS {
                assert_eq!(
                    (BITS << i).isolate_lowest_one(),
                    (LEAST_SIG_ONE << i).isolate_lowest_one(),
                );
                i += 1;
            }
        }

        #[test]
        fn test_highest_one() {
            const ZERO: $T = 0;
            const ONE: $T = 1;

            assert_eq!(ZERO.highest_one(), None);

            for i in 0..<$T>::BITS {
                // Set single bit.
                assert_eq!((ONE << i).highest_one(), Some(i));
                // Set lowest bits.
                assert_eq!((<$T>::MAX >> i).highest_one(), Some(<$T>::BITS - i - 1));
                // Set highest bits.
                assert_eq!((<$T>::MAX << i).highest_one(), Some(<$T>::BITS - 1));
            }
        }

        #[test]
        fn test_lowest_one() {
            const ZERO: $T = 0;
            const ONE: $T = 1;

            assert_eq!(ZERO.lowest_one(), None);

            for i in 0..<$T>::BITS {
                // Set single bit.
                assert_eq!((ONE << i).lowest_one(), Some(i));
                // Set lowest bits.
                assert_eq!((<$T>::MAX >> i).lowest_one(), Some(0));
                // Set highest bits.
                assert_eq!((<$T>::MAX << i).lowest_one(), Some(i));
            }
        }

        fn from_str<T: core::str::FromStr>(t: &str) -> Option<T> {
            core::str::FromStr::from_str(t).ok()
        }

        #[test]
        pub fn test_from_str() {
            assert_eq!(from_str::<$T>("0"), Some(0 as $T));
            assert_eq!(from_str::<$T>("3"), Some(3 as $T));
            assert_eq!(from_str::<$T>("10"), Some(10 as $T));
            assert_eq!(from_str::<u32>("123456789"), Some(123456789 as u32));
            assert_eq!(from_str::<$T>("00100"), Some(100 as $T));

            assert_eq!(from_str::<$T>(""), None);
            assert_eq!(from_str::<$T>(" "), None);
            assert_eq!(from_str::<$T>("x"), None);
        }

        test_runtime_and_compiletime! {
            fn test_parse_bytes() {
                assert_eq_const_safe!(Result<$T, ParseIntError>: $T::from_str_radix("123", 10), Ok(123 as $T));
                assert_eq_const_safe!(Result<$T, ParseIntError>: $T::from_str_radix("1001", 2), Ok(9 as $T));
                assert_eq_const_safe!(Result<$T, ParseIntError>: $T::from_str_radix("123", 8), Ok(83 as $T));
                assert_eq_const_safe!(Result<u16, ParseIntError>: u16::from_str_radix("123", 16), Ok(291 as u16));
                assert_eq_const_safe!(Result<u16, ParseIntError>: u16::from_str_radix("ffff", 16), Ok(65535 as u16));
                assert_eq_const_safe!(Result<$T, ParseIntError>: $T::from_str_radix("z", 36), Ok(35 as $T));

                assert!($T::from_str_radix("Z", 10).is_err());
                assert!($T::from_str_radix("_", 2).is_err());
            }

            fn test_pow() {
                {
                    const R: $T = 2;
                    assert_eq_const_safe!($T: R.pow(2), 4 as $T);
                    assert_eq_const_safe!($T: R.pow(0), 1 as $T);
                    assert_eq_const_safe!($T: R.wrapping_pow(2), 4 as $T);
                    assert_eq_const_safe!($T: R.wrapping_pow(0), 1 as $T);
                    assert_eq_const_safe!(Option<$T>: R.checked_pow(2), Some(4 as $T));
                    assert_eq_const_safe!(Option<$T>: R.checked_pow(0), Some(1 as $T));
                    assert_eq_const_safe!(($T, bool): R.overflowing_pow(2), (4 as $T, false));
                    assert_eq_const_safe!(($T, bool): R.overflowing_pow(0), (1 as $T, false));
                    assert_eq_const_safe!($T: R.saturating_pow(2), 4 as $T);
                    assert_eq_const_safe!($T: R.saturating_pow(0), 1 as $T);
                }

                {
                    const R: $T = $T::MAX;
                    // use `^` to represent .pow() with no overflow.
                    // if itest::MAX == 2^j-1, then itest is a `j` bit int,
                    // so that `itest::MAX*itest::MAX == 2^(2*j)-2^(j+1)+1`,
                    // thussaturating_pow the overflowing result is exactly 1.
                    assert_eq_const_safe!($T: R.wrapping_pow(2), 1 as $T);
                    assert_eq_const_safe!(Option<$T>: R.checked_pow(2), None);
                    assert_eq_const_safe!(($T, bool): R.overflowing_pow(2), (1 as $T, true));
                    assert_eq_const_safe!($T: R.saturating_pow(2), MAX);
                }
            }

            fn test_isqrt() {
                assert_eq_const_safe!($T: (0 as $T).isqrt(), 0 as $T);
                assert_eq_const_safe!($T: (1 as $T).isqrt(), 1 as $T);
                assert_eq_const_safe!($T: (2 as $T).isqrt(), 1 as $T);
                assert_eq_const_safe!($T: (99 as $T).isqrt(), 9 as $T);
                assert_eq_const_safe!($T: (100 as $T).isqrt(), 10 as $T);
                assert_eq_const_safe!($T: $T::MAX.isqrt(), (1 << ($T::BITS / 2)) - 1);
            }
        }

        #[cfg(not(miri))] // Miri is too slow
        #[test]
        fn test_lots_of_isqrt() {
            let n_max: $T = (1024 * 1024).min($T::MAX as u128) as $T;
            for n in 0..=n_max {
                let isqrt: $T = n.isqrt();

                assert!(isqrt.pow(2) <= n);
                assert!(isqrt + 1 == (1 as $T) << ($T::BITS / 2) || (isqrt + 1).pow(2) > n);
            }

            for n in ($T::MAX - 255)..=$T::MAX {
                let isqrt: $T = n.isqrt();

                assert!(isqrt.pow(2) <= n);
                assert!(isqrt + 1 == (1 as $T) << ($T::BITS / 2) || (isqrt + 1).pow(2) > n);
            }
        }

        test_runtime_and_compiletime! {
            fn test_div_floor() {
                assert_eq_const_safe!($T: (8 as $T).div_floor(3), 2);
            }

            fn test_div_ceil() {
                assert_eq_const_safe!($T: (8 as $T).div_ceil(3), 3);
            }

            fn test_next_multiple_of() {
                assert_eq_const_safe!($T: (16 as $T).next_multiple_of(8), 16);
                assert_eq_const_safe!($T: (23 as $T).next_multiple_of(8), 24);
                assert_eq_const_safe!($T: MAX.next_multiple_of(1), MAX);
            }

            fn test_checked_next_multiple_of() {
                assert_eq_const_safe!(Option<$T>: (16 as $T).checked_next_multiple_of(8), Some(16));
                assert_eq_const_safe!(Option<$T>: (23 as $T).checked_next_multiple_of(8), Some(24));
                assert_eq_const_safe!(Option<$T>: (1 as $T).checked_next_multiple_of(0), None);
                assert_eq_const_safe!(Option<$T>: MAX.checked_next_multiple_of(2), None);
            }

            fn test_is_next_multiple_of() {
                assert!((12 as $T).is_multiple_of(4));
                assert!(!(12 as $T).is_multiple_of(5));
                assert!((0 as $T).is_multiple_of(0));
                assert!(!(12 as $T).is_multiple_of(0));
            }

            fn test_carrying_add() {
                assert_eq_const_safe!(($T, bool): $T::MAX.carrying_add(1, false), (0, true));
                assert_eq_const_safe!(($T, bool): $T::MAX.carrying_add(0, true), (0, true));
                assert_eq_const_safe!(($T, bool): $T::MAX.carrying_add(1, true), (1, true));

                assert_eq_const_safe!(($T, bool): $T::MIN.carrying_add($T::MAX, false), ($T::MAX, false));
                assert_eq_const_safe!(($T, bool): $T::MIN.carrying_add(0, true), (1, false));
                assert_eq_const_safe!(($T, bool): $T::MIN.carrying_add($T::MAX, true), (0, true));
            }

            fn test_borrowing_sub() {
                assert_eq_const_safe!(($T, bool): $T::MIN.borrowing_sub(1, false), ($T::MAX, true));
                assert_eq_const_safe!(($T, bool): $T::MIN.borrowing_sub(0, true), ($T::MAX, true));
                assert_eq_const_safe!(($T, bool): $T::MIN.borrowing_sub(1, true), ($T::MAX - 1, true));

                assert_eq_const_safe!(($T, bool): $T::MAX.borrowing_sub($T::MAX, false), (0, false));
                assert_eq_const_safe!(($T, bool): $T::MAX.borrowing_sub(0, true), ($T::MAX - 1, false));
                assert_eq_const_safe!(($T, bool): $T::MAX.borrowing_sub($T::MAX, true), ($T::MAX, true));
            }

            fn test_widening_mul() {
                assert_eq_const_safe!(($T, $T): $T::MAX.widening_mul($T::MAX), (1, $T::MAX - 1));
            }

            fn test_carrying_mul() {
                assert_eq_const_safe!(($T, $T): $T::MAX.carrying_mul($T::MAX, 0), (1, $T::MAX - 1));
                assert_eq_const_safe!(($T, $T): $T::MAX.carrying_mul($T::MAX, $T::MAX), (0, $T::MAX));
            }

            fn test_carrying_mul_add() {
                assert_eq_const_safe!(($T, $T): $T::MAX.carrying_mul_add($T::MAX, 0, 0), (1, $T::MAX - 1));
                assert_eq_const_safe!(($T, $T): $T::MAX.carrying_mul_add($T::MAX, $T::MAX, 0), (0, $T::MAX));
                assert_eq_const_safe!(($T, $T): $T::MAX.carrying_mul_add($T::MAX, $T::MAX, $T::MAX), ($T::MAX, $T::MAX));
            }

            fn test_midpoint() {
                assert_eq_const_safe!($T: <$T>::midpoint(1, 3), 2);
                assert_eq_const_safe!($T: <$T>::midpoint(3, 1), 2);

                assert_eq_const_safe!($T: <$T>::midpoint(0, 0), 0);
                assert_eq_const_safe!($T: <$T>::midpoint(0, 2), 1);
                assert_eq_const_safe!($T: <$T>::midpoint(2, 0), 1);
                assert_eq_const_safe!($T: <$T>::midpoint(2, 2), 2);

                assert_eq_const_safe!($T: <$T>::midpoint(1, 4), 2);
                assert_eq_const_safe!($T: <$T>::midpoint(4, 1), 2);
                assert_eq_const_safe!($T: <$T>::midpoint(3, 4), 3);
                assert_eq_const_safe!($T: <$T>::midpoint(4, 3), 3);

                assert_eq_const_safe!($T: <$T>::midpoint(<$T>::MIN, <$T>::MAX), (<$T>::MAX - <$T>::MIN) / 2);
                assert_eq_const_safe!($T: <$T>::midpoint(<$T>::MAX, <$T>::MIN), (<$T>::MAX - <$T>::MIN) / 2);
                assert_eq_const_safe!($T: <$T>::midpoint(<$T>::MIN, <$T>::MIN), <$T>::MIN);
                assert_eq_const_safe!($T: <$T>::midpoint(<$T>::MAX, <$T>::MAX), <$T>::MAX);

                assert_eq_const_safe!($T: <$T>::midpoint(<$T>::MIN, 6), <$T>::MIN / 2 + 3);
                assert_eq_const_safe!($T: <$T>::midpoint(6, <$T>::MIN), <$T>::MIN / 2 + 3);
                assert_eq_const_safe!($T: <$T>::midpoint(<$T>::MAX, 6), (<$T>::MAX - <$T>::MIN) / 2 + 3);
                assert_eq_const_safe!($T: <$T>::midpoint(6, <$T>::MAX), (<$T>::MAX - <$T>::MIN) / 2 + 3);
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
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_ONE), (<$T>::MIN << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_TWO), (<$T>::MIN << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_THREE), (<$T>::MIN << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MIN << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, 1), (<$T>::MIN << 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, 3), (<$T>::MIN << 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, 5), (<$T>::MIN << 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW3), 0);

                // <$T>::MAX
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_ONE), (<$T>::MAX << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_TWO), (<$T>::MAX << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_THREE), (<$T>::MAX << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MAX << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, 1), (<$T>::MAX << 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, 3), (<$T>::MAX << 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, 5), (<$T>::MAX << 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 1
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_ONE), (1 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_TWO), (1 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_THREE), (1 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_TEST_FOUR), (1 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, 1), (1 << 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, 3), (1 << 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, 5), (1 << 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(1, SHIFT_AMOUNT_OVERFLOW3), 0);

                // !0
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_TEST_ONE), (!0 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_TEST_TWO), (!0 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_TEST_THREE), (!0 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_TEST_FOUR), (!0 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, 1), (!0 << 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, 3), (!0 << 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, 5), (!0 << 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(!0, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 8
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_ONE), (8 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_TWO), (8 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_THREE), (8 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_TEST_FOUR), (8 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, 1), (8 << 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, 3), (8 << 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, 5), (8 << 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(8, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 17
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_ONE), (17 << SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_TWO), (17 << SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_THREE), (17 << SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_TEST_FOUR), (17 << SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, 1), (17 << 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, 3), (17 << 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, 5), (17 << 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shl(17, SHIFT_AMOUNT_OVERFLOW3), 0);
            }

            fn test_unbounded_shr() {
                // <$T>::MIN
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_ONE), (<$T>::MIN >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_TWO), (<$T>::MIN >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_THREE), (<$T>::MIN >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MIN >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, 1), (<$T>::MIN >> 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, 3), (<$T>::MIN >> 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, 5), (<$T>::MIN >> 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MIN, SHIFT_AMOUNT_OVERFLOW3), 0);

                // <$T>::MAX
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_ONE), (<$T>::MAX >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_TWO), (<$T>::MAX >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_THREE), (<$T>::MAX >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_TEST_FOUR), (<$T>::MAX >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, 1), (<$T>::MAX >> 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, 3), (<$T>::MAX >> 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, 5), (<$T>::MAX >> 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(<$T>::MAX, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 1
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_ONE), (1 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_TWO), (1 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_THREE), (1 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_TEST_FOUR), (1 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, 1), (1 >> 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, 3), (1 >> 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, 5), (1 >> 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(1, SHIFT_AMOUNT_OVERFLOW3), 0);

                // !0
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_TEST_ONE), (!0 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_TEST_TWO), (!0 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_TEST_THREE), (!0 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_TEST_FOUR), (!0 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, 1), (!0 >> 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, 3), (!0 >> 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, 5), (!0 >> 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(!0, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 8
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_ONE), (8 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_TWO), (8 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_THREE), (8 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_TEST_FOUR), (8 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, 1), (8 >> 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, 3), (8 >> 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, 5), (8 >> 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(8, SHIFT_AMOUNT_OVERFLOW3), 0);

                // 17
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_ONE), (17 >> SHIFT_AMOUNT_TEST_ONE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_TWO), (17 >> SHIFT_AMOUNT_TEST_TWO));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_THREE), (17 >> SHIFT_AMOUNT_TEST_THREE));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_TEST_FOUR), (17 >> SHIFT_AMOUNT_TEST_FOUR));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, 1), (17 >> 1));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, 3), (17 >> 3));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, 5), (17 >> 5));
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_OVERFLOW), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_OVERFLOW2), 0);
                assert_eq_const_safe!($T: <$T>::unbounded_shr(17, SHIFT_AMOUNT_OVERFLOW3), 0);
            }
        }

        const EXACT_DIV_SUCCESS_DIVIDEND1: $T = 42;
        const EXACT_DIV_SUCCESS_DIVISOR1: $T = 6;
        const EXACT_DIV_SUCCESS_QUOTIENT1: $T = 7;
        const EXACT_DIV_SUCCESS_DIVIDEND2: $T = 18;
        const EXACT_DIV_SUCCESS_DIVISOR2: $T = 3;
        const EXACT_DIV_SUCCESS_QUOTIENT2: $T = 6;

        test_runtime_and_compiletime! {
            fn test_exact_div() {
                // 42 / 6
                assert_eq_const_safe!(Option<$T>: <$T>::checked_exact_div(EXACT_DIV_SUCCESS_DIVIDEND1, EXACT_DIV_SUCCESS_DIVISOR1), Some(EXACT_DIV_SUCCESS_QUOTIENT1));
                assert_eq_const_safe!($T: <$T>::exact_div(EXACT_DIV_SUCCESS_DIVIDEND1, EXACT_DIV_SUCCESS_DIVISOR1), EXACT_DIV_SUCCESS_QUOTIENT1);

                // 18 / 3
                assert_eq_const_safe!(Option<$T>: <$T>::checked_exact_div(EXACT_DIV_SUCCESS_DIVIDEND2, EXACT_DIV_SUCCESS_DIVISOR2), Some(EXACT_DIV_SUCCESS_QUOTIENT2));
                assert_eq_const_safe!($T: <$T>::exact_div(EXACT_DIV_SUCCESS_DIVIDEND2, EXACT_DIV_SUCCESS_DIVISOR2), EXACT_DIV_SUCCESS_QUOTIENT2);

                // failures
                assert_eq_const_safe!(Option<$T>: <$T>::checked_exact_div(1, 2), None);
                assert_eq_const_safe!(Option<$T>: <$T>::checked_exact_div(0, 0), None);
            }
        }
    };
}
