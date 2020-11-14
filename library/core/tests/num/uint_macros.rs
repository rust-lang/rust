macro_rules! uint_module {
    ($T:ident, $T_i:ident) => {
        #[cfg(test)]
        mod tests {
            use core::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr};
            use core::$T_i::*;
            use std::str::FromStr;

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

            #[test]
            fn test_count_ones() {
                assert!(A.count_ones() == 3);
                assert!(B.count_ones() == 2);
                assert!(C.count_ones() == 5);
            }

            #[test]
            fn test_count_zeros() {
                assert!(A.count_zeros() == $T::BITS - 3);
                assert!(B.count_zeros() == $T::BITS - 2);
                assert!(C.count_zeros() == $T::BITS - 5);
            }

            #[test]
            fn test_leading_trailing_ones() {
                let a: $T = 0b0101_1111;
                assert_eq!(a.trailing_ones(), 5);
                assert_eq!((!a).leading_ones(), $T::BITS - 7);

                assert_eq!(a.reverse_bits().leading_ones(), 5);

                assert_eq!(_1.leading_ones(), $T::BITS);
                assert_eq!(_1.trailing_ones(), $T::BITS);

                assert_eq!((_1 << 1).trailing_ones(), 0);
                assert_eq!((_1 >> 1).leading_ones(), 0);

                assert_eq!((_1 << 1).leading_ones(), $T::BITS - 1);
                assert_eq!((_1 >> 1).trailing_ones(), $T::BITS - 1);

                assert_eq!(_0.leading_ones(), 0);
                assert_eq!(_0.trailing_ones(), 0);

                let x: $T = 0b0010_1100;
                assert_eq!(x.leading_ones(), 0);
                assert_eq!(x.trailing_ones(), 0);
            }

            #[test]
            fn test_rotate() {
                assert_eq!(A.rotate_left(6).rotate_right(2).rotate_right(4), A);
                assert_eq!(B.rotate_left(3).rotate_left(2).rotate_right(5), B);
                assert_eq!(C.rotate_left(6).rotate_right(2).rotate_right(4), C);

                // Rotating these should make no difference
                //
                // We test using 124 bits because to ensure that overlong bit shifts do
                // not cause undefined behaviour. See #10183.
                assert_eq!(_0.rotate_left(124), _0);
                assert_eq!(_1.rotate_left(124), _1);
                assert_eq!(_0.rotate_right(124), _0);
                assert_eq!(_1.rotate_right(124), _1);

                // Rotating by 0 should have no effect
                assert_eq!(A.rotate_left(0), A);
                assert_eq!(B.rotate_left(0), B);
                assert_eq!(C.rotate_left(0), C);
                // Rotating by a multiple of word size should also have no effect
                assert_eq!(A.rotate_left(128), A);
                assert_eq!(B.rotate_left(128), B);
                assert_eq!(C.rotate_left(128), C);
            }

            #[test]
            fn test_swap_bytes() {
                assert_eq!(A.swap_bytes().swap_bytes(), A);
                assert_eq!(B.swap_bytes().swap_bytes(), B);
                assert_eq!(C.swap_bytes().swap_bytes(), C);

                // Swapping these should make no difference
                assert_eq!(_0.swap_bytes(), _0);
                assert_eq!(_1.swap_bytes(), _1);
            }

            #[test]
            fn test_reverse_bits() {
                assert_eq!(A.reverse_bits().reverse_bits(), A);
                assert_eq!(B.reverse_bits().reverse_bits(), B);
                assert_eq!(C.reverse_bits().reverse_bits(), C);

                // Swapping these should make no difference
                assert_eq!(_0.reverse_bits(), _0);
                assert_eq!(_1.reverse_bits(), _1);
            }

            #[test]
            fn test_le() {
                assert_eq!($T::from_le(A.to_le()), A);
                assert_eq!($T::from_le(B.to_le()), B);
                assert_eq!($T::from_le(C.to_le()), C);
                assert_eq!($T::from_le(_0), _0);
                assert_eq!($T::from_le(_1), _1);
                assert_eq!(_0.to_le(), _0);
                assert_eq!(_1.to_le(), _1);
            }

            #[test]
            fn test_be() {
                assert_eq!($T::from_be(A.to_be()), A);
                assert_eq!($T::from_be(B.to_be()), B);
                assert_eq!($T::from_be(C.to_be()), C);
                assert_eq!($T::from_be(_0), _0);
                assert_eq!($T::from_be(_1), _1);
                assert_eq!(_0.to_be(), _0);
                assert_eq!(_1.to_be(), _1);
            }

            #[test]
            fn test_unsigned_checked_div() {
                assert!((10 as $T).checked_div(2) == Some(5));
                assert!((5 as $T).checked_div(0) == None);
            }

            fn from_str<T: FromStr>(t: &str) -> Option<T> {
                FromStr::from_str(t).ok()
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

            #[test]
            pub fn test_parse_bytes() {
                assert_eq!($T::from_str_radix("123", 10), Ok(123 as $T));
                assert_eq!($T::from_str_radix("1001", 2), Ok(9 as $T));
                assert_eq!($T::from_str_radix("123", 8), Ok(83 as $T));
                assert_eq!(u16::from_str_radix("123", 16), Ok(291 as u16));
                assert_eq!(u16::from_str_radix("ffff", 16), Ok(65535 as u16));
                assert_eq!($T::from_str_radix("z", 36), Ok(35 as $T));

                assert_eq!($T::from_str_radix("Z", 10).ok(), None::<$T>);
                assert_eq!($T::from_str_radix("_", 2).ok(), None::<$T>);
            }

            #[test]
            fn test_pow() {
                let mut r = 2 as $T;
                assert_eq!(r.pow(2), 4 as $T);
                assert_eq!(r.pow(0), 1 as $T);
                assert_eq!(r.wrapping_pow(2), 4 as $T);
                assert_eq!(r.wrapping_pow(0), 1 as $T);
                assert_eq!(r.checked_pow(2), Some(4 as $T));
                assert_eq!(r.checked_pow(0), Some(1 as $T));
                assert_eq!(r.overflowing_pow(2), (4 as $T, false));
                assert_eq!(r.overflowing_pow(0), (1 as $T, false));
                assert_eq!(r.saturating_pow(2), 4 as $T);
                assert_eq!(r.saturating_pow(0), 1 as $T);

                r = MAX;
                // use `^` to represent .pow() with no overflow.
                // if itest::MAX == 2^j-1, then itest is a `j` bit int,
                // so that `itest::MAX*itest::MAX == 2^(2*j)-2^(j+1)+1`,
                // thussaturating_pow the overflowing result is exactly 1.
                assert_eq!(r.wrapping_pow(2), 1 as $T);
                assert_eq!(r.checked_pow(2), None);
                assert_eq!(r.overflowing_pow(2), (1 as $T, true));
                assert_eq!(r.saturating_pow(2), MAX);
            }
        }
    };
}
