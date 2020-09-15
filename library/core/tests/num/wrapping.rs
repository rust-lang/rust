// run-pass

use std::num::Wrapping;

macro_rules! wrapping_operation {
    ($result:expr, $lhs:ident $op:tt $rhs:expr) => {
        assert_eq!($result, $lhs $op $rhs);
        assert_eq!($result, &$lhs $op $rhs);
        assert_eq!($result, $lhs $op &$rhs);
        assert_eq!($result, &$lhs $op &$rhs);
    };
    ($result:expr, $op:tt $expr:expr) => {
        assert_eq!($result, $op $expr);
        assert_eq!($result, $op &$expr);
    };
}

macro_rules! wrapping_assignment {
    ($result:expr, $lhs:ident $op:tt $rhs:expr) => {
        let mut lhs1 = $lhs;
        lhs1 $op $rhs;
        assert_eq!($result, lhs1);

        let mut lhs2 = $lhs;
        lhs2 $op &$rhs;
        assert_eq!($result, lhs2);
    };
}

macro_rules! wrapping_test {
    ($type:ty, $min:expr, $max:expr) => {
        let zero: Wrapping<$type> = Wrapping(0);
        let one: Wrapping<$type> = Wrapping(1);
        let min: Wrapping<$type> = Wrapping($min);
        let max: Wrapping<$type> = Wrapping($max);

        wrapping_operation!(min, max + one);
        wrapping_assignment!(min, max += one);
        wrapping_operation!(max, min - one);
        wrapping_assignment!(max, min -= one);
        wrapping_operation!(max, max * one);
        wrapping_assignment!(max, max *= one);
        wrapping_operation!(max, max / one);
        wrapping_assignment!(max, max /= one);
        wrapping_operation!(zero, max % one);
        wrapping_assignment!(zero, max %= one);
        wrapping_operation!(zero, zero & max);
        wrapping_assignment!(zero, zero &= max);
        wrapping_operation!(max, zero | max);
        wrapping_assignment!(max, zero |= max);
        wrapping_operation!(zero, max ^ max);
        wrapping_assignment!(zero, max ^= max);
        wrapping_operation!(zero, zero << 1usize);
        wrapping_assignment!(zero, zero <<= 1usize);
        wrapping_operation!(zero, zero >> 1usize);
        wrapping_assignment!(zero, zero >>= 1usize);
        wrapping_operation!(zero, -zero);
        wrapping_operation!(max, !min);
    };
}

fn main() {
    wrapping_test!(i8, std::i8::MIN, std::i8::MAX);
    wrapping_test!(i16, std::i16::MIN, std::i16::MAX);
    wrapping_test!(i32, std::i32::MIN, std::i32::MAX);
    wrapping_test!(i64, std::i64::MIN, std::i64::MAX);
    #[cfg(not(target_os = "emscripten"))]
    wrapping_test!(i128, std::i128::MIN, std::i128::MAX);
    wrapping_test!(isize, std::isize::MIN, std::isize::MAX);
    wrapping_test!(u8, std::u8::MIN, std::u8::MAX);
    wrapping_test!(u16, std::u16::MIN, std::u16::MAX);
    wrapping_test!(u32, std::u32::MIN, std::u32::MAX);
    wrapping_test!(u64, std::u64::MIN, std::u64::MAX);
    #[cfg(not(target_os = "emscripten"))]
    wrapping_test!(u128, std::u128::MIN, std::u128::MAX);
    wrapping_test!(usize, std::usize::MIN, std::usize::MAX);
}
