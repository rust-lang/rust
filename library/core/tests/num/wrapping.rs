use core::num::Wrapping;

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
        #[test]
        fn wrapping_$type() {
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
        }
    };
}

wrapping_test!(i8, i8::MIN, i8::MAX);
wrapping_test!(i16, i16::MIN, i16::MAX);
wrapping_test!(i32, i32::MIN, i32::MAX);
wrapping_test!(i64, i64::MIN, i64::MAX);
#[cfg(not(target_os = "emscripten"))]
wrapping_test!(i128, i128::MIN, i128::MAX);
wrapping_test!(isize, isize::MIN, isize::MAX);
wrapping_test!(u8, u8::MIN, u8::MAX);
wrapping_test!(u16, u16::MIN, u16::MAX);
wrapping_test!(u32, u32::MIN, u32::MAX);
wrapping_test!(u64, u64::MIN, u64::MAX);
#[cfg(not(target_os = "emscripten"))]
wrapping_test!(u128, u128::MIN, u128::MAX);
wrapping_test!(usize, usize::MIN, usize::MAX);
