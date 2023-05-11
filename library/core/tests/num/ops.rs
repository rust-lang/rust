use core::ops::*;

// For types L and R, checks that a trait implementation exists for
//   * binary ops: L op R, L op &R, &L op R and &L op &R
//   * assign ops: &mut L op R, &mut L op &R
macro_rules! impl_defined {
    ($op:ident, $method:ident($lhs:literal, $rhs:literal), $result:literal, $lt:ty, $rt:ty) => {
        let lhs = $lhs as $lt;
        let rhs = $rhs as $rt;
        assert_eq!($result as $lt, $op::$method(lhs, rhs));
        assert_eq!($result as $lt, $op::$method(lhs, &rhs));
        assert_eq!($result as $lt, $op::$method(&lhs, rhs));
        assert_eq!($result as $lt, $op::$method(&lhs, &rhs));
    };
    ($op:ident, $method:ident(&mut $lhs:literal, $rhs:literal), $result:literal, $lt:ty, $rt:ty) => {
        let rhs = $rhs as $rt;
        let mut lhs = $lhs as $lt;
        $op::$method(&mut lhs, rhs);
        assert_eq!($result as $lt, lhs);

        let mut lhs = $lhs as $lt;
        $op::$method(&mut lhs, &rhs);
        assert_eq!($result as $lt, lhs);
    };
}

// For all specified types T, checks that a trait implementation exists for
//   * binary ops: T op T, T op &T, &T op T and &T op &T
//   * assign ops: &mut T op T, &mut T op &T
//   * unary ops: op T and op &T
macro_rules! impls_defined {
    ($op:ident, $method:ident($lhs:literal, $rhs:literal), $result:literal, $($t:ty),+) => {$(
        impl_defined!($op, $method($lhs, $rhs), $result, $t, $t);
    )+};
    ($op:ident, $method:ident(&mut $lhs:literal, $rhs:literal), $result:literal, $($t:ty),+) => {$(
        impl_defined!($op, $method(&mut $lhs, $rhs), $result, $t, $t);
    )+};
    ($op:ident, $method:ident($operand:literal), $result:literal, $($t:ty),+) => {$(
        let operand = $operand as $t;
        assert_eq!($result as $t, $op::$method(operand));
        assert_eq!($result as $t, $op::$method(&operand));
    )+};
}

macro_rules! test_op {
    ($fn_name:ident, $op:ident::$method:ident($lhs:literal), $result:literal, $($t:ty),+) => {
        #[test]
        fn $fn_name() {
            impls_defined!($op, $method($lhs), $result, $($t),+);
        }
    };
}

test_op!(test_neg_defined, Neg::neg(0), 0, i8, i16, i32, i64, f32, f64);
#[cfg(not(target_os = "emscripten"))]
test_op!(test_neg_defined_128, Neg::neg(0), 0, i128);

test_op!(test_not_defined_bool, Not::not(true), false, bool);

macro_rules! test_arith_op {
    ($fn_name:ident, $op:ident::$method:ident($lhs:literal, $rhs:literal)) => {
        #[test]
        fn $fn_name() {
            impls_defined!(
                $op,
                $method($lhs, $rhs),
                0,
                i8,
                i16,
                i32,
                i64,
                isize,
                u8,
                u16,
                u32,
                u64,
                usize,
                f32,
                f64
            );
            #[cfg(not(target_os = "emscripten"))]
            impls_defined!($op, $method($lhs, $rhs), 0, i128, u128);
        }
    };
    ($fn_name:ident, $op:ident::$method:ident(&mut $lhs:literal, $rhs:literal)) => {
        #[test]
        fn $fn_name() {
            impls_defined!(
                $op,
                $method(&mut $lhs, $rhs),
                0,
                i8,
                i16,
                i32,
                i64,
                isize,
                u8,
                u16,
                u32,
                u64,
                usize,
                f32,
                f64
            );
            #[cfg(not(target_os = "emscripten"))]
            impls_defined!($op, $method(&mut $lhs, $rhs), 0, i128, u128);
        }
    };
}

test_arith_op!(test_add_defined, Add::add(0, 0));
test_arith_op!(test_add_assign_defined, AddAssign::add_assign(&mut 0, 0));
test_arith_op!(test_sub_defined, Sub::sub(0, 0));
test_arith_op!(test_sub_assign_defined, SubAssign::sub_assign(&mut 0, 0));
test_arith_op!(test_mul_defined, Mul::mul(0, 0));
test_arith_op!(test_mul_assign_defined, MulAssign::mul_assign(&mut 0, 0));
test_arith_op!(test_div_defined, Div::div(0, 1));
test_arith_op!(test_div_assign_defined, DivAssign::div_assign(&mut 0, 1));
test_arith_op!(test_rem_defined, Rem::rem(0, 1));
test_arith_op!(test_rem_assign_defined, RemAssign::rem_assign(&mut 0, 1));

macro_rules! test_bitop {
    ($test_name:ident, $op:ident::$method:ident) => {
        #[test]
        fn $test_name() {
            impls_defined!(
                $op,
                $method(0, 0),
                0,
                i8,
                i16,
                i32,
                i64,
                isize,
                u8,
                u16,
                u32,
                u64,
                usize
            );
            #[cfg(not(target_os = "emscripten"))]
            impls_defined!($op, $method(0, 0), 0, i128, u128);
            impls_defined!($op, $method(false, false), false, bool);
        }
    };
}
macro_rules! test_bitop_assign {
    ($test_name:ident, $op:ident::$method:ident) => {
        #[test]
        fn $test_name() {
            impls_defined!(
                $op,
                $method(&mut 0, 0),
                0,
                i8,
                i16,
                i32,
                i64,
                isize,
                u8,
                u16,
                u32,
                u64,
                usize
            );
            #[cfg(not(target_os = "emscripten"))]
            impls_defined!($op, $method(&mut 0, 0), 0, i128, u128);
            impls_defined!($op, $method(&mut false, false), false, bool);
        }
    };
}

test_bitop!(test_bitand_defined, BitAnd::bitand);
test_bitop_assign!(test_bitand_assign_defined, BitAndAssign::bitand_assign);
test_bitop!(test_bitor_defined, BitOr::bitor);
test_bitop_assign!(test_bitor_assign_defined, BitOrAssign::bitor_assign);
test_bitop!(test_bitxor_defined, BitXor::bitxor);
test_bitop_assign!(test_bitxor_assign_defined, BitXorAssign::bitxor_assign);

macro_rules! test_shift_inner {
    ($op:ident::$method:ident, $lt:ty, $($rt:ty),+) => {
        $(impl_defined!($op, $method(0,0), 0, $lt, $rt);)+
    };
    ($op:ident::$method:ident, $lt:ty) => {
        test_shift_inner!($op::$method, $lt, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
        #[cfg(not(target_os = "emscripten"))]
        test_shift_inner!($op::$method, $lt, i128, u128);
    };
}

macro_rules! test_shift {
    ($op:ident::$method:ident, $($lt:ty),+) => {
        $(test_shift_inner!($op::$method, $lt);)+
    };
    ($test_name:ident, $op:ident::$method:ident) => {
        #[test]
        fn $test_name() {
            test_shift!($op::$method, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
            #[cfg(not(target_os = "emscripten"))]
            test_shift!($op::$method, i128, u128);
        }
    };
}

macro_rules! test_shift_assign_inner {
    ($op:ident::$method:ident, $lt:ty, $($rt:ty),+) => {
        $(impl_defined!($op, $method(&mut 0,0), 0, $lt, $rt);)+
    };
    ($op:ident::$method:ident, $lt:ty) => {
        test_shift_assign_inner!($op::$method, $lt, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
        #[cfg(not(target_os = "emscripten"))]
        test_shift_assign_inner!($op::$method, $lt, i128, u128);
    };
}

macro_rules! test_shift_assign {
    ($op:ident::$method:ident, $($lt:ty),+) => {
        $(test_shift_assign_inner!($op::$method, $lt);)+
    };
    ($test_name:ident, $op:ident::$method:ident) => {
        #[test]
        fn $test_name() {
            test_shift_assign!($op::$method, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
            #[cfg(not(target_os = "emscripten"))]
            test_shift_assign!($op::$method, i128, u128);
        }
    };
}
test_shift!(test_shl_defined, Shl::shl);
test_shift_assign!(test_shl_assign_defined, ShlAssign::shl_assign);
test_shift!(test_shr_defined, Shr::shr);
test_shift_assign!(test_shr_assign_defined, ShrAssign::shr_assign);
