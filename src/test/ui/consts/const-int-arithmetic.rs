// run-pass

#![feature(const_int_checked)]
#![feature(const_int_euclidean)]
#![feature(const_int_overflowing)]
#![feature(const_int_saturating)]
#![feature(const_int_wrapping)]

macro_rules! assert_same_const {
    ($(const $ident:ident: $ty:ty = $exp:expr;)+) => {
        $(const $ident: $ty = $exp;)+

        pub fn main() {
            $(assert_eq!($exp, $ident);)+
        }
    }
}

assert_same_const! {
    // `const_int_checked`
    const CHECKED_ADD_I32_A: Option<i32> = 5i32.checked_add(2);
    const CHECKED_ADD_I8_A: Option<i8> = 127i8.checked_add(2);
    const CHECKED_ADD_U8_A: Option<u8> = 255u8.checked_add(2);

    const CHECKED_SUB_I32_A: Option<i32> = 5i32.checked_sub(2);
    const CHECKED_SUB_I8_A: Option<i8> = (-127 as i8).checked_sub(2);
    const CHECKED_SUB_U8_A: Option<u8> = 1u8.checked_sub(2);

    const CHECKED_MUL_I32_A: Option<i32> = 5i32.checked_mul(7777);
    const CHECKED_MUL_I8_A: Option<i8> = (-127 as i8).checked_mul(-99);
    const CHECKED_MUL_U8_A: Option<u8> = 1u8.checked_mul(3);

    const CHECKED_DIV_I32_A: Option<i32> = 5i32.checked_div(7777);
    const CHECKED_DIV_I8_A: Option<i8> = (-127 as i8).checked_div(-99);
    const CHECKED_DIV_U8_A: Option<u8> = 1u8.checked_div(3);

    const CHECKED_REM_I32_A: Option<i32> = 5i32.checked_rem(7777);
    const CHECKED_REM_I8_A: Option<i8> = (-127 as i8).checked_rem(-99);
    const CHECKED_REM_U8_A: Option<u8> = 1u8.checked_rem(3);
    const CHECKED_REM_U8_B: Option<u8> = 1u8.checked_rem(0);

    const CHECKED_NEG_I32_A: Option<i32> = 5i32.checked_neg();
    const CHECKED_NEG_I8_A: Option<i8> = (-127 as i8).checked_neg();
    const CHECKED_NEG_U8_A: Option<u8> = 1u8.checked_neg();
    const CHECKED_NEG_U8_B: Option<u8> = u8::min_value().checked_neg();

    const CHECKED_SHL_I32_A: Option<i32> = 5i32.checked_shl(77777);
    const CHECKED_SHL_I8_A: Option<i8> = (-127 as i8).checked_shl(2);
    const CHECKED_SHL_U8_A: Option<u8> = 1u8.checked_shl(8);
    const CHECKED_SHL_U8_B: Option<u8> = 1u8.checked_shl(0);

    const CHECKED_SHR_I32_A: Option<i32> = 5i32.checked_shr(77777);
    const CHECKED_SHR_I8_A: Option<i8> = (-127 as i8).checked_shr(2);
    const CHECKED_SHR_U8_A: Option<u8> = 1u8.checked_shr(8);
    const CHECKED_SHR_U8_B: Option<u8> = 1u8.checked_shr(0);

    const CHECKED_ABS_I32_A: Option<i32> = 5i32.checked_abs();
    const CHECKED_ABS_I8_A: Option<i8> = (-127 as i8).checked_abs();
    const CHECKED_ABS_I8_B: Option<i8> = 1i8.checked_abs();
    const CHECKED_ABS_I8_C: Option<i8> = i8::min_value().checked_abs();

    // `const_int_overflowing`
    const DIV_A: (i8, bool) = 8i8.overflowing_div(2);
    const DIV_B: (i8, bool) = 8i8.overflowing_div(3);
    const DIV_C: (i8, bool) = i8::min_value().overflowing_div(-1i8);
    const DIV_D: (u8, bool) = 8u8.overflowing_div(2);
    const DIV_E: (u8, bool) = 8u8.overflowing_div(3);

    const REM_A: (i8, bool) = 8i8.overflowing_rem(2);
    const REM_B: (i8, bool) = 8i8.overflowing_rem(3);
    const REM_C: (i8, bool) = i8::min_value().overflowing_rem(-1i8);
    const REM_D: (u8, bool) = 8u8.overflowing_rem(2);
    const REM_E: (u8, bool) = 8u8.overflowing_rem(3);

    // `const_int_saturating`
    const ADD_INT_U32_NO: u32 = (42 as u32).saturating_add(2);
    const ADD_INT_U32: u32 = u32::max_value().saturating_add(1);
    const ADD_INT_U128: u128 = u128::max_value().saturating_add(1);
    const ADD_INT_I128: i128 = i128::max_value().saturating_add(1);
    const ADD_INT_I128_NEG: i128 = i128::min_value().saturating_add(-1);

    const SUB_INT_U32_NO: u32 = (42 as u32).saturating_sub(2);
    const SUB_INT_U32: u32 = (1 as u32).saturating_sub(2);
    const SUB_INT_I32_NO: i32 = (-42 as i32).saturating_sub(2);
    const SUB_INT_I32_NEG: i32 = i32::min_value().saturating_sub(1);
    const SUB_INT_I32_POS: i32 = i32::max_value().saturating_sub(-1);
    const SUB_INT_U128: u128 = (0 as u128).saturating_sub(1);
    const SUB_INT_I128_NEG: i128 = i128::min_value().saturating_sub(1);
    const SUB_INT_I128_POS: i128 = i128::max_value().saturating_sub(-1);

    const MUL_INT_U32_NO: u32 = (42 as u32).saturating_mul(2);
    const MUL_INT_U32: u32 = (1 as u32).saturating_mul(2);
    const MUL_INT_I32_NO: i32 = (-42 as i32).saturating_mul(2);
    const MUL_INT_I32_NEG: i32 = i32::min_value().saturating_mul(1);
    const MUL_INT_I32_POS: i32 = i32::max_value().saturating_mul(2);
    const MUL_INT_U128: u128 = (0 as u128).saturating_mul(1);
    const MUL_INT_I128_NEG: i128 = i128::min_value().saturating_mul(2);
    const MUL_INT_I128_POS: i128 = i128::max_value().saturating_mul(2);

    const NEG_INT_I8: i8 = (-42i8).saturating_neg();
    const NEG_INT_I8_B: i8 = i8::min_value().saturating_neg();
    const NEG_INT_I32: i32 = i32::min_value().saturating_neg();
    const NEG_INT_I32_B: i32 = i32::max_value().saturating_neg();
    const NEG_INT_I128: i128 = i128::min_value().saturating_neg();
    const NEG_INT_I128_B: i128 = i128::max_value().saturating_neg();

    const ABS_INT_I8_A: i8 = 4i8.saturating_abs();
    const ABS_INT_I8_B: i8 = -4i8.saturating_abs();
    const ABS_INT_I8_C: i8 = i8::min_value().saturating_abs();
    const ABS_INT_I32_A: i32 = 4i32.saturating_abs();
    const ABS_INT_I32_B: i32 = -4i32.saturating_abs();
    const ABS_INT_I32_C: i32 = i32::min_value().saturating_abs();
    const ABS_INT_I128_A: i128 = 4i128.saturating_abs();
    const ABS_INT_I128_B: i128 = -4i128.saturating_abs();
    const ABS_INT_I128_C: i128 = i128::min_value().saturating_abs();

    // `const_int_euclidean`
    const CHECKED_DIV_I32_A: Option<i32> = 5i32.checked_div_euclid(7777);
    const CHECKED_DIV_I8_A: Option<i8> = (-127 as i8).checked_div_euclid(-99);
    const CHECKED_DIV_I8_B: Option<i8> = (-127 as i8).checked_div_euclid(1);
    const CHECKED_DIV_I8_C: Option<i8> = i8::min_value().checked_div_euclid(-1);
    const CHECKED_DIV_U8_A: Option<u8> = 1u8.checked_div_euclid(3);

    const CHECKED_REM_I32_A: Option<i32> = 5i32.checked_rem_euclid(7777);
    const CHECKED_REM_I8_A: Option<i8> = (-127 as i8).checked_rem_euclid(-99);
    const CHECKED_REM_I8_B: Option<i8> = (-127 as i8).checked_rem_euclid(0);
    const CHECKED_REM_I8_C: Option<i8> = i8::min_value().checked_rem_euclid(-1);
    const CHECKED_REM_U8_A: Option<u8> = 1u8.checked_rem_euclid(3);

    const WRAPPING_DIV_I32_A: i32 = 5i32.wrapping_div_euclid(7777);
    const WRAPPING_DIV_I8_A: i8 = (-127 as i8).wrapping_div_euclid(-99);
    const WRAPPING_DIV_I8_B: i8 = (-127 as i8).wrapping_div_euclid(1);
    const WRAPPING_DIV_I8_C: i8 = i8::min_value().wrapping_div_euclid(-1);
    const WRAPPING_DIV_U8_A: u8 = 1u8.wrapping_div_euclid(3);

    const WRAPPING_REM_I32_A: i32 = 5i32.wrapping_rem_euclid(7777);
    const WRAPPING_REM_I8_A: i8 = (-127 as i8).wrapping_rem_euclid(-99);
    const WRAPPING_REM_I8_B: i8 = (-127 as i8).wrapping_rem_euclid(1);
    const WRAPPING_REM_I8_C: i8 = i8::min_value().wrapping_rem_euclid(-1);
    const WRAPPING_REM_U8_A: u8 = 1u8.wrapping_rem_euclid(3);

    const OVERFLOWING_DIV_I32_A: (i32, bool) = 5i32.overflowing_div_euclid(7777);
    const OVERFLOWING_DIV_I8_A: (i8, bool) = (-127 as i8).overflowing_div_euclid(-99);
    const OVERFLOWING_DIV_I8_B: (i8, bool) = (-127 as i8).overflowing_div_euclid(1);
    const OVERFLOWING_DIV_I8_C: (i8, bool) = i8::min_value().overflowing_div_euclid(-1);
    const OVERFLOWING_DIV_U8_A: (u8, bool) = 1u8.overflowing_div_euclid(3);

    const OVERFLOWING_REM_I32_A: (i32, bool) = 5i32.overflowing_rem_euclid(7777);
    const OVERFLOWING_REM_I8_A: (i8, bool) = (-127 as i8).overflowing_rem_euclid(-99);
    const OVERFLOWING_REM_I8_B: (i8, bool) = (-127 as i8).overflowing_rem_euclid(1);
    const OVERFLOWING_REM_I8_C: (i8, bool) = i8::min_value().overflowing_rem_euclid(-1);
    const OVERFLOWING_REM_U8_A: (u8, bool) = 1u8.overflowing_rem_euclid(3);

    // `const_int_wrapping`
    const DIV_A: i8 = 8i8.wrapping_div(2);
    const DIV_B: i8 = 8i8.wrapping_div(3);
    const DIV_C: i8 = i8::min_value().wrapping_div(-1i8);
    const DIV_D: u8 = 8u8.wrapping_div(2);
    const DIV_E: u8 = 8u8.wrapping_div(3);

    const REM_A: i8 = 8i8.wrapping_rem(2);
    const REM_B: i8 = 8i8.wrapping_rem(3);
    const REM_C: i8 = i8::min_value().wrapping_rem(-1i8);
    const REM_D: u8 = 8u8.wrapping_rem(2);
    const REM_E: u8 = 8u8.wrapping_rem(3);
}
