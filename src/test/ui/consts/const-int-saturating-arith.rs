// run-pass
#![feature(const_saturating_int_methods)]
#![feature(const_int_saturating)]
#![feature(saturating_neg)]

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

fn main() {
    assert_eq!(ADD_INT_U32_NO, 44);
    assert_eq!(ADD_INT_U32, u32::max_value());
    assert_eq!(ADD_INT_U128, u128::max_value());
    assert_eq!(ADD_INT_I128, i128::max_value());
    assert_eq!(ADD_INT_I128_NEG, i128::min_value());

    assert_eq!(SUB_INT_U32_NO, 40);
    assert_eq!(SUB_INT_U32, 0);
    assert_eq!(SUB_INT_I32_NO, -44);
    assert_eq!(SUB_INT_I32_NEG, i32::min_value());
    assert_eq!(SUB_INT_I32_POS, i32::max_value());
    assert_eq!(SUB_INT_U128, 0);
    assert_eq!(SUB_INT_I128_NEG, i128::min_value());
    assert_eq!(SUB_INT_I128_POS, i128::max_value());

    assert_eq!(MUL_INT_U32_NO, 84);
    assert_eq!(MUL_INT_U32, 2);
    assert_eq!(MUL_INT_I32_NO, -84);
    assert_eq!(MUL_INT_I32_NEG, i32::min_value());
    assert_eq!(MUL_INT_I32_POS, i32::max_value());
    assert_eq!(MUL_INT_U128, 0);
    assert_eq!(MUL_INT_I128_NEG, i128::min_value());
    assert_eq!(MUL_INT_I128_POS, i128::max_value());

    assert_eq!(NEG_INT_I8, 42);
    assert_eq!(NEG_INT_I8_B, i8::max_value());
    assert_eq!(NEG_INT_I32, i32::max_value());
    assert_eq!(NEG_INT_I32_B, i32::min_value() + 1);
    assert_eq!(NEG_INT_I128, i128::max_value());
    assert_eq!(NEG_INT_I128_B, i128::min_value() + 1);

    assert_eq!(ABS_INT_I8_A, 4i8);
    assert_eq!(ABS_INT_I8_B, -4i8);
    assert_eq!(ABS_INT_I8_C, i8::max_value());
    assert_eq!(ABS_INT_I32_A, 4i32);
    assert_eq!(ABS_INT_I32_B, -4i32);
    assert_eq!(ABS_INT_I32_C, i32::max_value());
    assert_eq!(ABS_INT_I128_A, 4i128);
    assert_eq!(ABS_INT_I128_B, -4i128);
    assert_eq!(ABS_INT_I128_C, i128::max_value());
}
