// ignore-emscripten no i128 support
#![feature(const_saturating_int_methods)]

const INT_U32_NO: u32 = (42 as u32).saturating_add(2);
const INT_U32: u32 = u32::max_value().saturating_add(1);
const INT_U128: u128 = u128::max_value().saturating_add(1);
const INT_I128: i128 = i128::max_value().saturating_add(1);
const INT_I128_NEG: i128 = i128::min_value().saturating_add(-1);

const INT_U32_NO_SUB: u32 = (42 as u32).saturating_sub(2);
const INT_U32_SUB: u32 = (1 as u32).saturating_sub(2);
const INT_I32_NO_SUB: i32 = (-42 as i32).saturating_sub(2);
const INT_I32_NEG_SUB: i32 = i32::min_value().saturating_sub(1);
const INT_I32_POS_SUB: i32 = i32::max_value().saturating_sub(-1);
const INT_U128_SUB: u128 = (0 as u128).saturating_sub(1);
const INT_I128_NEG_SUB: i128 = i128::min_value().saturating_sub(1);
const INT_I128_POS_SUB: i128 = i128::max_value().saturating_sub(-1);

fn main() {
    assert_eq!(INT_U32_NO, 44);
    assert_eq!(INT_U32, u32::max_value());
    assert_eq!(INT_U128, u128::max_value());
    assert_eq!(INT_I128, i128::max_value());
    assert_eq!(INT_I128_NEG, i128::min_value());

    assert_eq!(INT_U32_NO_SUB, 40);
    assert_eq!(INT_U32_SUB, 0);
    assert_eq!(INT_I32_NO_SUB, -44);
    assert_eq!(INT_I32_NEG_SUB, i32::min_value());
    assert_eq!(INT_I32_POS_SUB, i32::max_value());
    assert_eq!(INT_U128_SUB, 0);
    assert_eq!(INT_I128_NEG_SUB, i128::min_value());
    assert_eq!(INT_I128_POS_SUB, i128::max_value());
}
