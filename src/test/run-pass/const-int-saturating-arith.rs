const INT_U32_NO: u32 = (42 as u32).saturating_add(2);
const INT_U32: u32 = u32::max_value().saturating_add(1);
const INT_U128: u128 = u128::max_value().saturating_add(1);
const INT_I128: i128 = i128::max_value().saturating_add(1);
const INT_I128_NEG: i128 = i128::min_value().saturating_add(-1);

fn main() {
    assert_eq!(INT_U32_NO, 44);
    assert_eq!(INT_U32, u32::max_value());
    assert_eq!(INT_U128, u128::max_value());
    assert_eq!(INT_I128, i128::max_value());
    assert_eq!(INT_I128_NEG, i128::min_value());
}