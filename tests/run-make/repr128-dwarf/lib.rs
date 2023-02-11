#![crate_type = "lib"]
#![feature(repr128)]

// Use .to_le() to ensure that the bytes are in the same order on both little- and big-endian
// platforms.

#[repr(u128)]
pub enum U128Enum {
    U128A = 0_u128.to_le(),
    U128B = 1_u128.to_le(),
    U128C = (u64::MAX as u128 + 1).to_le(),
    U128D = u128::MAX.to_le(),
}

#[repr(i128)]
pub enum I128Enum {
    I128A = 0_i128.to_le(),
    I128B = (-1_i128).to_le(),
    I128C = i128::MIN.to_le(),
    I128D = i128::MAX.to_le(),
}

pub fn f(_: U128Enum, _: I128Enum) {}
