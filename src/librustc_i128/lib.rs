#![allow(non_camel_case_types)]
#![cfg_attr(not(stage0), feature(i128_type))]

#[cfg(stage0)]
pub type i128 = i64;
#[cfg(stage0)]
pub type u128 = u64;

#[cfg(not(stage0))]
pub type i128 = int::_i128;
#[cfg(not(stage0))]
pub type u128 = int::_u128;
#[cfg(not(stage0))]
mod int {
    pub type _i128 = i128;
    pub type _u128 = u128;
}
