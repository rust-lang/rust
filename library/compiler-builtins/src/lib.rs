#![allow(unused_features)]
#![cfg_attr(not(test), no_std)]
#![feature(asm)]
#![feature(core_intrinsics)]
#![feature(naked_functions)]
// TODO(rust-lang/rust#35021) uncomment when that PR lands
// #![feature(rustc_builtins)]

#[cfg(test)]
extern crate core;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(test)]
mod test;

/// Trait for some basic operations on integers
trait Int {
    fn bits() -> usize;
}

// TODO: Once i128/u128 support lands, we'll want to add impls for those as well
impl Int for u32 {
    fn bits() -> usize { 32 }
}
impl Int for i32 {
    fn bits() -> usize { 32 }
}
impl Int for u64 {
    fn bits() -> usize { 64 }
}
impl Int for i64 {
    fn bits() -> usize { 64 }
}

/// Trait to convert an integer to/from smaller parts
trait LargeInt {
    type LowHalf;
    type HighHalf;

    fn low(self) -> Self::LowHalf;
    fn high(self) -> Self::HighHalf;
    fn from_parts(low: Self::LowHalf, high: Self::HighHalf) -> Self;
}

// TODO: Once i128/u128 support lands, we'll want to add impls for those as well
impl LargeInt for u64 {
    type LowHalf = u32;
    type HighHalf = u32;

    fn low(self) -> u32 {
        self as u32
    }
    fn high(self) -> u32 {
        (self >> 32) as u32
    }
    fn from_parts(low: u32, high: u32) -> u64 {
        low as u64 | ((high as u64) << 32)
    }
}
impl LargeInt for i64 {
    type LowHalf = u32;
    type HighHalf = i32;

    fn low(self) -> u32 {
        self as u32
    }
    fn high(self) -> i32 {
        (self >> 32) as i32
    }
    fn from_parts(low: u32, high: i32) -> i64 {
        low as i64 | ((high as i64) << 32)
    }
}

macro_rules! absv_i2 {
    ($intrinsic:ident : $ty:ty) => {
        #[no_mangle]
        pub extern "C" fn $intrinsic(x: $ty) -> $ty {
            let n = <$ty>::bits();
            if x == 1 << (n - 1) {
                panic!();
            }
            let y = x >> (n - 1);
            (x ^ y) - y
        }

    }
}

absv_i2!(__absvsi2: i32);
absv_i2!(__absvdi2: i64);
// TODO(rust-lang/35118)?
// absv_i2!(__absvti2, i128);
