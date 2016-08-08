#![allow(unused_features)]
#![cfg_attr(not(test), no_std)]
#![feature(asm)]
#![feature(core_intrinsics)]
#![feature(naked_functions)]
// TODO(rust-lang/rust#35021) uncomment when that PR lands
// #![feature(rustc_builtins)]

#[cfg(test)]
extern crate core;

use core::mem;

#[cfg(target_arch = "arm")]
pub mod arm;

#[cfg(test)]
mod test;

const CHAR_BITS: usize = 8;

macro_rules! absv_i2 {
    ($intrinsic:ident : $ty:ty) => {
        #[no_mangle]
        pub extern "C" fn $intrinsic(x: $ty) -> $ty {
            let n = mem::size_of::<$ty>() * CHAR_BITS;
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
