// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

use std::mem::MaybeUninit;

#[repr(transparent)]
pub struct MaybeUninitWrapper<const N: usize>(MaybeUninit<[u64; N]>);

fn main() {}
