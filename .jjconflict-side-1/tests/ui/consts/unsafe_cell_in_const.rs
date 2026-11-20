//! Ensure we do not complain about zero-sized `UnsafeCell` in a const in any form.
//! See <https://github.com/rust-lang/rust/issues/142948>.

//@ check-pass
use std::cell::UnsafeCell;

const X1: &mut UnsafeCell<[i32; 0]> = UnsafeCell::from_mut(&mut []);

const X2: &mut UnsafeCell<[i32]> = UnsafeCell::from_mut(&mut []);

trait Trait {}
impl Trait for [i32; 0] {}
const X3: &mut UnsafeCell<dyn Trait> = UnsafeCell::from_mut(&mut []);

fn main() {}
