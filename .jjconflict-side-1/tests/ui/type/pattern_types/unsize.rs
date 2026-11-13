//! Show that pattern-types with pointer base types can be part of unsizing coercions

//@ check-pass

#![feature(pattern_type_macro, pattern_types)]

use std::pat::pattern_type;

type NonNull<T> = pattern_type!(*const T is !null);

trait Trait {}
impl Trait for u32 {}
impl Trait for i32 {}

fn main() {
    let x: NonNull<u32> = unsafe { std::mem::transmute(std::ptr::dangling::<u32>()) };
    let x: NonNull<dyn Trait> = x;
}
