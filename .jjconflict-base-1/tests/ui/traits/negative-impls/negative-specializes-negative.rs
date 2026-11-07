#![feature(specialization)]
#![feature(negative_impls)]

// Test a negative impl that "specializes" another negative impl.
//
//@ check-pass

trait MyTrait {}

impl<T> !MyTrait for T {}
impl !MyTrait for u32 {}

fn main() {}
