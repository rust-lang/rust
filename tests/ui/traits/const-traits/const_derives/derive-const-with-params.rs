//@ known-bug: #110395
// FIXME(const_trait_impl) check-pass

#![feature(derive_const)]
#![feature(const_trait_impl)]

#[derive_const(PartialEq)]
pub struct Reverse<T>(T);

const fn foo(a: Reverse<i32>, b: Reverse<i32>) -> bool {
    a == b
}

fn main() {}
