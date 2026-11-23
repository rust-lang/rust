// Regression test for https://github.com/rust-lang/rust/issues/165471

//@ run-pass

#![feature(gca_min_const_items)]
#![feature(associated_type_defaults)]

use std::gca;

trait Trait {
    #[rustc_always_gca]
    const K: u8 = gca!(3);
}

struct Defaulted;
impl Trait for Defaulted {}

struct Overridden;
impl Trait for Overridden {
    const K: u8 = gca!(7);
}

fn main() {
    assert_eq!(<Defaulted as Trait>::K, 3);
    assert_eq!(<Overridden as Trait>::K, 7);
}
