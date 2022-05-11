// check-pass

#![feature(return_position_impl_trait_v2)]

use std::fmt::Debug;

fn foo() -> impl Debug {
    22
}

fn main() {
    assert_eq!("22", format!("{:?}", foo()));
}
