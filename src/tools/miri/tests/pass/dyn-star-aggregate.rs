#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

#[derive(Debug)]
#[repr(C)]
pub struct Foo(usize);

fn main() {
    let x = Foo(1) as dyn* Debug;
    assert_eq!(format!("{x:?}"), "Foo(1)");
}
