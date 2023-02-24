// run-pass

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete

use std::fmt::Debug;

#[derive(Debug)]
#[repr(C)]
pub struct Foo(usize);

fn main() {
    let x = Foo(1) as dyn* Debug;
    assert_eq!(format!("{x:?}"), "Foo(1)");
}
