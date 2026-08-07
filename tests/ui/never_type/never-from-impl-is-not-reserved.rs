// check that the `for<T> T: From<!>` impl is not reserved anymore

//@ check-pass
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=coherence

#![feature(never_type)]

pub struct MyFoo;
pub trait MyTrait {}

impl MyTrait for MyFoo {}
impl<T> MyTrait for T where T: From<!> {}

fn main() {}
