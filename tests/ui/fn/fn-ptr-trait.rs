#![feature(fn_static)]
//@ check-pass

use std::ops::FnPtr;

trait Foo {}
impl<T> Foo for T where T: FnPtr {}
impl Foo for i32 {} // works because `FnPtr` is `#[fundamental]`

fn main() {}
