#![feature(fn_ptr_trait)]
//@ check-pass

use std::marker::FnPtr;

trait Foo {}
impl<T> Foo for T where T: FnPtr {}
impl Foo for i32 {} // works because `FnPtr` is `#[fundamental]`

fn main() {}
