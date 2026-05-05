#![feature(fn_static)]
//@ check-pass

use std::ops::FnPtr;

trait Foo {}
impl<T> Foo for Vec<T> where T: FnPtr {}

fn main() {}
