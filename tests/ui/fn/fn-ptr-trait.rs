#![feature(fn_ptr_trait)]
//@ check-pass

use std::marker::FnPtr;

trait Foo {}
impl<T> Foo for Vec<T> where T: FnPtr {}

fn main() {}
