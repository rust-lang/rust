//@ check-pass

#![feature(tuple_trait)]

use std::marker::Tuple;

trait Foo {}
impl<T> Foo for T where T: Tuple {}
impl Foo for i32 {} // works because `Tuple` is `#[fundamental]`

fn main() {}
