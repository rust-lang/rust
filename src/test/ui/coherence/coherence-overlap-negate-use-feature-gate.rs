// check-pass

#![feature(negative_impls)]

use std::ops::DerefMut;

trait Foo {}
impl<T: DerefMut> Foo for T {}
impl<U> Foo for &U {}

fn main() {}
