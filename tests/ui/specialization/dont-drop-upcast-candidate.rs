#![feature(unsize)]

use std::marker::Unsize;
use std::ops::Deref;

trait Foo: Bar {}
trait Bar {}

impl<T> Bar for T where dyn Foo: Unsize<dyn Bar> {}
impl Bar for () {}
//~^ ERROR conflicting implementations of trait `Bar` for type `()`

fn main() {}
