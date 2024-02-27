trait Bar: Iterator {}

impl Bar for String {} //~ ERROR `String` is not an iterator [E0277]

impl<T> Bar for Vec<T> {} //~ ERROR `Vec<T>` is not an iterator [E0277]

use std::ops::IndexMut;
trait Foo<Idx>: IndexMut<Idx> {} 

impl<Idx> Foo<Idx> for String {} //~ ERROR the type `String` cannot be mutably indexed by `Idx` [E0277]

fn main() {}
