#![feature(trait_alias)]

trait Foo {}
trait A<T: Foo> {}
trait B<T> = A<T>; //~ ERROR trait `Foo` is not implemented for `T`

fn main() {}
