#![feature(trait_alias)]

trait Foo {}
trait A<T: Foo> {}
trait B<T> = A<T>; //~ ERROR `T: Foo` is not satisfied

fn main() {}
