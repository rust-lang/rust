use std::ops::DerefMut;

trait Foo {}
impl<T: DerefMut> Foo for T {}
impl<U> Foo for &U {}
//~^ ERROR: conflicting implementations of trait `Foo` for type `&_` [E0119]

fn main() {}
