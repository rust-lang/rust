trait Foo { }

impl<T: std::ops::DerefMut> Foo for T { }

impl<T> Foo for &T { }
//~^ ERROR conflicting implementations of trait `Foo` for type `&_` [E0119]

fn main() { }
