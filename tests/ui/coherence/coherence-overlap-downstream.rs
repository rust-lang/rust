// Tests that we consider `T: Sugar + Fruit` to be ambiguous, even
// though no impls are found.

pub trait Sugar {}
pub trait Fruit {}
pub trait Sweet {}
impl<T:Sugar> Sweet for T { }
impl<T:Fruit> Sweet for T { }
//~^ ERROR E0119

pub trait Foo<X> {}
pub trait Bar<X> {}
impl<X, T> Foo<X> for T where T: Bar<X> {}
impl<X> Foo<X> for i32 {}
//~^ ERROR E0119

fn main() { }
