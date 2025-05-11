//@ dont-require-annotations: NOTE

pub trait Trait1<X> {
    type Output;
}

pub trait Trait2<X> {}

pub struct A;

impl<X, T> Trait1<X> for T where T: Trait2<X> {
    type Output = ();
}

impl<X> Trait1<Box<X>> for A {
//~^ ERROR conflicting implementations of trait
//~| NOTE downstream crates may implement trait `Trait2<std::boxed::Box<_>>` for type `A`
    type Output = i32;
}

fn main() {}
