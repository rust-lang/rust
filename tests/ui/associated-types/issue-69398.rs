//@ check-pass
#![feature(trivial_bounds)]

pub trait Foo {
    type Bar;
}

pub trait Broken {
    type Assoc;
    fn broken(&self) where Self::Assoc: Foo;
}

impl<T> Broken for T {
    type Assoc = ();
    fn broken(&self) where Self::Assoc: Foo {
        let _x: <Self::Assoc as Foo>::Bar;
    }
}

fn main() {
    let _m: &dyn Broken<Assoc=()> = &();
}
