#![feature(rustc_attrs)]

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

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let _m: &Broken<Assoc=()> = &();
}
