#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]
#![deny(where_clauses_object_safety)]


const fn bar<T: ?Sized>() -> usize { 7 }

trait Foo {
    fn test(&self) where [u8; bar::<Self>()]: Sized;
    //~^ ERROR the trait `Foo` cannot be made into an object
    //~| WARN this was previously accepted by the compiler but is being phased out
}

impl Foo for () {
    fn test(&self) where [u8; bar::<Self>()]: Sized {}
}

fn use_dyn(v: &dyn Foo) {
    v.test();
}

fn main() {}
