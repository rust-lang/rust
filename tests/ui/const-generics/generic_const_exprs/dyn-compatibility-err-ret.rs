#![feature(generic_const_exprs)]
#![allow(incomplete_features)]


const fn bar<T: ?Sized>() -> usize { 7 }

trait Foo {
    fn test(&self) -> [u8; bar::<Self>()];
}

impl Foo for () {
    fn test(&self) -> [u8; bar::<Self>()] {
        [0; bar::<Self>()]
    }
}

fn use_dyn(v: &dyn Foo) { //~ERROR the trait `Foo` is not dyn compatible
    v.test();
}

fn main() {}
