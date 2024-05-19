#![feature(type_alias_impl_trait)]

//@ check-pass

trait Trait {}

type TAIT = impl Trait;

struct Concrete;
impl Trait for Concrete {}

fn tait() -> TAIT {
    Concrete
}

trait OuterTrait {
    type Item;
}
struct Dummy<T> {
    t: T,
}
impl<T> OuterTrait for Dummy<T> {
    type Item = T;
}

fn tait_and_impl_trait() -> impl OuterTrait<Item = (TAIT, impl Trait)> {
    Dummy {
        t: (tait(), Concrete),
    }
}

fn tait_and_dyn_trait() -> impl OuterTrait<Item = (TAIT, Box<dyn Trait>)> {
    let b: Box<dyn Trait> = Box::new(Concrete);
    Dummy { t: (tait(), b) }
}

fn main() {}
