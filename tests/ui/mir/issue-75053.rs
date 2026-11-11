//@ check-pass
//@ compile-flags: -Z mir-opt-level=3

#![feature(type_alias_impl_trait)]

use std::marker::PhantomData;

trait MyIndex<T> {
    type O;
    fn my_index(self) -> Self::O;
}
trait MyFrom<T>: Sized {
    type Error;
    fn my_from(value: T) -> Result<Self, Self::Error>;
}

pub trait F {}
impl F for () {}
pub type DummyT<T> = impl F;
#[define_opaque(DummyT)]
fn _dummy_t<T>() -> DummyT<T> {}

struct Phantom1<T>(PhantomData<T>);
struct Phantom2<T>(PhantomData<T>);
struct Scope<T>(Phantom2<DummyT<T>>);

impl<T> Scope<T> {
    fn new() -> Self {
        unimplemented!()
    }
}

impl<T> MyFrom<Phantom2<T>> for Phantom1<T> {
    type Error = ();
    fn my_from(_: Phantom2<T>) -> Result<Self, Self::Error> {
        unimplemented!()
    }
}

impl<T: MyFrom<Phantom2<DummyT<U>>>, U> MyIndex<Phantom1<T>> for Scope<U> {
    type O = T;
    fn my_index(self) -> Self::O {
        MyFrom::my_from(self.0).ok().unwrap()
    }
}

fn main() {
    let _pos: Phantom1<DummyT<()>> = Scope::new().my_index();
}
