// compile-flags: -Z mir-opt-level=3

// revisions: min_tait full_tait in_bindings
#![feature(min_type_alias_impl_trait, rustc_attrs)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
#![cfg_attr(in_bindings, feature(impl_trait_in_bindings))]
//[in_bindings]~^ WARN incomplete

use std::marker::PhantomData;

trait MyIndex<T> {
    type O;
    fn my_index(self) -> Self::O;
}
trait MyFrom<T>: Sized {
    type Error;
    fn my_from(value: T) -> Result<Self, Self::Error>;
}

trait F {}
impl F for () {}
type DummyT<T> = impl F;
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

#[rustc_error]
fn main() {
    let _pos: Phantom1<DummyT<()>> = Scope::new().my_index();
    //[min_tait,full_tait]~^ ERROR not permitted here
    //[in_bindings]~^^ ERROR type annotations needed
}
