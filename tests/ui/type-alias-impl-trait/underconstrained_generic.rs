#![feature(type_alias_impl_trait)]

use std::marker::PhantomData;

trait Trait {
    fn foo<T, U>(t: T) -> U;
}

trait ProofForConversion<X> {
    fn convert<T, U>(_: PhantomData<Self>, r: T) -> U;
}

impl<X: Trait> ProofForConversion<X> for () {
    fn convert<T, U>(_: PhantomData<Self>, r: T) -> U {
        X::foo(r)
    }
}

type Converter<T> = impl ProofForConversion<T>;

#[define_opaque(Converter)]
fn _defining_use<T: Trait>() -> Converter<T> {
    ()
    //~^ ERROR the trait bound `T: Trait` is not satisfied
}

fn main() {}
