//@ revisions: stock precise

#![feature(const_trait_impl, const_destruct)]
#![cfg_attr(precise, feature(const_precise_live_drops))]

use std::marker::{Destruct, PhantomData};

struct NonTrivialDrop;

impl Drop for NonTrivialDrop {
    fn drop(&mut self) {
        println!("Non trivial drop");
    }
}

#[const_trait]
trait A { fn a() { } }

impl A for NonTrivialDrop {}

const fn check<T: [const] Destruct>(_: T) {}

struct ConstDropImplWithBounds<T: A>(PhantomData<T>);

impl<T: [const] A> const Drop for ConstDropImplWithBounds<T> {
    fn drop(&mut self) {
        T::a();
    }
}

const _: () = check::<ConstDropImplWithBounds<NonTrivialDrop>>(
    //~^ ERROR the trait bound
    ConstDropImplWithBounds(PhantomData)
);

struct ConstDropImplWithNonConstBounds<T: A>(PhantomData<T>);

impl<T: [const] A> const Drop for ConstDropImplWithNonConstBounds<T> {
    fn drop(&mut self) {
        T::a();
    }
}

fn main() {}
