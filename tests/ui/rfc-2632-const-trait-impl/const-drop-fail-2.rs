// revisions: stock precise
#![feature(const_trait_impl)]
#![feature(const_mut_refs)]
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

struct ConstDropImplWithBounds<T: ~const A>(PhantomData<T>);

impl<T: ~const A> const Drop for ConstDropImplWithBounds<T> {
    fn drop(&mut self) {
        T::a();
    }
}

const fn check<T: ~const Destruct>(_: T) {}

const _: () = check::<ConstDropImplWithBounds<NonTrivialDrop>>(
    //~^ ERROR the trait bound
    ConstDropImplWithBounds(PhantomData)
    //~^ ERROR the trait bound
);

struct ConstDropImplWithNonConstBounds<T: A>(PhantomData<T>);

impl<T: ~const A> const Drop for ConstDropImplWithNonConstBounds<T> {
//~^ ERROR `Drop` impl requires `T: ~const A` but the struct it is implemented for does not
    fn drop(&mut self) {
        T::a();
    }
}

fn main() {}
