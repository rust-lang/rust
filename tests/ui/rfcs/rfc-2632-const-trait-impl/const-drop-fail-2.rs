// known-bug: #110395
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

const fn check<T: ~const Destruct>(_: T) {}


/* FIXME(effects)
struct ConstDropImplWithBounds<T: ~const A>(PhantomData<T>);

impl<T: ~const A> const Drop for ConstDropImplWithBounds<T> {
    fn drop(&mut self) {
        T::a();
    }
}

const _: () = check::<ConstDropImplWithBounds<NonTrivialDrop>>(
    ConstDropImplWithBounds(PhantomData)
);
*/

struct ConstDropImplWithNonConstBounds<T: A>(PhantomData<T>);

impl<T: ~const A> const Drop for ConstDropImplWithNonConstBounds<T> {
    fn drop(&mut self) {
        T::a();
    }
}

fn main() {}
