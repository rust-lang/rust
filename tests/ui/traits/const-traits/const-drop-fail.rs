//@[new_precise] compile-flags: -Znext-solver
//@[new_stock] compile-flags: -Znext-solver
//@ revisions: new_stock old_stock new_precise old_precise
//@ ignore-backends: gcc

#![feature(const_trait_impl, const_destruct)]
#![cfg_attr(any(new_precise, old_precise), feature(const_precise_live_drops))]

use std::marker::{Destruct, PhantomData};

struct NonTrivialDrop;

impl Drop for NonTrivialDrop {
    fn drop(&mut self) {
        println!("Non trivial drop");
    }
}

struct ConstImplWithDropGlue(NonTrivialDrop);

impl const Drop for ConstImplWithDropGlue {
    fn drop(&mut self) {}
}

const fn check<T: [const] Destruct>(_: T) {}

macro_rules! check_all {
    ($($exp:expr),*$(,)?) => {$(
        const _: () = check($exp);
    )*};
}

check_all! {
    NonTrivialDrop,
    //~^ ERROR the trait bound `NonTrivialDrop: const Destruct` is not satisfied
    ConstImplWithDropGlue(NonTrivialDrop),
    //~^ ERROR the trait bound `NonTrivialDrop: const Destruct` is not satisfied
}

fn main() {}
