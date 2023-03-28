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

struct ConstImplWithDropGlue(NonTrivialDrop);

impl const Drop for ConstImplWithDropGlue {
    fn drop(&mut self) {}
}

const fn check<T: ~const Destruct>(_: T) {}

macro_rules! check_all {
    ($($exp:expr),*$(,)?) => {$(
        const _: () = check($exp);
        //~^ ERROR can't drop
        //~| ERROR can't drop
    )*};
}

check_all! {
    NonTrivialDrop,
    ConstImplWithDropGlue(NonTrivialDrop),
}

fn main() {}
