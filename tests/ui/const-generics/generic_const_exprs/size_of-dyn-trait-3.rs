//! Regression test for #114663
//@ edition:2021

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use core::fmt::Debug;
use core::marker::PhantomData;

struct Inline<T>
where
    [u8; ::core::mem::size_of::<T>() + 1]:,
{
    _phantom: PhantomData<T>,
}

fn main() {
    let dst = Inline::<dyn Debug>::new(0);
    //~^ ERROR the size for values of type `dyn Debug` cannot be known at compilation time
    //~| ERROR no function or associated item named `new` found for struct `Inline<T>`
}
