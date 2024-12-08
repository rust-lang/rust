//@ known-bug: #114663
//@ edition:2021

#![feature(generic_const_exprs)]

use core::fmt::Debug;

struct Inline<T>
where
    [u8; ::core::mem::size_of::<T>() + 1]:,
{
    _phantom: PhantomData<T>,
}

fn main() {
    let dst = Inline::<dyn Debug>::new(0); // BANG!
}
