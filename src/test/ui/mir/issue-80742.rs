// check-fail

// This test used to cause an ICE in rustc_mir::interpret::step::eval_rvalue_into_place

#![allow(incomplete_features)]
#![feature(const_evaluatable_checked)]
#![feature(const_generics)]

use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::size_of;

struct Inline<T>
where
    [u8; size_of::<T>() + 1]: ,
{
    _phantom: PhantomData<T>,
    buf: [u8; size_of::<T>() + 1],
}

impl<T> Inline<T>
where
    [u8; size_of::<T>() + 1]: ,
{
    pub fn new(val: T) -> Inline<T> {
        todo!()
    }
}

fn main() {
    let dst = Inline::<dyn Debug>::new(0); //~ ERROR
    //~^ ERROR
}
