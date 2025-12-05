//@ check-fail

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

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
    let dst = Inline::<dyn Debug>::new(0);
    //~^ ERROR the size for values of type `dyn Debug` cannot be known at compilation time
    //~| ERROR the function or associated item `new` exists for struct `Inline<dyn Debug>`
}
