#![feature(min_const_generics)]

use std::marker::PhantomData;

use std::mem::{self, MaybeUninit};

struct Bug<S> {
    //~^ ERROR parameter `S` is never used
    A: [(); {
        let x: S = MaybeUninit::uninit();
        //~^ ERROR generic parameters must not be used inside of non-trivial constant values
        let b = &*(&x as *const _ as *const S);
        //~^ ERROR generic parameters must not be used inside of non-trivial constant values
        0
    }],
}

fn main() {}
