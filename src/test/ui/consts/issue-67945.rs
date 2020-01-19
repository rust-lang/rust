#![feature(type_ascription)]

use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};

struct Bug1<S> {
    //~^ ERROR parameter `S` is never used [E0392]
    A: [(); {
        let x: S = MaybeUninit::uninit();
        //~^ ERROR type parameters cannot appear within an array length expression [E0747]
        let b = &*(&x as *const _ as *const S);
        //~^ ERROR type parameters cannot appear within an array length expression [E0747]
        0
    }],
}

struct Bug2<S> {
    //~^ ERROR parameter `S` is never used [E0392]
    A: [(); {
        let x: Option<S> = None;
        //~^ ERROR type parameters cannot appear within an array length expression [E0747]
        0
    }],
}

struct Bug3<S> {
    //~^ ERROR parameter `S` is never used [E0392]
    A: [(); {
        let x: Option<Box<S>> = None;
        //~^ ERROR type parameters cannot appear within an array length expression [E0747]
        0
    }],
}

enum Bug4<S> {
    //~^ ERROR parameter `S` is never used [E0392]
    Var = {
        let x: S = 0;
        //~^ ERROR type parameters cannot appear within an enum discriminant [E0747]
        0
    },
}

enum Bug5<S> {
    //~^ ERROR parameter `S` is never used [E0392]
    Var = 0: S,
    //~^ ERROR type parameters cannot appear within an enum discriminant [E0747]
}

fn main() {}
