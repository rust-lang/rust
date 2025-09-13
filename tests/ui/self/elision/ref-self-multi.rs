#![feature(arbitrary_self_types)]
#![allow(non_snake_case)]
#![allow(unused)]

use std::marker::PhantomData;
use std::ops::{Deref, Receiver};

struct Struct {}

struct Wrap<T, P>(T, PhantomData<P>);

impl<T, P> Deref for Wrap<T, P> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}
impl<T, P> Receiver for Wrap<T, P> {
    type Target = T;
}

impl Struct {
    fn ref_box_ref_Self(self: &Box<&Self>, f: &u32) -> &u32 {
        //~^ ERROR missing lifetime specifier
        f
    }

    fn ref_wrap_ref_Self(self: &Wrap<&Self, u32>, f: &u32) -> &u32 {
        //~^ ERROR missing lifetime specifier
        f
    }
}

fn main() {}
