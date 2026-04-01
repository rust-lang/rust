//@ run-rustfix
//@ edition:2018

#![feature(arbitrary_self_types)]
#![allow(non_snake_case, dead_code)]

use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;

struct Struct {}

struct Wrap<T, P>(T, PhantomData<P>);

impl<T, P> Deref for Wrap<T, P> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl Struct {
    // Test using `&self` sugar:

    fn ref_self(&self, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    // Test using `&Self` explicitly:

    fn ref_Self(self: &Self, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    fn box_ref_Self(self: Box<&Self>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    fn pin_ref_Self(self: Pin<&Self>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    fn box_box_ref_Self(self: Box<Box<&Self>>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    fn box_pin_ref_Self(self: Box<Pin<&Self>>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    fn wrap_ref_Self_Self(self: Wrap<&Self, Self>, f: &u8) -> &u8 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    fn ref_box_Self(self: &Box<Self>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
