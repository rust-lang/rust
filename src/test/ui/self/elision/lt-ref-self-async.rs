// edition:2018
// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

#![allow(non_snake_case)]

use std::pin::Pin;

struct Struct<'a> { data: &'a u32 }

impl<'a> Struct<'a> {
    // Test using `&self` sugar:

    async fn ref_self(&self, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    // Test using `&Self` explicitly:

    async fn ref_Self(self: &Self, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn box_ref_Self(self: Box<&Self>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn pin_ref_Self(self: Pin<&Self>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn box_box_ref_Self(self: Box<Box<&Self>>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn box_pin_Self(self: Box<Pin<&Self>>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}

fn main() { }
