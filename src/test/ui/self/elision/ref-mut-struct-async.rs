// edition:2018
// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

#![allow(non_snake_case)]

use std::pin::Pin;

struct Struct { }

impl Struct {
    // Test using `&mut Struct` explicitly:

    async fn ref_Struct(self: &mut Struct, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn box_ref_Struct(self: Box<&mut Struct>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn pin_ref_Struct(self: Pin<&mut Struct>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn box_box_ref_Struct(self: Box<Box<&mut Struct>>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }

    async fn box_pin_ref_Struct(self: Box<Pin<&mut Struct>>, f: &u32) -> &u32 {
        f
        //[base]~^ ERROR lifetime mismatch
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}

fn main() { }
