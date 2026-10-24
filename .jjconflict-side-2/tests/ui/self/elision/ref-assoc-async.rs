//@ edition:2018

#![allow(non_snake_case)]

use std::pin::Pin;

trait Trait {
    type AssocType;
}

struct Struct { }

impl Trait for Struct {
    type AssocType = Self;
}

impl Struct {
    async fn ref_AssocType(self: &<Struct as Trait>::AssocType, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    async fn box_ref_AssocType(self: Box<&<Struct as Trait>::AssocType>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    async fn pin_ref_AssocType(self: Pin<&<Struct as Trait>::AssocType>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    async fn box_box_ref_AssocType(self: Box<Box<&<Struct as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }

    async fn box_pin_ref_AssocType(self: Box<Pin<&<Struct as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() { }
