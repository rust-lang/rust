//@ check-pass
//@ edition:2018

#![allow(non_snake_case)]

use std::rc::Rc;

trait Trait {
    type AssocType;
}

struct Struct { }

impl Trait for Struct {
    type AssocType = Self;
}

impl Struct {
    async fn assoc(self: <Struct as Trait>::AssocType, f: &u32) -> &u32 {
        f
    }

    async fn box_AssocType(self: Box<<Struct as Trait>::AssocType>, f: &u32) -> &u32 {
        f
    }

    async fn rc_AssocType(self: Rc<<Struct as Trait>::AssocType>, f: &u32) -> &u32 {
        f
    }

    async fn box_box_AssocType(self: Box<Box<<Struct as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
    }

    async fn box_rc_AssocType(self: Box<Rc<<Struct as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
