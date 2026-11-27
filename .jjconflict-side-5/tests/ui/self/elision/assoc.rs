//@ check-pass

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
    fn assoc(self: <Struct as Trait>::AssocType, f: &u32) -> &u32 {
        f
    }

    fn box_AssocType(self: Box<<Struct as Trait>::AssocType>, f: &u32) -> &u32 {
        f
    }

    fn rc_AssocType(self: Rc<<Struct as Trait>::AssocType>, f: &u32) -> &u32 {
        f
    }

    fn box_box_AssocType(self: Box<Box<<Struct as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
    }

    fn box_rc_AssocType(self: Box<Rc<<Struct as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
