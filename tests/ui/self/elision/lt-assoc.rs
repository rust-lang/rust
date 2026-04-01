//@ check-pass

#![allow(non_snake_case)]

use std::rc::Rc;

trait Trait {
    type AssocType;
}

struct Struct<'a> { x: &'a u32 }

impl<'a> Trait for Struct<'a> {
    type AssocType = Self;
}

impl<'a> Struct<'a> {
    fn take_self(self, f: &u32) -> &u32 {
        f
    }

    fn take_AssocType(self: <Struct<'a> as Trait>::AssocType, f: &u32) -> &u32 {
        f
    }

    fn take_Box_AssocType(self: Box<<Struct<'a> as Trait>::AssocType>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Box_AssocType(self: Box<Box<<Struct<'a> as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
    }

    fn take_Rc_AssocType(self: Rc<<Struct<'a> as Trait>::AssocType>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Rc_AssocType(self: Box<Rc<<Struct<'a> as Trait>::AssocType>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
