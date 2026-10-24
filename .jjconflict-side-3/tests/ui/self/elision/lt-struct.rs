//@ check-pass

#![allow(non_snake_case)]

use std::rc::Rc;

struct Struct<'a> { x: &'a u32 }

impl<'a> Struct<'a> {
    fn take_self(self, f: &u32) -> &u32 {
        f
    }

    fn take_Struct(self: Struct<'a>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Struct(self: Box<Struct<'a>>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Box_Struct(self: Box<Box<Struct<'a>>>, f: &u32) -> &u32 {
        f
    }

    fn take_Rc_Struct(self: Rc<Struct<'a>>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Rc_Struct(self: Box<Rc<Struct<'a>>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
