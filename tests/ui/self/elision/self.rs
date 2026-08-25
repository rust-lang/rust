//@ check-pass

#![allow(non_snake_case)]

use std::rc::Rc;

struct Struct { }

impl Struct {
    fn take_self(self, f: &u32) -> &u32 {
        f
    }

    fn take_Self(self: Self, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Self(self: Box<Self>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Box_Self(self: Box<Box<Self>>, f: &u32) -> &u32 {
        f
    }

    fn take_Rc_Self(self: Rc<Self>, f: &u32) -> &u32 {
        f
    }

    fn take_Box_Rc_Self(self: Box<Rc<Self>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
