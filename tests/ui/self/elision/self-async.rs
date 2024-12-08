//@ check-pass
//@ edition:2018

#![allow(non_snake_case)]

use std::rc::Rc;

struct Struct { }

impl Struct {
    async fn take_self(self, f: &u32) -> &u32 {
        f
    }

    async fn take_Self(self: Self, f: &u32) -> &u32 {
        f
    }

    async fn take_Box_Self(self: Box<Self>, f: &u32) -> &u32 {
        f
    }

    async fn take_Box_Box_Self(self: Box<Box<Self>>, f: &u32) -> &u32 {
        f
    }

    async fn take_Rc_Self(self: Rc<Self>, f: &u32) -> &u32 {
        f
    }

    async fn take_Box_Rc_Self(self: Box<Rc<Self>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
