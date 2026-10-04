//@ check-pass
//@ edition:2018

#![allow(non_snake_case)]

use std::rc::Rc;

struct Struct { }

impl Struct {
    async fn ref_Struct(self: Struct, f: &u32) -> &u32 {
        f
    }

    async fn box_Struct(self: Box<Struct>, f: &u32) -> &u32 {
        f
    }

    async fn rc_Struct(self: Rc<Struct>, f: &u32) -> &u32 {
        f
    }

    async fn box_box_Struct(self: Box<Box<Struct>>, f: &u32) -> &u32 {
        f
    }

    async fn box_rc_Struct(self: Box<Rc<Struct>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
