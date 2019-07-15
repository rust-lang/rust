// check-pass

#![feature(arbitrary_self_types)]
#![allow(non_snake_case)]

use std::rc::Rc;

struct Struct { }

impl Struct {
    // Test using `&mut Struct` explicitly:

    fn ref_Struct(self: Struct, f: &u32) -> &u32 {
        f //~ ERROR lifetime mismatch
    }

    fn box_Struct(self: Box<Struct>, f: &u32) -> &u32 {
        f //~ ERROR lifetime mismatch
    }

    fn rc_Struct(self: Rc<Struct>, f: &u32) -> &u32 {
        f //~ ERROR lifetime mismatch
    }

    fn box_box_Struct(self: Box<Box<Struct>>, f: &u32) -> &u32 {
        f //~ ERROR lifetime mismatch
    }

    fn box_rc_Struct(self: Box<Rc<Struct>>, f: &u32) -> &u32 {
        f //~ ERROR lifetime mismatch
    }
}

fn main() { }
