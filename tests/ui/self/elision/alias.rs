//@ check-pass

#![allow(non_snake_case)]

use std::rc::Rc;

struct Struct { }

type Alias = Struct;

impl Struct {
    // Test using an alias for `Struct`:

    fn alias(self: Alias, f: &u32) -> &u32 {
        f
    }

    fn box_Alias(self: Box<Alias>, f: &u32) -> &u32 {
        f
    }

    fn rc_Alias(self: Rc<Alias>, f: &u32) -> &u32 {
        f
    }

    fn box_box_Alias(self: Box<Box<Alias>>, f: &u32) -> &u32 {
        f
    }

    fn box_rc_Alias(self: Box<Rc<Alias>>, f: &u32) -> &u32 {
        f
    }
}

fn main() { }
