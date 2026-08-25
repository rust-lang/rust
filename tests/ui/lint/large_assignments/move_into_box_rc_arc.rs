#![deny(large_assignments)]
#![feature(large_assignments)]
#![move_size_limit = "1000"]
//@ build-fail
//@ only-64bit

//@ edition:2018
//@ compile-flags: -Zmir-opt-level=0

use std::{sync::Arc, rc::Rc};

fn main() {
    // Looking at --emit mir, we can see that all parameters below are passed
    // by move.
    let _ = Arc::new([0; 9999]); // OK!
    let _ = Box::new([0; 9999]); // OK!
    let _ = Rc::new([0; 9999]); // OK!

    // Looking at --emit llvm-ir, we can see that no memcpy is involved in the
    // parameter passing. Instead, a pointer is passed. This is typically what
    // we get when moving parameter into functions. So we don't want the lint to
    // trigger here.
    let _ = NotBox::new([0; 9999]); // OK (compare with copy_into_box_rc_arc.rs)
}

struct NotBox {
    data: [u8; 9999],
}

impl NotBox {
    fn new(data: [u8; 9999]) -> Self {
        Self {
            // Looking at --emit llvm-ir, we can see that a memcpy is involved.
            // So we want the lint to trigger here.
            data, //~ ERROR large_assignments
        }
    }
}
