#![deny(large_assignments)]
#![feature(large_assignments)]
#![move_size_limit = "1000"]
//@ build-fail
//@ only-64bit

//@ edition:2018
//@ compile-flags: -Zmir-opt-level=1

use std::{sync::Arc, rc::Rc};

fn main() {
    let data = [0; 9999];

    // Looking at --emit mir, we can see that all parameters below are passed by
    // copy. But it requires at least mir-opt-level=1.
    let _ = Arc::new(data); // OK!
    let _ = Box::new(data); // OK!
    let _ = Rc::new(data); // OK!

    // Looking at --emit llvm-ir, we can see that a memcpy is involved in the
    // parameter passing. So we want the lint to trigger here.
    let _ = NotBox::new(data); //~ ERROR large_assignments
}

struct NotBox {
    data: [u8; 9999],
}

impl NotBox {
    fn new(data: [u8; 9999]) -> Self {
        // Looking at --emit llvm-ir, we can see that a memcpy is involved.
        // So we want the lint to trigger here.
        Self { //~ ERROR large_assignments
            data,
        }
    }
}
