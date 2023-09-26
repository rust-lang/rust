#![deny(large_assignments)]
#![feature(large_assignments)]
#![move_size_limit = "1000"]
// build-fail
// only-x86_64

// edition:2018
// compile-flags: -Zmir-opt-level=0

use std::{sync::Arc, rc::Rc};

fn main() {
    let _ = Arc::new([0; 9999]); // OK!
    let _ = Box::new([0; 9999]); // OK!
    let _ = Rc::new([0; 9999]); // OK!
    let _ = NotBox::new([0; 9999]); //~ ERROR large_assignments
}

struct NotBox {
    data: [u8; 9999],
}

impl NotBox {
    fn new(data: [u8; 9999]) -> Self {
        Self {
            data, //~ ERROR large_assignments
        }
    }
}
