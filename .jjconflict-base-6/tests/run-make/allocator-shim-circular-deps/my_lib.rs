#![crate_type = "lib"]

use std::alloc::System;

#[global_allocator]
static ALLOCATOR: System = System;

pub fn do_something() {
    format!("allocating a string!");
}
