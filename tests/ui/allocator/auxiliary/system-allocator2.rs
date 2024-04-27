//@ no-prefer-dynamic

#![crate_type = "rlib"]

use std::alloc::System;

#[global_allocator]
static A: System = System;
