#![feature(thread_local)]

use std::alloc::System;

#[global_allocator]
#[thread_local]
//~^ ERROR: allocators cannot be `#[thread_local]`
static A: System = System;

fn main() {}
