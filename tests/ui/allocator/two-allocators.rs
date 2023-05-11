use std::alloc::System;

#[global_allocator]
static A: System = System;
#[global_allocator]
static B: System = System;
//~^ ERROR: cannot define multiple global allocators

fn main() {}
