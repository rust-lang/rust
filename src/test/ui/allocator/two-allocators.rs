use std::alloc::System;

#[global_allocator]
static A: System = System;
#[global_allocator]
static B: System = System;
//~^ ERROR: cannot define more than one #[global_allocator]

fn main() {}
