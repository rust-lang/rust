#[global_allocator]
fn foo() {} //~ ERROR: allocators must be statics

fn main() {}
