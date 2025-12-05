//@ revisions: u w
//@[u] only-unix
//@[w] only-windows
#[global_allocator]
static A: usize = 0;
//~^ ERROR E0277
//~| ERROR E0277
//~| ERROR E0277
//~| ERROR E0277

fn main() {}
