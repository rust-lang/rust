#[w = { extern crate alloc; }]
//~^ ERROR attribute value must be a literal
//~| ERROR cannot find attribute `w` in this scope
fn f() {}

fn main() {}
