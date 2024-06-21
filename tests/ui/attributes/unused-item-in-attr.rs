#[w = { extern crate alloc; }]
//~^ ERROR attribute value must be a literal
//~| ERROR cannot find attribute `w`
fn f() {}

fn main() {}
