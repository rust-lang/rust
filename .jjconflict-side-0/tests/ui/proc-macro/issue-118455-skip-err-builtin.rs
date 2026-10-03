#![some_nonexistent_attribute]
//~^ ERROR cannot find attribute `some_nonexistent_attribute` in this scope
#[derive(Debug)]
pub struct SomeUserCode;

fn main() {}
