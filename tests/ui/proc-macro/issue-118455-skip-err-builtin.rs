#![some_nonexistent_attribute]
//~^ ERROR cannot find attribute `some_nonexistent_attribute`
#[derive(Debug)]
pub struct SomeUserCode;

fn main() {}
