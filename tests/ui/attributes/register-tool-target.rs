//@ check-pass
#![feature(register_tool)]

#[register_tool(no_valid_target)]
//~^ WARN crate-level attribute should be an inner attribute
fn main() {

}
