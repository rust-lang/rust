#![feature(isa_attribute)]

#[instruction_set()] //~ ERROR
fn no_isa_defined() {
}

fn main() {
}
