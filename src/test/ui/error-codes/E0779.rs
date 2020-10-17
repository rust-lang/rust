#![feature(isa_attribute)]

#[instruction_set(arm::magic)] //~ ERROR
fn main() {

}
