#![feature(diagnostic_c_buffer_length)]
#![deny(misplaced_diagnostic_attributes)]

#[diagnostic::c_buffer_length(buffer, length)]
//~^ ERROR attribute cannot be used on
struct Misplaced;

fn main() {}
