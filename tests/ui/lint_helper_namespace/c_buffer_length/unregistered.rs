#![feature(lint_c_buffer_length)]

#[lint::c_buffer_length(buffer, length)]
//~^ ERROR cannot find module or crate `lint` in this scope
fn unregistered(buffer: *const u8, length: usize) {}

fn main() {}
