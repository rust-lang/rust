#![feature(lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

#[lint::c_buffer_length(buffer, length)]
#[lint::c_buffer_length(buffer, length)]
//~^ ERROR duplicate `lint::c_buffer_length` attributes
fn duplicate(buffer: *const u8, length: usize) {}

fn main() {}
