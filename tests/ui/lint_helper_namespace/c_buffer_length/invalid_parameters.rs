#![feature(lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

#[lint::c_buffer_length(buffer, length)]
//~^ ERROR invalid parameter `buffer`
fn not_pointer(buffer: &[u8], length: usize) {}

#[lint::c_buffer_length(buffer, length)]
//~^ ERROR invalid parameter `length`
fn not_integer(buffer: *const u8, length: bool) {}

#[lint::c_buffer_length(missing, length)]
//~^ ERROR invalid parameter `missing`
fn missing_pointer(buffer: *const u8, length: usize) {}

#[lint::c_buffer_length(buffer, missing)]
//~^ ERROR invalid parameter `missing`
fn missing_length(buffer: *const u8, length: usize) {}

fn main() {}
