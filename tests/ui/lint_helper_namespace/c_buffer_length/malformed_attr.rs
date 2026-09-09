#![feature(lint_c_buffer_length, register_tool)]
#![register_tool(lint)]

#[lint::c_buffer_length]
//~^ ERROR expects two distinct parameter names
fn no_args() {}

#[lint::c_buffer_length(buffer, buffer)]
//~^ ERROR expects two distinct parameter names
fn same(buffer: *const u8) {}

fn main() {}
