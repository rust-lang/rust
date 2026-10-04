#![feature(diagnostic_c_buffer_length)]
#![deny(malformed_diagnostic_attributes)]

#[diagnostic::c_buffer_length]
//~^ ERROR expects two distinct parameter names
fn no_args() {}

#[diagnostic::c_buffer_length(buffer)]
//~^ ERROR expects two distinct parameter names
fn one(buffer: *const u8) {}

#[diagnostic::c_buffer_length(buffer, length, extra)]
//~^ ERROR expects two distinct parameter names
fn three(buffer: *const u8, length: usize, extra: usize) {}

#[diagnostic::c_buffer_length(buffer, length)]
//~^ ERROR invalid parameter `buffer`
fn not_pointer(buffer: &[u8], length: usize) {}

#[diagnostic::c_buffer_length(buffer, length)]
//~^ ERROR invalid parameter `length`
fn not_integer(buffer: *const u8, length: bool) {}

fn main() {}
