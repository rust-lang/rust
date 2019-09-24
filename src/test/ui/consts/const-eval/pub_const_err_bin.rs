// build-pass (FIXME(62277): could be check-pass?)
#![warn(const_err)]

pub const Z: u32 = 0 - 1;
//~^ WARN any use of this value will cause an error

pub type Foo = [i32; 0 - 1];

fn main() {}
