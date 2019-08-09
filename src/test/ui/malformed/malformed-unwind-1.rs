#![feature(unwind_attributes)]

#[unwind] //~ ERROR malformed `unwind` attribute
extern "C" fn f1() {}

#[unwind = ""] //~ ERROR malformed `unwind` attribute
extern "C" fn f2() {}

fn main() {}
