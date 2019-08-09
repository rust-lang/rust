#![feature(unwind_attributes)]

#[unwind(allowed, aborts)]
//~^ ERROR malformed `unwind` attribute
extern "C" fn f1() {}

#[unwind(unsupported)]
//~^ ERROR malformed `unwind` attribute
extern "C" fn f2() {}

fn main() {}
