#![feature(unwind_attributes)]

#[unwind]
//~^ ERROR attribute must be of the form
extern "C" fn f1() {}

#[unwind = ""]
//~^ ERROR attribute must be of the form
extern "C" fn f2() {}

fn main() {}
