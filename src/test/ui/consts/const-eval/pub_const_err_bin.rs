// check-pass
#![warn(const_err)]

pub const Z: u32 = 0 - 1;
//~^ WARN any use of this value will cause an error
//~| WARN this was previously accepted by the compiler but is being phased out

pub type Foo = [i32; 0 - 1];

fn main() {}
