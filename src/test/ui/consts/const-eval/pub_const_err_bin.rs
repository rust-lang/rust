// compile-pass
#![warn(const_err)]

pub const Z: u32 = 0 - 1;
//~^ WARN this constant cannot be used

pub type Foo = [i32; 0 - 1];
//~^ WARN attempt to subtract with overflow
//~| WARN this array length cannot be used

fn main() {}
