// Test for https://github.com/rust-lang/rust/issues/153126

#[repr(packed, C)]
#[derive(PartialEq)]
struct Thing(u8, String);
//~^ ERROR cannot move out of a shared reference
//~| ERROR cannot move out of a shared reference

fn main() {}
