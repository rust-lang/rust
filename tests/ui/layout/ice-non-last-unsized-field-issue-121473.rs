// Regression test for #121473
// Checks that no ICE occurs when `size_of`
// is applied to a struct that has an unsized
// field which is not its last field

use std::mem::size_of;

pub struct BadStruct {
    pub field1: i32,
    pub field2: str, // Unsized field that is not the last field
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    pub field3: [u8; 16],
}

pub fn main() {
    // The ICE occurs only in promoted MIR
    let _x = &size_of::<BadStruct>();
    assert_eq!(size_of::<BadStruct>(), 21);
}
