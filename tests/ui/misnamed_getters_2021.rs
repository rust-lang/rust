//@edition: 2021
#![allow(unused)]
#![allow(clippy::struct_field_names)]
#![warn(clippy::misnamed_getters)]

// Edition 2021 specific check, where `unsafe` blocks are not required
// inside `unsafe fn`.

union B {
    a: u8,
    b: u8,
}

impl B {
    unsafe fn a(&self) -> &u8 {
        //~^ misnamed_getters

        &self.b
    }
}

fn main() {
    // test code goes here
}
