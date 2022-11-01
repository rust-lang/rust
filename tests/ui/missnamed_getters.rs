#![allow(unused)]
#![warn(clippy::missnamed_getters)]

struct A {
    a: u8,
    b: u8,
}

impl A {
    fn a(&self) -> &u8 {
        &self.b
    }
}

union B {
    a: u8,
    b: u8,
}

impl B {
    unsafe fn a(&self) -> &u8 {
        &self.b
    }
}

fn main() {
    // test code goes here
}
