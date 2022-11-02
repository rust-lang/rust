#![allow(unused)]
#![warn(clippy::misnamed_getters)]

struct A {
    a: u8,
    b: u8,
    c: u8,
}

impl A {
    fn a(&self) -> &u8 {
        &self.b
    }
    fn a_mut(&mut self) -> &mut u8 {
        &mut self.b
    }

    fn b(self) -> u8 {
        self.a
    }

    fn b_mut(&mut self) -> &mut u8 {
        &mut self.a
    }

    fn c(&self) -> &u8 {
        &self.b
    }

    fn c_mut(&mut self) -> &mut u8 {
        &mut self.a
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
    unsafe fn a_mut(&mut self) -> &mut u8 {
        &mut self.b
    }

    unsafe fn b(self) -> u8 {
        self.a
    }

    unsafe fn b_mut(&mut self) -> &mut u8 {
        &mut self.a
    }

    unsafe fn c(&self) -> &u8 {
        &self.b
    }

    unsafe fn c_mut(&mut self) -> &mut u8 {
        &mut self.a
    }

    unsafe fn a_unchecked(&self) -> &u8 {
        &self.b
    }
    unsafe fn a_unchecked_mut(&mut self) -> &mut u8 {
        &mut self.b
    }

    unsafe fn b_unchecked(self) -> u8 {
        self.a
    }

    unsafe fn b_unchecked_mut(&mut self) -> &mut u8 {
        &mut self.a
    }

    unsafe fn c_unchecked(&self) -> &u8 {
        &self.b
    }

    unsafe fn c_unchecked_mut(&mut self) -> &mut u8 {
        &mut self.a
    }
}

fn main() {
    // test code goes here
}
