//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub const trait Plus {
    fn plus(self, rhs: Self) -> Self;
}


pub const unsafe trait Minus {
    fn minus(self, rhs: Self) -> Self;
}

const impl Plus for i32 {
    fn plus(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Plus for u32 {
    fn plus(self, rhs: Self) -> Self {
        self + rhs
    }
}


const unsafe impl Minus for i32 {
    fn minus(self, rhs: Self) -> Self {
        self - rhs
    }
}

unsafe impl Minus for u32 {
    fn minus(self, rhs: Self) -> Self {
        self - rhs
    }
}


pub const fn add_i32(a: i32, b: i32) -> i32 {
    a.plus(b) // ok
}

pub const fn add_u32(a: u32, b: u32) -> u32 {
    a.plus(b)
    //~^ ERROR the trait bound `u32: [const] Plus`
}


pub const unsafe fn sub_i32(a: i32, b: i32) -> i32 {
    a.minus(b) // ok
}

pub const unsafe fn sub_u32(a: u32, b: u32) -> u32 {
    a.minus(b)
    //~^ ERROR the trait bound `u32: [const] Minus`
}


fn main() {}
