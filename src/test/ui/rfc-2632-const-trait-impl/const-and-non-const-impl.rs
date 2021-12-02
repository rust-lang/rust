#![feature(const_trait_impl)]

pub struct Int(i32);

impl const std::ops::Add for i32 { //~ ERROR type annotations needed
    //~^ ERROR only traits defined in the current crate can be implemented for arbitrary types
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl std::ops::Add for Int { //~ ERROR type annotations needed
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Int(self.0 + rhs.0)
    }
}

impl const std::ops::Add for Int { //~ ERROR type annotations needed
    //~^ ERROR conflicting implementations of trait
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Int(self.0 + rhs.0)
    }
}

fn main() {}
