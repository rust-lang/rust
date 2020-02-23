#![allow(incomplete_features)]
#![feature(const_trait_impl)]

pub struct Int(i32);

impl const std::ops::Add for i32 {
    //~^ ERROR conflicting implementations of trait
    //~| ERROR only traits defined in the current crate can be implemented for arbitrary types
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl std::ops::Add for Int {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Int(self.0 + rhs.0)
    }
}

impl const std::ops::Add for Int {
    //~^ ERROR conflicting implementations of trait
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Int(self.0 + rhs.0)
    }
}

fn main() {}
