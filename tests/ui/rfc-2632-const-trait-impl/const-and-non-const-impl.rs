// known-bug: #110395

#![feature(const_trait_impl)]

pub struct Int(i32);

impl const std::ops::Add for i32 {
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
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Int(self.0 + rhs.0)
    }
}

fn main() {}
