


#![warn(suspicious_arithmetic_impl)]
use std::ops::{Add, AddAssign, Mul, Sub, Div};

#[derive(Copy, Clone)]
struct Foo(u32);

impl Add for Foo {
    type Output = Foo;

    fn add(self, other: Self) -> Self {
        Foo(self.0 - other.0)
    }
}

impl AddAssign for Foo {
    fn add_assign(&mut self, other: Foo) {
        *self = *self - other;
    }
}

impl Mul for Foo {
    type Output = Foo;

    fn mul(self, other: Foo) -> Foo {
        Foo(self.0 * other.0 % 42) // OK: BiRem part of BiExpr as parent node
    }
}

impl Sub for Foo {
    type Output = Foo;

    fn sub(self, other: Self) -> Self {
        Foo(self.0 * other.0 - 42) // OK: BiMul part of BiExpr as child node
    }
}

impl Div for Foo {
    type Output = Foo;

    fn div(self, other: Self) -> Self {
        Foo(do_nothing(self.0 + other.0) / 42) // OK: BiAdd part of BiExpr as child node
    }
}

fn main() {}

fn do_nothing(x: u32) -> u32 {
    x
}
