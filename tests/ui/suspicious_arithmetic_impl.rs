#![warn(clippy::suspicious_arithmetic_impl)]
use std::ops::{Add, AddAssign, Div, Mul, Sub};

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
        Foo(self.0 * other.0 % 42) // OK: BinOpKind::Rem part of BiExpr as parent node
    }
}

impl Sub for Foo {
    type Output = Foo;

    fn sub(self, other: Self) -> Self {
        Foo(self.0 * other.0 - 42) // OK: BinOpKind::Mul part of BiExpr as child node
    }
}

impl Div for Foo {
    type Output = Foo;

    fn div(self, other: Self) -> Self {
        Foo(do_nothing(self.0 + other.0) / 42) // OK: BinOpKind::Add part of BiExpr as child node
    }
}

struct Bar(i32);

impl Add for Bar {
    type Output = Bar;

    fn add(self, other: Self) -> Self {
        Bar(self.0 & !other.0) // OK: UnNot part of BiExpr as child node
    }
}

impl Sub for Bar {
    type Output = Bar;

    fn sub(self, other: Self) -> Self {
        if self.0 <= other.0 {
            Bar(-(self.0 & other.0)) // OK: UnNeg part of BiExpr as parent node
        } else {
            Bar(0)
        }
    }
}

fn main() {}

fn do_nothing(x: u32) -> u32 {
    x
}
