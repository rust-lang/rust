#![allow(clippy::legacy_numeric_constants)]
#![warn(clippy::suspicious_arithmetic_impl)]
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, Div, DivAssign, Mul, MulAssign, Rem, Shl, Shr, Sub,
};

#[derive(Copy, Clone)]
struct Foo(u32);

impl Add for Foo {
    type Output = Foo;

    fn add(self, other: Self) -> Self {
        Foo(self.0 - other.0)
        //~^ ERROR: suspicious use of `-` in `Add` impl
        //~| NOTE: `-D clippy::suspicious-arithmetic-impl` implied by `-D warnings`
    }
}

impl AddAssign for Foo {
    fn add_assign(&mut self, other: Foo) {
        *self = *self - other;
        //~^ ERROR: suspicious use of `-` in `AddAssign` impl
        //~| NOTE: `-D clippy::suspicious-op-assign-impl` implied by `-D warnings`
    }
}

impl BitOrAssign for Foo {
    fn bitor_assign(&mut self, other: Foo) {
        let idx = other.0;
        self.0 |= 1 << idx; // OK: BinOpKind::Shl part of AssignOp as child node
    }
}

impl MulAssign for Foo {
    fn mul_assign(&mut self, other: Foo) {
        self.0 /= other.0;
        //~^ ERROR: suspicious use of `/` in `MulAssign` impl
    }
}

impl DivAssign for Foo {
    fn div_assign(&mut self, other: Foo) {
        self.0 /= other.0; // OK: BinOpKind::Div == DivAssign
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

impl Rem for Foo {
    type Output = Foo;

    fn rem(self, other: Self) -> Self {
        Foo(self.0 / other.0)
        //~^ ERROR: suspicious use of `/` in `Rem` impl
    }
}

impl BitAnd for Foo {
    type Output = Foo;

    fn bitand(self, other: Self) -> Self {
        Foo(self.0 | other.0)
        //~^ ERROR: suspicious use of `|` in `BitAnd` impl
    }
}

impl BitOr for Foo {
    type Output = Foo;

    fn bitor(self, other: Self) -> Self {
        Foo(self.0 ^ other.0)
        //~^ ERROR: suspicious use of `^` in `BitOr` impl
    }
}

impl BitXor for Foo {
    type Output = Foo;

    fn bitxor(self, other: Self) -> Self {
        Foo(self.0 & other.0)
        //~^ ERROR: suspicious use of `&` in `BitXor` impl
    }
}

impl Shl for Foo {
    type Output = Foo;

    fn shl(self, other: Self) -> Self {
        Foo(self.0 >> other.0)
        //~^ ERROR: suspicious use of `>>` in `Shl` impl
    }
}

impl Shr for Foo {
    type Output = Foo;

    fn shr(self, other: Self) -> Self {
        Foo(self.0 << other.0)
        //~^ ERROR: suspicious use of `<<` in `Shr` impl
    }
}

struct Bar(i32);

impl Add for Bar {
    type Output = Bar;

    fn add(self, other: Self) -> Self {
        Bar(self.0 & !other.0) // OK: Not part of BiExpr as child node
    }
}

impl Sub for Bar {
    type Output = Bar;

    fn sub(self, other: Self) -> Self {
        if self.0 <= other.0 {
            Bar(-(self.0 & other.0)) // OK: Neg part of BiExpr as parent node
        } else {
            Bar(0)
        }
    }
}

fn main() {}

fn do_nothing(x: u32) -> u32 {
    x
}

struct MultipleBinops(u32);

impl Add for MultipleBinops {
    type Output = MultipleBinops;

    // OK: multiple Binops in `add` impl
    fn add(self, other: Self) -> Self::Output {
        let mut result = self.0 + other.0;
        if result >= u32::max_value() {
            result -= u32::max_value();
        }
        MultipleBinops(result)
    }
}

impl Mul for MultipleBinops {
    type Output = MultipleBinops;

    // OK: multiple Binops in `mul` impl
    fn mul(self, other: Self) -> Self::Output {
        let mut result: u32 = 0;
        let size = std::cmp::max(self.0, other.0) as usize;
        let mut v = vec![0; size + 1];
        for i in 0..size + 1 {
            result *= i as u32;
        }
        MultipleBinops(result)
    }
}
