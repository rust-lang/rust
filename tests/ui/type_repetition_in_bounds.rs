#![deny(clippy::type_repetition_in_bounds)]

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub fn foo<T>(_t: T)
where
    T: Copy,
    T: Clone,
{
    unimplemented!();
}

pub fn bar<T, U>(_t: T, _u: U)
where
    T: Copy,
    U: Clone,
{
    unimplemented!();
}

trait LintBounds
where
    Self: Clone,
    Self: Copy + Default + Ord,
    Self: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign,
{
}

trait LotsOfBounds
where
    Self: Clone + Copy + Default + Ord,
    Self: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign,
{
}

fn main() {}
