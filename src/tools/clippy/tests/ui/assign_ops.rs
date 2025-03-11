#![allow(clippy::useless_vec)]
#![warn(clippy::assign_op_pattern)]
#![feature(const_trait_impl, const_ops)]

use core::num::Wrapping;
use std::ops::{Mul, MulAssign};

fn main() {
    let mut a = 5;
    a = a + 1;
    //~^ assign_op_pattern
    a = 1 + a;
    //~^ assign_op_pattern
    a = a - 1;
    //~^ assign_op_pattern
    a = a * 99;
    //~^ assign_op_pattern
    a = 42 * a;
    //~^ assign_op_pattern
    a = a / 2;
    //~^ assign_op_pattern
    a = a % 5;
    //~^ assign_op_pattern
    a = a & 1;
    //~^ assign_op_pattern
    a = 1 - a;
    a = 5 / a;
    a = 42 % a;
    a = 6 << a;
    let mut s = String::new();
    s = s + "bla";
    //~^ assign_op_pattern

    // Issue #9180
    let mut a = Wrapping(0u32);
    a = a + Wrapping(1u32);
    //~^ assign_op_pattern
    let mut v = vec![0u32, 1u32];
    v[0] = v[0] + v[1];
    //~^ assign_op_pattern
    let mut v = vec![Wrapping(0u32), Wrapping(1u32)];
    v[0] = v[0] + v[1];
    let _ = || v[0] = v[0] + v[1];
}

fn cow_add_assign() {
    use std::borrow::Cow;
    let mut buf = Cow::Owned(String::from("bar"));
    let cows = Cow::Borrowed("foo");

    // this can be linted
    buf = buf + cows.clone();
    //~^ assign_op_pattern

    // this should not as cow<str> Add is not commutative
    buf = cows + buf;
}

// check that we don't lint on op assign impls, because that's just the way to impl them

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Wrap(i64);

impl Mul<i64> for Wrap {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self {
        Wrap(self.0 * rhs)
    }
}

impl MulAssign<i64> for Wrap {
    fn mul_assign(&mut self, rhs: i64) {
        *self = *self * rhs
    }
}

mod issue14871 {

    use std::ops::{Add, AddAssign};

    pub trait Number: Copy + Add<Self, Output = Self> + AddAssign {
        const ZERO: Self;
        const ONE: Self;
    }

    #[const_trait]
    pub trait NumberConstants {
        fn constant(value: usize) -> Self;
    }

    impl<T> const NumberConstants for T
    where
        T: Number + [const] core::ops::Add,
    {
        fn constant(value: usize) -> Self {
            let mut res = Self::ZERO;
            let mut count = 0;
            while count < value {
                res = res + Self::ONE;
                count += 1;
            }
            res
        }
    }
}
