//@ run-rustfix

#![allow(dead_code)]

use std::ops::Add;

struct A<B>(B);

impl<B> Add for A<B> where B: Add {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        A(self.0 + rhs.0) //~ ERROR mismatched types
    }
}

struct C<B>(B);

impl<B: Add> Add for C<B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0) //~ ERROR mismatched types
    }
}

struct D<B>(B);

impl<B> Add for D<B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0) //~ ERROR cannot add `B` to `B`
    }
}

struct E<B>(B);

impl<B: Add> Add for E<B> where <B as Add>::Output = B {
    //~^ ERROR equality constraints are not yet supported in `where` clauses
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0) //~ ERROR mismatched types
    }
}

fn main() {}
