//@ check-pass

// Tests for nested self-reference which caused a stack overflow.

use std::fmt::Debug;
use std::ops::*;

fn gen() -> impl PartialOrd + PartialEq + Debug { }

struct Bar {}
trait Foo<T = Self> {}
trait FooNested<T = Option<Self>> {}
impl Foo for Bar {}
impl FooNested for Bar {}

fn foo() -> impl Foo + FooNested {
    Bar {}
}

fn test_impl_ops() -> impl Add + Sub + Mul + Div { 1 }
fn test_impl_assign_ops() -> impl AddAssign + SubAssign + MulAssign + DivAssign { 1 }

fn main() {}
