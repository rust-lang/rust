#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

pub trait Yay {}
impl Yay for u32 {}

fn foo() {
    type Foo = impl Debug;
    is_yay::<Foo>(); //~ ERROR: the trait bound `Foo: Yay` is not satisfied
}

fn is_yay<T: Yay>() {}

fn main() {}
