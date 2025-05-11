#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

type Foo = impl Debug;

pub trait Yay {}
impl Yay for u32 {}

#[define_opaque(Foo)]
fn foo() {
    is_yay::<Foo>(); //~ ERROR: the trait bound `Foo: Yay` is not satisfied
}

fn is_yay<T: Yay>() {}

fn main() {}
