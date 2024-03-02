#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

type Foo = impl Debug;

pub trait Yay { }
impl Yay for u32 { }

fn foo() {
    is_yay::<Foo>(); //~ ERROR trait `Yay` is not implemented for `Foo`
}

fn is_yay<T: Yay>() { }

fn main() {}
