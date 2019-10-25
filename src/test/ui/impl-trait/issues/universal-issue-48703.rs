#![feature(universal_impl_trait)]

use std::fmt::Debug;

fn foo<T>(x: impl Debug) { }

fn main() {
    foo::<String>('a'); //~ ERROR cannot provide explicit type parameters
}
