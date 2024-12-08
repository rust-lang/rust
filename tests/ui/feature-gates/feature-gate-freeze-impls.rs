#![feature(freeze, negative_impls)]

use std::marker::Freeze;

struct Foo;

unsafe impl Freeze for Foo {}
//~^ explicit impls for the `Freeze` trait are not permitted

struct Bar;

impl !Freeze for Bar {}
//~^ explicit impls for the `Freeze` trait are not permitted

fn main() {}
