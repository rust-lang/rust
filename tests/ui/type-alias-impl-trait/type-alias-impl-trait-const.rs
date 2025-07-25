#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

pub type Foo = impl Debug;

#[define_opaque(Foo)]
const _FOO: Foo = 5;

#[define_opaque(Foo)]
static _BAR: Foo = 22_i32;
//~^ ERROR cycle detected when computing type of `_BAR`
//~| ERROR cycle detected when computing type of `_BAR`
//~| ERROR cycle detected when computing type of `_BAR`

fn main() {}
