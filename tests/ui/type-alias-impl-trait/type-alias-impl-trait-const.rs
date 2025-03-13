#![feature(type_alias_impl_trait)]
// Ensures that `const` items can not constrain an opaque `impl Trait`.

use std::fmt::Debug;

pub type Foo = impl Debug;
//~^ ERROR unconstrained opaque type

#[define_opaque(Foo)]
//~^ ERROR only functions and methods can define opaque types
const _FOO: Foo = 5;
//~^ ERROR mismatched types

#[define_opaque(Foo)]
//~^ ERROR only functions and methods can define opaque types
static _BAR: Foo = 22_u32;
//~^ ERROR mismatched types

fn main() {}
