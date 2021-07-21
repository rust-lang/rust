// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

// Ensures that `const` items can constrain an opaque `impl Trait`.

use std::fmt::Debug;

pub type Foo = impl Debug;
//~^ ERROR could not find defining uses

const _FOO: Foo = 5;
//~^ ERROR mismatched types [E0308]

fn main() {}
