#![feature(type_alias_impl_trait)]

// Ensures that `const` items can constrain an opaque `impl Trait`.

use std::fmt::Debug;

pub type Foo = impl Debug;
//~^ ERROR could not find defining uses

const _FOO: Foo = 5;
//~^ ERROR mismatched types [E0308]

fn main() {}
