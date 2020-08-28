// check-pass

#![feature(type_alias_impl_trait)]
// Currently, the `type_alias_impl_trait` feature implicitly
// depends on `impl_trait_in_bindings` in order to work properly.
// Specifically, this line requires `impl_trait_in_bindings` to be enabled:
// https://github.com/rust-lang/rust/blob/481068a707679257e2a738b40987246e0420e787/compiler/rustc_typeck/check/mod.rs#L856
#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete

// Ensures that `const` items can constrain an opaque `impl Trait`.

use std::fmt::Debug;

pub type Foo = impl Debug;

const _FOO: Foo = 5;

fn main() {
}
