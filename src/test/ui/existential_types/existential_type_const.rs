// check-pass

#![feature(existential_type)]
// Currently, the `existential_type` feature implicitly
// depends on `impl_trait_in_bindings` in order to work properly.
// Specifically, this line requires `impl_trait_in_bindings` to be enabled:
// https://github.com/rust-lang/rust/blob/481068a707679257e2a738b40987246e0420e787/src/librustc_typeck/check/mod.rs#L856
#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

// Ensures that `const` items can constrain an `existential type`.

use std::fmt::Debug;

pub existential type Foo: Debug;

const _FOO: Foo = 5;

fn main() {
}
