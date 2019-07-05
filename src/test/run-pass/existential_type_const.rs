#![feature(existential_type)]
#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

// Ensures that consts can constrain an existential type

use std::fmt::Debug;

// Type `Foo` refers to a type that implements the `Debug` trait.
// The concrete type to which `Foo` refers is inferred from this module,
// and this concrete type is hidden from outer modules (but not submodules).
pub existential type Foo: Debug;

const _FOO: Foo = 5;

fn main() {
}
