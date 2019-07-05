// check-pass

#![feature(existential_type)]
#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete and may cause the compiler to crash

// Ensures that `const` items can constrain an `existential type`.

use std::fmt::Debug;

pub existential type Foo: Debug;

const _FOO: Foo = 5;

fn main() {
}
