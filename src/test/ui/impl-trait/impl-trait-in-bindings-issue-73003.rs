// check-pass

#![feature(impl_trait_in_bindings)]
//~^ WARN the feature `impl_trait_in_bindings` is incomplete

const _: impl Fn() = ||();

fn main() {}
