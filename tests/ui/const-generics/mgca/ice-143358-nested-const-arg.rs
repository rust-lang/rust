//! Regression test for <https://github.com/rust-lang/rust/issues/143358>
//!
//! Using deeply nested const generic function calls in the type position of a
//! const parameter with both `generic_const_exprs` and `min_generic_const_args`
//! used to ICE with "unhandled node ConstArg" in `generics_of`.
//! Fixed by <https://github.com/rust-lang/rust/pull/149136>.

#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete
#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete

fn identity<const T: identity<{ identity::<{ identity::<{}> }>() }>>();
//~^ ERROR free function without a body
//~| ERROR expected type, found function `identity`
//~| ERROR complex const arguments must be placed inside of a `const` block

fn main() {}
