//! regression test for <https://github.com/rust-lang/rust/issues/143358>
#![expect(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]

fn identity<const T: identity<{ identity::<{ identity::<{}> }>() }>>();
//~^ ERROR: free function without a body
//~| ERROR: expected type, found function `identity`
//~| ERROR: complex const arguments must be placed inside of a `const` block

fn main() {}
