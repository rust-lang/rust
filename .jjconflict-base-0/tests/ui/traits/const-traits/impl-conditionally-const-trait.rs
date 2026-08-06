//! This test ensures that we can only implement `const Trait` for a type
//! and not have the conditionally const syntax in that position.

#![feature(const_trait_impl)]

struct S;
trait T {}

impl [const] T for S {}
//~^ ERROR expected identifier, found `]`

fn main() {}
