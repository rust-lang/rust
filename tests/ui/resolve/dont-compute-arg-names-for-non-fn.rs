//@ aux-build:foreign-trait-with-assoc.rs

extern crate foreign_trait_with_assoc;
use foreign_trait_with_assoc::Foo;

// Make sure we don't try to call `fn_arg_idents` on a non-fn item.

impl Foo for Bar {}
//~^ ERROR cannot find type `Bar` in this scope

fn main() {}
