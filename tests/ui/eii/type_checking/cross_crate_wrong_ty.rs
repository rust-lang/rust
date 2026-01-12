//@ compile-flags: --crate-type rlib
//@ aux-build: cross_crate_eii_declaration.rs
// Tests whether the type checking still works properly when the declaration is in another crate.
#![feature(extern_item_impls)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

extern crate cross_crate_eii_declaration;

#[unsafe(cross_crate_eii_declaration::foo)]
fn other() -> u64 {
//~^ ERROR `other` has 0 parameters but #[foo] requires it to have 1
    0
}

fn main() {
    cross_crate_eii_declaration::bar(0);
}
