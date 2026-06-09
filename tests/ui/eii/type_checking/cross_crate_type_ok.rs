//@ compile-flags: --crate-type rlib
//@ check-pass
//@ aux-build: cross_crate_eii_declaration.rs
// Tests whether calling EIIs works with the declaration in another crate.
#![feature(extern_item_impls)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

extern crate cross_crate_eii_declaration;

#[unsafe(cross_crate_eii_declaration::foo)]
fn other(x: u64) -> u64 {
    x
}

fn main() {
    cross_crate_eii_declaration::bar(0);
}
