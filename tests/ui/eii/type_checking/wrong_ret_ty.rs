//@ compile-flags: --crate-type rlib
// Uses manual desugaring of EII internals: tests whether the return type matches.
#![feature(extern_item_impls)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_declaration(bar)]
#[rustc_builtin_macro(eii_shared_macro)]
macro foo() {}

unsafe extern "Rust" {
    safe fn bar(x: u64) -> u64;
}

#[foo]
fn other(_x: u64) {
    //~^ ERROR function `other` has a type that is incompatible with the declaration
}

fn main() {
    bar(0);
}
