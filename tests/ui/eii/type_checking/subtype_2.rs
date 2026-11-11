//@ compile-flags: --crate-type rlib
// Uses manual desugaring of EII internals:
// Tests whether it's not ok when the implementation is less general.
#![feature(extern_item_impls)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_declaration(bar)]
#[rustc_builtin_macro(eii_shared_macro)]
macro foo() {}

unsafe extern "Rust" {
    safe fn bar<'a>(x: &'static u64) -> &'a u64;
}

#[foo]
fn other<'a>(x: &'a u64) -> &'static u64 {
    //~^ ERROR lifetime parameters or bounds of `other` do not match the declaration
    &0
}

fn main() {
    bar(&0);
}
