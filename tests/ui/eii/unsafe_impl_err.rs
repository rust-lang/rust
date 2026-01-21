//@ compile-flags: --crate-type rlib
// Tests whether it's an error to implement an unsafe EII safely.
#![feature(extern_item_impls)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_declaration(bar, "unsafe")]
#[rustc_builtin_macro(eii_shared_macro)]
macro foo() {}

unsafe extern "Rust" {
    safe fn bar(x: u64) -> u64;
}

#[foo] //~ ERROR `#[foo]` is unsafe to implement
fn other(x: u64) -> u64 {
    x
}

fn main() {
    bar(0);
}
