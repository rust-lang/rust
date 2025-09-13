//@ compile-flags: --crate-type rlib
//@ check-pass
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_extern_target(bar)]
#[rustc_builtin_macro(eii_shared_macro)]
macro foo() {}

unsafe extern "Rust" {
    safe fn bar<'a>(x: &'a u64) -> &'a u64;
}

#[foo]
fn other<'a>(x: &'a u64) -> &'static u64 {
    &0
}

fn main() {
    bar(&0);
}
