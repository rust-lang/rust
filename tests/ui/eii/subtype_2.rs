//@ compile-flags: --crate-type rlib
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_macro_for(bar)]
#[rustc_builtin_macro(eii_macro)]
macro foo() {

}

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
