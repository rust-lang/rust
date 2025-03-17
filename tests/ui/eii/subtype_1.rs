//@ compile-flags: --crate-type rlib
//FIXME: known ICE
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_macro_for(bar)]
#[rustc_builtin_macro(eii_macro)]
macro foo() {

}

unsafe extern "Rust" {
    safe fn bar<'a, 'b>(x: &'b u64) -> &'a u64;
}

#[foo]
fn other<'a, 'b>(x: &'b u64) -> &'b u64 {
//~^ ERROR lifetime parameters or bounds `other` do not match the declaration
    &0
}

fn main() {
    bar(&0);
}
