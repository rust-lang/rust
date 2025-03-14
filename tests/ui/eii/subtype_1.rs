//@ compile-flags: --crate-type rlib
//@ check-pass
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[core::eii_macro_for(bar)]
#[rustc_builtin_macro(eii_macro)]
macro foo() {

}

unsafe extern "Rust" {
    safe fn bar<'a, 'b>(x: &'b u64) -> &'a u64;
}

#[foo]
fn other<'a, 'b>(x: &'b u64) -> &'b u64 {
    &0
}

fn main() {
    bar(&0);
}
