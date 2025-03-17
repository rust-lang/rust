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
    safe fn bar(x: u64) -> u64;
}

#[foo]
fn other() -> u64 {
//~^ ERROR `other` has 0 parameters but #[foo] requires it to have
    3
}

fn main() {
    bar(0);
}
