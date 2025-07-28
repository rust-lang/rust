//@ compile-flags: --crate-type rlib
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_macro_for(bar, "unsafe")]
#[rustc_builtin_macro(eii_macro)]
macro foo() {

}

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
