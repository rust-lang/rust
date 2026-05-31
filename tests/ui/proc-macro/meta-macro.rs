//@ aux-build:make-macro.rs
//@ proc-macro: meta-macro.rs
//@ edition:2018
//@ compile-flags: -Z span-debug
//@ run-pass
//@ revisions: wasm native
//@[wasm] only-wasm-proc-macro
//@[native] ignore-wasm-proc-macro

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate meta_macro;

fn main() {
    meta_macro::print_def_site!();
}
