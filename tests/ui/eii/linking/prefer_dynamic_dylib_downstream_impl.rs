//@ run-pass
//@ check-run-results
//@ aux-build: prefer_dynamic_decl.rs
//@ aux-build: prefer_dynamic_dylib_intermediate.rs
//@ compile-flags: -C prefer-dynamic
//@ needs-dynamic-linking
//@ ignore-backends: gcc
//@ ignore-windows
// Tests that EII resolution works across the dylib boundary with -C prefer-dynamic.
// The dylib intermediate crate declares an EII but does not implement it,
// and the final executable provides the explicit implementation.

extern crate prefer_dynamic_dylib_intermediate as intermediate;
extern crate prefer_dynamic_decl as decl;

#[decl::eii1]
fn eii1_impl(x: u64) {
    println!("explicit: {x:?}")
}

fn main() {
    // Call through the EII alias — should use the explicit impl
    decl::decl1(42);
    // Call directly
    eii1_impl(21);
}
