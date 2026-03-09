// issue: <https://github.com/rust-lang/rust/issues/14344>
// Test that we can depend on an `no-prefer-dynamic` crate.
//@ run-pass
//@ aux-build:extern-crate-no-prefer-dynamic-aux-1.rs
//@ aux-build:extern-crate-no-prefer-dynamic-aux-2.rs

extern crate extern_crate_no_prefer_dynamic_aux_1;
extern crate extern_crate_no_prefer_dynamic_aux_2;

fn main() {
    extern_crate_no_prefer_dynamic_aux_1::foo();
    extern_crate_no_prefer_dynamic_aux_2::bar();
}
