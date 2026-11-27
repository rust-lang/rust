//! Regression test for https://github.com/rust-lang/rust/issues/32518
//@ run-pass
//@ no-prefer-dynamic
//@ aux-build:no-duplicate-symbols-with-codegen-units-cgu-test.rs
//@ aux-build:no-duplicate-symbols-with-codegen-units-cgu-test-a.rs
//@ aux-build:no-duplicate-symbols-with-codegen-units-cgu-test-b.rs

extern crate no_duplicate_symbols_with_codegen_units_cgu_test_a as cgu_test_a;
extern crate no_duplicate_symbols_with_codegen_units_cgu_test_b as cgu_test_b;

fn main() {
    cgu_test_a::a::a();
    cgu_test_b::a::a();
}
