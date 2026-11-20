//! Regression test for <https://github.com/rust-lang/rust/issues/155724>
//@ check-pass
//@ edition:2024
//@ proc-macro: closure-hir-span.rs

extern crate closure_hir_span;

fn main() {
    closure_hir_span::m!(move || {});
}
