//@ check-pass
//@ compile-flags: -Z span-debug --error-format human
//@ proc-macro: test-macros.rs

// Regression test for #132727. `collect_tokens` must support nested
// replacement ranges: expanding the inner `cfg_attr` on `$expr` produces a
// replacement range nested inside the one for the `let` statement, which is
// itself nested inside the one for `Foo`'s anonymous constant. This case was
// originally covered by `macro-rules-derive-cfg.rs`, but #129346 (which
// removed support for nested replacement ranges) simplified the nesting away
// from that test, and the revert in #132587 didn't restore it.

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro_rules! produce_it {
    ($expr:expr) => {
        #[derive(Print)]
        struct Foo {
            val: [bool; {
                let a = #[cfg_attr(not(FALSE), rustc_dummy(first))] $expr;
                0
            }]
        }
    }
}

produce_it!(#[cfg_attr(not(FALSE), rustc_dummy(second))] {
    #![cfg_attr(not(FALSE), rustc_dummy(third))]
    #[cfg_attr(not(FALSE), rustc_dummy(fourth))]
    30
});

fn main() {}
