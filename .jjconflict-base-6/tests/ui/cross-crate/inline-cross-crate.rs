//! Dpn't ice on using an inlined function from another crate
//! See <https://github.com/rust-lang/rust/issues/18502> and
//! <https://github.com/rust-lang/rust/issues/18501>

//@ run-pass
//@ aux-build:inline-cross-crate.rs

extern crate inline_cross_crate as fmt;

fn main() {
    ::fmt::baz();
}
