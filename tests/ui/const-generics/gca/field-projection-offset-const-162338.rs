//! Regression test for <https://github.com/rust-lang/rust/issues/162338>.
//@ build-pass
//@ compile-flags: -Znext-solver=globally
#![feature(gca_const_items, gca_min_const_items, field_projections)]

use std::field::{Field, field_of};

struct Struct {
    b: i64,
}

fn project_ref() {
    <field_of!(Struct, b)>::OFFSET;
}

fn main() {}
