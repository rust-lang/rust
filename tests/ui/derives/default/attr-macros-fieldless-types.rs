//! Regression test for https://github.com/rust-lang/rust/issues/164078,
//! where some attribute macros relied on the generated code always being
//! `Unit {}` and not `Unit()` or `Unit`.

//@ check-pass
//@ proc-macro: rewrite-fieldless-type-kind.rs

extern crate rewrite_fieldless_type_kind;
use rewrite_fieldless_type_kind::*;

#[derive(Default)]
#[to_unit]
struct UnitToUnit;

#[derive(Default)]
#[to_tuple]
struct UnitToTuple;

#[derive(Default)]
#[to_braced]
struct UnitToBraced;

#[derive(Default)]
#[to_unit]
struct TupleToUnit();

#[derive(Default)]
#[to_tuple]
struct TupleToTuple();

#[derive(Default)]
#[to_braced]
struct TupleToBraced();

#[derive(Default)]
#[to_unit]
struct BracedToUnit {}

#[derive(Default)]
#[to_tuple]
struct BracedToTuple {}

#[derive(Default)]
#[to_braced]
struct BracedToBraced {}

fn main() {}
