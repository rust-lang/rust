//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
//@ run-pass
#![feature(field_projections, freeze)]
#![allow(incomplete_features, dead_code)]
use std::field::field_of;
use std::marker::{Freeze, Unpin};

struct Struct {
    field: u32,
}

union Union {
    field: u32,
}

enum Enum {
    Variant1 { field: u32 },
    Variant2(u32),
}

fn assert_traits<T: Send + Sync + Unpin + Copy + Clone + Sized + Freeze>() {}

fn main() {
    assert_traits::<field_of!(Struct, field)>();
    assert_traits::<field_of!(Union, field)>();
    assert_traits::<field_of!(Enum, Variant1.field)>();
    assert_traits::<field_of!(Enum, Variant2.0)>();
}
