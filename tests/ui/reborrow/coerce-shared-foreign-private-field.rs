//@ check-pass

//@ aux-build: reborrow_foreign_private.rs

//! Test that CoerceShared cannot be implemented targeting a foreign struct with private fields.

#![feature(reborrow)]

extern crate reborrow_foreign_private;

use reborrow_foreign_private::ForeignRef;
use std::marker::{CoerceShared, Reborrow};

struct LocalMut<'a> {
    value: &'a mut i32,
}

impl<'a> Reborrow for LocalMut<'a> {}

// Should error: ForeignRef has private fields.
impl<'a> CoerceShared<ForeignRef<'a>> for LocalMut<'a> {}

fn main() {}
