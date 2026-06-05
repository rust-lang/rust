//@ normalize-stderr: "\n\n\z" -> "\n"

//@ aux-build: reborrow_foreign_private.rs

#![feature(reborrow)]

extern crate reborrow_foreign_private;

use reborrow_foreign_private::ForeignRef;
use std::marker::{CoerceShared, Reborrow};

struct LocalMut<'a> {
    value: &'a mut i32,
}

impl<'a> Reborrow for LocalMut<'a> {}

impl<'a> CoerceShared<ForeignRef<'a>> for LocalMut<'a> {}
//~^ ERROR

fn main() {}
