//@ check-pass

//! Regression test for <https://github.com/rust-lang/rust/issues/165604>.
//!
//! `CoerceShared` through a shared reference is allowed, and used to cause
//! an ICE in borrowck.

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct CustomMut<'a>(&'a mut ());
impl<'a> Reborrow for CustomMut<'a> {}

#[derive(Clone, Copy)]
struct CustomRef<'a>(&'a ());
impl<'a> CoerceShared<CustomRef<'a>> for CustomMut<'a> {}

fn method(_: CustomRef) {}

fn main() {
    let mut x = ();
    let a = CustomMut(&mut x);
    let b = &a;
    method(*b);
}
