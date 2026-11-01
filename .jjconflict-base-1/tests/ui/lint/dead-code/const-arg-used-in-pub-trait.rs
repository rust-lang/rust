//! Regression test for <https://github.com/rust-lang/rust/issues/128617>
//@ check-pass

#![deny(dead_code)]

pub struct Value<const NUMBER: u8> {}

pub trait Trait {}

const CONST: u8 = 11;

impl Trait for Value<CONST> {}

fn main() {}
