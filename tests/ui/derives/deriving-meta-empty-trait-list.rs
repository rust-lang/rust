//@ check-pass

#![deny(unused)]

#[derive()] // OK
struct _Bar;

pub fn main() {}
