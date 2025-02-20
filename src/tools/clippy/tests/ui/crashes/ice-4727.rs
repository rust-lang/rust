//@ check-pass

#![warn(clippy::use_self)]

#[path = "auxiliary/ice-4727-aux.rs"]
mod aux;

fn main() {}
