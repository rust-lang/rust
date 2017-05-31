#![feature(rustc_private)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

extern crate rustc;
extern crate syntax_pos;

pub mod semcheck;
