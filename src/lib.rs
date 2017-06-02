#![feature(rustc_private)]
#![feature(rustc_diagnostic_macros)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

extern crate rustc;
extern crate rustc_errors;
extern crate syntax;
extern crate syntax_pos;

pub mod semcheck;
