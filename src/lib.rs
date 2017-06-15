#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

extern crate rustc;
extern crate rustc_errors;
extern crate semver;
extern crate syntax;
extern crate syntax_pos;

pub mod semcheck;
