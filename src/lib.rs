#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]

#[macro_use]
extern crate log;

#[cfg(test)]
extern crate quickcheck;

extern crate rustc;
extern crate semver;
extern crate syntax;
extern crate syntax_pos;

pub mod semcheck;
