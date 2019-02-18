#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]

extern crate rustc;
extern crate syntax;
extern crate syntax_pos;

mod changes;
mod mapping;
mod mismatch;
mod translate;
mod traverse;
mod typeck;

pub use self::traverse::run_analysis;
