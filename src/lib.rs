#![feature(rustc_private)]
#![allow(clippy::similar_names)]
#![allow(clippy::single_match_else)]
#![deny(warnings)]
extern crate rustc;
extern crate syntax;
extern crate syntax_pos;

mod changes;
mod mapping;
mod mismatch;
mod translate;
mod traverse;
mod typeck;

pub use self::traverse::{run_analysis, run_traversal};
