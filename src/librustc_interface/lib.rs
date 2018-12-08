#![feature(box_syntax)]
#![feature(set_stdio)]
#![feature(nll)]
#![feature(arbitrary_self_types)]
#![feature(generator_trait)]
#![feature(generators)]
#![cfg_attr(unix, feature(libc))]

#![allow(unused_imports)]

#![recursion_limit="256"]

#[cfg(unix)]
extern crate libc;
#[macro_use]
extern crate log;
extern crate rustc;
extern crate rustc_codegen_utils;
extern crate rustc_allocator;
extern crate rustc_borrowck;
extern crate rustc_incremental;
extern crate rustc_traits;
#[macro_use]
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_lint;
extern crate rustc_metadata;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate rustc_plugin;
extern crate rustc_privacy;
extern crate rustc_rayon as rayon;
extern crate rustc_resolve;
extern crate rustc_typeck;
extern crate smallvec;
extern crate serialize;
extern crate syntax;
extern crate syntax_pos;
extern crate syntax_ext;

pub mod interface;
mod passes;
mod queries;
pub mod util;
mod proc_macro_decls;
mod profile;

pub use interface::{run_compiler, Config};
