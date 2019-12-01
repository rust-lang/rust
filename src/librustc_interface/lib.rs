#![feature(box_syntax)]
#![feature(set_stdio)]
#![feature(nll)]
#![feature(arbitrary_self_types)]
#![feature(generator_trait)]
#![feature(generators)]
#![cfg_attr(unix, feature(libc))]

#![recursion_limit="256"]

#[cfg(unix)]
extern crate libc;

pub mod interface;
mod passes;
mod queries;
pub mod util;
mod proc_macro_decls;

pub use interface::{run_compiler, Config};
pub use queries::Queries;

#[cfg(test)]
mod tests;
