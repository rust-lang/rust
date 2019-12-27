#![feature(bool_to_option)]
#![feature(box_syntax)]
#![feature(set_stdio)]
#![feature(nll)]
#![feature(arbitrary_self_types)]
#![feature(generator_trait)]
#![feature(generators)]
#![cfg_attr(unix, feature(libc))]
#![recursion_limit = "256"]

#[cfg(unix)]
extern crate libc;

mod callbacks;
pub mod interface;
mod passes;
mod proc_macro_decls;
mod queries;
pub mod util;

pub use interface::{run_compiler, Config};
pub use queries::Queries;

#[cfg(test)]
mod tests;
