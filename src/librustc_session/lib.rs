#![feature(crate_visibility_modifier)]
#![feature(or_patterns)]

#[macro_use]
extern crate bitflags;

pub mod cgu_reuse_tracker;
pub mod utils;
#[macro_use]
pub mod lint;
pub mod parse;

mod code_stats;
#[macro_use]
pub mod config;
pub mod filesearch;
mod options;
pub mod search_paths;

mod session;
pub use session::*;

pub mod output;

pub use getopts;
