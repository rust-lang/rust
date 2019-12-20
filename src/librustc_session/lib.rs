#![feature(crate_visibility_modifier)]
#![feature(test)]

// Use the test crate here so we depend on getopts through it. This allow tools to link to both
// librustc_session and libtest.
extern crate test as _;
extern crate getopts;

pub mod cgu_reuse_tracker;
pub mod utils;
#[macro_use]
pub mod lint;
pub mod node_id;
pub mod parse;

mod code_stats;
#[macro_use]
pub mod config;
mod options;
pub mod filesearch;
pub mod search_paths;

mod session;
pub use session::*;
