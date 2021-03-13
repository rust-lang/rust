#![feature(crate_visibility_modifier)]
#![feature(once_cell)]
#![feature(or_patterns)]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate rustc_macros;

pub mod cgu_reuse_tracker;
pub mod utils;
pub use lint::{declare_lint, declare_lint_pass, declare_tool_lint, impl_lint_pass};
pub use rustc_lint_defs as lint;
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
