#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(once_cell)]
#![feature(option_get_or_insert_default)]
#![feature(rustc_attrs)]
#![feature(map_many_mut)]
#![recursion_limit = "256"]
#![allow(rustc::potential_query_instability)]

#[macro_use]
extern crate rustc_macros;
pub mod errors;

pub mod cgu_reuse_tracker;
pub mod utils;
pub use lint::{declare_lint, declare_lint_pass, declare_tool_lint, impl_lint_pass};
pub use rustc_lint_defs as lint;
pub mod parse;

mod code_stats;
#[macro_use]
pub mod config;
pub mod cstore;
pub mod filesearch;
mod options;
pub mod search_paths;

mod session;
pub use session::*;

pub mod output;

pub use getopts;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext: rustc_ast::HashStableContext + rustc_hir::HashStableContext {}
