// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![feature(map_many_mut)]
#![feature(rustc_attrs)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

pub mod errors;

pub mod utils;
pub use lint::{declare_lint, declare_lint_pass, declare_tool_lint, impl_lint_pass};
pub use rustc_lint_defs as lint;
pub mod parse;

pub mod code_stats;
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

mod version;
pub use version::RustcVersion;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext: rustc_ast::HashStableContext + rustc_hir::HashStableContext {}
