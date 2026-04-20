// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(const_option_ops)]
#![feature(const_trait_impl)]
#![feature(default_field_values)]
#![feature(iter_intersperse)]
#![feature(macro_derive)]
#![feature(macro_metavar_expr)]
#![feature(rustc_attrs)]
// To generate CodegenOptionsTargetModifiers and UnstableOptionsTargetModifiers enums
// with macro_rules, it is necessary to use recursive mechanic ("Incremental TT Munchers").
#![recursion_limit = "256"]
// tidy-alphabetical-end

pub use getopts;
pub use lint::{declare_lint, declare_lint_pass, declare_tool_lint, impl_lint_pass};
pub use rustc_lint_defs as lint;
pub use session::*;

pub mod code_stats;
pub mod errors;
pub mod parse;
pub mod utils;
#[macro_use]
pub mod config;
pub mod cstore;
pub mod filesearch;
mod macros;
mod options;
pub mod output;
pub mod search_paths;
mod session;
