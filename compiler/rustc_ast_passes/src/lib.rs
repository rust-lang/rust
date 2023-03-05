//! The `rustc_ast_passes` crate contains passes which validate the AST in `syntax`
//! parsed by `rustc_parse` and then lowered, after the passes in this crate,
//! by `rustc_ast_lowering`.
//!
//! The crate also contains other misc AST visitors, e.g. `node_count` and `show_span`.

#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(iter_is_partitioned)]
#![feature(let_chains)]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_macros::fluent_messages;

pub mod ast_validation;
mod errors;
pub mod feature_gate;
pub mod node_count;
pub mod show_span;

fluent_messages! { "../locales/en-US.ftl" }
