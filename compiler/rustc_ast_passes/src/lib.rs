//! The `rustc_ast_passes` crate contains passes which validate the AST in `syntax`
//! parsed by `rustc_parse` and then lowered, after the passes in this crate,
//! by `rustc_ast_lowering`.
//!
//! The crate also contains other misc AST visitors, e.g. `node_count` and `show_span`.

#![allow(rustc::potential_query_instability)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(iter_is_partitioned)]
#![feature(let_chains)]
#![feature(let_else)]
#![recursion_limit = "256"]

pub mod ast_validation;
mod errors;
pub mod feature_gate;
pub mod node_count;
pub mod show_span;
