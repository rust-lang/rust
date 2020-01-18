//! The `rustc_ast_passes` crate contains passes which validate the AST in `syntax`
//! parsed by `rustc_parse` and then lowered, after the passes in this crate,
//! by `rustc_ast_lowering`.

#![cfg_attr(bootstrap, feature(slice_patterns))]

pub mod ast_validation;
pub mod feature_gate;
pub mod show_span;
