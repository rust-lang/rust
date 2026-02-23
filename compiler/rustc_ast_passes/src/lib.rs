//! The `rustc_ast_passes` crate contains passes which validate the AST in `syntax`
//! parsed by `rustc_parse` and then lowered, after the passes in this crate,
//! by `rustc_ast_lowering`.

// tidy-alphabetical-start
#![cfg_attr(bootstrap, feature(if_let_guard))]
#![feature(box_patterns)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
// tidy-alphabetical-end

pub mod ast_validation;
mod errors;
pub mod feature_gate;
