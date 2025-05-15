//! The `rustc_ast_passes` crate contains passes which validate the AST in `syntax`
//! parsed by `rustc_parse` and then lowered, after the passes in this crate,
//! by `rustc_ast_lowering`.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(box_patterns)]
#![feature(if_let_guard)]
#![feature(iter_is_partitioned)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

pub mod ast_validation;
mod errors;
pub mod feature_gate;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
