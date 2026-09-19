//! The `rustc_ast_passes` crate contains passes which validate the AST in `syntax`
//! parsed by `rustc_parse` and then lowered, after the passes in this crate,
//! by `rustc_ast_lowering`.

// tidy-alphabetical-start
#![feature(deref_patterns)]
#![feature(iter_intersperse)]
#![feature(iter_is_partitioned)]
#![feature(option_into_flat_iter)]
// tidy-alphabetical-end

pub mod ast_validation;
mod diagnostics;
pub mod feature_gate;
