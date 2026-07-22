//! Utility code in this crate can be used by bootstrap's `rustc` and `rustdoc`
//! shims, while also being shared by bootstrap and other bootstrap tools.
//!
//! Try to keep this crate small, to avoid bloating bootstrap build times.
//!
//! Ideally, any code in this crate should be used by at least one of the shims,
//! and at least one other crate (possibly the other shim).

mod arg_file_command;
pub mod proc_macro_deps;
mod shared_helpers;

pub use crate::arg_file_command::ArgFileCommand;
pub use crate::shared_helpers::{
    collect_args, dylib_path, dylib_path_var, exe, maybe_dump, parse_rustc_stage,
    parse_rustc_verbose, parse_value_from_args,
};
