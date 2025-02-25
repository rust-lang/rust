//! Functions and types dealing with attributes and meta items.
//!
//! FIXME(Centril): For now being, much of the logic is still in `rustc_ast::attr`.
//! The goal is to move the definition of `MetaItem` and things that don't need to be in `syntax`
//! to this crate.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

mod attributes;
mod session_diagnostics;

pub use attributes::*;
pub use rustc_attr_data_structures::*;
pub use util::{find_crate_name, is_builtin_attr, parse_version};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
