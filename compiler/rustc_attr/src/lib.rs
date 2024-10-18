//! Centralized logic for parsing and validating all attributes used after HIR.
//!
//! History: Check out [#131229](https://github.com/rust-lang/rust/issues/131229).
//! There used to be only one definition of attributes in the compiler: `ast::Attribute`.
//! These were then parsed or validated or both in places distributed all over the compiler.
//!
//! TODO(jdonszelmann): update devguide for best practices on attributes
//! TODO(jdonszelmann): rename to `rustc_attr` in the future, integrating it into this crate.
//!
//! To define a new builtin, first add it

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

mod builtin;
mod session_diagnostics;

// TODO: remove reexports
pub use IntType::*;
pub use ReprAttr::*;
pub use StabilityLevel::*;
pub use builtin::*;
pub(crate) use rustc_session::HashStableContext;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

use rustc_ast::{self as ast};
use rustc_hir::attribute::{Attribute, ParsedAttributeKind};

pub fn parse_attribute_list(_attr: &[ast::Attribute]) -> Vec<Attribute> {
    todo!()
}

pub fn parse_attribute(attr: &ast::Attribute) -> Option<ParsedAttributeKind> {
    match attr {
        _ => None,
    }
}
