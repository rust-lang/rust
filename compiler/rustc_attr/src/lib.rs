//! Functions and types dealing with attributes and meta items.
//!
//! FIXME(Centril): For now being, much of the logic is still in `rustc_ast::attr`.
//! The goal is to move the definition of `MetaItem` and things that don't need to be in `syntax`
//! to this crate.

#![feature(let_chains)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate rustc_macros;

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_macros::fluent_messages;

mod builtin;
mod session_diagnostics;

pub use builtin::*;
pub use IntType::*;
pub use ReprAttr::*;
pub use StabilityLevel::*;

pub use rustc_ast::attr::*;

pub(crate) use rustc_ast::HashStableContext;

fluent_messages! { "../locales/en-US.ftl" }
