//! Centralized logic for parsing and attributes.
//!
//! ## Architecture
//! This crate is part of a series of crates and modules that handle attribute processing.
//! - [rustc_hir::attrs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/index.html): Defines the data structures that store parsed attributes
//! - [rustc_attr_parsing](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/index.html): This crate, handles the parsing of attributes
//! - (planned) rustc_attr_validation: Will handle attribute validation, logic currently handled in `rustc_passes`
//!
//! The separation between data structures and parsing follows the principle of separation of concerns.
//! Data structures (`rustc_hir::attrs`) define what attributes look like after parsing.
//! This crate (`rustc_attr_parsing`) handles how to convert raw tokens into those structures.
//! This split allows other parts of the compiler to use the data structures without needing
//! the parsing logic, making the codebase more modular and maintainable.
//!
//! ## Background
//! Previously, the compiler had a single attribute definition (`ast::Attribute`) with parsing and
//! validation scattered throughout the codebase. This was reorganized for better maintainability
//! (see [#131229](https://github.com/rust-lang/rust/issues/131229)).
//!
//! ## Types of Attributes
//! In Rust, attributes are markers that can be attached to items. They come in two main categories.
//!
//! ### 1. Active Attributes
//! These are attribute-like proc-macros that expand into other Rust code.
//! They can be either user-defined or compiler-provided. Examples of compiler-provided active attributes:
//!   - `#[derive(...)]`: Expands into trait implementations
//!   - `#[cfg()]`: Expands based on configuration
//!   - `#[cfg_attr()]`: Conditional attribute application
//!
//! ### 2. Inert Attributes
//! These are pure markers that don't expand into other code. They guide the compilation process.
//! They can be user-defined (in proc-macro helpers) or built-in. Examples of built-in inert attributes:
//!   - `#[stable()]`: Marks stable API items
//!   - `#[inline()]`: Suggests function inlining
//!   - `#[repr()]`: Controls type representation
//!
//! ```text
//!                      Active                 Inert
//!              ┌──────────────────────┬──────────────────────┐
//!              │     (mostly in)      │    these are parsed  │
//!              │ rustc_builtin_macros │        here!         │
//!              │                      │                      │
//!              │    #[derive(...)]    │    #[stable()]       │
//!     Built-in │    #[cfg()]          │    #[inline()]       │
//!              │    #[cfg_attr()]     │    #[repr()]         │
//!              │                      │                      │
//!              ├──────────────────────┼──────────────────────┤
//!              │                      │                      │
//!              │                      │       `b` in         │
//!              │                      │ #[proc_macro_derive( │
//! User created │ #[proc_macro_attr()] │    a,                │
//!              │                      │    attributes(b)     │
//!              │                      │ ]                    │
//!              └──────────────────────┴──────────────────────┘
//! ```
//!
//! ## How This Crate Works
//! In this crate, syntactical attributes (sequences of tokens that look like
//! `#[something(something else)]`) are parsed into more semantic attributes, markers on items.
//! Multiple syntactic attributes might influence a single semantic attribute. For example,
//! `#[stable(...)]` and `#[unstable()]` cannot occur together, and both semantically define
//! a "stability" of an item. So, the stability attribute has an
//! [`AttributeParser`](attributes::AttributeParser) that recognizes both the `#[stable()]`
//! and `#[unstable()]` syntactic attributes, and at the end produce a single
//! [`AttributeKind::Stability`](rustc_hir::attrs::AttributeKind::Stability).
//!
//! When multiple instances of the same attribute are allowed, they're combined into a single
//! semantic attribute. For example:
//!
//! ```rust
//! #[repr(C)]
//! #[repr(packed)]
//! struct Meow {}
//! ```
//!
//! This is equivalent to `#[repr(C, packed)]` and results in a single `AttributeKind::Repr`
//! containing both `C` and `packed` annotations.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(decl_macro)]
#![feature(rustdoc_internals)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

#[macro_use]
/// All the individual attribute parsers for each of rustc's built-in attributes.
mod attributes;

/// All the important types given to attribute parsers when parsing
pub(crate) mod context;

/// Code that other crates interact with, to actually parse a list (or sometimes single)
/// attribute.
mod interface;

/// Despite this entire module called attribute parsing and the term being a little overloaded,
/// in this module the code lives that actually breaks up tokenstreams into semantic pieces of attributes,
/// like lists or name-value pairs.
pub mod parser;

mod lints;
mod session_diagnostics;
mod target_checking;
pub mod validate_attr;

pub use attributes::cfg::{CFG_TEMPLATE, EvalConfigResult, eval_config_entry, parse_cfg_attr};
pub use attributes::cfg_old::*;
pub use attributes::util::{is_builtin_attr, is_doc_alias_attrs_contain_symbol, parse_version};
pub use context::{Early, Late, OmitDoc, ShouldEmit};
pub use interface::AttributeParser;
pub use lints::emit_attribute_lint;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
