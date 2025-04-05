//! Centralized logic for parsing and attributes.
//!
//! Part of a series of crates:
//! - rustc_attr_data_structures: contains types that the parsers parse into
//! - rustc_attr_parsing: this crate
//! - (in the future): rustc_attr_validation
//!
//! History: Check out [#131229](https://github.com/rust-lang/rust/issues/131229).
//! There used to be only one definition of attributes in the compiler: `ast::Attribute`.
//! These were then parsed or validated or both in places distributed all over the compiler.
//! This was a mess...
//!
//! Attributes are markers on items.
//! Many of them are actually attribute-like proc-macros, and are expanded to some other rust syntax.
//! This could either be a user provided proc macro, or something compiler provided.
//! `derive` is an example of one that the compiler provides.
//! These are built-in, but they have a valid expansion to Rust tokens and are thus called "active".
//! I personally like calling these *active* compiler-provided attributes, built-in *macros*,
//! because they still expand, and this helps to differentiate them from built-in *attributes*.
//! However, I'll be the first to admit that the naming here can be confusing.
//!
//! The alternative to active attributes, are inert attributes.
//! These can occur in user code (proc-macro helper attributes).
//! But what's important is, many built-in attributes are inert like this.
//! There is nothing they expand to during the macro expansion process,
//! sometimes because they literally cannot expand to something that is valid Rust.
//! They are really just markers to guide the compilation process.
//! An example is `#[inline(...)]` which changes how code for functions is generated.
//!
//! ```text
//!                      Active                 Inert
//!              ┌──────────────────────┬──────────────────────┐
//!              │     (mostly in)      │    these are parsed  │
//!              │ rustc_builtin_macros │        here!         │
//!              │                      │                      │
//!              │                      │                      │
//!              │    #[derive(...)]    │    #[stable()]       │
//!     Built-in │    #[cfg()]          │    #[inline()]       │
//!              │    #[cfg_attr()]     │    #[repr()]         │
//!              │                      │                      │
//!              │                      │                      │
//!              │                      │                      │
//!              ├──────────────────────┼──────────────────────┤
//!              │                      │                      │
//!              │                      │                      │
//!              │                      │       `b` in         │
//!              │                      │ #[proc_macro_derive( │
//! User created │ #[proc_macro_attr()] │    a,                │
//!              │                      │    attributes(b)     │
//!              │                      │ ]                    │
//!              │                      │                      │
//!              │                      │                      │
//!              │                      │                      │
//!              └──────────────────────┴──────────────────────┘
//! ```
//!
//! In this crate, syntactical attributes (sequences of tokens that look like
//! `#[something(something else)]`) are parsed into more semantic attributes, markers on items.
//! Multiple syntactic attributes might influence a single semantic attribute. For example,
//! `#[stable(...)]` and `#[unstable()]` cannot occur together, and both semantically define
//! a "stability" of an item. So, the stability attribute has an
//! [`AttributeParser`](attributes::AttributeParser) that recognizes both the `#[stable()]`
//! and `#[unstable()]` syntactic attributes, and at the end produce a single [`AttributeKind::Stability`].
//!
//! As a rule of thumb, when a syntactical attribute can be applied more than once, they should be
//! combined into a single semantic attribute. For example:
//!
//! ```
//! #[repr(C)]
//! #[repr(packed)]
//! struct Meow {}
//! ```
//!
//! should result in a single `AttributeKind::Repr` containing a list of repr annotations, in this
//! case `C` and `packed`. This is equivalent to writing `#[repr(C, packed)]` in a single
//! syntactical annotation.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

#[macro_use]
mod attributes;
mod context;
pub mod parser;
mod session_diagnostics;

pub use attributes::cfg::*;
pub use attributes::util::{find_crate_name, is_builtin_attr, parse_version};
pub use context::{AttributeParser, OmitDoc};
pub use rustc_attr_data_structures::*;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
