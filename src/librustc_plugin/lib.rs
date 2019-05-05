//! Infrastructure for compiler plugins.
//!
//! Plugins are Rust libraries which extend the behavior of `rustc`
//! in various ways.
//!
//! Plugin authors will use the `Registry` type re-exported by
//! this module, along with its methods. The rest of the module
//! is for use by `rustc` itself.
//!
//! To define a plugin, build a dylib crate with a
//! `#[plugin_registrar]` function:
//!
//! ```no_run
//! #![crate_name = "myplugin"]
//! #![crate_type = "dylib"]
//! #![feature(plugin_registrar)]
//! #![feature(rustc_private)]
//!
//! extern crate rustc_plugin;
//! extern crate syntax;
//! extern crate syntax_pos;
//!
//! use rustc_plugin::Registry;
//! use syntax::ext::base::{ExtCtxt, MacResult};
//! use syntax_pos::Span;
//! use syntax::tokenstream::TokenTree;
//!
//! #[plugin_registrar]
//! pub fn plugin_registrar(reg: &mut Registry) {
//!     reg.register_macro("mymacro", expand_mymacro);
//! }
//!
//! fn expand_mymacro(cx: &mut ExtCtxt, span: Span, tt: &[TokenTree]) -> Box<MacResult> {
//!     unimplemented!()
//! }
//!
//! # fn main() {}
//! ```
//!
//! WARNING: We currently don't check that the registrar function
//! has the appropriate type!
//!
//! To use a plugin while compiling another crate:
//!
//! ```rust
//! #![feature(plugin)]
//! #![plugin(myplugin)]
//! ```
//!
//! See the [`plugin`
//! feature](https://doc.rust-lang.org/nightly/unstable-book/language-features/plugin.html)
//! of the Unstable Book for more examples.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(nll)]
#![feature(rustc_diagnostic_macros)]

#![recursion_limit="256"]

#![deny(rust_2018_idioms)]

pub use registry::Registry;

mod error_codes;
pub mod registry;
pub mod load;
pub mod build;

__build_diagnostic_array! { librustc_plugin, DIAGNOSTICS }
