//! Infrastructure for compiler plugins.
//!
//! Plugins are a deprecated way to extend the behavior of `rustc` in various ways.
//!
//! See the [`plugin`
//! feature](https://doc.rust-lang.org/nightly/unstable-book/language-features/plugin.html)
//! of the Unstable Book for some examples.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(nll)]

#![recursion_limit="256"]

pub use registry::Registry;

pub mod registry;
pub mod load;
pub mod build;
