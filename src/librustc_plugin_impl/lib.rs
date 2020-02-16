//! Infrastructure for compiler plugins.
//!
//! Plugins are a deprecated way to extend the behavior of `rustc` in various ways.
//!
//! See the [`plugin`
//! feature](https://doc.rust-lang.org/nightly/unstable-book/language-features/plugin.html)
//! of the Unstable Book for some examples.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]
#![feature(nll)]

use rustc_lint::LintStore;

pub mod build;
pub mod load;

/// Structure used to register plugins.
///
/// A plugin registrar function takes an `&mut Registry` and should call
/// methods to register its plugins.
pub struct Registry<'a> {
    /// The `LintStore` allows plugins to register new lints.
    pub lint_store: &'a mut LintStore,
}
