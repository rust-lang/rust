//! Infrastructure for compiler plugins.
//!
//! Plugins are a deprecated way to extend the behavior of `rustc` in various ways.
//!
//! See the [`plugin`
//! feature](https://doc.rust-lang.org/nightly/unstable-book/language-features/plugin.html)
//! of the Unstable Book for some examples.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![recursion_limit = "256"]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

use rustc_errors::{DiagnosticMessage, SubdiagnosticMessage};
use rustc_lint::LintStore;
use rustc_macros::fluent_messages;

mod errors;
pub mod load;

fluent_messages! { "../messages.ftl" }

/// Structure used to register plugins.
///
/// A plugin registrar function takes an `&mut Registry` and should call
/// methods to register its plugins.
pub struct Registry<'a> {
    /// The `LintStore` allows plugins to register new lints.
    pub lint_store: &'a mut LintStore,
}
