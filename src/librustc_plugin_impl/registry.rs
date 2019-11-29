//! Used by plugin crates to tell `rustc` about the plugins they provide.

use rustc::lint::LintStore;
use rustc::session::Session;
use syntax_pos::Span;

use std::borrow::ToOwned;

/// Structure used to register plugins.
///
/// A plugin registrar function takes an `&mut Registry` and should call
/// methods to register its plugins.
///
/// This struct has public fields and other methods for use by `rustc`
/// itself. They are not documented here, and plugin authors should
/// not use them.
pub struct Registry<'a> {
    /// Compiler session. Useful if you want to emit diagnostic messages
    /// from the plugin registrar.
    pub sess: &'a Session,

    /// The `LintStore` allows plugins to register new lints.
    pub lint_store: &'a mut LintStore,

    #[doc(hidden)]
    pub krate_span: Span,

    #[doc(hidden)]
    pub llvm_passes: Vec<String>,
}

impl<'a> Registry<'a> {
    #[doc(hidden)]
    pub fn new(sess: &'a Session, lint_store: &'a mut LintStore, krate_span: Span) -> Registry<'a> {
        Registry {
            sess,
            lint_store,
            krate_span,
            llvm_passes: vec![],
        }
    }

    /// Register an LLVM pass.
    ///
    /// Registration with LLVM itself is handled through static C++ objects with
    /// constructors. This method simply adds a name to the list of passes to
    /// execute.
    pub fn register_llvm_pass(&mut self, name: &str) {
        self.llvm_passes.push(name.to_owned());
    }
}
