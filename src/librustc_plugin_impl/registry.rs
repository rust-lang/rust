//! Used by plugin crates to tell `rustc` about the plugins they provide.

use rustc::lint::LintStore;
use rustc::session::Session;

use syntax_expand::base::{SyntaxExtension, SyntaxExtensionKind, NamedSyntaxExtension};
use syntax_expand::base::MacroExpanderFn;
use syntax::symbol::Symbol;
use syntax::ast;
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
    pub args_hidden: Option<Vec<ast::NestedMetaItem>>,

    #[doc(hidden)]
    pub krate_span: Span,

    #[doc(hidden)]
    pub syntax_exts: Vec<NamedSyntaxExtension>,

    #[doc(hidden)]
    pub llvm_passes: Vec<String>,
}

impl<'a> Registry<'a> {
    #[doc(hidden)]
    pub fn new(sess: &'a Session, lint_store: &'a mut LintStore, krate_span: Span) -> Registry<'a> {
        Registry {
            sess,
            lint_store,
            args_hidden: None,
            krate_span,
            syntax_exts: vec![],
            llvm_passes: vec![],
        }
    }

    /// Gets the plugin's arguments, if any.
    ///
    /// These are specified inside the `plugin` crate attribute as
    ///
    /// ```no_run
    /// #![plugin(my_plugin_name(... args ...))]
    /// ```
    ///
    /// Returns empty slice in case the plugin was loaded
    /// with `--extra-plugins`
    pub fn args(&self) -> &[ast::NestedMetaItem] {
        self.args_hidden.as_ref().map(|v| &v[..]).unwrap_or(&[])
    }

    /// Register a syntax extension of any kind.
    ///
    /// This is the most general hook into `libsyntax`'s expansion behavior.
    pub fn register_syntax_extension(&mut self, name: ast::Name, extension: SyntaxExtension) {
        self.syntax_exts.push((name, extension));
    }

    /// Register a macro of the usual kind.
    ///
    /// This is a convenience wrapper for `register_syntax_extension`.
    /// It builds for you a `SyntaxExtensionKind::LegacyBang` that calls `expander`,
    /// and also takes care of interning the macro's name.
    pub fn register_macro(&mut self, name: &str, expander: MacroExpanderFn) {
        let kind = SyntaxExtensionKind::LegacyBang(Box::new(expander));
        let ext = SyntaxExtension::default(kind, self.sess.edition());
        self.register_syntax_extension(Symbol::intern(name), ext);
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
