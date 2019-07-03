//! Used by plugin crates to tell `rustc` about the plugins they provide.

use rustc::lint::{EarlyLintPassObject, LateLintPassObject, LintId, Lint};
use rustc::session::Session;
use rustc::util::nodemap::FxHashMap;

use syntax::ext::base::{SyntaxExtension, SyntaxExtensionKind, NamedSyntaxExtension};
use syntax::ext::base::MacroExpanderFn;
use syntax::symbol::Symbol;
use syntax::ast;
use syntax::feature_gate::AttributeType;
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

    #[doc(hidden)]
    pub args_hidden: Option<Vec<ast::NestedMetaItem>>,

    #[doc(hidden)]
    pub krate_span: Span,

    #[doc(hidden)]
    pub syntax_exts: Vec<NamedSyntaxExtension>,

    #[doc(hidden)]
    pub early_lint_passes: Vec<EarlyLintPassObject>,

    #[doc(hidden)]
    pub late_lint_passes: Vec<LateLintPassObject>,

    #[doc(hidden)]
    pub lint_groups: FxHashMap<&'static str, (Vec<LintId>, Option<&'static str>)>,

    #[doc(hidden)]
    pub llvm_passes: Vec<String>,

    #[doc(hidden)]
    pub attributes: Vec<(Symbol, AttributeType)>,
}

impl<'a> Registry<'a> {
    #[doc(hidden)]
    pub fn new(sess: &'a Session, krate_span: Span) -> Registry<'a> {
        Registry {
            sess,
            args_hidden: None,
            krate_span,
            syntax_exts: vec![],
            early_lint_passes: vec![],
            late_lint_passes: vec![],
            lint_groups: FxHashMap::default(),
            llvm_passes: vec![],
            attributes: vec![],
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
    pub fn register_syntax_extension(&mut self, name: ast::Name, mut extension: SyntaxExtension) {
        if extension.def_info.is_none() {
            extension.def_info = Some((ast::CRATE_NODE_ID, self.krate_span));
        }
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

    /// Register a compiler lint pass.
    pub fn register_early_lint_pass(&mut self, lint_pass: EarlyLintPassObject) {
        self.early_lint_passes.push(lint_pass);
    }

    /// Register a compiler lint pass.
    pub fn register_late_lint_pass(&mut self, lint_pass: LateLintPassObject) {
        self.late_lint_passes.push(lint_pass);
    }
    /// Register a lint group.
    pub fn register_lint_group(
        &mut self,
        name: &'static str,
        deprecated_name: Option<&'static str>,
        to: Vec<&'static Lint>
    ) {
        self.lint_groups.insert(name,
                                (to.into_iter().map(|x| LintId::of(x)).collect(),
                                 deprecated_name));
    }

    /// Register an LLVM pass.
    ///
    /// Registration with LLVM itself is handled through static C++ objects with
    /// constructors. This method simply adds a name to the list of passes to
    /// execute.
    pub fn register_llvm_pass(&mut self, name: &str) {
        self.llvm_passes.push(name.to_owned());
    }

    /// Register an attribute with an attribute type.
    ///
    /// Registered attributes will bypass the `custom_attribute` feature gate.
    /// `Whitelisted` attributes will additionally not trigger the `unused_attribute`
    /// lint. `CrateLevel` attributes will not be allowed on anything other than a crate.
    pub fn register_attribute(&mut self, name: Symbol, ty: AttributeType) {
        self.attributes.push((name, ty));
    }
}
