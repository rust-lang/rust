use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::in_macro;
use if_chain::if_chain;
use rustc_ast::{Crate, Item, ItemKind, UseTreeKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::symbol::kw;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// **What it does:** Checking for imports with single component use path.
    ///
    /// **Why is this bad?** Import with single component use path such as `use cratename;`
    /// is not necessary, and thus should be removed.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust,ignore
    /// use regex;
    ///
    /// fn main() {
    ///     regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    /// }
    /// ```
    /// Better as
    /// ```rust,ignore
    /// fn main() {
    ///     regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    /// }
    /// ```
    pub SINGLE_COMPONENT_PATH_IMPORTS,
    style,
    "imports with single component path are redundant"
}

#[derive(Default)]
pub struct SingleComponentPathImports {
    /// keep track of imports reused with `self` keyword,
    /// such as `self::crypto_hash` in the example below
    ///
    /// ```rust,ignore
    /// use self::crypto_hash::{Algorithm, Hasher};
    /// ```
    imports_reused_with_self: Vec<Symbol>,
    /// keep track of single use statements
    /// such as `crypto_hash` in the example below
    ///
    /// ```rust,ignore
    /// use crypto_hash;
    /// ```
    single_use_usages: Vec<(Symbol, Span)>,
}

impl_lint_pass!(SingleComponentPathImports => [SINGLE_COMPONENT_PATH_IMPORTS]);

impl EarlyLintPass for SingleComponentPathImports {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        if cx.sess.opts.edition < Edition::Edition2018 {
            return;
        }
        for item in &krate.items {
            self.track_uses(&item);
        }
        for single_use in &self.single_use_usages {
            if !self.imports_reused_with_self.contains(&single_use.0) {
                span_lint_and_sugg(
                    cx,
                    SINGLE_COMPONENT_PATH_IMPORTS,
                    single_use.1,
                    "this import is redundant",
                    "remove it entirely",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}

impl SingleComponentPathImports {
    fn track_uses(&mut self, item: &Item) {
        if_chain! {
            if !in_macro(item.span);
            if !item.vis.kind.is_pub();
            if let ItemKind::Use(use_tree) = &item.kind;
            if let segments = &use_tree.prefix.segments;

            then {
                // keep track of `use some_module;` usages
                if segments.len() == 1 {
                    if let UseTreeKind::Simple(None, _, _) = use_tree.kind {
                        let ident = &segments[0].ident;
                        self.single_use_usages.push((ident.name, item.span));
                    }
                    return;
                }

                // keep track of `use self::some_module` usages
                if segments[0].ident.name == kw::SelfLower {
                    // simple case such as `use self::module::SomeStruct`
                    if segments.len() > 1 {
                        self.imports_reused_with_self.push(segments[1].ident.name);
                        return;
                    }

                    // nested case such as `use self::{module1::Struct1, module2::Struct2}`
                    if let UseTreeKind::Nested(trees) = &use_tree.kind {
                        for tree in trees {
                            let segments = &tree.0.prefix.segments;
                            if !segments.is_empty() {
                                self.imports_reused_with_self.push(segments[0].ident.name);
                            }
                        }
                    }
                }
            }
        }
    }
}
