use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use rustc_ast::{ptr::P, Crate, Item, ItemKind, MacroDef, ModKind, UseTreeKind, VisibilityKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{edition::Edition, symbol::kw, Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checking for imports with single component use path.
    ///
    /// ### Why is this bad?
    /// Import with single component use path such as `use cratename;`
    /// is not necessary, and thus should be removed.
    ///
    /// ### Example
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
    #[clippy::version = "1.43.0"]
    pub SINGLE_COMPONENT_PATH_IMPORTS,
    style,
    "imports with single component path are redundant"
}

declare_lint_pass!(SingleComponentPathImports => [SINGLE_COMPONENT_PATH_IMPORTS]);

impl EarlyLintPass for SingleComponentPathImports {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        if cx.sess.opts.edition < Edition::Edition2018 {
            return;
        }
        check_mod(cx, &krate.items);
    }
}

fn check_mod(cx: &EarlyContext<'_>, items: &[P<Item>]) {
    // keep track of imports reused with `self` keyword,
    // such as `self::crypto_hash` in the example below
    // ```rust,ignore
    // use self::crypto_hash::{Algorithm, Hasher};
    // ```
    let mut imports_reused_with_self = Vec::new();

    // keep track of single use statements
    // such as `crypto_hash` in the example below
    // ```rust,ignore
    // use crypto_hash;
    // ```
    let mut single_use_usages = Vec::new();

    // keep track of macros defined in the module as we don't want it to trigger on this (#7106)
    // ```rust,ignore
    // macro_rules! foo { () => {} };
    // pub(crate) use foo;
    // ```
    let mut macros = Vec::new();

    for item in items {
        track_uses(
            cx,
            item,
            &mut imports_reused_with_self,
            &mut single_use_usages,
            &mut macros,
        );
    }

    for single_use in &single_use_usages {
        if !imports_reused_with_self.contains(&single_use.0) {
            let can_suggest = single_use.2;
            if can_suggest {
                span_lint_and_sugg(
                    cx,
                    SINGLE_COMPONENT_PATH_IMPORTS,
                    single_use.1,
                    "this import is redundant",
                    "remove it entirely",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            } else {
                span_lint_and_help(
                    cx,
                    SINGLE_COMPONENT_PATH_IMPORTS,
                    single_use.1,
                    "this import is redundant",
                    None,
                    "remove this import",
                );
            }
        }
    }
}

fn track_uses(
    cx: &EarlyContext<'_>,
    item: &Item,
    imports_reused_with_self: &mut Vec<Symbol>,
    single_use_usages: &mut Vec<(Symbol, Span, bool)>,
    macros: &mut Vec<Symbol>,
) {
    if item.span.from_expansion() || item.vis.kind.is_pub() {
        return;
    }

    match &item.kind {
        ItemKind::Mod(_, ModKind::Loaded(ref items, ..)) => {
            check_mod(cx, items);
        },
        ItemKind::MacroDef(MacroDef { macro_rules: true, .. }) => {
            macros.push(item.ident.name);
        },
        ItemKind::Use(use_tree) => {
            let segments = &use_tree.prefix.segments;

            let should_report =
                |name: &Symbol| !macros.contains(name) || matches!(item.vis.kind, VisibilityKind::Inherited);

            // keep track of `use some_module;` usages
            if segments.len() == 1 {
                if let UseTreeKind::Simple(None, _, _) = use_tree.kind {
                    let name = segments[0].ident.name;
                    if should_report(&name) {
                        single_use_usages.push((name, item.span, true));
                    }
                }
                return;
            }

            if segments.is_empty() {
                // keep track of `use {some_module, some_other_module};` usages
                if let UseTreeKind::Nested(trees) = &use_tree.kind {
                    for tree in trees {
                        let segments = &tree.0.prefix.segments;
                        if segments.len() == 1 {
                            if let UseTreeKind::Simple(None, _, _) = tree.0.kind {
                                let name = segments[0].ident.name;
                                if should_report(&name) {
                                    single_use_usages.push((name, tree.0.span, false));
                                }
                            }
                        }
                    }
                }
            } else {
                // keep track of `use self::some_module` usages
                if segments[0].ident.name == kw::SelfLower {
                    // simple case such as `use self::module::SomeStruct`
                    if segments.len() > 1 {
                        imports_reused_with_self.push(segments[1].ident.name);
                        return;
                    }

                    // nested case such as `use self::{module1::Struct1, module2::Struct2}`
                    if let UseTreeKind::Nested(trees) = &use_tree.kind {
                        for tree in trees {
                            let segments = &tree.0.prefix.segments;
                            if !segments.is_empty() {
                                imports_reused_with_self.push(segments[0].ident.name);
                            }
                        }
                    }
                }
            }
        },
        _ => {},
    }
}
