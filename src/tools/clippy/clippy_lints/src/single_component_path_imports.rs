use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use rustc_ast::node_id::{NodeId, NodeMap};
use rustc_ast::ptr::P;
use rustc_ast::visit::{Visitor, walk_expr};
use rustc_ast::{Crate, Expr, ExprKind, Item, ItemKind, MacroDef, ModKind, Ty, TyKind, UseTreeKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::edition::Edition;
use rustc_span::symbol::kw;
use rustc_span::{Span, Symbol};

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

#[derive(Default)]
pub struct SingleComponentPathImports {
    /// Buffer found usages to emit when visiting that item so that `#[allow]` works as expected
    found: NodeMap<Vec<SingleUse>>,
}

struct SingleUse {
    name: Symbol,
    span: Span,
    item_id: NodeId,
    can_suggest: bool,
}

impl_lint_pass!(SingleComponentPathImports => [SINGLE_COMPONENT_PATH_IMPORTS]);

impl EarlyLintPass for SingleComponentPathImports {
    fn check_crate(&mut self, cx: &EarlyContext<'_>, krate: &Crate) {
        if cx.sess().opts.edition < Edition::Edition2018 {
            return;
        }

        self.check_mod(&krate.items);
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        for SingleUse { span, can_suggest, .. } in self.found.remove(&item.id).into_iter().flatten() {
            if can_suggest {
                span_lint_and_sugg(
                    cx,
                    SINGLE_COMPONENT_PATH_IMPORTS,
                    span,
                    "this import is redundant",
                    "remove it entirely",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            } else {
                span_lint_and_help(
                    cx,
                    SINGLE_COMPONENT_PATH_IMPORTS,
                    span,
                    "this import is redundant",
                    None,
                    "remove this import",
                );
            }
        }
    }
}

#[derive(Default)]
struct ImportUsageVisitor {
    // keep track of imports reused with `self` keyword, such as `self::std` in the example below.
    // Removing the `use std;` would make this a compile error (#10549)
    // ```
    // use std;
    //
    // fn main() {
    //     let _ = self::std::io::stdout();
    // }
    // ```
    imports_referenced_with_self: Vec<Symbol>,
}

impl Visitor<'_> for ImportUsageVisitor {
    fn visit_expr(&mut self, expr: &Expr) {
        if let ExprKind::Path(_, path) = &expr.kind
            && path.segments.len() > 1
            && path.segments[0].ident.name == kw::SelfLower
        {
            self.imports_referenced_with_self.push(path.segments[1].ident.name);
        }
        walk_expr(self, expr);
    }

    fn visit_ty(&mut self, ty: &Ty) {
        if let TyKind::Path(_, path) = &ty.kind
            && path.segments.len() > 1
            && path.segments[0].ident.name == kw::SelfLower
        {
            self.imports_referenced_with_self.push(path.segments[1].ident.name);
        }
    }
}

impl SingleComponentPathImports {
    fn check_mod(&mut self, items: &[P<Item>]) {
        // keep track of imports reused with `self` keyword, such as `self::crypto_hash` in the example
        // below. Removing the `use crypto_hash;` would make this a compile error
        // ```
        // use crypto_hash;
        //
        // use self::crypto_hash::{Algorithm, Hasher};
        // ```
        let mut imports_reused_with_self = Vec::new();

        // keep track of single use statements such as `crypto_hash` in the example below
        // ```
        // use crypto_hash;
        // ```
        let mut single_use_usages = Vec::new();

        // keep track of macros defined in the module as we don't want it to trigger on this (#7106)
        // ```
        // macro_rules! foo { () => {} };
        // pub(crate) use foo;
        // ```
        let mut macros = Vec::new();

        let mut import_usage_visitor = ImportUsageVisitor::default();
        for item in items {
            self.track_uses(item, &mut imports_reused_with_self, &mut single_use_usages, &mut macros);
            import_usage_visitor.visit_item(item);
        }

        for usage in single_use_usages {
            if !imports_reused_with_self.contains(&usage.name)
                && !import_usage_visitor.imports_referenced_with_self.contains(&usage.name)
            {
                self.found.entry(usage.item_id).or_default().push(usage);
            }
        }
    }

    fn track_uses(
        &mut self,
        item: &Item,
        imports_reused_with_self: &mut Vec<Symbol>,
        single_use_usages: &mut Vec<SingleUse>,
        macros: &mut Vec<Symbol>,
    ) {
        if item.span.from_expansion() || item.vis.kind.is_pub() {
            return;
        }

        match &item.kind {
            ItemKind::Mod(_, _, ModKind::Loaded(items, ..)) => {
                self.check_mod(items);
            },
            ItemKind::MacroDef(ident, MacroDef { macro_rules: true, .. }) => {
                macros.push(ident.name);
            },
            ItemKind::Use(use_tree) => {
                let segments = &use_tree.prefix.segments;

                // keep track of `use some_module;` usages
                if segments.len() == 1 {
                    if let UseTreeKind::Simple(None) = use_tree.kind {
                        let name = segments[0].ident.name;
                        if !macros.contains(&name) {
                            single_use_usages.push(SingleUse {
                                name,
                                span: item.span,
                                item_id: item.id,
                                can_suggest: true,
                            });
                        }
                    }
                    return;
                }

                if segments.is_empty() {
                    // keep track of `use {some_module, some_other_module};` usages
                    if let UseTreeKind::Nested { items, .. } = &use_tree.kind {
                        for tree in items {
                            let segments = &tree.0.prefix.segments;
                            if segments.len() == 1
                                && let UseTreeKind::Simple(None) = tree.0.kind
                            {
                                let name = segments[0].ident.name;
                                if !macros.contains(&name) {
                                    single_use_usages.push(SingleUse {
                                        name,
                                        span: tree.0.span,
                                        item_id: item.id,
                                        can_suggest: false,
                                    });
                                }
                            }
                        }
                    }
                }
                // keep track of `use self::some_module` usages
                else if segments[0].ident.name == kw::SelfLower {
                    // simple case such as `use self::module::SomeStruct`
                    if segments.len() > 1 {
                        imports_reused_with_self.push(segments[1].ident.name);
                        return;
                    }

                    // nested case such as `use self::{module1::Struct1, module2::Struct2}`
                    if let UseTreeKind::Nested { items, .. } = &use_tree.kind {
                        for tree in items {
                            let segments = &tree.0.prefix.segments;
                            if !segments.is_empty() {
                                imports_reused_with_self.push(segments[0].ident.name);
                            }
                        }
                    }
                }
            },
            _ => {},
        }
    }
}
