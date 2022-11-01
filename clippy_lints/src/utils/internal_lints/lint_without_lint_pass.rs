use crate::utils::internal_lints::metadata_collector::is_deprecated_lint;
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::macros::root_macro_call_first_node;
use clippy_utils::{is_lint_allowed, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast as ast;
use rustc_ast::ast::LitKind;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::hir_id::CRATE_HIR_ID;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{ExprKind, HirId, Item, MutTy, Mutability, Path, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Symbol;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Ensures every lint is associated to a `LintPass`.
    ///
    /// ### Why is this bad?
    /// The compiler only knows lints via a `LintPass`. Without
    /// putting a lint to a `LintPass::get_lints()`'s return, the compiler will not
    /// know the name of the lint.
    ///
    /// ### Known problems
    /// Only checks for lints associated using the
    /// `declare_lint_pass!`, `impl_lint_pass!`, and `lint_array!` macros.
    ///
    /// ### Example
    /// ```rust,ignore
    /// declare_lint! { pub LINT_1, ... }
    /// declare_lint! { pub LINT_2, ... }
    /// declare_lint! { pub FORGOTTEN_LINT, ... }
    /// // ...
    /// declare_lint_pass!(Pass => [LINT_1, LINT_2]);
    /// // missing FORGOTTEN_LINT
    /// ```
    pub LINT_WITHOUT_LINT_PASS,
    internal,
    "declaring a lint without associating it in a LintPass"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for cases of an auto-generated lint without an updated description,
    /// i.e. `default lint description`.
    ///
    /// ### Why is this bad?
    /// Indicates that the lint is not finished.
    ///
    /// ### Example
    /// ```rust,ignore
    /// declare_lint! { pub COOL_LINT, nursery, "default lint description" }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// declare_lint! { pub COOL_LINT, nursery, "a great new lint" }
    /// ```
    pub DEFAULT_LINT,
    internal,
    "found 'default lint description' in a lint declaration"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for invalid `clippy::version` attributes.
    ///
    /// Valid values are:
    /// * "pre 1.29.0"
    /// * any valid semantic version
    pub INVALID_CLIPPY_VERSION_ATTRIBUTE,
    internal,
    "found an invalid `clippy::version` attribute"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for declared clippy lints without the `clippy::version` attribute.
    ///
    pub MISSING_CLIPPY_VERSION_ATTRIBUTE,
    internal,
    "found clippy lint without `clippy::version` attribute"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for cases of an auto-generated deprecated lint without an updated reason,
    /// i.e. `"default deprecation note"`.
    ///
    /// ### Why is this bad?
    /// Indicates that the documentation is incomplete.
    ///
    /// ### Example
    /// ```rust,ignore
    /// declare_deprecated_lint! {
    ///     /// ### What it does
    ///     /// Nothing. This lint has been deprecated.
    ///     ///
    ///     /// ### Deprecation reason
    ///     /// TODO
    ///     #[clippy::version = "1.63.0"]
    ///     pub COOL_LINT,
    ///     "default deprecation note"
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// declare_deprecated_lint! {
    ///     /// ### What it does
    ///     /// Nothing. This lint has been deprecated.
    ///     ///
    ///     /// ### Deprecation reason
    ///     /// This lint has been replaced by `cooler_lint`
    ///     #[clippy::version = "1.63.0"]
    ///     pub COOL_LINT,
    ///     "this lint has been replaced by `cooler_lint`"
    /// }
    /// ```
    pub DEFAULT_DEPRECATION_REASON,
    internal,
    "found 'default deprecation note' in a deprecated lint declaration"
}

#[derive(Clone, Debug, Default)]
pub struct LintWithoutLintPass {
    declared_lints: FxHashMap<Symbol, Span>,
    registered_lints: FxHashSet<Symbol>,
}

impl_lint_pass!(LintWithoutLintPass => [DEFAULT_LINT, LINT_WITHOUT_LINT_PASS, INVALID_CLIPPY_VERSION_ATTRIBUTE, MISSING_CLIPPY_VERSION_ATTRIBUTE, DEFAULT_DEPRECATION_REASON]);

impl<'tcx> LateLintPass<'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if is_lint_allowed(cx, DEFAULT_LINT, item.hir_id())
            || is_lint_allowed(cx, DEFAULT_DEPRECATION_REASON, item.hir_id())
        {
            return;
        }

        if let hir::ItemKind::Static(ty, Mutability::Not, body_id) = item.kind {
            let is_lint_ref_ty = is_lint_ref_type(cx, ty);
            if is_deprecated_lint(cx, ty) || is_lint_ref_ty {
                check_invalid_clippy_version_attribute(cx, item);

                let expr = &cx.tcx.hir().body(body_id).value;
                let fields;
                if is_lint_ref_ty {
                    if let ExprKind::AddrOf(_, _, inner_exp) = expr.kind
                        && let ExprKind::Struct(_, struct_fields, _) = inner_exp.kind {
                            fields = struct_fields;
                    } else {
                        return;
                    }
                } else if let ExprKind::Struct(_, struct_fields, _) = expr.kind {
                    fields = struct_fields;
                } else {
                    return;
                }

                let field = fields
                    .iter()
                    .find(|f| f.ident.as_str() == "desc")
                    .expect("lints must have a description field");

                if let ExprKind::Lit(Spanned {
                    node: LitKind::Str(ref sym, _),
                    ..
                }) = field.expr.kind
                {
                    let sym_str = sym.as_str();
                    if is_lint_ref_ty {
                        if sym_str == "default lint description" {
                            span_lint(
                                cx,
                                DEFAULT_LINT,
                                item.span,
                                &format!("the lint `{}` has the default lint description", item.ident.name),
                            );
                        }

                        self.declared_lints.insert(item.ident.name, item.span);
                    } else if sym_str == "default deprecation note" {
                        span_lint(
                            cx,
                            DEFAULT_DEPRECATION_REASON,
                            item.span,
                            &format!("the lint `{}` has the default deprecation reason", item.ident.name),
                        );
                    }
                }
            }
        } else if let Some(macro_call) = root_macro_call_first_node(cx, item) {
            if !matches!(
                cx.tcx.item_name(macro_call.def_id).as_str(),
                "impl_lint_pass" | "declare_lint_pass"
            ) {
                return;
            }
            if let hir::ItemKind::Impl(hir::Impl {
                of_trait: None,
                items: impl_item_refs,
                ..
            }) = item.kind
            {
                let mut collector = LintCollector {
                    output: &mut self.registered_lints,
                    cx,
                };
                let body_id = cx.tcx.hir().body_owned_by(
                    cx.tcx.hir().local_def_id(
                        impl_item_refs
                            .iter()
                            .find(|iiref| iiref.ident.as_str() == "get_lints")
                            .expect("LintPass needs to implement get_lints")
                            .id
                            .hir_id(),
                    ),
                );
                collector.visit_expr(cx.tcx.hir().body(body_id).value);
            }
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        if is_lint_allowed(cx, LINT_WITHOUT_LINT_PASS, CRATE_HIR_ID) {
            return;
        }

        for (lint_name, &lint_span) in &self.declared_lints {
            // When using the `declare_tool_lint!` macro, the original `lint_span`'s
            // file points to "<rustc macros>".
            // `compiletest-rs` thinks that's an error in a different file and
            // just ignores it. This causes the test in compile-fail/lint_pass
            // not able to capture the error.
            // Therefore, we need to climb the macro expansion tree and find the
            // actual span that invoked `declare_tool_lint!`:
            let lint_span = lint_span.ctxt().outer_expn_data().call_site;

            if !self.registered_lints.contains(lint_name) {
                span_lint(
                    cx,
                    LINT_WITHOUT_LINT_PASS,
                    lint_span,
                    &format!("the lint `{lint_name}` is not added to any `LintPass`"),
                );
            }
        }
    }
}

pub(super) fn is_lint_ref_type(cx: &LateContext<'_>, ty: &hir::Ty<'_>) -> bool {
    if let TyKind::Rptr(
        _,
        MutTy {
            ty: inner,
            mutbl: Mutability::Not,
        },
    ) = ty.kind
    {
        if let TyKind::Path(ref path) = inner.kind {
            if let Res::Def(DefKind::Struct, def_id) = cx.qpath_res(path, inner.hir_id) {
                return match_def_path(cx, def_id, &paths::LINT);
            }
        }
    }

    false
}

fn check_invalid_clippy_version_attribute(cx: &LateContext<'_>, item: &'_ Item<'_>) {
    if let Some(value) = extract_clippy_version_value(cx, item) {
        // The `sym!` macro doesn't work as it only expects a single token.
        // It's better to keep it this way and have a direct `Symbol::intern` call here.
        if value == Symbol::intern("pre 1.29.0") {
            return;
        }

        if RustcVersion::parse(value.as_str()).is_err() {
            span_lint_and_help(
                cx,
                INVALID_CLIPPY_VERSION_ATTRIBUTE,
                item.span,
                "this item has an invalid `clippy::version` attribute",
                None,
                "please use a valid semantic version, see `doc/adding_lints.md`",
            );
        }
    } else {
        span_lint_and_help(
            cx,
            MISSING_CLIPPY_VERSION_ATTRIBUTE,
            item.span,
            "this lint is missing the `clippy::version` attribute or version value",
            None,
            "please use a `clippy::version` attribute, see `doc/adding_lints.md`",
        );
    }
}

/// This function extracts the version value of a `clippy::version` attribute if the given value has
/// one
pub(super) fn extract_clippy_version_value(cx: &LateContext<'_>, item: &'_ Item<'_>) -> Option<Symbol> {
    let attrs = cx.tcx.hir().attrs(item.hir_id());
    attrs.iter().find_map(|attr| {
        if_chain! {
            // Identify attribute
            if let ast::AttrKind::Normal(ref attr_kind) = &attr.kind;
            if let [tool_name, attr_name] = &attr_kind.item.path.segments[..];
            if tool_name.ident.name == sym::clippy;
            if attr_name.ident.name == sym::version;
            if let Some(version) = attr.value_str();
            then { Some(version) } else { None }
        }
    })
}

struct LintCollector<'a, 'tcx> {
    output: &'a mut FxHashSet<Symbol>,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LintCollector<'a, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn visit_path(&mut self, path: &'tcx Path<'_>, _: HirId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].ident.name);
        }
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}
