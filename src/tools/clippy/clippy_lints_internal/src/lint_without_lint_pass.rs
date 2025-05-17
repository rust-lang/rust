use crate::internal_paths;
use clippy_utils::diagnostics::{span_lint, span_lint_and_help};
use clippy_utils::is_lint_allowed;
use clippy_utils::macros::root_macro_call_first_node;
use rustc_ast::ast::LitKind;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::hir_id::CRATE_HIR_ID;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{ExprKind, HirId, Item, MutTy, Mutability, Path, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_middle::hir::nested_filter;
use rustc_session::impl_lint_pass;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Symbol;
use rustc_span::{Span, sym};

declare_tool_lint! {
    /// ### What it does
    /// Ensures every lint is associated to a `LintPass`.
    ///
    /// ### Why is this bad?
    /// The compiler only knows lints via a `LintPass`. Without
    /// putting a lint to a `LintPass::lint_vec()`'s return, the compiler will not
    /// know the name of the lint.
    ///
    /// ### Known problems
    /// Only checks for lints associated using the `declare_lint_pass!` and
    /// `impl_lint_pass!` macros.
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
    pub clippy::LINT_WITHOUT_LINT_PASS,
    Warn,
    "declaring a lint without associating it in a LintPass",
    report_in_external_macro: true

}

declare_tool_lint! {
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
    pub clippy::DEFAULT_LINT,
    Warn,
    "found 'default lint description' in a lint declaration",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// ### What it does
    /// Checks for invalid `clippy::version` attributes.
    ///
    /// Valid values are:
    /// * "pre 1.29.0"
    /// * any valid semantic version
    pub clippy::INVALID_CLIPPY_VERSION_ATTRIBUTE,
    Warn,
    "found an invalid `clippy::version` attribute",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// ### What it does
    /// Checks for declared clippy lints without the `clippy::version` attribute.
    pub clippy::MISSING_CLIPPY_VERSION_ATTRIBUTE,
    Warn,
    "found clippy lint without `clippy::version` attribute",
    report_in_external_macro: true
}

#[derive(Clone, Debug, Default)]
pub struct LintWithoutLintPass {
    declared_lints: FxIndexMap<Symbol, Span>,
    registered_lints: FxIndexSet<Symbol>,
}

impl_lint_pass!(LintWithoutLintPass => [
    DEFAULT_LINT,
    LINT_WITHOUT_LINT_PASS,
    INVALID_CLIPPY_VERSION_ATTRIBUTE,
    MISSING_CLIPPY_VERSION_ATTRIBUTE,
]);

impl<'tcx> LateLintPass<'tcx> for LintWithoutLintPass {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let hir::ItemKind::Static(ident, ty, Mutability::Not, body_id) = item.kind {
            if is_lint_ref_type(cx, ty) {
                check_invalid_clippy_version_attribute(cx, item);

                let expr = &cx.tcx.hir_body(body_id).value;
                let fields = if let ExprKind::AddrOf(_, _, inner_exp) = expr.kind
                    && let ExprKind::Struct(_, struct_fields, _) = inner_exp.kind
                {
                    struct_fields
                } else {
                    return;
                };

                let field = fields
                    .iter()
                    .find(|f| f.ident.as_str() == "desc")
                    .expect("lints must have a description field");

                if let ExprKind::Lit(Spanned {
                    node: LitKind::Str(sym, _),
                    ..
                }) = field.expr.kind
                {
                    let sym_str = sym.as_str();
                    if sym_str == "default lint description" {
                        span_lint(
                            cx,
                            DEFAULT_LINT,
                            item.span,
                            format!("the lint `{}` has the default lint description", ident.name),
                        );
                    }
                    self.declared_lints.insert(ident.name, item.span);
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
                let body = cx.tcx.hir_body_owned_by(
                    impl_item_refs
                        .iter()
                        .find(|iiref| iiref.ident.as_str() == "lint_vec")
                        .expect("LintPass needs to implement lint_vec")
                        .id
                        .owner_id
                        .def_id,
                );
                collector.visit_expr(body.value);
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
                    format!("the lint `{lint_name}` is not added to any `LintPass`"),
                );
            }
        }
    }
}

pub(super) fn is_lint_ref_type(cx: &LateContext<'_>, ty: &hir::Ty<'_>) -> bool {
    if let TyKind::Ref(
        _,
        MutTy {
            ty: inner,
            mutbl: Mutability::Not,
        },
    ) = ty.kind
        && let TyKind::Path(ref path) = inner.kind
        && let Res::Def(DefKind::Struct, def_id) = cx.qpath_res(path, inner.hir_id)
    {
        internal_paths::LINT.matches(cx, def_id)
    } else {
        false
    }
}

fn check_invalid_clippy_version_attribute(cx: &LateContext<'_>, item: &'_ Item<'_>) {
    if let Some(value) = extract_clippy_version_value(cx, item) {
        if value.as_str() == "pre 1.29.0" {
            return;
        }

        if rustc_attr_parsing::parse_version(value).is_none() {
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
    let attrs = cx.tcx.hir_attrs(item.hir_id());
    attrs.iter().find_map(|attr| {
        if let hir::Attribute::Unparsed(attr_kind) = &attr
            // Identify attribute
            && let [tool_name, attr_name] = &attr_kind.path.segments[..]
            && tool_name.name == sym::clippy
            && attr_name.name == sym::version
            && let Some(version) = attr.value_str()
        {
            Some(version)
        } else {
            None
        }
    })
}

struct LintCollector<'a, 'tcx> {
    output: &'a mut FxIndexSet<Symbol>,
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> Visitor<'tcx> for LintCollector<'_, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn visit_path(&mut self, path: &Path<'_>, _: HirId) {
        if path.segments.len() == 1 {
            self.output.insert(path.segments[0].ident.name);
        }
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}
