use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::indent_of;
use clippy_utils::{is_default_equivalent, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{
    def::{CtorKind, CtorOf, DefKind, Res},
    Body, Expr, ExprKind, GenericArg, Impl, ImplItemKind, Item, ItemKind, Node, PathSegment, QPath, Ty, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{AdtDef, DefIdTree};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects manual `std::default::Default` implementations that are identical to a derived implementation.
    ///
    /// ### Why is this bad?
    /// It is less concise.
    ///
    /// ### Example
    /// ```rust
    /// struct Foo {
    ///     bar: bool
    /// }
    ///
    /// impl Default for Foo {
    ///     fn default() -> Self {
    ///         Self {
    ///             bar: false
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// #[derive(Default)]
    /// struct Foo {
    ///     bar: bool
    /// }
    /// ```
    ///
    /// ### Known problems
    /// Derive macros [sometimes use incorrect bounds](https://github.com/rust-lang/rust/issues/26925)
    /// in generic types and the user defined `impl` may be more generalized or
    /// specialized than what derive will produce. This lint can't detect the manual `impl`
    /// has exactly equal bounds, and therefore this lint is disabled for types with
    /// generic parameters.
    #[clippy::version = "1.57.0"]
    pub DERIVABLE_IMPLS,
    complexity,
    "manual implementation of the `Default` trait which is equal to a derive"
}

pub struct DerivableImpls {
    msrv: Msrv,
}

impl DerivableImpls {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        DerivableImpls { msrv }
    }
}

impl_lint_pass!(DerivableImpls => [DERIVABLE_IMPLS]);

fn is_path_self(e: &Expr<'_>) -> bool {
    if let ExprKind::Path(QPath::Resolved(_, p)) = e.kind {
        matches!(p.res, Res::SelfCtor(..) | Res::Def(DefKind::Ctor(..), _))
    } else {
        false
    }
}

fn check_struct<'tcx>(
    cx: &LateContext<'tcx>,
    item: &'tcx Item<'_>,
    self_ty: &Ty<'_>,
    func_expr: &Expr<'_>,
    adt_def: AdtDef<'_>,
) {
    if let TyKind::Path(QPath::Resolved(_, p)) = self_ty.kind {
        if let Some(PathSegment { args: Some(a), .. }) = p.segments.last() {
            for arg in a.args {
                if !matches!(arg, GenericArg::Lifetime(_)) {
                    return;
                }
            }
        }
    }
    let should_emit = match peel_blocks(func_expr).kind {
        ExprKind::Tup(fields) => fields.iter().all(|e| is_default_equivalent(cx, e)),
        ExprKind::Call(callee, args) if is_path_self(callee) => args.iter().all(|e| is_default_equivalent(cx, e)),
        ExprKind::Struct(_, fields, _) => fields.iter().all(|ef| is_default_equivalent(cx, ef.expr)),
        _ => false,
    };

    if should_emit {
        let struct_span = cx.tcx.def_span(adt_def.did());
        span_lint_and_then(cx, DERIVABLE_IMPLS, item.span, "this `impl` can be derived", |diag| {
            diag.span_suggestion_hidden(
                item.span,
                "remove the manual implementation...",
                String::new(),
                Applicability::MachineApplicable,
            );
            diag.span_suggestion(
                struct_span.shrink_to_lo(),
                "...and instead derive it",
                "#[derive(Default)]\n".to_string(),
                Applicability::MachineApplicable,
            );
        });
    }
}

fn check_enum<'tcx>(cx: &LateContext<'tcx>, item: &'tcx Item<'_>, func_expr: &Expr<'_>, adt_def: AdtDef<'_>) {
    if_chain! {
        if let ExprKind::Path(QPath::Resolved(None, p)) = &peel_blocks(func_expr).kind;
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Const), id) = p.res;
        if let variant_id = cx.tcx.parent(id);
        if let Some(variant_def) = adt_def.variants().iter().find(|v| v.def_id == variant_id);
        if variant_def.fields.is_empty();
        if !variant_def.is_field_list_non_exhaustive();

        then {
            let enum_span = cx.tcx.def_span(adt_def.did());
            let indent_enum = indent_of(cx, enum_span).unwrap_or(0);
            let variant_span = cx.tcx.def_span(variant_def.def_id);
            let indent_variant = indent_of(cx, variant_span).unwrap_or(0);
            span_lint_and_then(
                cx,
                DERIVABLE_IMPLS,
                item.span,
                "this `impl` can be derived",
                |diag| {
                    diag.span_suggestion_hidden(
                        item.span,
                        "remove the manual implementation...",
                        String::new(),
                        Applicability::MachineApplicable
                    );
                    diag.span_suggestion(
                        enum_span.shrink_to_lo(),
                        "...and instead derive it...",
                        format!(
                            "#[derive(Default)]\n{indent}",
                            indent = " ".repeat(indent_enum),
                        ),
                        Applicability::MachineApplicable
                    );
                    diag.span_suggestion(
                        variant_span.shrink_to_lo(),
                        "...and mark the default variant",
                        format!(
                            "#[default]\n{indent}",
                            indent = " ".repeat(indent_variant),
                        ),
                        Applicability::MachineApplicable
                    );
                }
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for DerivableImpls {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if_chain! {
            if let ItemKind::Impl(Impl {
                of_trait: Some(ref trait_ref),
                items: [child],
                self_ty,
                ..
            }) = item.kind;
            if !cx.tcx.has_attr(item.owner_id.to_def_id(), sym::automatically_derived);
            if !item.span.from_expansion();
            if let Some(def_id) = trait_ref.trait_def_id();
            if cx.tcx.is_diagnostic_item(sym::Default, def_id);
            if let impl_item_hir = child.id.hir_id();
            if let Some(Node::ImplItem(impl_item)) = cx.tcx.hir().find(impl_item_hir);
            if let ImplItemKind::Fn(_, b) = &impl_item.kind;
            if let Body { value: func_expr, .. } = cx.tcx.hir().body(*b);
            if let Some(adt_def) = cx.tcx.bound_type_of(item.owner_id).subst_identity().ty_adt_def();
            if let attrs = cx.tcx.hir().attrs(item.hir_id());
            if !attrs.iter().any(|attr| attr.doc_str().is_some());
            if let child_attrs = cx.tcx.hir().attrs(impl_item_hir);
            if !child_attrs.iter().any(|attr| attr.doc_str().is_some());

            then {
                if adt_def.is_struct() {
                    check_struct(cx, item, self_ty, func_expr, adt_def);
                } else if adt_def.is_enum() && self.msrv.meets(msrvs::DEFAULT_ENUM_ATTRIBUTE) {
                    check_enum(cx, item, func_expr, adt_def);
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}
