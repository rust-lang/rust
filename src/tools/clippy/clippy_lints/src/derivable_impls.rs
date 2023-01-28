use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_default_equivalent, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{
    def::{DefKind, Res},
    Body, Expr, ExprKind, GenericArg, Impl, ImplItemKind, Item, ItemKind, Node, PathSegment, QPath, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
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

declare_lint_pass!(DerivableImpls => [DERIVABLE_IMPLS]);

fn is_path_self(e: &Expr<'_>) -> bool {
    if let ExprKind::Path(QPath::Resolved(_, p)) = e.kind {
        matches!(p.res, Res::SelfCtor(..) | Res::Def(DefKind::Ctor(..), _))
    } else {
        false
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
            if let Some(adt_def) = cx.tcx.type_of(item.owner_id).ty_adt_def();
            if let attrs = cx.tcx.hir().attrs(item.hir_id());
            if !attrs.iter().any(|attr| attr.doc_str().is_some());
            if let child_attrs = cx.tcx.hir().attrs(impl_item_hir);
            if !child_attrs.iter().any(|attr| attr.doc_str().is_some());
            if adt_def.is_struct();
            then {
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
                    ExprKind::Call(callee, args)
                        if is_path_self(callee) => args.iter().all(|e| is_default_equivalent(cx, e)),
                    ExprKind::Struct(_, fields, _) => fields.iter().all(|ef| is_default_equivalent(cx, ef.expr)),
                    _ => false,
                };

                if should_emit {
                    let struct_span = cx.tcx.def_span(adt_def.did());
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
                                struct_span.shrink_to_lo(),
                                "...and instead derive it",
                                "#[derive(Default)]\n".to_string(),
                                Applicability::MachineApplicable
                            );
                        }
                    );
                }
            }
        }
    }
}
