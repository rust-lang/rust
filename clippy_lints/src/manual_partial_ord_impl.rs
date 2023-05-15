use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{
    def_path_def_ids, diagnostics::span_lint_and_sugg, get_trait_def_id, match_def_path, path_res, ty::implements_trait,
};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::*;
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::def_id::DefId;
use std::cell::OnceCell;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual implementations of both `PartialOrd` and `Ord` when only `Ord` is
    /// necessary.
    ///
    /// ### Why is this bad?
    /// If both `PartialOrd` and `Ord` are implemented, `PartialOrd` will wrap the returned value of
    /// `Ord::cmp` in `Some`. Not doing this may silently introduce an error.
    ///
    /// ### Example
    /// ```rust
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Ord for A {
    ///     fn cmp(&self, other: &Self) -> Ordering {
    ///         todo!();
    ///     }
    /// }
    /// 
    /// impl PartialOrd for A {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    ///         todo!();
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// #[derive(Eq, PartialEq)]
    /// struct A(u32);
    ///
    /// impl Ord for A {
    ///     fn cmp(&self, other: &Self) -> Ordering {
    ///         todo!();
    ///     }
    /// }
    /// 
    /// impl PartialOrd for A {
    ///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    ///         Some(self.cmp(other))
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.71.0"]
    pub MANUAL_PARTIAL_ORD_IMPL,
    nursery,
    "default lint description"
}
impl_lint_pass!(ManualPartialOrdImpl => [MANUAL_PARTIAL_ORD_IMPL]);

#[derive(Clone)]
pub struct ManualPartialOrdImpl {
    pub ord_def_id: OnceCell<DefId>,
}

impl LateLintPass<'_> for ManualPartialOrdImpl {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if_chain! {
            if let ItemKind::Impl(imp) = &item.kind;
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(item.owner_id);
            if cx.tcx.is_diagnostic_item(sym!(PartialOrd), impl_trait_ref.skip_binder().def_id);
            then {
                lint_impl_body(self, cx, imp, item);
            }
        }
    }
}

fn lint_impl_body(conf: &mut ManualPartialOrdImpl, cx: &LateContext<'_>, imp: &Impl<'_>, item: &Item<'_>) {
    for imp_item in imp.items {
        if_chain! {
            if imp_item.ident.name == sym!(partial_cmp);
            if let ImplItemKind::Fn(_, id) = cx.tcx.hir().impl_item(imp_item.id).kind;
            then {
                let body = cx.tcx.hir().body(id);
                let ord_def_id = conf.ord_def_id.get_or_init(|| get_trait_def_id(cx, &["core", "cmp", "Ord"]).unwrap());
                if let ExprKind::Block(block, ..)
                    = body.value.kind && implements_trait(cx, hir_ty_to_ty(cx.tcx, imp.self_ty), *ord_def_id, &[])
                {
                    if_chain! {
                        if block.stmts.is_empty();
                        if let Some(expr) = block.expr;
                        if let ExprKind::Call(Expr { kind: ExprKind::Path(path), ..}, [cmp_expr]) = expr.kind;
                        if let QPath::Resolved(_, some_path) = path;
                        if let Some(some_seg_one) = some_path.segments.get(0);
                        if some_seg_one.ident.name == sym!(Some);
                        if let ExprKind::MethodCall(cmp_path, _, [other_expr], ..) = cmp_expr.kind;
                        if cmp_path.ident.name == sym!(cmp);
                        if let Res::Local(..) = path_res(cx, other_expr);
                        then {}
                        else {
                            span_lint_and_then(
                                cx,
                                MANUAL_PARTIAL_ORD_IMPL,
                                item.span,
                                "manual implementation of `PartialOrd` when `Ord` is already implemented",
                                |diag| {
                                    if let Some(param) = body.params.get(1)
                                        && let PatKind::Binding(_, _, param_ident, ..) = param.pat.kind
                                    {
                                        diag.span_suggestion(
                                            block.span,
                                            "change this to",
                                            format!("{{ Some(self.cmp({})) }}",
                                            param_ident.name),
                                            Applicability::MaybeIncorrect
                                        );
                                    } else {
                                        diag.help("return the value of `self.cmp` wrapped in `Some` instead");
                                    };
                                }
                            );
                        }
                    }
                }
            }
        }
    }
}
