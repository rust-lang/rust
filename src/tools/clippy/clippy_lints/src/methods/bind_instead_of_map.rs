use super::{BIND_INSTEAD_OF_MAP, contains_return};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::peel_blocks;
use clippy_utils::source::{snippet, snippet_with_context};
use clippy_utils::visitors::find_all_ret_expressions;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{LangItem, QPath};
use rustc_lint::LateContext;
use rustc_span::Span;

pub(super) fn check_and_then_some(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    arg: &hir::Expr<'_>,
) -> bool {
    BindInsteadOfMap {
        variant_lang_item: LangItem::OptionSome,
        bad_method_name: "and_then",
        good_method_name: "map",
    }
    .check(cx, expr, recv, arg)
}

pub(super) fn check_and_then_ok(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    arg: &hir::Expr<'_>,
) -> bool {
    BindInsteadOfMap {
        variant_lang_item: LangItem::ResultOk,
        bad_method_name: "and_then",
        good_method_name: "map",
    }
    .check(cx, expr, recv, arg)
}

pub(super) fn check_or_else_err(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    recv: &hir::Expr<'_>,
    arg: &hir::Expr<'_>,
) -> bool {
    BindInsteadOfMap {
        variant_lang_item: LangItem::ResultErr,
        bad_method_name: "or_else",
        good_method_name: "map_err",
    }
    .check(cx, expr, recv, arg)
}

struct BindInsteadOfMap {
    variant_lang_item: LangItem,
    bad_method_name: &'static str,
    good_method_name: &'static str,
}

impl BindInsteadOfMap {
    fn no_op_msg(&self, cx: &LateContext<'_>) -> Option<String> {
        let variant_id = cx.tcx.lang_items().get(self.variant_lang_item)?;
        let item_id = cx.tcx.parent(variant_id);
        Some(format!(
            "using `{}.{}({})`, which is a no-op",
            cx.tcx.item_name(item_id),
            self.bad_method_name,
            cx.tcx.item_name(variant_id),
        ))
    }

    fn lint_msg(&self, cx: &LateContext<'_>) -> Option<String> {
        let variant_id = cx.tcx.lang_items().get(self.variant_lang_item)?;
        let item_id = cx.tcx.parent(variant_id);
        Some(format!(
            "using `{}.{}(|x| {}(y))`, which is more succinctly expressed as `{}(|x| y)`",
            cx.tcx.item_name(item_id),
            self.bad_method_name,
            cx.tcx.item_name(variant_id),
            self.good_method_name,
        ))
    }

    fn lint_closure_autofixable(
        &self,
        cx: &LateContext<'_>,
        expr: &hir::Expr<'_>,
        recv: &hir::Expr<'_>,
        closure_expr: &hir::Expr<'_>,
        closure_args_span: Span,
    ) -> bool {
        if let hir::ExprKind::Call(some_expr, [inner_expr]) = closure_expr.kind
            && let hir::ExprKind::Path(QPath::Resolved(_, path)) = some_expr.kind
            && self.is_variant(cx, path.res)
            && !contains_return(inner_expr)
            && let Some(msg) = self.lint_msg(cx)
        {
            let mut app = Applicability::MachineApplicable;
            let some_inner_snip = snippet_with_context(cx, inner_expr.span, closure_expr.span.ctxt(), "_", &mut app).0;

            let closure_args_snip = snippet(cx, closure_args_span, "..");
            let option_snip = snippet(cx, recv.span, "..");
            let note = format!(
                "{option_snip}.{}({closure_args_snip} {some_inner_snip})",
                self.good_method_name
            );
            span_lint_and_sugg(cx, BIND_INSTEAD_OF_MAP, expr.span, msg, "try", note, app);
            true
        } else {
            false
        }
    }

    fn lint_closure(&self, cx: &LateContext<'_>, expr: &hir::Expr<'_>, closure_expr: &hir::Expr<'_>) -> bool {
        let mut suggs = Vec::new();
        let can_sugg: bool = find_all_ret_expressions(cx, closure_expr, |ret_expr| {
            if !ret_expr.span.from_expansion()
                && let hir::ExprKind::Call(func_path, [arg]) = ret_expr.kind
                && let hir::ExprKind::Path(QPath::Resolved(_, path)) = func_path.kind
                && self.is_variant(cx, path.res)
                && !contains_return(arg)
            {
                suggs.push((ret_expr.span, arg.span.source_callsite()));
                true
            } else {
                false
            }
        });
        let (span, msg) = if can_sugg
            && let hir::ExprKind::MethodCall(segment, ..) = expr.kind
            && let Some(msg) = self.lint_msg(cx)
        {
            (segment.ident.span, msg)
        } else {
            return false;
        };
        span_lint_and_then(cx, BIND_INSTEAD_OF_MAP, expr.span, msg, |diag| {
            diag.multipart_suggestion(
                format!("use `{}` instead", self.good_method_name),
                std::iter::once((span, self.good_method_name.into()))
                    .chain(
                        suggs
                            .into_iter()
                            .map(|(span1, span2)| (span1, snippet(cx, span2, "_").into())),
                    )
                    .collect(),
                Applicability::MachineApplicable,
            );
        });
        true
    }

    /// Lint use of `_.and_then(|x| Some(y))` for `Option`s
    fn check(&self, cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, arg: &hir::Expr<'_>) -> bool {
        if let Some(adt) = cx.typeck_results().expr_ty(recv).ty_adt_def()
            && let Some(vid) = cx.tcx.lang_items().get(self.variant_lang_item)
            && adt.did() == cx.tcx.parent(vid)
        {
        } else {
            return false;
        }

        match arg.kind {
            hir::ExprKind::Closure(&hir::Closure { body, fn_decl_span, .. }) => {
                let closure_body = cx.tcx.hir_body(body);
                let closure_expr = peel_blocks(closure_body.value);

                if self.lint_closure_autofixable(cx, expr, recv, closure_expr, fn_decl_span) {
                    true
                } else {
                    self.lint_closure(cx, expr, closure_expr)
                }
            },
            // `_.and_then(Some)` case, which is no-op.
            hir::ExprKind::Path(QPath::Resolved(_, path)) if self.is_variant(cx, path.res) => {
                if let Some(msg) = self.no_op_msg(cx) {
                    span_lint_and_sugg(
                        cx,
                        BIND_INSTEAD_OF_MAP,
                        expr.span,
                        msg,
                        "use the expression directly",
                        snippet(cx, recv.span, "..").into(),
                        Applicability::MachineApplicable,
                    );
                }
                true
            },
            _ => false,
        }
    }

    fn is_variant(&self, cx: &LateContext<'_>, res: Res) -> bool {
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fn), id) = res
            && let Some(variant_id) = cx.tcx.lang_items().get(self.variant_lang_item)
        {
            return cx.tcx.parent(id) == variant_id;
        }
        false
    }
}
