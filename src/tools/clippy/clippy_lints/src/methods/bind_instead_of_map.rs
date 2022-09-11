use super::{contains_return, BIND_INSTEAD_OF_MAP};
use clippy_utils::diagnostics::{multispan_sugg_with_applicability, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_with_macro_callsite};
use clippy_utils::{peel_blocks, visitors::find_all_ret_expressions};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{LangItem, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::DefIdTree;
use rustc_span::Span;

pub(crate) struct OptionAndThenSome;

impl BindInsteadOfMap for OptionAndThenSome {
    const VARIANT_LANG_ITEM: LangItem = LangItem::OptionSome;
    const BAD_METHOD_NAME: &'static str = "and_then";
    const GOOD_METHOD_NAME: &'static str = "map";
}

pub(crate) struct ResultAndThenOk;

impl BindInsteadOfMap for ResultAndThenOk {
    const VARIANT_LANG_ITEM: LangItem = LangItem::ResultOk;
    const BAD_METHOD_NAME: &'static str = "and_then";
    const GOOD_METHOD_NAME: &'static str = "map";
}

pub(crate) struct ResultOrElseErrInfo;

impl BindInsteadOfMap for ResultOrElseErrInfo {
    const VARIANT_LANG_ITEM: LangItem = LangItem::ResultErr;
    const BAD_METHOD_NAME: &'static str = "or_else";
    const GOOD_METHOD_NAME: &'static str = "map_err";
}

pub(crate) trait BindInsteadOfMap {
    const VARIANT_LANG_ITEM: LangItem;
    const BAD_METHOD_NAME: &'static str;
    const GOOD_METHOD_NAME: &'static str;

    fn no_op_msg(cx: &LateContext<'_>) -> Option<String> {
        let variant_id = cx.tcx.lang_items().require(Self::VARIANT_LANG_ITEM).ok()?;
        let item_id = cx.tcx.parent(variant_id);
        Some(format!(
            "using `{}.{}({})`, which is a no-op",
            cx.tcx.item_name(item_id),
            Self::BAD_METHOD_NAME,
            cx.tcx.item_name(variant_id),
        ))
    }

    fn lint_msg(cx: &LateContext<'_>) -> Option<String> {
        let variant_id = cx.tcx.lang_items().require(Self::VARIANT_LANG_ITEM).ok()?;
        let item_id = cx.tcx.parent(variant_id);
        Some(format!(
            "using `{}.{}(|x| {}(y))`, which is more succinctly expressed as `{}(|x| y)`",
            cx.tcx.item_name(item_id),
            Self::BAD_METHOD_NAME,
            cx.tcx.item_name(variant_id),
            Self::GOOD_METHOD_NAME
        ))
    }

    fn lint_closure_autofixable(
        cx: &LateContext<'_>,
        expr: &hir::Expr<'_>,
        recv: &hir::Expr<'_>,
        closure_expr: &hir::Expr<'_>,
        closure_args_span: Span,
    ) -> bool {
        if_chain! {
            if let hir::ExprKind::Call(some_expr, [inner_expr]) = closure_expr.kind;
            if let hir::ExprKind::Path(QPath::Resolved(_, path)) = some_expr.kind;
            if Self::is_variant(cx, path.res);
            if !contains_return(inner_expr);
            if let Some(msg) = Self::lint_msg(cx);
            then {
                let some_inner_snip = if inner_expr.span.from_expansion() {
                    snippet_with_macro_callsite(cx, inner_expr.span, "_")
                } else {
                    snippet(cx, inner_expr.span, "_")
                };

                let closure_args_snip = snippet(cx, closure_args_span, "..");
                let option_snip = snippet(cx, recv.span, "..");
                let note = format!("{}.{}({} {})", option_snip, Self::GOOD_METHOD_NAME, closure_args_snip, some_inner_snip);
                span_lint_and_sugg(
                    cx,
                    BIND_INSTEAD_OF_MAP,
                    expr.span,
                    &msg,
                    "try this",
                    note,
                    Applicability::MachineApplicable,
                );
                true
            } else {
                false
            }
        }
    }

    fn lint_closure(cx: &LateContext<'_>, expr: &hir::Expr<'_>, closure_expr: &hir::Expr<'_>) -> bool {
        let mut suggs = Vec::new();
        let can_sugg: bool = find_all_ret_expressions(cx, closure_expr, |ret_expr| {
            if_chain! {
                if !ret_expr.span.from_expansion();
                if let hir::ExprKind::Call(func_path, [arg]) = ret_expr.kind;
                if let hir::ExprKind::Path(QPath::Resolved(_, path)) = func_path.kind;
                if Self::is_variant(cx, path.res);
                if !contains_return(arg);
                then {
                    suggs.push((ret_expr.span, arg.span.source_callsite()));
                    true
                } else {
                    false
                }
            }
        });
        let (span, msg) = if_chain! {
            if can_sugg;
            if let hir::ExprKind::MethodCall(segment, ..) = expr.kind;
            if let Some(msg) = Self::lint_msg(cx);
            then { (segment.ident.span, msg) } else { return false; }
        };
        span_lint_and_then(cx, BIND_INSTEAD_OF_MAP, expr.span, &msg, |diag| {
            multispan_sugg_with_applicability(
                diag,
                "try this",
                Applicability::MachineApplicable,
                std::iter::once((span, Self::GOOD_METHOD_NAME.into())).chain(
                    suggs
                        .into_iter()
                        .map(|(span1, span2)| (span1, snippet(cx, span2, "_").into())),
                ),
            );
        });
        true
    }

    /// Lint use of `_.and_then(|x| Some(y))` for `Option`s
    fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, arg: &hir::Expr<'_>) -> bool {
        if_chain! {
            if let Some(adt) = cx.typeck_results().expr_ty(recv).ty_adt_def();
            if let Ok(vid) = cx.tcx.lang_items().require(Self::VARIANT_LANG_ITEM);
            if adt.did() == cx.tcx.parent(vid);
            then {} else { return false; }
        }

        match arg.kind {
            hir::ExprKind::Closure(&hir::Closure { body, fn_decl_span, .. }) => {
                let closure_body = cx.tcx.hir().body(body);
                let closure_expr = peel_blocks(closure_body.value);

                if Self::lint_closure_autofixable(cx, expr, recv, closure_expr, fn_decl_span) {
                    true
                } else {
                    Self::lint_closure(cx, expr, closure_expr)
                }
            },
            // `_.and_then(Some)` case, which is no-op.
            hir::ExprKind::Path(QPath::Resolved(_, path)) if Self::is_variant(cx, path.res) => {
                if let Some(msg) = Self::no_op_msg(cx) {
                    span_lint_and_sugg(
                        cx,
                        BIND_INSTEAD_OF_MAP,
                        expr.span,
                        &msg,
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

    fn is_variant(cx: &LateContext<'_>, res: Res) -> bool {
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fn), id) = res {
            if let Ok(variant_id) = cx.tcx.lang_items().require(Self::VARIANT_LANG_ITEM) {
                return cx.tcx.parent(id) == variant_id;
            }
        }
        false
    }
}
