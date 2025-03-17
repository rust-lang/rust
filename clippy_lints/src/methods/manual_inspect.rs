use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{IntoSpan, SpanRangeExt};
use clippy_utils::ty::get_field_by_name;
use clippy_utils::visitors::{for_each_expr, for_each_expr_without_closures};
use clippy_utils::{ExprUseNode, expr_use_ctxt, is_diag_item_method, is_diag_trait_item, path_to_local_id, sym};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{BindingMode, BorrowKind, ByRef, ClosureKind, Expr, ExprKind, Mutability, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::{Adjust, Adjustment, AutoBorrow, AutoBorrowMutability};
use rustc_span::{DUMMY_SP, Span, Symbol};

use super::MANUAL_INSPECT;

#[expect(clippy::too_many_lines)]
pub(crate) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, arg: &Expr<'_>, name: Symbol, name_span: Span, msrv: Msrv) {
    if let ExprKind::Closure(c) = arg.kind
        && matches!(c.kind, ClosureKind::Closure)
        && let typeck = cx.typeck_results()
        && let Some(fn_id) = typeck.type_dependent_def_id(expr.hir_id)
        && (is_diag_trait_item(cx, fn_id, sym::Iterator)
            || ((is_diag_item_method(cx, fn_id, sym::Option) || is_diag_item_method(cx, fn_id, sym::Result))
                && msrv.meets(cx, msrvs::OPTION_RESULT_INSPECT)))
        && let body = cx.tcx.hir_body(c.body)
        && let [param] = body.params
        && let PatKind::Binding(BindingMode(ByRef::No, Mutability::Not), arg_id, _, None) = param.pat.kind
        && let arg_ty = typeck.node_type(arg_id)
        && let ExprKind::Block(block, _) = body.value.kind
        && let Some(final_expr) = block.expr
        && !block.stmts.is_empty()
        && path_to_local_id(final_expr, arg_id)
        && typeck.expr_adjustments(final_expr).is_empty()
    {
        let mut requires_copy = false;
        let mut requires_deref = false;

        // The number of unprocessed return expressions.
        let mut ret_count = 0u32;

        // The uses for which processing is delayed until after the visitor.
        let mut delayed = vec![];

        let ctxt = arg.span.ctxt();
        let can_lint = for_each_expr_without_closures(block.stmts, |e| {
            if let ExprKind::Closure(c) = e.kind {
                // Nested closures don't need to treat returns specially.
                let _: Option<!> = for_each_expr(cx, cx.tcx.hir_body(c.body).value, |e| {
                    if path_to_local_id(e, arg_id) {
                        let (kind, same_ctxt) = check_use(cx, e);
                        match (kind, same_ctxt && e.span.ctxt() == ctxt) {
                            (_, false) | (UseKind::Deref | UseKind::Return(..), true) => {
                                requires_copy = true;
                                requires_deref = true;
                            },
                            (UseKind::AutoBorrowed, true) => {},
                            (UseKind::WillAutoDeref, true) => {
                                requires_copy = true;
                            },
                            (kind, true) => delayed.push(kind),
                        }
                    }
                    ControlFlow::Continue(())
                });
            } else if matches!(e.kind, ExprKind::Ret(_)) {
                ret_count += 1;
            } else if path_to_local_id(e, arg_id) {
                let (kind, same_ctxt) = check_use(cx, e);
                match (kind, same_ctxt && e.span.ctxt() == ctxt) {
                    (UseKind::Return(..), false) => {
                        return ControlFlow::Break(());
                    },
                    (_, false) | (UseKind::Deref, true) => {
                        requires_copy = true;
                        requires_deref = true;
                    },
                    (UseKind::AutoBorrowed, true) => {},
                    (UseKind::WillAutoDeref, true) => {
                        requires_copy = true;
                    },
                    (kind @ UseKind::Return(_), true) => {
                        ret_count -= 1;
                        delayed.push(kind);
                    },
                    (kind, true) => delayed.push(kind),
                }
            }
            ControlFlow::Continue(())
        })
        .is_none();

        if ret_count != 0 {
            // A return expression that didn't return the original value was found.
            return;
        }

        let mut edits = Vec::with_capacity(delayed.len() + 3);
        let mut addr_of_edits = Vec::with_capacity(delayed.len());
        for x in delayed {
            match x {
                UseKind::Return(s) => edits.push((s.with_leading_whitespace(cx).with_ctxt(s.ctxt()), String::new())),
                UseKind::Borrowed(s) => {
                    let range = s.map_range(cx, |_, src, range| {
                        let src = src.get(range.clone())?;
                        let trimmed = src.trim_start_matches([' ', '\t', '\n', '\r', '(']);
                        trimmed.starts_with('&').then(|| {
                            let pos = range.start + src.len() - trimmed.len();
                            pos..pos + 1
                        })
                    });
                    if let Some(range) = range {
                        addr_of_edits.push((range.with_ctxt(s.ctxt()), String::new()));
                    } else {
                        requires_copy = true;
                        requires_deref = true;
                    }
                },
                UseKind::FieldAccess(name, e) => {
                    let Some(mut ty) = get_field_by_name(cx.tcx, arg_ty.peel_refs(), name) else {
                        requires_copy = true;
                        continue;
                    };
                    let mut prev_expr = e;

                    for (_, parent) in cx.tcx.hir_parent_iter(e.hir_id) {
                        if let Node::Expr(e) = parent {
                            match e.kind {
                                ExprKind::Field(_, name)
                                    if let Some(fty) = get_field_by_name(cx.tcx, ty.peel_refs(), name.name) =>
                                {
                                    ty = fty;
                                    prev_expr = e;
                                    continue;
                                },
                                ExprKind::AddrOf(BorrowKind::Ref, ..) => break,
                                _ if matches!(
                                    typeck.expr_adjustments(prev_expr).first(),
                                    Some(Adjustment {
                                        kind: Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Not))
                                            | Adjust::Deref(_),
                                        ..
                                    })
                                ) =>
                                {
                                    break;
                                },
                                _ => {},
                            }
                        }
                        requires_copy |= !cx.type_is_copy_modulo_regions(ty);
                        break;
                    }
                },
                // Already processed uses.
                UseKind::AutoBorrowed | UseKind::WillAutoDeref | UseKind::Deref => {},
            }
        }

        if can_lint
            && (!requires_copy || cx.type_is_copy_modulo_regions(arg_ty))
            // This case could be handled, but a fair bit of care would need to be taken.
            && (!requires_deref || arg_ty.is_freeze(cx.tcx, cx.typing_env()))
        {
            if requires_deref {
                edits.push((param.span.shrink_to_lo(), "&".into()));
            } else {
                edits.extend(addr_of_edits);
            }
            let edit = match name {
                sym::map => "inspect",
                sym::map_err => "inspect_err",
                _ => return,
            };
            edits.push((name_span, edit.to_string()));
            edits.push((
                final_expr
                    .span
                    .with_leading_whitespace(cx)
                    .with_ctxt(final_expr.span.ctxt()),
                String::new(),
            ));
            let app = if edits.iter().any(|(s, _)| s.from_expansion()) {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };
            span_lint_and_then(
                cx,
                MANUAL_INSPECT,
                name_span,
                format!("using `{name}` over `{edit}`"),
                |diag| {
                    diag.multipart_suggestion("try", edits, app);
                },
            );
        }
    }
}

enum UseKind<'tcx> {
    AutoBorrowed,
    WillAutoDeref,
    Deref,
    Return(Span),
    Borrowed(Span),
    FieldAccess(Symbol, &'tcx Expr<'tcx>),
}

/// Checks how the value is used, and whether it was used in the same `SyntaxContext`.
fn check_use<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) -> (UseKind<'tcx>, bool) {
    let use_cx = expr_use_ctxt(cx, e);
    if use_cx
        .adjustments
        .first()
        .is_some_and(|a| matches!(a.kind, Adjust::Deref(_)))
    {
        return (UseKind::AutoBorrowed, use_cx.same_ctxt);
    }
    let res = match use_cx.use_node(cx) {
        ExprUseNode::Return(_) => {
            if let ExprKind::Ret(Some(e)) = use_cx.node.expect_expr().kind {
                UseKind::Return(e.span)
            } else {
                return (UseKind::Return(DUMMY_SP), false);
            }
        },
        ExprUseNode::FieldAccess(name) => UseKind::FieldAccess(name.name, use_cx.node.expect_expr()),
        ExprUseNode::Callee | ExprUseNode::MethodArg(_, _, 0)
            if use_cx
                .adjustments
                .first()
                .is_some_and(|a| matches!(a.kind, Adjust::Borrow(AutoBorrow::Ref(AutoBorrowMutability::Not)))) =>
        {
            UseKind::AutoBorrowed
        },
        ExprUseNode::Callee | ExprUseNode::MethodArg(_, _, 0) => UseKind::WillAutoDeref,
        ExprUseNode::AddrOf(BorrowKind::Ref, _) => UseKind::Borrowed(use_cx.node.expect_expr().span),
        _ => UseKind::Deref,
    };
    (res, use_cx.same_ctxt)
}
