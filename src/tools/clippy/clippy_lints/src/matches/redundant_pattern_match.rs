use super::REDUNDANT_PATTERN_MATCHING;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::needs_ordered_drop;
use clippy_utils::{higher, match_def_path};
use clippy_utils::{is_lang_ctor, is_trait_method, paths};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, PollPending};
use rustc_hir::{
    intravisit::{walk_expr, Visitor},
    Arm, Block, Expr, ExprKind, Node, Pat, PatKind, QPath, UnOp,
};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, subst::GenericArgKind, DefIdTree, Ty};
use rustc_span::sym;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let Some(higher::IfLet {
        if_else,
        let_pat,
        let_expr,
        ..
    }) = higher::IfLet::hir(cx, expr)
    {
        find_sugg_for_if_let(cx, expr, let_pat, let_expr, "if", if_else.is_some());
    } else if let Some(higher::WhileLet { let_pat, let_expr, .. }) = higher::WhileLet::hir(expr) {
        find_sugg_for_if_let(cx, expr, let_pat, let_expr, "while", false);
    }
}

// Extract the generic arguments out of a type
fn try_get_generic_ty(ty: Ty<'_>, index: usize) -> Option<Ty<'_>> {
    if_chain! {
        if let ty::Adt(_, subs) = ty.kind();
        if let Some(sub) = subs.get(index);
        if let GenericArgKind::Type(sub_ty) = sub.unpack();
        then {
            Some(sub_ty)
        } else {
            None
        }
    }
}

// Checks if there are any temporaries created in the given expression for which drop order
// matters.
fn temporaries_need_ordered_drop<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    struct V<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        res: bool,
    }
    impl<'a, 'tcx> Visitor<'tcx> for V<'a, 'tcx> {
        fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
            match expr.kind {
                // Taking the reference of a value leaves a temporary
                // e.g. In `&String::new()` the string is a temporary value.
                // Remaining fields are temporary values
                // e.g. In `(String::new(), 0).1` the string is a temporary value.
                ExprKind::AddrOf(_, _, expr) | ExprKind::Field(expr, _) => {
                    if !matches!(expr.kind, ExprKind::Path(_)) {
                        if needs_ordered_drop(self.cx, self.cx.typeck_results().expr_ty(expr)) {
                            self.res = true;
                        } else {
                            self.visit_expr(expr);
                        }
                    }
                },
                // the base type is alway taken by reference.
                // e.g. In `(vec![0])[0]` the vector is a temporary value.
                ExprKind::Index(base, index) => {
                    if !matches!(base.kind, ExprKind::Path(_)) {
                        if needs_ordered_drop(self.cx, self.cx.typeck_results().expr_ty(base)) {
                            self.res = true;
                        } else {
                            self.visit_expr(base);
                        }
                    }
                    self.visit_expr(index);
                },
                // Method calls can take self by reference.
                // e.g. In `String::new().len()` the string is a temporary value.
                ExprKind::MethodCall(_, [self_arg, args @ ..], _) => {
                    if !matches!(self_arg.kind, ExprKind::Path(_)) {
                        let self_by_ref = self
                            .cx
                            .typeck_results()
                            .type_dependent_def_id(expr.hir_id)
                            .map_or(false, |id| self.cx.tcx.fn_sig(id).skip_binder().inputs()[0].is_ref());
                        if self_by_ref && needs_ordered_drop(self.cx, self.cx.typeck_results().expr_ty(self_arg)) {
                            self.res = true;
                        } else {
                            self.visit_expr(self_arg);
                        }
                    }
                    args.iter().for_each(|arg| self.visit_expr(arg));
                },
                // Either explicitly drops values, or changes control flow.
                ExprKind::DropTemps(_)
                | ExprKind::Ret(_)
                | ExprKind::Break(..)
                | ExprKind::Yield(..)
                | ExprKind::Block(Block { expr: None, .. }, _)
                | ExprKind::Loop(..) => (),

                // Only consider the final expression.
                ExprKind::Block(Block { expr: Some(expr), .. }, _) => self.visit_expr(expr),

                _ => walk_expr(self, expr),
            }
        }
    }

    let mut v = V { cx, res: false };
    v.visit_expr(expr);
    v.res
}

fn find_sugg_for_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    keyword: &'static str,
    has_else: bool,
) {
    // also look inside refs
    // if we have &None for example, peel it so we can detect "if let None = x"
    let check_pat = match let_pat.kind {
        PatKind::Ref(inner, _mutability) => inner,
        _ => let_pat,
    };
    let op_ty = cx.typeck_results().expr_ty(let_expr);
    // Determine which function should be used, and the type contained by the corresponding
    // variant.
    let (good_method, inner_ty) = match check_pat.kind {
        PatKind::TupleStruct(ref qpath, [sub_pat], _) => {
            if let PatKind::Wild = sub_pat.kind {
                let res = cx.typeck_results().qpath_res(qpath, check_pat.hir_id);
                let Some(id) = res.opt_def_id().map(|ctor_id| cx.tcx.parent(ctor_id)) else { return };
                let lang_items = cx.tcx.lang_items();
                if Some(id) == lang_items.result_ok_variant() {
                    ("is_ok()", try_get_generic_ty(op_ty, 0).unwrap_or(op_ty))
                } else if Some(id) == lang_items.result_err_variant() {
                    ("is_err()", try_get_generic_ty(op_ty, 1).unwrap_or(op_ty))
                } else if Some(id) == lang_items.option_some_variant() {
                    ("is_some()", op_ty)
                } else if Some(id) == lang_items.poll_ready_variant() {
                    ("is_ready()", op_ty)
                } else if match_def_path(cx, id, &paths::IPADDR_V4) {
                    ("is_ipv4()", op_ty)
                } else if match_def_path(cx, id, &paths::IPADDR_V6) {
                    ("is_ipv6()", op_ty)
                } else {
                    return;
                }
            } else {
                return;
            }
        },
        PatKind::Path(ref path) => {
            let method = if is_lang_ctor(cx, path, OptionNone) {
                "is_none()"
            } else if is_lang_ctor(cx, path, PollPending) {
                "is_pending()"
            } else {
                return;
            };
            // `None` and `Pending` don't have an inner type.
            (method, cx.tcx.types.unit)
        },
        _ => return,
    };

    // If this is the last expression in a block or there is an else clause then the whole
    // type needs to be considered, not just the inner type of the branch being matched on.
    // Note the last expression in a block is dropped after all local bindings.
    let check_ty = if has_else
        || (keyword == "if" && matches!(cx.tcx.hir().parent_iter(expr.hir_id).next(), Some((_, Node::Block(..)))))
    {
        op_ty
    } else {
        inner_ty
    };

    // All temporaries created in the scrutinee expression are dropped at the same time as the
    // scrutinee would be, so they have to be considered as well.
    // e.g. in `if let Some(x) = foo.lock().unwrap().baz.as_ref() { .. }` the lock will be held
    // for the duration if body.
    let needs_drop = needs_ordered_drop(cx, check_ty) || temporaries_need_ordered_drop(cx, let_expr);

    // check that `while_let_on_iterator` lint does not trigger
    if_chain! {
        if keyword == "while";
        if let ExprKind::MethodCall(method_path, _, _) = let_expr.kind;
        if method_path.ident.name == sym::next;
        if is_trait_method(cx, let_expr, sym::Iterator);
        then {
            return;
        }
    }

    let result_expr = match &let_expr.kind {
        ExprKind::AddrOf(_, _, borrowed) => borrowed,
        ExprKind::Unary(UnOp::Deref, deref) => deref,
        _ => let_expr,
    };

    span_lint_and_then(
        cx,
        REDUNDANT_PATTERN_MATCHING,
        let_pat.span,
        &format!("redundant pattern matching, consider using `{}`", good_method),
        |diag| {
            // if/while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            let expr_span = expr.span;

            // if/while let ... = ... { ... }
            //                 ^^^
            let op_span = result_expr.span.source_callsite();

            // if/while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^
            let span = expr_span.until(op_span.shrink_to_hi());

            let app = if needs_drop {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };

            let sugg = Sugg::hir_with_macro_callsite(cx, result_expr, "_")
                .maybe_par()
                .to_string();

            diag.span_suggestion(span, "try this", format!("{} {}.{}", keyword, sugg, good_method), app);

            if needs_drop {
                diag.note("this will change drop order of the result, as well as all temporaries");
                diag.note("add `#[allow(clippy::redundant_pattern_matching)]` if this is important");
            }
        },
    );
}

pub(super) fn check_match<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, op: &Expr<'_>, arms: &[Arm<'_>]) {
    if arms.len() == 2 {
        let node_pair = (&arms[0].pat.kind, &arms[1].pat.kind);

        let found_good_method = match node_pair {
            (
                PatKind::TupleStruct(ref path_left, patterns_left, _),
                PatKind::TupleStruct(ref path_right, patterns_right, _),
            ) if patterns_left.len() == 1 && patterns_right.len() == 1 => {
                if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].kind, &patterns_right[0].kind) {
                    find_good_method_for_match(
                        cx,
                        arms,
                        path_left,
                        path_right,
                        &paths::RESULT_OK,
                        &paths::RESULT_ERR,
                        "is_ok()",
                        "is_err()",
                    )
                    .or_else(|| {
                        find_good_method_for_match(
                            cx,
                            arms,
                            path_left,
                            path_right,
                            &paths::IPADDR_V4,
                            &paths::IPADDR_V6,
                            "is_ipv4()",
                            "is_ipv6()",
                        )
                    })
                } else {
                    None
                }
            },
            (PatKind::TupleStruct(ref path_left, patterns, _), PatKind::Path(ref path_right))
            | (PatKind::Path(ref path_left), PatKind::TupleStruct(ref path_right, patterns, _))
                if patterns.len() == 1 =>
            {
                if let PatKind::Wild = patterns[0].kind {
                    find_good_method_for_match(
                        cx,
                        arms,
                        path_left,
                        path_right,
                        &paths::OPTION_SOME,
                        &paths::OPTION_NONE,
                        "is_some()",
                        "is_none()",
                    )
                    .or_else(|| {
                        find_good_method_for_match(
                            cx,
                            arms,
                            path_left,
                            path_right,
                            &paths::POLL_READY,
                            &paths::POLL_PENDING,
                            "is_ready()",
                            "is_pending()",
                        )
                    })
                } else {
                    None
                }
            },
            _ => None,
        };

        if let Some(good_method) = found_good_method {
            let span = expr.span.to(op.span);
            let result_expr = match &op.kind {
                ExprKind::AddrOf(_, _, borrowed) => borrowed,
                _ => op,
            };
            span_lint_and_then(
                cx,
                REDUNDANT_PATTERN_MATCHING,
                expr.span,
                &format!("redundant pattern matching, consider using `{}`", good_method),
                |diag| {
                    diag.span_suggestion(
                        span,
                        "try this",
                        format!("{}.{}", snippet(cx, result_expr.span, "_"), good_method),
                        Applicability::MaybeIncorrect, // snippet
                    );
                },
            );
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn find_good_method_for_match<'a>(
    cx: &LateContext<'_>,
    arms: &[Arm<'_>],
    path_left: &QPath<'_>,
    path_right: &QPath<'_>,
    expected_left: &[&str],
    expected_right: &[&str],
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<&'a str> {
    let left_id = cx
        .typeck_results()
        .qpath_res(path_left, arms[0].pat.hir_id)
        .opt_def_id()?;
    let right_id = cx
        .typeck_results()
        .qpath_res(path_right, arms[1].pat.hir_id)
        .opt_def_id()?;
    let body_node_pair = if match_def_path(cx, left_id, expected_left) && match_def_path(cx, right_id, expected_right) {
        (&(*arms[0].body).kind, &(*arms[1].body).kind)
    } else if match_def_path(cx, right_id, expected_left) && match_def_path(cx, right_id, expected_right) {
        (&(*arms[1].body).kind, &(*arms[0].body).kind)
    } else {
        return None;
    };

    match body_node_pair {
        (ExprKind::Lit(ref lit_left), ExprKind::Lit(ref lit_right)) => match (&lit_left.node, &lit_right.node) {
            (LitKind::Bool(true), LitKind::Bool(false)) => Some(should_be_left),
            (LitKind::Bool(false), LitKind::Bool(true)) => Some(should_be_right),
            _ => None,
        },
        _ => None,
    }
}
