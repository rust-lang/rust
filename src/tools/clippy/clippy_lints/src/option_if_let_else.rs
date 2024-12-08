use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{
    CaptureKind, can_move_expr_to_closure, eager_or_lazy, higher, is_else_clause, is_in_const_context,
    is_res_lang_ctor, peel_blocks, peel_hir_expr_while,
};
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome, ResultErr, ResultOk};
use rustc_hir::def::Res;
use rustc_hir::{Arm, BindingMode, Expr, ExprKind, MatchSource, Mutability, Pat, PatKind, Path, QPath, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::SyntaxContext;

declare_clippy_lint! {
    /// ### What it does
    /// Lints usage of `if let Some(v) = ... { y } else { x }` and
    /// `match .. { Some(v) => y, None/_ => x }` which are more
    /// idiomatically done with `Option::map_or` (if the else bit is a pure
    /// expression) or `Option::map_or_else` (if the else bit is an impure
    /// expression).
    ///
    /// ### Why is this bad?
    /// Using the dedicated functions of the `Option` type is clearer and
    /// more concise than an `if let` expression.
    ///
    /// ### Notes
    /// This lint uses a deliberately conservative metric for checking if the
    /// inside of either body contains loop control expressions `break` or
    /// `continue` (which cannot be used within closures). If these are found,
    /// this lint will not be raised.
    ///
    /// ### Example
    /// ```no_run
    /// # let optional: Option<u32> = Some(0);
    /// # fn do_complicated_function() -> u32 { 5 };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     5
    /// };
    /// let _ = match optional {
    ///     Some(val) => val + 1,
    ///     None => 5
    /// };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     let y = do_complicated_function();
    ///     y*y
    /// };
    /// ```
    ///
    /// should be
    ///
    /// ```no_run
    /// # let optional: Option<u32> = Some(0);
    /// # fn do_complicated_function() -> u32 { 5 };
    /// let _ = optional.map_or(5, |foo| foo);
    /// let _ = optional.map_or(5, |val| val + 1);
    /// let _ = optional.map_or_else(||{
    ///     let y = do_complicated_function();
    ///     y*y
    /// }, |foo| foo);
    /// ```
    // FIXME: Before moving this lint out of nursery, the lint name needs to be updated. It now also
    // covers matches and `Result`.
    #[clippy::version = "1.47.0"]
    pub OPTION_IF_LET_ELSE,
    nursery,
    "reimplementation of Option::map_or"
}

declare_lint_pass!(OptionIfLetElse => [OPTION_IF_LET_ELSE]);

/// A struct containing information about occurrences of construct that this lint detects
///
/// Such as:
///
/// ```ignore
/// if let Some(..) = {..} else {..}
/// ```
/// or
/// ```ignore
/// match x {
///     Some(..) => {..},
///     None/_ => {..}
/// }
/// ```
struct OptionOccurrence {
    option: String,
    method_sugg: String,
    some_expr: String,
    none_expr: String,
}

fn format_option_in_sugg(cond_sugg: Sugg<'_>, as_ref: bool, as_mut: bool) -> String {
    format!(
        "{}{}",
        cond_sugg.maybe_par(),
        if as_mut {
            ".as_mut()"
        } else if as_ref {
            ".as_ref()"
        } else {
            ""
        }
    )
}

fn try_get_option_occurrence<'tcx>(
    cx: &LateContext<'tcx>,
    ctxt: SyntaxContext,
    pat: &Pat<'tcx>,
    expr: &Expr<'_>,
    if_then: &'tcx Expr<'_>,
    if_else: &'tcx Expr<'_>,
) -> Option<OptionOccurrence> {
    let cond_expr = match expr.kind {
        ExprKind::Unary(UnOp::Deref, inner_expr) | ExprKind::AddrOf(_, _, inner_expr) => inner_expr,
        _ => expr,
    };
    let (inner_pat, is_result) = try_get_inner_pat_and_is_result(cx, pat)?;
    if let PatKind::Binding(bind_annotation, _, id, None) = inner_pat.kind
        && let Some(some_captures) = can_move_expr_to_closure(cx, if_then)
        && let Some(none_captures) = can_move_expr_to_closure(cx, if_else)
        && some_captures
            .iter()
            .filter_map(|(id, &c)| none_captures.get(id).map(|&c2| (c, c2)))
            .all(|(x, y)| x.is_imm_ref() && y.is_imm_ref())
    {
        let capture_mut = if bind_annotation == BindingMode::MUT {
            "mut "
        } else {
            ""
        };
        let some_body = peel_blocks(if_then);
        let none_body = peel_blocks(if_else);
        let method_sugg = if eager_or_lazy::switch_to_eager_eval(cx, none_body) {
            "map_or"
        } else {
            "map_or_else"
        };
        let capture_name = id.name.to_ident_string();
        let (as_ref, as_mut) = match &expr.kind {
            ExprKind::AddrOf(_, Mutability::Not, _) => (true, false),
            ExprKind::AddrOf(_, Mutability::Mut, _) => (false, true),
            _ if let Some(mutb) = cx.typeck_results().expr_ty(expr).ref_mutability() => {
                (mutb == Mutability::Not, mutb == Mutability::Mut)
            },
            _ => (
                bind_annotation == BindingMode::REF,
                bind_annotation == BindingMode::REF_MUT,
            ),
        };

        // Check if captures the closure will need conflict with borrows made in the scrutinee.
        // TODO: check all the references made in the scrutinee expression. This will require interacting
        // with the borrow checker. Currently only `<local>[.<field>]*` is checked for.
        if as_ref || as_mut {
            let e = peel_hir_expr_while(cond_expr, |e| match e.kind {
                ExprKind::Field(e, _) | ExprKind::AddrOf(_, _, e) => Some(e),
                _ => None,
            });
            if let ExprKind::Path(QPath::Resolved(
                None,
                Path {
                    res: Res::Local(local_id),
                    ..
                },
            )) = e.kind
            {
                match some_captures.get(local_id).or_else(|| {
                    (method_sugg == "map_or_else")
                        .then_some(())
                        .and_then(|()| none_captures.get(local_id))
                }) {
                    Some(CaptureKind::Value | CaptureKind::Ref(Mutability::Mut)) => return None,
                    Some(CaptureKind::Ref(Mutability::Not)) if as_mut => return None,
                    Some(CaptureKind::Ref(Mutability::Not)) | None => (),
                }
            }
        }

        let mut app = Applicability::Unspecified;

        let (none_body, is_argless_call) = match none_body.kind {
            ExprKind::Call(call_expr, []) if !none_body.span.from_expansion() => (call_expr, true),
            _ => (none_body, false),
        };

        return Some(OptionOccurrence {
            option: format_option_in_sugg(
                Sugg::hir_with_context(cx, cond_expr, ctxt, "..", &mut app),
                as_ref,
                as_mut,
            ),
            method_sugg: method_sugg.to_string(),
            some_expr: format!(
                "|{capture_mut}{capture_name}| {}",
                Sugg::hir_with_context(cx, some_body, ctxt, "..", &mut app),
            ),
            none_expr: format!(
                "{}{}",
                if method_sugg == "map_or" || is_argless_call {
                    ""
                } else if is_result {
                    "|_| "
                } else {
                    "|| "
                },
                Sugg::hir_with_context(cx, none_body, ctxt, "..", &mut app),
            ),
        });
    }

    None
}

fn try_get_inner_pat_and_is_result<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'tcx>) -> Option<(&'tcx Pat<'tcx>, bool)> {
    if let PatKind::TupleStruct(ref qpath, [inner_pat], ..) = pat.kind {
        let res = cx.qpath_res(qpath, pat.hir_id);
        if is_res_lang_ctor(cx, res, OptionSome) {
            return Some((inner_pat, false));
        } else if is_res_lang_ctor(cx, res, ResultOk) {
            return Some((inner_pat, true));
        }
    }
    None
}

/// If this expression is the option if let/else construct we're detecting, then
/// this function returns an `OptionOccurrence` struct with details if
/// this construct is found, or None if this construct is not found.
fn detect_option_if_let_else<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<OptionOccurrence> {
    if let Some(higher::IfLet {
        let_pat,
        let_expr,
        if_then,
        if_else: Some(if_else),
        ..
    }) = higher::IfLet::hir(cx, expr)
        && !cx.typeck_results().expr_ty(expr).is_unit()
        && !is_else_clause(cx.tcx, expr)
    {
        try_get_option_occurrence(cx, expr.span.ctxt(), let_pat, let_expr, if_then, if_else)
    } else {
        None
    }
}

fn detect_option_match<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<OptionOccurrence> {
    if let ExprKind::Match(ex, arms, MatchSource::Normal) = expr.kind
        && !cx.typeck_results().expr_ty(expr).is_unit()
        && let Some((let_pat, if_then, if_else)) = try_convert_match(cx, arms)
    {
        try_get_option_occurrence(cx, expr.span.ctxt(), let_pat, ex, if_then, if_else)
    } else {
        None
    }
}

fn try_convert_match<'tcx>(
    cx: &LateContext<'tcx>,
    arms: &[Arm<'tcx>],
) -> Option<(&'tcx Pat<'tcx>, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    if let [first_arm, second_arm] = arms
        && first_arm.guard.is_none()
        && second_arm.guard.is_none()
    {
        return if is_none_or_err_arm(cx, second_arm) {
            Some((first_arm.pat, first_arm.body, second_arm.body))
        } else if is_none_or_err_arm(cx, first_arm) {
            Some((second_arm.pat, second_arm.body, first_arm.body))
        } else {
            None
        };
    }
    None
}

fn is_none_or_err_arm(cx: &LateContext<'_>, arm: &Arm<'_>) -> bool {
    match arm.pat.kind {
        PatKind::Path(ref qpath) => is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), OptionNone),
        PatKind::TupleStruct(ref qpath, [first_pat], _) => {
            is_res_lang_ctor(cx, cx.qpath_res(qpath, arm.pat.hir_id), ResultErr)
                && matches!(first_pat.kind, PatKind::Wild)
        },
        PatKind::Wild => true,
        _ => false,
    }
}

impl<'tcx> LateLintPass<'tcx> for OptionIfLetElse {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        // Don't lint macros and constants
        if expr.span.from_expansion() || is_in_const_context(cx) {
            return;
        }

        let detection = detect_option_if_let_else(cx, expr).or_else(|| detect_option_match(cx, expr));
        if let Some(det) = detection {
            span_lint_and_sugg(
                cx,
                OPTION_IF_LET_ELSE,
                expr.span,
                format!("use Option::{} instead of an if let/else", det.method_sugg),
                "try",
                format!(
                    "{}.{}({}, {})",
                    det.option, det.method_sugg, det.none_expr, det.some_expr
                ),
                Applicability::MaybeIncorrect,
            );
        }
    }
}
