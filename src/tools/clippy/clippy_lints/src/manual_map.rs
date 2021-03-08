use crate::{
    map_unit_fn::OPTION_MAP_UNIT_FN,
    matches::MATCH_AS_REF,
    utils::{
        is_allowed, is_type_diagnostic_item, match_def_path, match_var, paths, peel_hir_expr_refs,
        peel_mid_ty_refs_is_mutable, snippet_with_applicability, span_lint_and_sugg,
    },
};
use rustc_ast::util::parser::PREC_POSTFIX;
use rustc_errors::Applicability;
use rustc_hir::{Arm, BindingAnnotation, Block, Expr, ExprKind, Mutability, Pat, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::{sym, Ident};

declare_clippy_lint! {
    /// **What it does:** Checks for usages of `match` which could be implemented using `map`
    ///
    /// **Why is this bad?** Using the `map` method is clearer and more concise.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// match Some(0) {
    ///     Some(x) => Some(x + 1),
    ///     None => None,
    /// };
    /// ```
    /// Use instead:
    /// ```rust
    /// Some(0).map(|x| x + 1);
    /// ```
    pub MANUAL_MAP,
    style,
    "reimplementation of `map`"
}

declare_lint_pass!(ManualMap => [MANUAL_MAP]);

impl LateLintPass<'_> for ManualMap {
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Match(scrutinee, [arm1 @ Arm { guard: None, .. }, arm2 @ Arm { guard: None, .. }], _) =
            expr.kind
        {
            let (scrutinee_ty, ty_ref_count, ty_mutability) =
                peel_mid_ty_refs_is_mutable(cx.typeck_results().expr_ty(scrutinee));
            if !is_type_diagnostic_item(cx, scrutinee_ty, sym::option_type)
                || !is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::option_type)
            {
                return;
            }

            let (some_expr, some_pat, pat_ref_count, is_wild_none) =
                match (try_parse_pattern(cx, arm1.pat), try_parse_pattern(cx, arm2.pat)) {
                    (Some(OptionPat::Wild), Some(OptionPat::Some { pattern, ref_count }))
                        if is_none_expr(cx, arm1.body) =>
                    {
                        (arm2.body, pattern, ref_count, true)
                    },
                    (Some(OptionPat::None), Some(OptionPat::Some { pattern, ref_count }))
                        if is_none_expr(cx, arm1.body) =>
                    {
                        (arm2.body, pattern, ref_count, false)
                    },
                    (Some(OptionPat::Some { pattern, ref_count }), Some(OptionPat::Wild))
                        if is_none_expr(cx, arm2.body) =>
                    {
                        (arm1.body, pattern, ref_count, true)
                    },
                    (Some(OptionPat::Some { pattern, ref_count }), Some(OptionPat::None))
                        if is_none_expr(cx, arm2.body) =>
                    {
                        (arm1.body, pattern, ref_count, false)
                    },
                    _ => return,
                };

            // Top level or patterns aren't allowed in closures.
            if matches!(some_pat.kind, PatKind::Or(_)) {
                return;
            }

            let some_expr = match get_some_expr(cx, some_expr) {
                Some(expr) => expr,
                None => return,
            };

            if cx.typeck_results().expr_ty(some_expr) == cx.tcx.types.unit
                && !is_allowed(cx, OPTION_MAP_UNIT_FN, expr.hir_id)
            {
                return;
            }

            // Determine which binding mode to use.
            let explicit_ref = some_pat.contains_explicit_ref_binding();
            let binding_ref = explicit_ref.or_else(|| (ty_ref_count != pat_ref_count).then(|| ty_mutability));

            let as_ref_str = match binding_ref {
                Some(Mutability::Mut) => ".as_mut()",
                Some(Mutability::Not) => ".as_ref()",
                None => "",
            };

            let mut app = Applicability::MachineApplicable;

            // Remove address-of expressions from the scrutinee. `as_ref` will be called,
            // the type is copyable, or the option is being passed by value.
            let scrutinee = peel_hir_expr_refs(scrutinee).0;
            let scrutinee_str = snippet_with_applicability(cx, scrutinee.span, "_", &mut app);
            let scrutinee_str = if expr.precedence().order() < PREC_POSTFIX {
                // Parens are needed to chain method calls.
                format!("({})", scrutinee_str)
            } else {
                scrutinee_str.into()
            };

            let body_str = if let PatKind::Binding(annotation, _, some_binding, None) = some_pat.kind {
                if let Some(func) = can_pass_as_func(cx, some_binding, some_expr) {
                    snippet_with_applicability(cx, func.span, "..", &mut app).into_owned()
                } else {
                    if match_var(some_expr, some_binding.name)
                        && !is_allowed(cx, MATCH_AS_REF, expr.hir_id)
                        && binding_ref.is_some()
                    {
                        return;
                    }

                    // `ref` and `ref mut` annotations were handled earlier.
                    let annotation = if matches!(annotation, BindingAnnotation::Mutable) {
                        "mut "
                    } else {
                        ""
                    };
                    format!(
                        "|{}{}| {}",
                        annotation,
                        some_binding,
                        snippet_with_applicability(cx, some_expr.span, "..", &mut app)
                    )
                }
            } else if !is_wild_none && explicit_ref.is_none() {
                // TODO: handle explicit reference annotations.
                format!(
                    "|{}| {}",
                    snippet_with_applicability(cx, some_pat.span, "..", &mut app),
                    snippet_with_applicability(cx, some_expr.span, "..", &mut app)
                )
            } else {
                // Refutable bindings and mixed reference annotations can't be handled by `map`.
                return;
            };

            span_lint_and_sugg(
                cx,
                MANUAL_MAP,
                expr.span,
                "manual implementation of `Option::map`",
                "try this",
                format!("{}{}.map({})", scrutinee_str, as_ref_str, body_str),
                app,
            );
        }
    }
}

// Checks whether the expression could be passed as a function, or whether a closure is needed.
// Returns the function to be passed to `map` if it exists.
fn can_pass_as_func(cx: &LateContext<'tcx>, binding: Ident, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    match expr.kind {
        ExprKind::Call(func, [arg])
            if match_var(arg, binding.name) && cx.typeck_results().expr_adjustments(arg).is_empty() =>
        {
            Some(func)
        },
        _ => None,
    }
}

enum OptionPat<'a> {
    Wild,
    None,
    Some {
        // The pattern contained in the `Some` tuple.
        pattern: &'a Pat<'a>,
        // The number of references before the `Some` tuple.
        // e.g. `&&Some(_)` has a ref count of 2.
        ref_count: usize,
    },
}

// Try to parse into a recognized `Option` pattern.
// i.e. `_`, `None`, `Some(..)`, or a reference to any of those.
fn try_parse_pattern(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>) -> Option<OptionPat<'tcx>> {
    fn f(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, ref_count: usize) -> Option<OptionPat<'tcx>> {
        match pat.kind {
            PatKind::Wild => Some(OptionPat::Wild),
            PatKind::Ref(pat, _) => f(cx, pat, ref_count + 1),
            PatKind::Path(QPath::Resolved(None, path))
                if path
                    .res
                    .opt_def_id()
                    .map_or(false, |id| match_def_path(cx, id, &paths::OPTION_NONE)) =>
            {
                Some(OptionPat::None)
            },
            PatKind::TupleStruct(QPath::Resolved(None, path), [pattern], _)
                if path
                    .res
                    .opt_def_id()
                    .map_or(false, |id| match_def_path(cx, id, &paths::OPTION_SOME)) =>
            {
                Some(OptionPat::Some { pattern, ref_count })
            },
            _ => None,
        }
    }
    f(cx, pat, 0)
}

// Checks for an expression wrapped by the `Some` constructor. Returns the contained expression.
fn get_some_expr(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    // TODO: Allow more complex expressions.
    match expr.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(QPath::Resolved(None, path)),
                ..
            },
            [arg],
        ) => {
            if match_def_path(cx, path.res.opt_def_id()?, &paths::OPTION_SOME) {
                Some(arg)
            } else {
                None
            }
        },
        ExprKind::Block(
            Block {
                stmts: [],
                expr: Some(expr),
                ..
            },
            _,
        ) => get_some_expr(cx, expr),
        _ => None,
    }
}

// Checks for the `None` value.
fn is_none_expr(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Path(QPath::Resolved(None, path)) => path
            .res
            .opt_def_id()
            .map_or(false, |id| match_def_path(cx, id, &paths::OPTION_NONE)),
        ExprKind::Block(
            Block {
                stmts: [],
                expr: Some(expr),
                ..
            },
            _,
        ) => is_none_expr(cx, expr),
        _ => false,
    }
}
