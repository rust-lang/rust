use crate::{
    map_unit_fn::OPTION_MAP_UNIT_FN,
    matches::MATCH_AS_REF,
    utils::{
        can_partially_move_ty, is_allowed, is_type_diagnostic_item, match_def_path, match_var, paths,
        peel_hir_expr_refs, peel_mid_ty_refs_is_mutable, snippet_with_applicability, snippet_with_context,
        span_lint_and_sugg,
    },
};
use rustc_ast::util::parser::PREC_POSTFIX;
use rustc_errors::Applicability;
use rustc_hir::{
    def::Res,
    intravisit::{walk_expr, ErasedMap, NestedVisitorMap, Visitor},
    Arm, BindingAnnotation, Block, Expr, ExprKind, Mutability, Pat, PatKind, Path, QPath,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{
    symbol::{sym, Ident},
    SyntaxContext,
};

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
            if !(is_type_diagnostic_item(cx, scrutinee_ty, sym::option_type)
                && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::option_type))
            {
                return;
            }

            let expr_ctxt = expr.span.ctxt();
            let (some_expr, some_pat, pat_ref_count, is_wild_none) = match (
                try_parse_pattern(cx, arm1.pat, expr_ctxt),
                try_parse_pattern(cx, arm2.pat, expr_ctxt),
            ) {
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

            let some_expr = match get_some_expr(cx, some_expr, expr_ctxt) {
                Some(expr) => expr,
                None => return,
            };

            if cx.typeck_results().expr_ty(some_expr) == cx.tcx.types.unit
                && !is_allowed(cx, OPTION_MAP_UNIT_FN, expr.hir_id)
            {
                return;
            }

            if !can_move_expr_to_closure(cx, some_expr) {
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

            // Remove address-of expressions from the scrutinee. Either `as_ref` will be called, or
            // it's being passed by value.
            let scrutinee = peel_hir_expr_refs(scrutinee).0;
            let scrutinee_str = snippet_with_context(cx, scrutinee.span, expr_ctxt, "..", &mut app);
            let scrutinee_str =
                if scrutinee.span.ctxt() == expr.span.ctxt() && scrutinee.precedence().order() < PREC_POSTFIX {
                    format!("({})", scrutinee_str)
                } else {
                    scrutinee_str.into()
                };

            let body_str = if let PatKind::Binding(annotation, _, some_binding, None) = some_pat.kind {
                match can_pass_as_func(cx, some_binding, some_expr) {
                    Some(func) if func.span.ctxt() == some_expr.span.ctxt() => {
                        snippet_with_applicability(cx, func.span, "..", &mut app).into_owned()
                    },
                    _ => {
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
                            snippet_with_context(cx, some_expr.span, expr_ctxt, "..", &mut app)
                        )
                    },
                }
            } else if !is_wild_none && explicit_ref.is_none() {
                // TODO: handle explicit reference annotations.
                format!(
                    "|{}| {}",
                    snippet_with_context(cx, some_pat.span, expr_ctxt, "..", &mut app),
                    snippet_with_context(cx, some_expr.span, expr_ctxt, "..", &mut app)
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

// Checks if the expression can be moved into a closure as is.
fn can_move_expr_to_closure(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    struct V<'cx, 'tcx> {
        cx: &'cx LateContext<'tcx>,
        make_closure: bool,
    }
    impl Visitor<'tcx> for V<'_, 'tcx> {
        type Map = ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            match e.kind {
                ExprKind::Break(..)
                | ExprKind::Continue(_)
                | ExprKind::Ret(_)
                | ExprKind::Yield(..)
                | ExprKind::InlineAsm(_)
                | ExprKind::LlvmInlineAsm(_) => {
                    self.make_closure = false;
                },
                // Accessing a field of a local value can only be done if the type isn't
                // partially moved.
                ExprKind::Field(base_expr, _)
                    if matches!(
                        base_expr.kind,
                        ExprKind::Path(QPath::Resolved(_, Path { res: Res::Local(_), .. }))
                    ) && can_partially_move_ty(self.cx, self.cx.typeck_results().expr_ty(base_expr)) =>
                {
                    // TODO: check if the local has been partially moved. Assume it has for now.
                    self.make_closure = false;
                    return;
                }
                _ => (),
            };
            walk_expr(self, e);
        }
    }

    let mut v = V { cx, make_closure: true };
    v.visit_expr(expr);
    v.make_closure
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
fn try_parse_pattern(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, ctxt: SyntaxContext) -> Option<OptionPat<'tcx>> {
    fn f(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, ref_count: usize, ctxt: SyntaxContext) -> Option<OptionPat<'tcx>> {
        match pat.kind {
            PatKind::Wild => Some(OptionPat::Wild),
            PatKind::Ref(pat, _) => f(cx, pat, ref_count + 1, ctxt),
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
                    .map_or(false, |id| match_def_path(cx, id, &paths::OPTION_SOME))
                    && pat.span.ctxt() == ctxt =>
            {
                Some(OptionPat::Some { pattern, ref_count })
            },
            _ => None,
        }
    }
    f(cx, pat, 0, ctxt)
}

// Checks for an expression wrapped by the `Some` constructor. Returns the contained expression.
fn get_some_expr(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, ctxt: SyntaxContext) -> Option<&'tcx Expr<'tcx>> {
    // TODO: Allow more complex expressions.
    match expr.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(QPath::Resolved(None, path)),
                ..
            },
            [arg],
        ) if ctxt == expr.span.ctxt() => {
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
        ) => get_some_expr(cx, expr, ctxt),
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
