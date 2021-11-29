use crate::{map_unit_fn::OPTION_MAP_UNIT_FN, matches::MATCH_AS_REF};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::higher::IfLetOrMatch;
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::ty::{is_type_diagnostic_item, peel_mid_ty_refs_is_mutable, type_is_unsafe_function};
use clippy_utils::{
    can_move_expr_to_closure, in_constant, is_else_clause, is_lang_ctor, is_lint_allowed, path_to_local_id,
    peel_blocks, peel_hir_expr_refs, peel_hir_expr_while, CaptureKind,
};
use rustc_ast::util::parser::PREC_POSTFIX;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{
    def::Res, Arm, BindingAnnotation, Block, BlockCheckMode, Expr, ExprKind, HirId, Mutability, Pat, PatKind, Path,
    QPath, UnsafeSource,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of `match` which could be implemented using `map`
    ///
    /// ### Why is this bad?
    /// Using the `map` method is clearer and more concise.
    ///
    /// ### Example
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
    #[clippy::version = "1.52.0"]
    pub MANUAL_MAP,
    style,
    "reimplementation of `map`"
}

declare_lint_pass!(ManualMap => [MANUAL_MAP]);

impl LateLintPass<'_> for ManualMap {
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (scrutinee, then_pat, then_body, else_pat, else_body) = match IfLetOrMatch::parse(cx, expr) {
            Some(IfLetOrMatch::IfLet(scrutinee, pat, body, Some(r#else))) => (scrutinee, pat, body, None, r#else),
            Some(IfLetOrMatch::Match(
                scrutinee,
                [arm1 @ Arm { guard: None, .. }, arm2 @ Arm { guard: None, .. }],
                _,
            )) => (scrutinee, arm1.pat, arm1.body, Some(arm2.pat), arm2.body),
            _ => return,
        };
        if in_external_macro(cx.sess(), expr.span) || in_constant(cx, expr.hir_id) {
            return;
        }

        let (scrutinee_ty, ty_ref_count, ty_mutability) =
            peel_mid_ty_refs_is_mutable(cx.typeck_results().expr_ty(scrutinee));
        if !(is_type_diagnostic_item(cx, scrutinee_ty, sym::Option)
            && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::Option))
        {
            return;
        }

        let expr_ctxt = expr.span.ctxt();
        let (some_expr, some_pat, pat_ref_count, is_wild_none) = match (
            try_parse_pattern(cx, then_pat, expr_ctxt),
            else_pat.map_or(Some(OptionPat::Wild), |p| try_parse_pattern(cx, p, expr_ctxt)),
        ) {
            (Some(OptionPat::Wild), Some(OptionPat::Some { pattern, ref_count })) if is_none_expr(cx, then_body) => {
                (else_body, pattern, ref_count, true)
            },
            (Some(OptionPat::None), Some(OptionPat::Some { pattern, ref_count })) if is_none_expr(cx, then_body) => {
                (else_body, pattern, ref_count, false)
            },
            (Some(OptionPat::Some { pattern, ref_count }), Some(OptionPat::Wild)) if is_none_expr(cx, else_body) => {
                (then_body, pattern, ref_count, true)
            },
            (Some(OptionPat::Some { pattern, ref_count }), Some(OptionPat::None)) if is_none_expr(cx, else_body) => {
                (then_body, pattern, ref_count, false)
            },
            _ => return,
        };

        // Top level or patterns aren't allowed in closures.
        if matches!(some_pat.kind, PatKind::Or(_)) {
            return;
        }

        let some_expr = match get_some_expr(cx, some_expr, false, expr_ctxt) {
            Some(expr) => expr,
            None => return,
        };

        // These two lints will go back and forth with each other.
        if cx.typeck_results().expr_ty(some_expr.expr) == cx.tcx.types.unit
            && !is_lint_allowed(cx, OPTION_MAP_UNIT_FN, expr.hir_id)
        {
            return;
        }

        // `map` won't perform any adjustments.
        if !cx.typeck_results().expr_adjustments(some_expr.expr).is_empty() {
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

        match can_move_expr_to_closure(cx, some_expr.expr) {
            Some(captures) => {
                // Check if captures the closure will need conflict with borrows made in the scrutinee.
                // TODO: check all the references made in the scrutinee expression. This will require interacting
                // with the borrow checker. Currently only `<local>[.<field>]*` is checked for.
                if let Some(binding_ref_mutability) = binding_ref {
                    let e = peel_hir_expr_while(scrutinee, |e| match e.kind {
                        ExprKind::Field(e, _) | ExprKind::AddrOf(_, _, e) => Some(e),
                        _ => None,
                    });
                    if let ExprKind::Path(QPath::Resolved(None, Path { res: Res::Local(l), .. })) = e.kind {
                        match captures.get(l) {
                            Some(CaptureKind::Value | CaptureKind::Ref(Mutability::Mut)) => return,
                            Some(CaptureKind::Ref(Mutability::Not)) if binding_ref_mutability == Mutability::Mut => {
                                return;
                            },
                            Some(CaptureKind::Ref(Mutability::Not)) | None => (),
                        }
                    }
                }
            },
            None => return,
        };

        let mut app = Applicability::MachineApplicable;

        // Remove address-of expressions from the scrutinee. Either `as_ref` will be called, or
        // it's being passed by value.
        let scrutinee = peel_hir_expr_refs(scrutinee).0;
        let (scrutinee_str, _) = snippet_with_context(cx, scrutinee.span, expr_ctxt, "..", &mut app);
        let scrutinee_str =
            if scrutinee.span.ctxt() == expr.span.ctxt() && scrutinee.precedence().order() < PREC_POSTFIX {
                format!("({})", scrutinee_str)
            } else {
                scrutinee_str.into()
            };

        let body_str = if let PatKind::Binding(annotation, id, some_binding, None) = some_pat.kind {
            if_chain! {
                if !some_expr.needs_unsafe_block;
                if let Some(func) = can_pass_as_func(cx, id, some_expr.expr);
                if func.span.ctxt() == some_expr.expr.span.ctxt();
                then {
                    snippet_with_applicability(cx, func.span, "..", &mut app).into_owned()
                } else {
                    if path_to_local_id(some_expr.expr, id)
                        && !is_lint_allowed(cx, MATCH_AS_REF, expr.hir_id)
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
                    let expr_snip = snippet_with_context(cx, some_expr.expr.span, expr_ctxt, "..", &mut app).0;
                    if some_expr.needs_unsafe_block {
                        format!("|{}{}| unsafe {{ {} }}", annotation, some_binding, expr_snip)
                    } else {
                        format!("|{}{}| {}", annotation, some_binding, expr_snip)
                    }
                }
            }
        } else if !is_wild_none && explicit_ref.is_none() {
            // TODO: handle explicit reference annotations.
            let pat_snip = snippet_with_context(cx, some_pat.span, expr_ctxt, "..", &mut app).0;
            let expr_snip = snippet_with_context(cx, some_expr.expr.span, expr_ctxt, "..", &mut app).0;
            if some_expr.needs_unsafe_block {
                format!("|{}| unsafe {{ {} }}", pat_snip, expr_snip)
            } else {
                format!("|{}| {}", pat_snip, expr_snip)
            }
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
            if else_pat.is_none() && is_else_clause(cx.tcx, expr) {
                format!("{{ {}{}.map({}) }}", scrutinee_str, as_ref_str, body_str)
            } else {
                format!("{}{}.map({})", scrutinee_str, as_ref_str, body_str)
            },
            app,
        );
    }
}

// Checks whether the expression could be passed as a function, or whether a closure is needed.
// Returns the function to be passed to `map` if it exists.
fn can_pass_as_func(cx: &LateContext<'tcx>, binding: HirId, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    match expr.kind {
        ExprKind::Call(func, [arg])
            if path_to_local_id(arg, binding)
                && cx.typeck_results().expr_adjustments(arg).is_empty()
                && !type_is_unsafe_function(cx, cx.typeck_results().expr_ty(func).peel_refs()) =>
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

struct SomeExpr<'tcx> {
    expr: &'tcx Expr<'tcx>,
    needs_unsafe_block: bool,
}

// Try to parse into a recognized `Option` pattern.
// i.e. `_`, `None`, `Some(..)`, or a reference to any of those.
fn try_parse_pattern(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, ctxt: SyntaxContext) -> Option<OptionPat<'tcx>> {
    fn f(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, ref_count: usize, ctxt: SyntaxContext) -> Option<OptionPat<'tcx>> {
        match pat.kind {
            PatKind::Wild => Some(OptionPat::Wild),
            PatKind::Ref(pat, _) => f(cx, pat, ref_count + 1, ctxt),
            PatKind::Path(ref qpath) if is_lang_ctor(cx, qpath, OptionNone) => Some(OptionPat::None),
            PatKind::TupleStruct(ref qpath, [pattern], _)
                if is_lang_ctor(cx, qpath, OptionSome) && pat.span.ctxt() == ctxt =>
            {
                Some(OptionPat::Some { pattern, ref_count })
            },
            _ => None,
        }
    }
    f(cx, pat, 0, ctxt)
}

// Checks for an expression wrapped by the `Some` constructor. Returns the contained expression.
fn get_some_expr(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    needs_unsafe_block: bool,
    ctxt: SyntaxContext,
) -> Option<SomeExpr<'tcx>> {
    // TODO: Allow more complex expressions.
    match expr.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(ref qpath),
                ..
            },
            [arg],
        ) if ctxt == expr.span.ctxt() && is_lang_ctor(cx, qpath, OptionSome) => Some(SomeExpr {
            expr: arg,
            needs_unsafe_block,
        }),
        ExprKind::Block(
            Block {
                stmts: [],
                expr: Some(expr),
                rules,
                ..
            },
            _,
        ) => get_some_expr(
            cx,
            expr,
            needs_unsafe_block || *rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
            ctxt,
        ),
        _ => None,
    }
}

// Checks for the `None` value.
fn is_none_expr(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    matches!(peel_blocks(expr).kind, ExprKind::Path(ref qpath) if is_lang_ctor(cx, qpath, OptionNone))
}
