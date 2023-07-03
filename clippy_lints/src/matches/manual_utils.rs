use crate::map_unit_fn::OPTION_MAP_UNIT_FN;
use crate::matches::MATCH_AS_REF;
use clippy_utils::source::{snippet_with_applicability, snippet_with_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{is_copy, is_type_diagnostic_item, peel_mid_ty_refs_is_mutable, type_is_unsafe_function};
use clippy_utils::{
    can_move_expr_to_closure, is_else_clause, is_lint_allowed, is_res_lang_ctor, path_res, path_to_local_id,
    peel_blocks, peel_hir_expr_refs, peel_hir_expr_while, CaptureKind,
};
use rustc_ast::util::parser::PREC_POSTFIX;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{BindingAnnotation, Expr, ExprKind, HirId, Mutability, Pat, PatKind, Path, QPath};
use rustc_lint::LateContext;
use rustc_span::{sym, SyntaxContext};

#[expect(clippy::too_many_arguments)]
#[expect(clippy::too_many_lines)]
pub(super) fn check_with<'tcx, F>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    scrutinee: &'tcx Expr<'_>,
    then_pat: &'tcx Pat<'_>,
    then_body: &'tcx Expr<'_>,
    else_pat: Option<&'tcx Pat<'_>>,
    else_body: &'tcx Expr<'_>,
    get_some_expr_fn: F,
) -> Option<SuggInfo<'tcx>>
where
    F: Fn(&LateContext<'tcx>, &'tcx Pat<'_>, &'tcx Expr<'_>, SyntaxContext) -> Option<SomeExpr<'tcx>>,
{
    let (scrutinee_ty, ty_ref_count, ty_mutability) =
        peel_mid_ty_refs_is_mutable(cx.typeck_results().expr_ty(scrutinee));
    if !(is_type_diagnostic_item(cx, scrutinee_ty, sym::Option)
        && is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr), sym::Option))
    {
        return None;
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
        _ => return None,
    };

    // Top level or patterns aren't allowed in closures.
    if matches!(some_pat.kind, PatKind::Or(_)) {
        return None;
    }

    let Some(some_expr) = get_some_expr_fn(cx, some_pat, some_expr, expr_ctxt) else {
        return None;
    };

    // These two lints will go back and forth with each other.
    if cx.typeck_results().expr_ty(some_expr.expr) == cx.tcx.types.unit
        && !is_lint_allowed(cx, OPTION_MAP_UNIT_FN, expr.hir_id)
    {
        return None;
    }

    // `map` won't perform any adjustments.
    if !cx.typeck_results().expr_adjustments(some_expr.expr).is_empty() {
        return None;
    }

    // Determine which binding mode to use.
    let explicit_ref = some_pat.contains_explicit_ref_binding();
    let binding_ref = explicit_ref.or_else(|| (ty_ref_count != pat_ref_count).then_some(ty_mutability));

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
                        Some(CaptureKind::Value | CaptureKind::Ref(Mutability::Mut)) => return None,
                        Some(CaptureKind::Ref(Mutability::Not)) if binding_ref_mutability == Mutability::Mut => {
                            return None;
                        },
                        Some(CaptureKind::Ref(Mutability::Not)) | None => (),
                    }
                }
            }
        },
        None => return None,
    };

    let mut app = Applicability::MachineApplicable;

    // Remove address-of expressions from the scrutinee. Either `as_ref` will be called, or
    // it's being passed by value.
    let scrutinee = peel_hir_expr_refs(scrutinee).0;
    let (scrutinee_str, _) = snippet_with_context(cx, scrutinee.span, expr_ctxt, "..", &mut app);
    let scrutinee_str = if scrutinee.span.ctxt() == expr.span.ctxt() && scrutinee.precedence().order() < PREC_POSTFIX {
        format!("({scrutinee_str})")
    } else {
        scrutinee_str.into()
    };

    let closure_expr_snip = some_expr.to_snippet_with_context(cx, expr_ctxt, &mut app);
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
                    return None;
                }

                // `ref` and `ref mut` annotations were handled earlier.
                let annotation = if matches!(annotation, BindingAnnotation::MUT) {
                    "mut "
                } else {
                    ""
                };

                if some_expr.needs_unsafe_block {
                    format!("|{annotation}{some_binding}| unsafe {{ {closure_expr_snip} }}")
                } else {
                    format!("|{annotation}{some_binding}| {closure_expr_snip}")
                }
            }
        }
    } else if !is_wild_none && explicit_ref.is_none() {
        // TODO: handle explicit reference annotations.
        let pat_snip = snippet_with_context(cx, some_pat.span, expr_ctxt, "..", &mut app).0;
        if some_expr.needs_unsafe_block {
            format!("|{pat_snip}| unsafe {{ {closure_expr_snip} }}")
        } else {
            format!("|{pat_snip}| {closure_expr_snip}")
        }
    } else {
        // Refutable bindings and mixed reference annotations can't be handled by `map`.
        return None;
    };

    // relies on the fact that Option<T>: Copy where T: copy
    let scrutinee_impl_copy = is_copy(cx, scrutinee_ty);

    Some(SuggInfo {
        needs_brackets: else_pat.is_none() && is_else_clause(cx.tcx, expr),
        scrutinee_impl_copy,
        scrutinee_str,
        as_ref_str,
        body_str,
        app,
    })
}

pub struct SuggInfo<'a> {
    pub needs_brackets: bool,
    pub scrutinee_impl_copy: bool,
    pub scrutinee_str: String,
    pub as_ref_str: &'a str,
    pub body_str: String,
    pub app: Applicability,
}

// Checks whether the expression could be passed as a function, or whether a closure is needed.
// Returns the function to be passed to `map` if it exists.
fn can_pass_as_func<'tcx>(cx: &LateContext<'tcx>, binding: HirId, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
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

#[derive(Debug)]
pub(super) enum OptionPat<'a> {
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

pub(super) struct SomeExpr<'tcx> {
    pub expr: &'tcx Expr<'tcx>,
    pub needs_unsafe_block: bool,
    pub needs_negated: bool, // for `manual_filter` lint
}

impl<'tcx> SomeExpr<'tcx> {
    pub fn new_no_negated(expr: &'tcx Expr<'tcx>, needs_unsafe_block: bool) -> Self {
        Self {
            expr,
            needs_unsafe_block,
            needs_negated: false,
        }
    }

    pub fn to_snippet_with_context(
        &self,
        cx: &LateContext<'tcx>,
        ctxt: SyntaxContext,
        app: &mut Applicability,
    ) -> Sugg<'tcx> {
        let sugg = Sugg::hir_with_context(cx, self.expr, ctxt, "..", app);
        if self.needs_negated { !sugg } else { sugg }
    }
}

// Try to parse into a recognized `Option` pattern.
// i.e. `_`, `None`, `Some(..)`, or a reference to any of those.
pub(super) fn try_parse_pattern<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    ctxt: SyntaxContext,
) -> Option<OptionPat<'tcx>> {
    fn f<'tcx>(
        cx: &LateContext<'tcx>,
        pat: &'tcx Pat<'_>,
        ref_count: usize,
        ctxt: SyntaxContext,
    ) -> Option<OptionPat<'tcx>> {
        match pat.kind {
            PatKind::Wild => Some(OptionPat::Wild),
            PatKind::Ref(pat, _) => f(cx, pat, ref_count + 1, ctxt),
            PatKind::Path(ref qpath) if is_res_lang_ctor(cx, cx.qpath_res(qpath, pat.hir_id), OptionNone) => {
                Some(OptionPat::None)
            },
            PatKind::TupleStruct(ref qpath, [pattern], _)
                if is_res_lang_ctor(cx, cx.qpath_res(qpath, pat.hir_id), OptionSome) && pat.span.ctxt() == ctxt =>
            {
                Some(OptionPat::Some { pattern, ref_count })
            },
            _ => None,
        }
    }
    f(cx, pat, 0, ctxt)
}

// Checks for the `None` value.
fn is_none_expr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    is_res_lang_ctor(cx, path_res(cx, peel_blocks(expr)), OptionNone)
}
