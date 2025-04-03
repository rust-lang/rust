use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::numeric_literal::NumericLiteral;
use clippy_utils::source::{SpanRangeExt, snippet_opt};
use clippy_utils::visitors::{Visitable, for_each_expr_without_closures};
use clippy_utils::{get_parent_expr, is_hir_ty_cfg_dependant, is_ty_alias, path_to_local};
use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, Lit, Node, Path, QPath, TyKind, UnOp};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty::{self, FloatTy, InferTy, Ty};
use std::ops::ControlFlow;

use super::UNNECESSARY_CAST;

#[expect(clippy::too_many_lines)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cast_expr: &Expr<'tcx>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
) -> bool {
    let cast_str = snippet_opt(cx, cast_expr.span).unwrap_or_default();

    if let ty::RawPtr(..) = cast_from.kind()
        // check both mutability and type are the same
        && cast_from.kind() == cast_to.kind()
        && let ExprKind::Cast(_, cast_to_hir) = expr.kind
        // Ignore casts to e.g. type aliases and infer types
        // - p as pointer_alias
        // - p as _
        && let TyKind::Ptr(to_pointee) = cast_to_hir.kind
    {
        match to_pointee.ty.kind {
            // Ignore casts to pointers that are aliases or cfg dependant, e.g.
            // - p as *const std::ffi::c_char (alias)
            // - p as *const std::os::raw::c_char (cfg dependant)
            TyKind::Path(qpath) => {
                if is_ty_alias(&qpath) || is_hir_ty_cfg_dependant(cx, to_pointee.ty) {
                    return false;
                }
            },
            // Ignore `p as *const _`
            TyKind::Infer(()) => return false,
            _ => {},
        }

        span_lint_and_sugg(
            cx,
            UNNECESSARY_CAST,
            expr.span,
            format!(
                "casting raw pointers to the same type and constness is unnecessary (`{cast_from}` -> `{cast_to}`)"
            ),
            "try",
            cast_str.clone(),
            Applicability::MaybeIncorrect,
        );
    }

    // skip cast of local that is a type alias
    if let ExprKind::Cast(inner, ..) = expr.kind
        && let ExprKind::Path(qpath) = inner.kind
        && let QPath::Resolved(None, Path { res, .. }) = qpath
        && let Res::Local(hir_id) = res
        && let parent = cx.tcx.parent_hir_node(*hir_id)
        && let Node::LetStmt(local) = parent
    {
        if let Some(ty) = local.ty
            && let TyKind::Path(qpath) = ty.kind
            && is_ty_alias(&qpath)
        {
            return false;
        }

        if let Some(expr) = local.init
            && let ExprKind::Cast(.., cast_to) = expr.kind
            && let TyKind::Path(qpath) = cast_to.kind
            && is_ty_alias(&qpath)
        {
            return false;
        }
    }

    // skip cast to non-primitive type
    if let ExprKind::Cast(_, cast_to) = expr.kind
        && let TyKind::Path(QPath::Resolved(_, path)) = &cast_to.kind
        && let Res::PrimTy(_) = path.res
    {
    } else {
        return false;
    }

    // skip cast of fn call that returns type alias
    if let ExprKind::Cast(inner, ..) = expr.kind
        && is_cast_from_ty_alias(cx, inner, cast_from)
    {
        return false;
    }

    if let Some(lit) = get_numeric_literal(cast_expr) {
        let literal_str = &cast_str;

        if let LitKind::Int(n, _) = lit.node
            && let Some(src) = cast_expr.span.get_source_text(cx)
            && cast_to.is_floating_point()
            && let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node)
            && let from_nbits = 128 - n.get().leading_zeros()
            && let to_nbits = fp_ty_mantissa_nbits(cast_to)
            && from_nbits != 0
            && to_nbits != 0
            && from_nbits <= to_nbits
            && num_lit.is_decimal()
        {
            lint_unnecessary_cast(cx, expr, num_lit.integer, cast_from, cast_to);
            return true;
        }

        match lit.node {
            LitKind::Int(_, LitIntType::Unsuffixed) if cast_to.is_integral() => {
                lint_unnecessary_cast(cx, expr, literal_str, cast_from, cast_to);
                return false;
            },
            LitKind::Float(_, LitFloatType::Unsuffixed) if cast_to.is_floating_point() => {
                lint_unnecessary_cast(cx, expr, literal_str, cast_from, cast_to);
                return false;
            },
            LitKind::Int(_, LitIntType::Signed(_) | LitIntType::Unsigned(_))
            | LitKind::Float(_, LitFloatType::Suffixed(_))
                if cast_from.kind() == cast_to.kind() =>
            {
                if let Some(src) = cast_expr.span.get_source_text(cx)
                    && let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node)
                {
                    lint_unnecessary_cast(cx, expr, num_lit.integer, cast_from, cast_to);
                    return true;
                }
            },
            _ => {},
        }
    }

    if cast_from.kind() == cast_to.kind() && !expr.span.in_external_macro(cx.sess().source_map()) {
        if let Some(id) = path_to_local(cast_expr)
            && !cx.tcx.hir_span(id).eq_ctxt(cast_expr.span)
        {
            // Binding context is different than the identifiers context.
            // Weird macro wizardry could be involved here.
            return false;
        }

        // If the whole cast expression is a unary expression (`(*x as T)`) or an addressof
        // expression (`(&x as T)`), then not surrounding the suggestion into a block risks us
        // changing the precedence of operators if the cast expression is followed by an operation
        // with higher precedence than the unary operator (`(*x as T).foo()` would become
        // `*x.foo()`, which changes what the `*` applies on).
        // The same is true if the expression encompassing the cast expression is a unary
        // expression or an addressof expression.
        let needs_block = matches!(cast_expr.kind, ExprKind::Unary(..) | ExprKind::AddrOf(..))
            || get_parent_expr(cx, expr).is_some_and(|e| matches!(e.kind, ExprKind::Unary(..) | ExprKind::AddrOf(..)));

        span_lint_and_sugg(
            cx,
            UNNECESSARY_CAST,
            expr.span,
            format!("casting to the same type is unnecessary (`{cast_from}` -> `{cast_to}`)"),
            "try",
            if needs_block {
                format!("{{ {cast_str} }}")
            } else {
                cast_str
            },
            Applicability::MachineApplicable,
        );
        return true;
    }

    false
}

fn lint_unnecessary_cast(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    raw_literal_str: &str,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
) {
    let literal_kind_name = if cast_from.is_integral() { "integer" } else { "float" };
    // first we remove all matches so `-(1)` become `-1`, and remove trailing dots, so `1.` become `1`
    let literal_str = raw_literal_str
        .replace(['(', ')'], "")
        .trim_end_matches('.')
        .to_string();
    // we know need to check if the parent is a method call, to add parenthesis accordingly (eg:
    // (-1).foo() instead of -1.foo())
    let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr)
        && let ExprKind::MethodCall(..) = parent_expr.kind
        && literal_str.starts_with('-')
    {
        format!("({literal_str}_{cast_to})")
    } else {
        format!("{literal_str}_{cast_to}")
    };

    span_lint_and_sugg(
        cx,
        UNNECESSARY_CAST,
        expr.span,
        format!("casting {literal_kind_name} literal to `{cast_to}` is unnecessary"),
        "try",
        sugg,
        Applicability::MachineApplicable,
    );
}

fn get_numeric_literal<'e>(expr: &'e Expr<'e>) -> Option<&'e Lit> {
    match expr.kind {
        ExprKind::Lit(lit) => Some(lit),
        ExprKind::Unary(UnOp::Neg, e) => {
            if let ExprKind::Lit(lit) = e.kind {
                Some(lit)
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Returns the mantissa bits wide of a fp type.
/// Will return 0 if the type is not a fp
fn fp_ty_mantissa_nbits(typ: Ty<'_>) -> u32 {
    match typ.kind() {
        ty::Float(FloatTy::F32) => 23,
        ty::Float(FloatTy::F64) | ty::Infer(InferTy::FloatVar(_)) => 52,
        _ => 0,
    }
}

/// Finds whether an `Expr` returns a type alias.
///
/// TODO: Maybe we should move this to `clippy_utils` so others won't need to go down this dark,
/// dark path reimplementing this (or something similar).
fn is_cast_from_ty_alias<'tcx>(cx: &LateContext<'tcx>, expr: impl Visitable<'tcx>, cast_from: Ty<'tcx>) -> bool {
    for_each_expr_without_closures(expr, |expr| {
        // Calls are a `Path`, and usage of locals are a `Path`. So, this checks
        // - call() as i32
        // - local as i32
        if let ExprKind::Path(qpath) = expr.kind {
            let res = cx.qpath_res(&qpath, expr.hir_id);
            // Function call
            if let Res::Def(DefKind::Fn, def_id) = res {
                let Some(snippet) = cx.tcx.def_span(def_id).get_source_text(cx) else {
                    return ControlFlow::Continue(());
                };
                // This is the worst part of this entire function. This is the only way I know of to
                // check whether a function returns a type alias. Sure, you can get the return type
                // from a function in the current crate as an hir ty, but how do you get it for
                // external functions?? Simple: It's impossible. So, we check whether a part of the
                // function's declaration snippet is exactly equal to the `Ty`. That way, we can
                // see whether it's a type alias.
                //
                // FIXME: This won't work if the type is given an alias through `use`, should we
                // consider this a type alias as well?
                if !snippet
                    .split("->")
                    .skip(1)
                    .any(|s| snippet_eq_ty(s, cast_from) || s.split("where").any(|ty| snippet_eq_ty(ty, cast_from)))
                {
                    return ControlFlow::Break(());
                }
            // Local usage
            } else if let Res::Local(hir_id) = res
                && let Node::LetStmt(l) = cx.tcx.parent_hir_node(hir_id)
            {
                if let Some(e) = l.init
                    && is_cast_from_ty_alias(cx, e, cast_from)
                {
                    return ControlFlow::Break::<()>(());
                }

                if let Some(ty) = l.ty
                    && let TyKind::Path(qpath) = ty.kind
                    && is_ty_alias(&qpath)
                {
                    return ControlFlow::Break::<()>(());
                }
            }
        }

        ControlFlow::Continue(())
    })
    .is_some()
}

fn snippet_eq_ty(snippet: &str, ty: Ty<'_>) -> bool {
    snippet.trim() == ty.to_string() || snippet.trim().contains(&format!("::{ty}"))
}
