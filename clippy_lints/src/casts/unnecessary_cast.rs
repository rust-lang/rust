use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::numeric_literal::NumericLiteral;
use clippy_utils::source::snippet_opt;
use if_chain::if_chain;
use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, Lit, QPath, TyKind, UnOp};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, FloatTy, InferTy, Ty};

use super::UNNECESSARY_CAST;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'tcx>,
    cast_expr: &Expr<'tcx>,
    cast_from: Ty<'tcx>,
    cast_to: Ty<'tcx>,
) -> bool {
    // skip non-primitive type cast
    if_chain! {
        if let ExprKind::Cast(_, cast_to) = expr.kind;
        if let TyKind::Path(QPath::Resolved(_, path)) = &cast_to.kind;
        if let Res::PrimTy(_) = path.res;
        then {}
        else {
            return false
        }
    }

    if let Some(lit) = get_numeric_literal(cast_expr) {
        let literal_str = snippet_opt(cx, cast_expr.span).unwrap_or_default();

        if_chain! {
            if let LitKind::Int(n, _) = lit.node;
            if let Some(src) = snippet_opt(cx, cast_expr.span);
            if cast_to.is_floating_point();
            if let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node);
            let from_nbits = 128 - n.leading_zeros();
            let to_nbits = fp_ty_mantissa_nbits(cast_to);
            if from_nbits != 0 && to_nbits != 0 && from_nbits <= to_nbits && num_lit.is_decimal();
            then {
                lint_unnecessary_cast(cx, expr, num_lit.integer, cast_from, cast_to);
                return true
            }
        }

        match lit.node {
            LitKind::Int(_, LitIntType::Unsuffixed) if cast_to.is_integral() => {
                lint_unnecessary_cast(cx, expr, &literal_str, cast_from, cast_to);
            },
            LitKind::Float(_, LitFloatType::Unsuffixed) if cast_to.is_floating_point() => {
                lint_unnecessary_cast(cx, expr, &literal_str, cast_from, cast_to);
            },
            LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed) => {},
            LitKind::Int(_, LitIntType::Signed(_) | LitIntType::Unsigned(_))
            | LitKind::Float(_, LitFloatType::Suffixed(_))
                if cast_from.kind() == cast_to.kind() =>
            {
                if let Some(src) = snippet_opt(cx, cast_expr.span) {
                    if let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node) {
                        lint_unnecessary_cast(cx, expr, num_lit.integer, cast_from, cast_to);
                    }
                }
            },
            _ => {
                if cast_from.kind() == cast_to.kind() && !in_external_macro(cx.sess(), expr.span) {
                    span_lint_and_sugg(
                        cx,
                        UNNECESSARY_CAST,
                        expr.span,
                        &format!(
                            "casting to the same type is unnecessary (`{}` -> `{}`)",
                            cast_from, cast_to
                        ),
                        "try",
                        literal_str,
                        Applicability::MachineApplicable,
                    );
                    return true;
                }
            },
        }
    }

    false
}

fn lint_unnecessary_cast(cx: &LateContext<'_>, expr: &Expr<'_>, literal_str: &str, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    let literal_kind_name = if cast_from.is_integral() { "integer" } else { "float" };
    let replaced_literal;
    let matchless = if literal_str.contains(['(', ')']) {
        replaced_literal = literal_str.replace(['(', ')'], "");
        &replaced_literal
    } else {
        literal_str
    };
    span_lint_and_sugg(
        cx,
        UNNECESSARY_CAST,
        expr.span,
        &format!("casting {} literal to `{}` is unnecessary", literal_kind_name, cast_to),
        "try",
        format!("{}_{}", matchless.trim_end_matches('.'), cast_to),
        Applicability::MachineApplicable,
    );
}

fn get_numeric_literal<'e>(expr: &'e Expr<'e>) -> Option<&'e Lit> {
    match expr.kind {
        ExprKind::Lit(ref lit) => Some(lit),
        ExprKind::Unary(UnOp::Neg, e) => {
            if let ExprKind::Lit(ref lit) = e.kind {
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
