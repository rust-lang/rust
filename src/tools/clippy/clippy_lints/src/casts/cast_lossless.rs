use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_in_const_context;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_isize_or_usize;
use rustc_errors::Applicability;
use rustc_hir::{Expr, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, FloatTy, Ty};
use rustc_span::hygiene;

use super::{CAST_LOSSLESS, utils};

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_from_expr: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
    cast_to_hir: &rustc_hir::Ty<'_>,
    msrv: Msrv,
) {
    if !should_lint(cx, cast_from, cast_to, msrv) {
        return;
    }

    span_lint_and_then(
        cx,
        CAST_LOSSLESS,
        expr.span,
        format!("casts from `{cast_from}` to `{cast_to}` can be expressed infallibly using `From`"),
        |diag| {
            diag.help("an `as` cast can become silently lossy if the types change in the future");
            let mut applicability = Applicability::MachineApplicable;
            let from_sugg = Sugg::hir_with_context(cx, cast_from_expr, expr.span.ctxt(), "<from>", &mut applicability);
            let Some(ty) = hygiene::walk_chain(cast_to_hir.span, expr.span.ctxt()).get_source_text(cx) else {
                return;
            };
            match cast_to_hir.kind {
                TyKind::Infer(()) => {
                    diag.span_suggestion_verbose(
                        expr.span,
                        "use `Into::into` instead",
                        format!("{}.into()", from_sugg.maybe_paren()),
                        applicability,
                    );
                },
                // Don't suggest `A<_>::B::From(x)` or `macro!()::from(x)`
                kind if matches!(kind, TyKind::Path(QPath::Resolved(_, path)) if path.segments.iter().any(|s| s.args.is_some()))
                    || !cast_to_hir.span.eq_ctxt(expr.span) =>
                {
                    diag.span_suggestion_verbose(
                        expr.span,
                        format!("use `<{ty}>::from` instead"),
                        format!("<{ty}>::from({from_sugg})"),
                        applicability,
                    );
                },
                _ => {
                    diag.span_suggestion_verbose(
                        expr.span,
                        format!("use `{ty}::from` instead"),
                        format!("{ty}::from({from_sugg})"),
                        applicability,
                    );
                },
            }
        },
    );
}

fn should_lint(cx: &LateContext<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>, msrv: Msrv) -> bool {
    // Do not suggest using From in consts/statics until it is valid to do so (see #2267).
    if is_in_const_context(cx) {
        return false;
    }

    match (
        utils::int_ty_to_nbits(cx.tcx, cast_from),
        utils::int_ty_to_nbits(cx.tcx, cast_to),
    ) {
        (Some(from_nbits), Some(to_nbits)) => {
            let cast_signed_to_unsigned = cast_from.is_signed() && !cast_to.is_signed();
            !is_isize_or_usize(cast_from)
                && !is_isize_or_usize(cast_to)
                && from_nbits < to_nbits
                && !cast_signed_to_unsigned
        },

        (Some(from_nbits), None) => {
            // FIXME: handle `f16` and `f128`
            let to_nbits = if let ty::Float(FloatTy::F32) = cast_to.kind() {
                32
            } else {
                64
            };
            !is_isize_or_usize(cast_from) && from_nbits < to_nbits
        },
        (None, Some(_)) if cast_from.is_bool() && msrv.meets(cx, msrvs::FROM_BOOL) => true,
        _ => matches!(cast_from.kind(), ty::Float(FloatTy::F32)) && matches!(cast_to.kind(), ty::Float(FloatTy::F64)),
    }
}
