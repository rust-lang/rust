use clippy_config::msrvs::{self, Msrv};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::in_constant;
use clippy_utils::source::{snippet_opt, snippet_with_applicability};
use clippy_utils::ty::is_isize_or_usize;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, FloatTy, Ty};

use super::{utils, CAST_LOSSLESS};

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cast_op: &Expr<'_>,
    cast_from: Ty<'_>,
    cast_to: Ty<'_>,
    cast_to_hir: &rustc_hir::Ty<'_>,
    msrv: &Msrv,
) {
    if !should_lint(cx, expr, cast_from, cast_to, msrv) {
        return;
    }

    // The suggestion is to use a function call, so if the original expression
    // has parens on the outside, they are no longer needed.
    let mut app = Applicability::MachineApplicable;
    let opt = snippet_opt(cx, cast_op.span.source_callsite());
    let sugg = opt.as_ref().map_or_else(
        || {
            app = Applicability::HasPlaceholders;
            ".."
        },
        |snip| {
            if should_strip_parens(cast_op, snip) {
                &snip[1..snip.len() - 1]
            } else {
                snip.as_str()
            }
        },
    );

    // Display the type alias instead of the aliased type. Fixes #11285
    //
    // FIXME: Once `lazy_type_alias` is stabilized(?) we should use `rustc_middle` types instead,
    // this will allow us to display the right type with `cast_from` as well.
    let cast_to_fmt = if let TyKind::Path(QPath::Resolved(None, path)) = cast_to_hir.kind
        // It's a bit annoying but the turbofish is optional for types. A type in an `as` cast
        // shouldn't have these if they're primitives, which are the only things we deal with.
        //
        // This could be removed for performance if this check is determined to have a pretty major
        // effect.
        && path.segments.iter().all(|segment| segment.args.is_none())
    {
        snippet_with_applicability(cx, cast_to_hir.span, "..", &mut app)
    } else {
        cast_to.to_string().into()
    };

    let message = if cast_from.is_bool() {
        format!("casting `{cast_from}` to `{cast_to_fmt}` is more cleanly stated with `{cast_to_fmt}::from(_)`")
    } else {
        format!("casting `{cast_from}` to `{cast_to_fmt}` may become silently lossy if you later change the type")
    };

    span_lint_and_sugg(
        cx,
        CAST_LOSSLESS,
        expr.span,
        &message,
        "try",
        format!("{cast_to_fmt}::from({sugg})"),
        app,
    );
}

fn should_lint(cx: &LateContext<'_>, expr: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>, msrv: &Msrv) -> bool {
    // Do not suggest using From in consts/statics until it is valid to do so (see #2267).
    if in_constant(cx, expr.hir_id) {
        return false;
    }

    match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, true) => {
            let cast_signed_to_unsigned = cast_from.is_signed() && !cast_to.is_signed();
            let from_nbits = utils::int_ty_to_nbits(cast_from, cx.tcx);
            let to_nbits = utils::int_ty_to_nbits(cast_to, cx.tcx);
            !is_isize_or_usize(cast_from)
                && !is_isize_or_usize(cast_to)
                && from_nbits < to_nbits
                && !cast_signed_to_unsigned
        },

        (true, false) => {
            let from_nbits = utils::int_ty_to_nbits(cast_from, cx.tcx);
            let to_nbits = if let ty::Float(FloatTy::F32) = cast_to.kind() {
                32
            } else {
                64
            };
            !is_isize_or_usize(cast_from) && from_nbits < to_nbits
        },
        (false, true) if matches!(cast_from.kind(), ty::Bool) && msrv.meets(msrvs::FROM_BOOL) => true,
        (_, _) => {
            matches!(cast_from.kind(), ty::Float(FloatTy::F32)) && matches!(cast_to.kind(), ty::Float(FloatTy::F64))
        },
    }
}

fn should_strip_parens(cast_expr: &Expr<'_>, snip: &str) -> bool {
    if let ExprKind::Binary(_, _, _) = cast_expr.kind {
        if snip.starts_with('(') && snip.ends_with(')') {
            return true;
        }
    }
    false
}
