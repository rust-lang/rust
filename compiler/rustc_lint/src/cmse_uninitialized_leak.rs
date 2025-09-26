use rustc_abi::ExternAbi;
use rustc_hir::{self as hir, Expr, ExprKind};
use rustc_middle::ty::layout::{LayoutCx, TyAndLayout};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::{LateContext, LateLintPass, LintContext, lints};

declare_lint! {
    /// The `cmse_uninitialized_leak` lint detects values that may be (partially) uninitialized that
    /// cross the secure boundary.
    ///
    /// ### Example
    ///
    /// ```rust,ignore (ABI is only supported on thumbv8)
    /// extern "cmse-nonsecure-entry" fn foo() -> MaybeUninit<u64> {
    ///     MaybeUninit::uninit()
    /// }
    /// ```
    ///
    /// This will produce:
    ///
    /// ```text
    /// warning: passing a union across the security boundary may leak information
    ///   --> lint_example.rs:2:5
    ///    |
    ///  2 |     MaybeUninit::uninit()
    ///    |     ^^^^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = note: the bits not used by the current variant may contain stale secure data
    ///    = note: `#[warn(cmse_uninitialized_leak)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The cmse calling conventions normally take care of clearing registers to make sure that
    /// stale secure information is not observable from non-secure code. Uninitialized memory may
    /// still contain secret information, so the programmer must be careful when (partially)
    /// uninitialized values cross the secure boundary. This lint fires when a partially
    /// uninitialized value (e.g. a `union` value or a type with a niche) crosses the secure
    /// boundary, i.e.:
    ///
    /// - when returned from a `cmse-nonsecure-entry` function
    /// - when passed as an argument to a `cmse-nonsecure-call` function
    ///
    /// This lint is a best effort: not all cases of (partially) uninitialized data crossing the
    /// secure boundary are caught.
    pub CMSE_UNINITIALIZED_LEAK,
    Warn,
    "(partially) uninitialized value may leak secure information"
}

declare_lint_pass!(CmseUninitializedLeak => [CMSE_UNINITIALIZED_LEAK]);

impl<'tcx> LateLintPass<'tcx> for CmseUninitializedLeak {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        check_cmse_entry_return(cx, expr);
        check_cmse_call_call(cx, expr);
    }
}

fn check_cmse_call_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    let ExprKind::Call(callee, arguments) = expr.kind else {
        return;
    };

    // Determine the callee ABI.
    let callee_ty = cx.typeck_results().expr_ty(callee);
    let sig = match callee_ty.kind() {
        ty::FnPtr(poly_sig, header) if header.abi == ExternAbi::CmseNonSecureCall => {
            poly_sig.skip_binder()
        }
        _ => return,
    };

    let fn_sig = cx.tcx.erase_and_anonymize_regions(sig);
    let typing_env = ty::TypingEnv::fully_monomorphized();

    for (arg, ty) in arguments.iter().zip(fn_sig.inputs()) {
        // `impl Trait` is not allowed in the argument types.
        if ty.has_opaque_types() {
            continue;
        }

        let Ok(layout) = cx.tcx.layout_of(typing_env.as_query_input(*ty)) else {
            continue;
        };

        if layout_contains_union(cx.tcx, &layout) {
            cx.emit_span_lint(
                CMSE_UNINITIALIZED_LEAK,
                arg.span,
                lints::CmseUnionMayLeakInformation,
            );
        }
    }
}

fn check_cmse_entry_return<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    let owner = cx.tcx.hir_enclosing_body_owner(expr.hir_id);

    match cx.tcx.def_kind(owner) {
        hir::def::DefKind::Fn | hir::def::DefKind::AssocFn => {}
        _ => return,
    }

    // Only continue if the current expr is an (implicit) return.
    let body = cx.tcx.hir_body_owned_by(owner);
    let is_implicit_return = expr.hir_id == body.value.hir_id;
    if !(matches!(expr.kind, ExprKind::Ret(_)) || is_implicit_return) {
        return;
    }

    let sig = cx.tcx.fn_sig(owner).skip_binder();
    if sig.abi() != ExternAbi::CmseNonSecureEntry {
        return;
    }

    let fn_sig = cx.tcx.instantiate_bound_regions_with_erased(sig);
    let fn_sig = cx.tcx.erase_and_anonymize_regions(fn_sig);
    let return_type = fn_sig.output();

    // `impl Trait` is not allowed in the return type.
    if return_type.has_opaque_types() {
        return;
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let Ok(ret_layout) = cx.tcx.layout_of(typing_env.as_query_input(return_type)) else {
        return;
    };

    if layout_contains_union(cx.tcx, &ret_layout) {
        let return_expr_span = if is_implicit_return {
            match expr.kind {
                ExprKind::Block(block, _) => match block.expr {
                    Some(tail) => tail.span,
                    None => expr.span,
                },
                _ => expr.span,
            }
        } else {
            expr.span
        };

        cx.emit_span_lint(
            CMSE_UNINITIALIZED_LEAK,
            return_expr_span,
            lints::CmseUnionMayLeakInformation,
        );
    }
}

/// Check whether any part of the layout is a union, which may contain secure data still.
fn layout_contains_union<'tcx>(tcx: TyCtxt<'tcx>, layout: &TyAndLayout<'tcx>) -> bool {
    if layout.ty.is_union() {
        return true;
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let cx = LayoutCx::new(tcx, typing_env);

    match &layout.variants {
        rustc_abi::Variants::Single { .. } => {
            for i in 0..layout.fields.count() {
                if layout_contains_union(tcx, &layout.field(&cx, i)) {
                    return true;
                }
            }
        }

        rustc_abi::Variants::Multiple { variants, .. } => {
            for (variant_idx, _vdata) in variants.iter_enumerated() {
                let variant_layout = layout.for_variant(&cx, variant_idx);

                for i in 0..variant_layout.fields.count() {
                    if layout_contains_union(tcx, &variant_layout.field(&cx, i)) {
                        return true;
                    }
                }
            }
        }

        rustc_abi::Variants::Empty => {}
    }

    false
}
