use rustc_abi::ExternAbi;
use rustc_hir::{self as hir, Expr, ExprKind};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_session::{declare_lint, declare_lint_pass};

use crate::{LateContext, LateLintPass, LintContext, lints};

declare_lint! {
    /// The `cmse_uninitialized_leak` lint detects types that contain a `union` that cross a
    /// secure boundary. Such values may still contain secure data in their padding bytes.
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
    /// warning: this value crossing a secure boundary may contain (partially) uninitialized data which can leak information
    ///   --> lint_example.rs:2:5
    ///    |
    ///  2 |     MaybeUninit::uninit()
    ///    |     ^^^^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = note: enum and union values can have variant-dependent padding that may contain stale secure data
    ///    = note: `#[warn(cmse_uninitialized_leak)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The cmse calling conventions clear unused registers, to prevent stale secure information
    /// from leaking into the non-secure application. Padding in both `struct`s and `enum`s is
    /// similarly cleared. But for `union` values the compiler does not know what byte ranges are
    /// padding (and may hold stale secure data) and what bytes ranges are live data.
    ///
    /// This lint fires when a type that is or contains a `union` crosses the secure boundary:
    ///
    /// - when returned from a `cmse-nonsecure-entry` function
    /// - when passed as an argument to a `cmse-nonsecure-call` function
    pub CMSE_UNINITIALIZED_LEAK,
    Warn,
    "value containing a union may leak secure information"
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
        ty::FnPtr(poly_sig, header) if header.abi() == ExternAbi::CmseNonSecureCall => {
            poly_sig.skip_binder()
        }
        _ => return,
    };

    let fn_sig = cx.tcx.erase_and_anonymize_regions(sig);

    for (arg, ty) in arguments.iter().zip(fn_sig.inputs()) {
        // `impl Trait` is not allowed in the argument types.
        if ty.has_opaque_types() {
            continue;
        }

        if contains_unstable_or_variant_dependent_padding(cx, *ty) {
            // Some part of the source type may be uninitialized.
            cx.emit_span_lint(
                CMSE_UNINITIALIZED_LEAK,
                arg.span,
                lints::CmseUninitializedMayLeakInformation,
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

    if contains_unstable_or_variant_dependent_padding(cx, return_type) {
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

        // Some part of the source type may be uninitialized.
        cx.emit_span_lint(
            CMSE_UNINITIALIZED_LEAK,
            return_expr_span,
            lints::CmseUninitializedMayLeakInformation,
        );
    }
}

/// Traverse `T` for any `union` or `enum`, and check whether it contains any padding that is
/// variant-dependent or unstable.
fn contains_unstable_or_variant_dependent_padding<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
) -> bool {
    let tcx = cx.tcx;

    // Types cross the secure boundary fully monomorphized.
    let typing_env = ty::TypingEnv::fully_monomorphized();

    match ty.kind() {
        ty::Adt(adt_def, args) => {
            if adt_def.is_union() {
                // It is still unclear whether a union value where all fields are equally
                // large and allow the same bit patterns can be considered initialized
                // (see rust-lang/unsafe-code-guidelines#438), so for now we just warn on
                // any union.
                return true;
            }

            // For enums and structs we recurse into the fields.
            // A non-repr(C) struct already triggers `improper_ctypes`.
            adt_def.all_fields().any(|field| {
                let field_ty = tcx.normalize_erasing_regions(typing_env, field.ty(tcx, args));
                contains_unstable_or_variant_dependent_padding(cx, field_ty)
            })
        }

        ty::Tuple(elems) => {
            // Element types might contain unions.
            //
            // Passing a tuple already triggers `improper_ctypes`.
            elems.iter().any(|elem| contains_unstable_or_variant_dependent_padding(cx, elem))
        }
        ty::Array(elem, _) => {
            // The element type might contain unions.
            //
            // Passing an array already triggers `improper_ctypes`.
            contains_unstable_or_variant_dependent_padding(cx, *elem)
        }

        _ => {
            // Other types are either scalar (hence no padding), or behind some kind of indirection.
            //
            // Note that the system traps when dereferencing a pointer to secure memory while in
            // non-secure mode, so passing a value with indirection is useless in practice.
            false
        }
    }
}
