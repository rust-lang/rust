use rustc_abi::ExternAbi;
use rustc_hir::intravisit::{FnKind, Visitor};
use rustc_hir::{Expr, ExprKind};
use rustc_lint_defs::{declare_lint, declare_lint_pass};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

use crate::{LateContext, LateLintPass, LintContext, diagnostics};

declare_lint! {
    /// The `cmse_uninitialized_leaks` lint detects types that contain a `union` that cross a
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
    /// warning: value crossing a secure boundary contains a union which may leak secure information
    ///   --> lint_example.rs:2:5
    ///    |
    ///  2 |     MaybeUninit::uninit()
    ///    |     ^^^^^^^^^^^^^^^^^^^^^
    ///    |
    ///    = note: union values can have variant-dependent padding that may contain stale secure data
    ///    = note: `#[warn(cmse_uninitialized_leaks)]` on by default
    /// ```
    ///
    /// ### Explanation
    ///
    /// The cmse calling conventions clear unused registers, to prevent stale secure information
    /// from leaking into the non-secure application. Padding in both `struct`s and `enum`s is
    /// similarly cleared. But for `union` values the compiler does not know what byte ranges are
    /// padding (and may hold stale secure data) and what byte ranges are live data.
    ///
    /// This lint fires when a type that is or contains a `union` crosses the secure boundary:
    ///
    /// - when returned from a `cmse-nonsecure-entry` function
    /// - when passed as an argument to a `cmse-nonsecure-call` function
    pub CMSE_UNINITIALIZED_LEAKS,
    Warn,
    "value containing a union may leak secure information"
}

declare_lint_pass!(CmseUninitializedLeaks => [CMSE_UNINITIALIZED_LEAKS]);

impl<'tcx> LateLintPass<'tcx> for CmseUninitializedLeaks {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        check_cmse_call_call(cx, expr);
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        _: &'tcx rustc_hir::FnDecl<'tcx>,
        body: &'tcx rustc_hir::Body<'tcx>,
        _: rustc_span::Span,
        def_id: rustc_span::def_id::LocalDefId,
    ) {
        check_cmse_entry_return(cx, fn_kind, body, def_id);
    }
}

fn check_cmse_call_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    let ExprKind::Call(callee, arguments) = expr.kind else {
        return;
    };

    // Determine the callee ABI.
    let callee_ty = cx.typeck_results().expr_ty_adjusted(callee);
    let ty::FnPtr(sig_tys, header) = callee_ty.kind() else { return };
    if header.abi() != ExternAbi::CmseNonSecureCall {
        return;
    }

    let sig_tys = cx.tcx.instantiate_bound_regions_with_erased(*sig_tys);
    for (arg, ty) in arguments.iter().zip(sig_tys.inputs()) {
        // A function pointer type cannot contain `impl Trait` parameters.
        if ty.has_opaque_types() {
            continue;
        }

        if contains_union(cx, *ty) {
            // Some part of the source type may be uninitialized.
            cx.emit_span_lint(
                CMSE_UNINITIALIZED_LEAKS,
                arg.span,
                diagnostics::CmseUninitializedMayLeakInformation,
            );
        }
    }
}

fn check_cmse_entry_return<'tcx>(
    cx: &LateContext<'tcx>,
    fn_kind: FnKind<'tcx>,
    body: &'tcx rustc_hir::Body<'tcx>,
    def_id: rustc_span::def_id::LocalDefId,
) {
    use rustc_trait_selection::error_reporting::traits::suggestions::ReturnsVisitor;

    let abi = match fn_kind {
        FnKind::ItemFn(_, _, header) => header.abi,
        FnKind::Method(_, fn_sig) => fn_sig.header.abi,
        FnKind::Closure => return,
    };

    if !matches!(abi, ExternAbi::CmseNonSecureEntry) {
        return;
    }

    let sig = cx.tcx.fn_sig(def_id).instantiate_identity();
    let output = sig.map(|sig| cx.tcx.instantiate_bound_regions_with_erased(sig).output());
    let Ok(return_type) = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), output) else {
        return;
    };

    if return_type.has_opaque_types() || !contains_union(cx, return_type) {
        return;
    }

    let mut visitor = ReturnsVisitor::default();
    visitor.visit_body(body);

    for ret in visitor.returns {
        if cx.typeck_results().expr_ty(ret).is_never() {
            continue;
        }
        cx.tcx.emit_node_span_lint(
            CMSE_UNINITIALIZED_LEAKS,
            ret.hir_id,
            ret.span,
            diagnostics::CmseUninitializedMayLeakInformation,
        );
    }
}

/// Does `ty` contain a union?
fn contains_union<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> bool {
    let tcx = cx.tcx;

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
                match tcx.try_normalize_erasing_regions(cx.typing_env(), field.ty(tcx, *args)) {
                    Ok(field_ty) => contains_union(cx, field_ty),
                    Err(_) => false,
                }
            })
        }

        ty::Tuple(elems) => {
            // Element types might contain unions.
            //
            // Passing a tuple already triggers `improper_ctypes`.
            elems.iter().any(|elem| contains_union(cx, elem))
        }
        ty::Array(elem, _) => {
            // The element type might contain unions.
            //
            // Passing an array already triggers `improper_ctypes`.
            contains_union(cx, *elem)
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
