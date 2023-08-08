use crate::lints::CStringPtr;
use crate::LateContext;
use crate::LateLintPass;
use crate::LintContext;
use rustc_hir::{Expr, ExprKind};
use rustc_middle::ty;
use rustc_span::{symbol::sym, Span};

declare_lint! {
    /// The `temporary_cstring_as_ptr` lint detects getting the inner pointer of
    /// a temporary `CString`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # #![allow(unused)]
    /// # use std::ffi::CString;
    /// let c_str = CString::new("foo").unwrap().as_ptr();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The inner pointer of a `CString` lives only as long as the `CString` it
    /// points to. Getting the inner pointer of a *temporary* `CString` allows the `CString`
    /// to be dropped at the end of the statement, as it is not being referenced as far as the typesystem
    /// is concerned. This means outside of the statement the pointer will point to freed memory, which
    /// causes undefined behavior if the pointer is later dereferenced.
    pub TEMPORARY_CSTRING_AS_PTR,
    Warn,
    "detects getting the inner pointer of a temporary `CString`"
}

declare_lint_pass!(TemporaryCStringAsPtr => [TEMPORARY_CSTRING_AS_PTR]);

impl<'tcx> LateLintPass<'tcx> for TemporaryCStringAsPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::MethodCall(as_ptr_path, as_ptr_receiver, ..) = expr.kind
            && as_ptr_path.ident.name == sym::as_ptr
            && let ExprKind::MethodCall(unwrap_path, unwrap_receiver, ..) = as_ptr_receiver.kind
            && (unwrap_path.ident.name == sym::unwrap || unwrap_path.ident.name == sym::expect)
        {
            lint_cstring_as_ptr(cx, as_ptr_path.ident.span, unwrap_receiver, as_ptr_receiver);
        }
    }
}

fn lint_cstring_as_ptr(
    cx: &LateContext<'_>,
    as_ptr_span: Span,
    source: &rustc_hir::Expr<'_>,
    unwrap: &rustc_hir::Expr<'_>,
) {
    let source_type = cx.typeck_results().expr_ty(source);
    if let ty::Adt(def, args) = source_type.kind() {
        if cx.tcx.is_diagnostic_item(sym::Result, def.did()) {
            if let ty::Adt(adt, _) = args.type_at(0).kind() {
                if cx.tcx.is_diagnostic_item(sym::cstring_type, adt.did()) {
                    cx.emit_spanned_lint(
                        TEMPORARY_CSTRING_AS_PTR,
                        as_ptr_span,
                        CStringPtr { as_ptr: as_ptr_span, unwrap: unwrap.span },
                    );
                }
            }
        }
    }
}
