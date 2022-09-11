use crate::LateContext;
use crate::LateLintPass;
use crate::LintContext;
use rustc_errors::fluent;
use rustc_hir::{Expr, ExprKind, PathSegment};
use rustc_middle::ty;
use rustc_span::{symbol::sym, ExpnKind, Span};

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

fn in_macro(span: Span) -> bool {
    if span.from_expansion() {
        !matches!(span.ctxt().outer_expn_data().kind, ExpnKind::Desugaring(..))
    } else {
        false
    }
}

fn first_method_call<'tcx>(
    expr: &'tcx Expr<'tcx>,
) -> Option<(&'tcx PathSegment<'tcx>, &'tcx Expr<'tcx>)> {
    if let ExprKind::MethodCall(path, receiver, args, ..) = &expr.kind {
        if args.iter().any(|e| e.span.from_expansion()) || receiver.span.from_expansion() {
            None
        } else {
            Some((path, *receiver))
        }
    } else {
        None
    }
}

impl<'tcx> LateLintPass<'tcx> for TemporaryCStringAsPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_macro(expr.span) {
            return;
        }

        match first_method_call(expr) {
            Some((path, unwrap_arg)) if path.ident.name == sym::as_ptr => {
                let as_ptr_span = path.ident.span;
                match first_method_call(unwrap_arg) {
                    Some((path, receiver))
                        if path.ident.name == sym::unwrap || path.ident.name == sym::expect =>
                    {
                        lint_cstring_as_ptr(cx, as_ptr_span, receiver, unwrap_arg);
                    }
                    _ => return,
                }
            }
            _ => return,
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
    if let ty::Adt(def, substs) = source_type.kind() {
        if cx.tcx.is_diagnostic_item(sym::Result, def.did()) {
            if let ty::Adt(adt, _) = substs.type_at(0).kind() {
                if cx.tcx.is_diagnostic_item(sym::cstring_type, adt.did()) {
                    cx.struct_span_lint(TEMPORARY_CSTRING_AS_PTR, as_ptr_span, |diag| {
                        diag.build(fluent::lint::cstring_ptr)
                            .span_label(as_ptr_span, fluent::lint::as_ptr_label)
                            .span_label(unwrap.span, fluent::lint::unwrap_label)
                            .note(fluent::lint::note)
                            .help(fluent::lint::help)
                            .emit();
                    });
                }
            }
        }
    }
}
