use crate::LateContext;
use crate::LateLintPass;
use crate::LintContext;
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
) -> Option<(&'tcx PathSegment<'tcx>, &'tcx [Expr<'tcx>])> {
    if let ExprKind::MethodCall(path, args, _) = &expr.kind {
        if args.iter().any(|e| e.span.from_expansion()) { None } else { Some((path, *args)) }
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
            Some((path, args)) if path.ident.name == sym::as_ptr => {
                let unwrap_arg = &args[0];
                let as_ptr_span = path.ident.span;
                match first_method_call(unwrap_arg) {
                    Some((path, args))
                        if path.ident.name == sym::unwrap || path.ident.name == sym::expect =>
                    {
                        let source_arg = &args[0];
                        lint_cstring_as_ptr(cx, as_ptr_span, source_arg, unwrap_arg);
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
        if cx.tcx.is_diagnostic_item(sym::Result, def.did) {
            if let ty::Adt(adt, _) = substs.type_at(0).kind() {
                if cx.tcx.is_diagnostic_item(sym::cstring_type, adt.did) {
                    cx.struct_span_lint(TEMPORARY_CSTRING_AS_PTR, as_ptr_span, |diag| {
                        let mut diag = diag
                            .build("getting the inner pointer of a temporary `CString`");
                        diag.span_label(as_ptr_span, "this pointer will be invalid");
                        diag.span_label(
                            unwrap.span,
                            "this `CString` is deallocated at the end of the statement, bind it to a variable to extend its lifetime",
                        );
                        diag.note("pointers do not have a lifetime; when calling `as_ptr` the `CString` will be deallocated at the end of the statement because nothing is referencing it as far as the type system is concerned");
                        diag.help("for more information, see https://doc.rust-lang.org/reference/destructors.html");
                        diag.emit();
                    });
                }
            }
        }
    }
}
