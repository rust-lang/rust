use crate::LateContext;
use crate::LateLintPass;
use crate::LintContext;
use rustc_hir::{Expr, ExprKind};
use rustc_middle::ty;
use rustc_span::{
    symbol::{sym, Symbol, SymbolStr},
    ExpnKind, Span,
};

declare_lint! {
    pub TEMPORARY_CSTRING_AS_PTR,
    Deny,
    "detects getting the inner pointer of a temporary `CString`"
}

declare_lint_pass!(TemporaryCStringAsPtr => [TEMPORARY_CSTRING_AS_PTR]);

/// Returns the method names and argument list of nested method call expressions that make up
/// `expr`. method/span lists are sorted with the most recent call first.
pub fn method_calls<'tcx>(
    expr: &'tcx Expr<'tcx>,
    max_depth: usize,
) -> (Vec<Symbol>, Vec<&'tcx [Expr<'tcx>]>, Vec<Span>) {
    let mut method_names = Vec::with_capacity(max_depth);
    let mut arg_lists = Vec::with_capacity(max_depth);
    let mut spans = Vec::with_capacity(max_depth);

    let mut current = expr;
    for _ in 0..max_depth {
        if let ExprKind::MethodCall(path, span, args, _) = &current.kind {
            if args.iter().any(|e| e.span.from_expansion()) {
                break;
            }
            method_names.push(path.ident.name);
            arg_lists.push(&**args);
            spans.push(*span);
            current = &args[0];
        } else {
            break;
        }
    }

    (method_names, arg_lists, spans)
}

fn in_macro(span: Span) -> bool {
    if span.from_expansion() {
        !matches!(span.ctxt().outer_expn_data().kind, ExpnKind::Desugaring(..))
    } else {
        false
    }
}

impl<'tcx> LateLintPass<'tcx> for TemporaryCStringAsPtr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_macro(expr.span) {
            return;
        }

        let (method_names, arg_lists, _) = method_calls(expr, 2);
        let method_names: Vec<SymbolStr> = method_names.iter().map(|s| s.as_str()).collect();
        let method_names: Vec<&str> = method_names.iter().map(|s| &**s).collect();

        if let ["as_ptr", "unwrap" | "expect"] = method_names.as_slice() {
            lint_cstring_as_ptr(cx, expr, &arg_lists[1][0], &arg_lists[0][0]);
        }
    }
}

fn lint_cstring_as_ptr(
    cx: &LateContext<'_>,
    expr: &rustc_hir::Expr<'_>,
    source: &rustc_hir::Expr<'_>,
    unwrap: &rustc_hir::Expr<'_>,
) {
    let source_type = cx.typeck_results().expr_ty(source);
    if let ty::Adt(def, substs) = source_type.kind {
        if cx.tcx.is_diagnostic_item(Symbol::intern("result_type"), def.did) {
            if let ty::Adt(adt, _) = substs.type_at(0).kind {
                let path = [
                    sym::std,
                    Symbol::intern("ffi"),
                    Symbol::intern("c_str"),
                    Symbol::intern("CString"),
                ];
                if cx.match_def_path(adt.did, &path) {
                    cx.struct_span_lint(TEMPORARY_CSTRING_AS_PTR, expr.span, |diag| {
                        let mut diag = diag
                            .build("you are getting the inner pointer of a temporary `CString`");
                        diag.note("that pointer will be invalid outside this expression");
                        diag.span_help(
                            unwrap.span,
                            "assign the `CString` to a variable to extend its lifetime",
                        );
                        diag.emit();
                    });
                }
            }
        }
    }
}
