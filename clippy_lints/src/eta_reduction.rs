use rustc::lint::*;
use rustc::ty;
use rustc::hir::*;
use utils::{snippet_opt, span_lint_and_then, is_adjusted, iter_input_pats};

#[allow(missing_copy_implementations)]
pub struct EtaPass;


/// **What it does:** Checks for closures which just call another function where
/// the function can be called directly. `unsafe` functions or calls where types
/// get adjusted are ignored.
///
/// **Why is this bad?** Needlessly creating a closure adds code for no benefit
/// and gives the optimizer more work.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// xs.map(|x| foo(x))
/// ```
/// where `foo(_)` is a plain function that takes the exact argument type of
/// `x`.
declare_lint! {
    pub REDUNDANT_CLOSURE,
    Warn,
    "redundant closures, i.e. `|a| foo(a)` (which can be written as just `foo`)"
}

impl LintPass for EtaPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(REDUNDANT_CLOSURE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for EtaPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        match expr.node {
            ExprCall(_, ref args) |
            ExprMethodCall(_, _, ref args) => {
                for arg in args {
                    check_closure(cx, arg)
                }
            },
            _ => (),
        }
    }
}

fn check_closure(cx: &LateContext, expr: &Expr) {
    if let ExprClosure(_, ref decl, eid, _, _) = expr.node {
        let body = cx.tcx.hir.body(eid);
        let ex = &body.value;
        if let ExprCall(ref caller, ref args) = ex.node {
            if args.len() != decl.inputs.len() {
                // Not the same number of arguments, there
                // is no way the closure is the same as the function
                return;
            }
            if is_adjusted(cx, ex) || args.iter().any(|arg| is_adjusted(cx, arg)) {
                // Are the expression or the arguments type-adjusted? Then we need the closure
                return;
            }
            let fn_ty = cx.tables.expr_ty(caller);
            match fn_ty.sty {
                // Is it an unsafe function? They don't implement the closure traits
                ty::TyFnDef(..) | ty::TyFnPtr(_) => {
                    let sig = fn_ty.fn_sig(cx.tcx);
                    if sig.skip_binder().unsafety == Unsafety::Unsafe || sig.skip_binder().output().sty == ty::TyNever {
                        return;
                    }
                },
                _ => (),
            }
            for (a1, a2) in iter_input_pats(decl, body).zip(args) {
                if let PatKind::Binding(_, _, ident, _) = a1.pat.node {
                    // XXXManishearth Should I be checking the binding mode here?
                    if let ExprPath(QPath::Resolved(None, ref p)) = a2.node {
                        if p.segments.len() != 1 {
                            // If it's a proper path, it can't be a local variable
                            return;
                        }
                        if p.segments[0].name != ident.node {
                            // The two idents should be the same
                            return;
                        }
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            }
            span_lint_and_then(cx, REDUNDANT_CLOSURE, expr.span, "redundant closure found", |db| {
                if let Some(snippet) = snippet_opt(cx, caller.span) {
                    db.span_suggestion(expr.span, "remove closure as shown", snippet);
                }
            });
        }
    }
}
