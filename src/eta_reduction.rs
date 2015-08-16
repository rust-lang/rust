use rustc::lint::*;
use syntax::ast::*;
use syntax::print::pprust::expr_to_string;

use utils::span_lint;


#[allow(missing_copy_implementations)]
pub struct EtaPass;


declare_lint!(pub REDUNDANT_CLOSURE, Warn,
              "using redundant closures, i.e. `|a| foo(a)` (which can be written as just `foo`)");

impl LintPass for EtaPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(REDUNDANT_CLOSURE)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        match &expr.node {
            &ExprCall(_, ref args) |
            &ExprMethodCall(_, _, ref args) => {
                for arg in args {
                    check_closure(cx, &*arg)
                }
            },
            _ => (),
        }
    }
}

fn is_adjusted(cx: &Context, e: &Expr) -> bool {
    cx.tcx.tables.borrow().adjustments.get(&e.id).is_some()
}

fn check_closure(cx: &Context, expr: &Expr) {
    if let ExprClosure(_, ref decl, ref blk) = expr.node {
        if !blk.stmts.is_empty() {
            // || {foo(); bar()}; can't be reduced here
            return;
        }
        if let Some(ref ex) = blk.expr {
            if let ExprCall(ref caller, ref args) = ex.node {
                if args.len() != decl.inputs.len() {
                    // Not the same number of arguments, there
                    // is no way the closure is the same as the function
                    return;
                }
                if args.iter().any(|arg| is_adjusted(cx, arg)) { return; }
                for (ref a1, ref a2) in decl.inputs.iter().zip(args) {
                    if let PatIdent(_, ident, _) = a1.pat.node {
                        // XXXManishearth Should I be checking the binding mode here?
                        if let ExprPath(None, ref p) = a2.node {
                            if p.segments.len() != 1 {
                                // If it's a proper path, it can't be a local variable
                                return;
                            }
                            if p.segments[0].identifier != ident.node {
                                // The two idents should be the same
                                return
                            }
                        } else {
                            return
                        }
                    } else {
                        return
                    }
                }
                span_lint(cx, REDUNDANT_CLOSURE, expr.span,
                             &format!("redundant closure found. Consider using `{}` in its place",
                                      expr_to_string(caller))[..])
            }
        }
    }
}
