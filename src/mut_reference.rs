use rustc::lint::*;
use rustc_front::hir::*;
use utils::span_lint;
use rustc::middle::ty::{TypeAndMut, TypeVariants, MethodCall, TyS};
use syntax::ptr::P;

declare_lint! {
    pub UNNECESSARY_MUT_PASSED,
    Warn,
    "an argument is passed as a mutable reference although the function/method only demands an \
     immutable reference"
}


#[derive(Copy,Clone)]
pub struct UnnecessaryMutPassed;

impl LintPass for UnnecessaryMutPassed {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNNECESSARY_MUT_PASSED)
    }
}

impl LateLintPass for UnnecessaryMutPassed {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        let borrowed_table = cx.tcx.tables.borrow();
        match e.node {
            ExprCall(ref fn_expr, ref arguments) => {
                match borrowed_table.node_types.get(&fn_expr.id) {
                    Some(function_type) => {
                        if let ExprPath(_, ref path) = fn_expr.node {
                            check_arguments(cx, &arguments, function_type, 
                                            &format!("{}", path));
                        }
                    }
                    None => unreachable!(), // A function with unknown type is called.
                                            // If this happened the compiler would have aborted the
                                            // compilation long ago.
                };


            }
            ExprMethodCall(ref name, _, ref arguments) => {
                let method_call = MethodCall::expr(e.id);
                match borrowed_table.method_map.get(&method_call) {
                    Some(method_type) => check_arguments(cx, &arguments, method_type.ty, 
                                                         &format!("{}", name.node.as_str())),
                    None => unreachable!(), // Just like above, this should never happen.
                };
            }
            _ => {}
        }
    }
}

fn check_arguments(cx: &LateContext, arguments: &[P<Expr>], type_definition: &TyS, name: &str) {
    if let TypeVariants::TyBareFn(_, ref fn_type) = type_definition.sty {
        let parameters = &fn_type.sig.skip_binder().inputs;
        for (argument, parameter) in arguments.iter().zip(parameters.iter()) {
            match parameter.sty {
                TypeVariants::TyRef(_, TypeAndMut {ty: _, mutbl: MutImmutable}) |
                TypeVariants::TyRawPtr(TypeAndMut {ty: _, mutbl: MutImmutable}) => {
                    if let ExprAddrOf(MutMutable, _) = argument.node {
                        span_lint(cx, UNNECESSARY_MUT_PASSED, 
                                  argument.span, &format!("The function/method \"{}\" \
                                  doesn't need a mutable reference", 
                                  name));
                    }
                }
                _ => {}
            }
        }
    }
}
