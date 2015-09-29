use rustc::lint::*;
use rustc_front::hir::*;
use utils::span_lint;
use rustc::middle::ty::{TypeAndMut, TypeVariants, MethodCall};

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
        match e.node {
            ExprCall(ref fn_expr, ref arguments) => {
                let borrowed_table = cx.tcx.tables.borrow();
                let funtion_type = match borrowed_table.node_types.get(&fn_expr.id) {
                    Some(funtion_type) => funtion_type,
                    None => unreachable!(), // A function with unknown type is called.
                                            // If this happened the compiler would have aborted the
                                            // compilation long ago.
                };
                if let TypeVariants::TyBareFn(_, ref b) = funtion_type.sty {
                    let parameters = b.sig.skip_binder().inputs.clone();
                    for (argument, parameter) in arguments.iter().zip(parameters.iter()) {
                        match parameter.sty {
                            TypeVariants::TyRef(_, TypeAndMut {ty: _, mutbl: MutImmutable}) | 
                            TypeVariants::TyRawPtr(TypeAndMut {ty: _, mutbl: MutImmutable}) => {
                                if let Expr_::ExprAddrOf(MutMutable, _) = argument.node {
                                    if let ExprPath(_, path) = fn_expr.node.clone() {
                                        span_lint(cx, UNNECESSARY_MUT_PASSED, 
                                                  argument.span, &format!("This argument of the \
                                                  function \"{}\" doesn't need to be mutable", path));
                                    }
                                }
                            },
                            _ => {}
                        }
                    }
                }
            },
            ExprMethodCall(ref name, _, ref arguments) => {
                let method_call = MethodCall::expr(e.id);
                let borrowed_table = cx.tcx.tables.borrow();
                let method_type = match borrowed_table.method_map.get(&method_call) {
                    Some(method_type) => method_type,
                    None => unreachable!(), // Just like above, this should never happen.
                };
                if let TypeVariants::TyBareFn(_, ref b) = method_type.ty.sty {
                    let parameters = b.sig.skip_binder().inputs.iter().clone();
                    for (argument, parameter) in arguments.iter().zip(parameters).skip(1) {
                        // Skip the first argument and the first parameter because it is the
                        // struct the function is called on.
                        match parameter.sty {
                            TypeVariants::TyRef(_, TypeAndMut {ty: _, mutbl: MutImmutable}) |
                            TypeVariants::TyRawPtr(TypeAndMut {ty: _, mutbl: MutImmutable}) => {
                                if let Expr_::ExprAddrOf(MutMutable, _) = argument.node {
                                    span_lint(cx, UNNECESSARY_MUT_PASSED, 
                                              argument.span, &format!("This argument of the \
                                              method \"{}\" doesn't need to be mutable", 
                                              name.node.as_str()));
                                }
                            },
                            _ => {}
                        }
                    }
                }
            },
            _ => {}
        }
    }
}
