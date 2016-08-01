use rustc::lint::*;
use rustc::ty::{TypeAndMut, TypeVariants, MethodCall, TyS};
use rustc::hir::*;
use syntax::ptr::P;
use utils::span_lint;

/// **What it does:** This lint detects giving a mutable reference to a function that only requires
/// an immutable reference.
///
/// **Why is this bad?** The immutable reference rules out all other references to the value. Also
/// the code misleads about the intent of the call site.
///
/// **Known problems:** None
///
/// **Example**
/// ```rust
/// my_vec.push(&mut value)
/// ```
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
                let function_type = borrowed_table.node_types
                                                  .get(&fn_expr.id)
                                                  .expect("A function with an unknown type is called. \
                                                           If this happened, the compiler would have \
                                                           aborted the compilation long ago");
                if let ExprPath(_, ref path) = fn_expr.node {
                    check_arguments(cx, arguments, function_type, &path.to_string());
                }
            }
            ExprMethodCall(ref name, _, ref arguments) => {
                let method_call = MethodCall::expr(e.id);
                let method_type = borrowed_table.method_map.get(&method_call).expect("This should never happen.");
                check_arguments(cx, arguments, method_type.ty, &name.node.as_str())
            }
            _ => (),
        }
    }
}

fn check_arguments(cx: &LateContext, arguments: &[P<Expr>], type_definition: &TyS, name: &str) {
    match type_definition.sty {
        TypeVariants::TyFnDef(_, _, fn_type) |
        TypeVariants::TyFnPtr(fn_type) => {
            let parameters = &fn_type.sig.skip_binder().inputs;
            for (argument, parameter) in arguments.iter().zip(parameters.iter()) {
                match parameter.sty {
                    TypeVariants::TyRef(_, TypeAndMut { mutbl: MutImmutable, .. }) |
                    TypeVariants::TyRawPtr(TypeAndMut { mutbl: MutImmutable, .. }) => {
                        if let ExprAddrOf(MutMutable, _) = argument.node {
                            span_lint(cx,
                                      UNNECESSARY_MUT_PASSED,
                                      argument.span,
                                      &format!("The function/method \"{}\" doesn't need a mutable reference", name));
                        }
                    }
                    _ => (),
                }
            }
        }
        _ => (),
    }
}
