use crate::utils::span_lint;
use rustc::declare_lint_pass;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty::subst::Subst;
use rustc::ty::{self, Ty};
use rustc_session::declare_tool_lint;

declare_clippy_lint! {
    /// **What it does:** Detects giving a mutable reference to a function that only
    /// requires an immutable reference.
    ///
    /// **Why is this bad?** The immutable reference rules out all other references
    /// to the value. Also the code misleads about the intent of the call site.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// my_vec.push(&mut value)
    /// ```
    pub UNNECESSARY_MUT_PASSED,
    style,
    "an argument passed as a mutable reference although the callee only demands an immutable reference"
}

declare_lint_pass!(UnnecessaryMutPassed => [UNNECESSARY_MUT_PASSED]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnnecessaryMutPassed {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        match e.kind {
            ExprKind::Call(ref fn_expr, ref arguments) => {
                if let ExprKind::Path(ref path) = fn_expr.kind {
                    check_arguments(
                        cx,
                        arguments,
                        cx.tables.expr_ty(fn_expr),
                        &print::to_string(print::NO_ANN, |s| s.print_qpath(path, false)),
                    );
                }
            },
            ExprKind::MethodCall(ref path, _, ref arguments) => {
                let def_id = cx.tables.type_dependent_def_id(e.hir_id).unwrap();
                let substs = cx.tables.node_substs(e.hir_id);
                let method_type = cx.tcx.type_of(def_id).subst(cx.tcx, substs);
                check_arguments(cx, arguments, method_type, &path.ident.as_str())
            },
            _ => (),
        }
    }
}

fn check_arguments<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, arguments: &[Expr], type_definition: Ty<'tcx>, name: &str) {
    match type_definition.kind {
        ty::FnDef(..) | ty::FnPtr(_) => {
            let parameters = type_definition.fn_sig(cx.tcx).skip_binder().inputs();
            for (argument, parameter) in arguments.iter().zip(parameters.iter()) {
                match parameter.kind {
                    ty::Ref(_, _, Mutability::Immutable)
                    | ty::RawPtr(ty::TypeAndMut {
                        mutbl: Mutability::Immutable,
                        ..
                    }) => {
                        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mutable, _) = argument.kind {
                            span_lint(
                                cx,
                                UNNECESSARY_MUT_PASSED,
                                argument.span,
                                &format!("The function/method `{}` doesn't need a mutable reference", name),
                            );
                        }
                    },
                    _ => (),
                }
            }
        },
        _ => (),
    }
}
