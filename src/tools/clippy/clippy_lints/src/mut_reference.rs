use clippy_utils::diagnostics::span_lint;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Detects passing a mutable reference to a function that only
    /// requires an immutable reference.
    ///
    /// ### Why is this bad?
    /// The mutable reference rules out all other references to
    /// the value. Also the code misleads about the intent of the call site.
    ///
    /// ### Example
    /// ```ignore
    /// // Bad
    /// my_vec.push(&mut value)
    ///
    /// // Good
    /// my_vec.push(&value)
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_MUT_PASSED,
    style,
    "an argument passed as a mutable reference although the callee only demands an immutable reference"
}

declare_lint_pass!(UnnecessaryMutPassed => [UNNECESSARY_MUT_PASSED]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryMutPassed {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        match e.kind {
            ExprKind::Call(fn_expr, arguments) => {
                if let ExprKind::Path(ref path) = fn_expr.kind {
                    check_arguments(
                        cx,
                        arguments,
                        cx.typeck_results().expr_ty(fn_expr),
                        &rustc_hir_pretty::to_string(rustc_hir_pretty::NO_ANN, |s| s.print_qpath(path, false)),
                        "function",
                    );
                }
            },
            ExprKind::MethodCall(path, arguments, _) => {
                let def_id = cx.typeck_results().type_dependent_def_id(e.hir_id).unwrap();
                let substs = cx.typeck_results().node_substs(e.hir_id);
                let method_type = cx.tcx.type_of(def_id).subst(cx.tcx, substs);
                check_arguments(cx, arguments, method_type, path.ident.as_str(), "method");
            },
            _ => (),
        }
    }
}

fn check_arguments<'tcx>(
    cx: &LateContext<'tcx>,
    arguments: &[Expr<'_>],
    type_definition: Ty<'tcx>,
    name: &str,
    fn_kind: &str,
) {
    match type_definition.kind() {
        ty::FnDef(..) | ty::FnPtr(_) => {
            let parameters = type_definition.fn_sig(cx.tcx).skip_binder().inputs();
            for (argument, parameter) in iter::zip(arguments, parameters) {
                match parameter.kind() {
                    ty::Ref(_, _, Mutability::Not)
                    | ty::RawPtr(ty::TypeAndMut {
                        mutbl: Mutability::Not, ..
                    }) => {
                        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, _) = argument.kind {
                            span_lint(
                                cx,
                                UNNECESSARY_MUT_PASSED,
                                argument.span,
                                &format!("the {} `{}` doesn't need a mutable reference", fn_kind, name),
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
