use clippy_utils::diagnostics::span_lint;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::declare_lint_pass;
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
    /// ```no_run
    /// # let mut vec = Vec::new();
    /// # let mut value = 5;
    /// vec.push(&mut value);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let mut vec = Vec::new();
    /// # let value = 5;
    /// vec.push(&value);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_MUT_PASSED,
    style,
    "an argument passed as a mutable reference although the callee only demands an immutable reference"
}

declare_lint_pass!(UnnecessaryMutPassed => [UNNECESSARY_MUT_PASSED]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryMutPassed {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            // Issue #11268
            return;
        }

        match e.kind {
            ExprKind::Call(fn_expr, arguments) => {
                if let ExprKind::Path(ref path) = fn_expr.kind {
                    check_arguments(
                        cx,
                        &mut arguments.iter(),
                        cx.typeck_results().expr_ty(fn_expr),
                        &rustc_hir_pretty::qpath_to_string(&cx.tcx, path),
                        "function",
                    );
                }
            },
            ExprKind::MethodCall(path, receiver, arguments, _) => {
                let def_id = cx.typeck_results().type_dependent_def_id(e.hir_id).unwrap();
                let args = cx.typeck_results().node_args(e.hir_id);
                let method_type = cx.tcx.type_of(def_id).instantiate(cx.tcx, args);
                check_arguments(
                    cx,
                    &mut iter::once(receiver).chain(arguments.iter()),
                    method_type,
                    path.ident.as_str(),
                    "method",
                );
            },
            _ => (),
        }
    }
}

fn check_arguments<'tcx>(
    cx: &LateContext<'tcx>,
    arguments: &mut dyn Iterator<Item = &'tcx Expr<'tcx>>,
    type_definition: Ty<'tcx>,
    name: &str,
    fn_kind: &str,
) {
    match type_definition.kind() {
        ty::FnDef(..) | ty::FnPtr(..) => {
            let parameters = type_definition.fn_sig(cx.tcx).skip_binder().inputs();
            for (argument, parameter) in iter::zip(arguments, parameters) {
                match parameter.kind() {
                    ty::Ref(_, _, Mutability::Not) | ty::RawPtr(_, Mutability::Not) => {
                        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, _) = argument.kind {
                            span_lint(
                                cx,
                                UNNECESSARY_MUT_PASSED,
                                argument.span,
                                format!("the {fn_kind} `{name}` doesn't need a mutable reference"),
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
