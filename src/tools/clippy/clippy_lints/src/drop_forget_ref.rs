use clippy_utils::diagnostics::span_lint_and_note;
use clippy_utils::ty::{is_copy, is_must_use_ty, is_type_lang_item};
use clippy_utils::{get_parent_node, is_must_use_func_call};
use rustc_hir::{Arm, Expr, ExprKind, LangItem, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `std::mem::drop` with a value that does not implement `Drop`.
    ///
    /// ### Why is this bad?
    /// Calling `std::mem::drop` is no different than dropping such a type. A different value may
    /// have been intended.
    ///
    /// ### Example
    /// ```rust
    /// struct Foo;
    /// let x = Foo;
    /// std::mem::drop(x);
    /// ```
    #[clippy::version = "1.62.0"]
    pub DROP_NON_DROP,
    suspicious,
    "call to `std::mem::drop` with a value which does not implement `Drop`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `std::mem::forget` with a value that does not implement `Drop`.
    ///
    /// ### Why is this bad?
    /// Calling `std::mem::forget` is no different than dropping such a type. A different value may
    /// have been intended.
    ///
    /// ### Example
    /// ```rust
    /// struct Foo;
    /// let x = Foo;
    /// std::mem::forget(x);
    /// ```
    #[clippy::version = "1.62.0"]
    pub FORGET_NON_DROP,
    suspicious,
    "call to `std::mem::forget` with a value which does not implement `Drop`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `std::mem::forget(t)` where `t` is
    /// `Drop` or has a field that implements `Drop`.
    ///
    /// ### Why is this bad?
    /// `std::mem::forget(t)` prevents `t` from running its
    /// destructor, possibly causing leaks.
    ///
    /// ### Example
    /// ```rust
    /// # use std::mem;
    /// # use std::rc::Rc;
    /// mem::forget(Rc::new(55))
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MEM_FORGET,
    restriction,
    "`mem::forget` usage on `Drop` types, likely to cause memory leaks"
}

const DROP_NON_DROP_SUMMARY: &str = "call to `std::mem::drop` with a value that does not implement `Drop`. \
                                 Dropping such a type only extends its contained lifetimes";
const FORGET_NON_DROP_SUMMARY: &str = "call to `std::mem::forget` with a value that does not implement `Drop`. \
                                   Forgetting such a type is the same as dropping it";

declare_lint_pass!(DropForgetRef => [
    DROP_NON_DROP,
    FORGET_NON_DROP,
    MEM_FORGET,
]);

impl<'tcx> LateLintPass<'tcx> for DropForgetRef {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(path, [arg]) = expr.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && let Some(fn_name) = cx.tcx.get_diagnostic_name(def_id)
        {
            let arg_ty = cx.typeck_results().expr_ty(arg);
            let is_copy = is_copy(cx, arg_ty);
            let drop_is_single_call_in_arm = is_single_call_in_arm(cx, arg, expr);
            let (lint, msg, note_span) = match fn_name {
                // early return for uplifted lints: dropping_references, dropping_copy_types, forgetting_references, forgetting_copy_types
                sym::mem_drop if arg_ty.is_ref() && !drop_is_single_call_in_arm => return,
                sym::mem_forget if arg_ty.is_ref() => return,
                sym::mem_drop if is_copy && !drop_is_single_call_in_arm => return,
                sym::mem_forget if is_copy => return,
                sym::mem_drop if is_type_lang_item(cx, arg_ty, LangItem::ManuallyDrop) => return,
                sym::mem_drop
                    if !(arg_ty.needs_drop(cx.tcx, cx.param_env)
                        || is_must_use_func_call(cx, arg)
                        || is_must_use_ty(cx, arg_ty)
                        || drop_is_single_call_in_arm
                        ) =>
                {
                    (DROP_NON_DROP, DROP_NON_DROP_SUMMARY.into(), Some(arg.span))
                },
                sym::mem_forget => {
                    if arg_ty.needs_drop(cx.tcx, cx.param_env) {
                        (
                            MEM_FORGET,
                            Cow::Owned(format!(
                                "usage of `mem::forget` on {}",
                                if arg_ty.ty_adt_def().map_or(false, |def| def.has_dtor(cx.tcx)) {
                                    "`Drop` type"
                                } else {
                                    "type with `Drop` fields"
                                }
                            )),
                            None,
                        )
                    } else {
                        (FORGET_NON_DROP, FORGET_NON_DROP_SUMMARY.into(), Some(arg.span))
                    }
                }
                _ => return,
            };
            span_lint_and_note(
                cx,
                lint,
                expr.span,
                &msg,
                note_span,
                &format!("argument has type `{arg_ty}`"),
            );
        }
    }
}

// dropping returned value of a function like in the following snippet is considered idiomatic, see
// #9482 for examples match <var> {
//     <pat> => drop(fn_with_side_effect_and_returning_some_value()),
//     ..
// }
fn is_single_call_in_arm<'tcx>(cx: &LateContext<'tcx>, arg: &'tcx Expr<'_>, drop_expr: &'tcx Expr<'_>) -> bool {
    if matches!(arg.kind, ExprKind::Call(..) | ExprKind::MethodCall(..)) {
        let parent_node = get_parent_node(cx.tcx, drop_expr.hir_id);
        if let Some(Node::Arm(Arm { body, .. })) = &parent_node {
            return body.hir_id == drop_expr.hir_id;
        }
    }
    false
}
