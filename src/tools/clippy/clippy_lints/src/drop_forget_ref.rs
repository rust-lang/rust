use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_note};
use clippy_utils::get_parent_node;
use clippy_utils::is_must_use_func_call;
use clippy_utils::ty::{is_copy, is_must_use_ty, is_type_lang_item};
use rustc_hir::{Arm, Expr, ExprKind, LangItem, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `std::mem::drop` with a reference
    /// instead of an owned value.
    ///
    /// ### Why is this bad?
    /// Calling `drop` on a reference will only drop the
    /// reference itself, which is a no-op. It will not call the `drop` method (from
    /// the `Drop` trait implementation) on the underlying referenced value, which
    /// is likely what was intended.
    ///
    /// ### Example
    /// ```ignore
    /// let mut lock_guard = mutex.lock();
    /// std::mem::drop(&lock_guard) // Should have been drop(lock_guard), mutex
    /// // still locked
    /// operation_that_requires_mutex_to_be_unlocked();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DROP_REF,
    correctness,
    "calls to `std::mem::drop` with a reference instead of an owned value"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `std::mem::forget` with a reference
    /// instead of an owned value.
    ///
    /// ### Why is this bad?
    /// Calling `forget` on a reference will only forget the
    /// reference itself, which is a no-op. It will not forget the underlying
    /// referenced
    /// value, which is likely what was intended.
    ///
    /// ### Example
    /// ```rust
    /// let x = Box::new(1);
    /// std::mem::forget(&x) // Should have been forget(x), x will still be dropped
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FORGET_REF,
    correctness,
    "calls to `std::mem::forget` with a reference instead of an owned value"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `std::mem::drop` with a value
    /// that derives the Copy trait
    ///
    /// ### Why is this bad?
    /// Calling `std::mem::drop` [does nothing for types that
    /// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html), since the
    /// value will be copied and moved into the function on invocation.
    ///
    /// ### Example
    /// ```rust
    /// let x: i32 = 42; // i32 implements Copy
    /// std::mem::drop(x) // A copy of x is passed to the function, leaving the
    ///                   // original unaffected
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DROP_COPY,
    correctness,
    "calls to `std::mem::drop` with a value that implements Copy"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calls to `std::mem::forget` with a value that
    /// derives the Copy trait
    ///
    /// ### Why is this bad?
    /// Calling `std::mem::forget` [does nothing for types that
    /// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html) since the
    /// value will be copied and moved into the function on invocation.
    ///
    /// An alternative, but also valid, explanation is that Copy types do not
    /// implement
    /// the Drop trait, which means they have no destructors. Without a destructor,
    /// there
    /// is nothing for `std::mem::forget` to ignore.
    ///
    /// ### Example
    /// ```rust
    /// let x: i32 = 42; // i32 implements Copy
    /// std::mem::forget(x) // A copy of x is passed to the function, leaving the
    ///                     // original unaffected
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub FORGET_COPY,
    correctness,
    "calls to `std::mem::forget` with a value that implements Copy"
}

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
    /// Prevents the safe `std::mem::drop` function from being called on `std::mem::ManuallyDrop`.
    ///
    /// ### Why is this bad?
    /// The safe `drop` function does not drop the inner value of a `ManuallyDrop`.
    ///
    /// ### Known problems
    /// Does not catch cases if the user binds `std::mem::drop`
    /// to a different name and calls it that way.
    ///
    /// ### Example
    /// ```rust
    /// struct S;
    /// drop(std::mem::ManuallyDrop::new(S));
    /// ```
    /// Use instead:
    /// ```rust
    /// struct S;
    /// unsafe {
    ///     std::mem::ManuallyDrop::drop(&mut std::mem::ManuallyDrop::new(S));
    /// }
    /// ```
    #[clippy::version = "1.49.0"]
    pub UNDROPPED_MANUALLY_DROPS,
    correctness,
    "use of safe `std::mem::drop` function to drop a std::mem::ManuallyDrop, which will not drop the inner value"
}

const DROP_REF_SUMMARY: &str = "calls to `std::mem::drop` with a reference instead of an owned value. \
                                Dropping a reference does nothing";
const FORGET_REF_SUMMARY: &str = "calls to `std::mem::forget` with a reference instead of an owned value. \
                                  Forgetting a reference does nothing";
const DROP_COPY_SUMMARY: &str = "calls to `std::mem::drop` with a value that implements `Copy`. \
                                 Dropping a copy leaves the original intact";
const FORGET_COPY_SUMMARY: &str = "calls to `std::mem::forget` with a value that implements `Copy`. \
                                   Forgetting a copy leaves the original intact";
const DROP_NON_DROP_SUMMARY: &str = "call to `std::mem::drop` with a value that does not implement `Drop`. \
                                 Dropping such a type only extends its contained lifetimes";
const FORGET_NON_DROP_SUMMARY: &str = "call to `std::mem::forget` with a value that does not implement `Drop`. \
                                   Forgetting such a type is the same as dropping it";

declare_lint_pass!(DropForgetRef => [
    DROP_REF,
    FORGET_REF,
    DROP_COPY,
    FORGET_COPY,
    DROP_NON_DROP,
    FORGET_NON_DROP,
    UNDROPPED_MANUALLY_DROPS
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
            let (lint, msg) = match fn_name {
                sym::mem_drop if arg_ty.is_ref() && !drop_is_single_call_in_arm => (DROP_REF, DROP_REF_SUMMARY),
                sym::mem_forget if arg_ty.is_ref() => (FORGET_REF, FORGET_REF_SUMMARY),
                sym::mem_drop if is_copy && !drop_is_single_call_in_arm => (DROP_COPY, DROP_COPY_SUMMARY),
                sym::mem_forget if is_copy => (FORGET_COPY, FORGET_COPY_SUMMARY),
                sym::mem_drop if is_type_lang_item(cx, arg_ty, LangItem::ManuallyDrop) => {
                    span_lint_and_help(
                        cx,
                        UNDROPPED_MANUALLY_DROPS,
                        expr.span,
                        "the inner value of this ManuallyDrop will not be dropped",
                        None,
                        "to drop a `ManuallyDrop<T>`, use std::mem::ManuallyDrop::drop",
                    );
                    return;
                }
                sym::mem_drop
                    if !(arg_ty.needs_drop(cx.tcx, cx.param_env)
                        || is_must_use_func_call(cx, arg)
                        || is_must_use_ty(cx, arg_ty)
                        || drop_is_single_call_in_arm
                        ) =>
                {
                    (DROP_NON_DROP, DROP_NON_DROP_SUMMARY)
                },
                sym::mem_forget if !arg_ty.needs_drop(cx.tcx, cx.param_env) => {
                    (FORGET_NON_DROP, FORGET_NON_DROP_SUMMARY)
                },
                _ => return,
            };
            span_lint_and_note(
                cx,
                lint,
                expr.span,
                msg,
                Some(arg.span),
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
