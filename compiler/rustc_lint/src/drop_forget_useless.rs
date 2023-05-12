use rustc_hir::{Arm, Expr, ExprKind, Node};
use rustc_middle::ty;
use rustc_span::sym;

use crate::{
    lints::{
        DropCopyDiag, DropRefDiag, ForgetCopyDiag, ForgetRefDiag, UndroppedManuallyDropsDiag,
        UndroppedManuallyDropsSuggestion,
    },
    LateContext, LateLintPass, LintContext,
};

declare_lint! {
    /// The `dropping_references` lint checks for calls to `std::mem::drop` with a reference
    /// instead of an owned value.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # fn operation_that_requires_mutex_to_be_unlocked() {} // just to make it compile
    /// # let mutex = std::sync::Mutex::new(1); // just to make it compile
    /// let mut lock_guard = mutex.lock();
    /// std::mem::drop(&lock_guard); // Should have been drop(lock_guard), mutex
    /// // still locked
    /// operation_that_requires_mutex_to_be_unlocked();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Calling `drop` on a reference will only drop the
    /// reference itself, which is a no-op. It will not call the `drop` method (from
    /// the `Drop` trait implementation) on the underlying referenced value, which
    /// is likely what was intended.
    pub DROPPING_REFERENCES,
    Warn,
    "calls to `std::mem::drop` with a reference instead of an owned value"
}

declare_lint! {
    /// The `forgetting_references` lint checks for calls to `std::mem::forget` with a reference
    /// instead of an owned value.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = Box::new(1);
    /// std::mem::forget(&x); // Should have been forget(x), x will still be dropped
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Calling `forget` on a reference will only forget the
    /// reference itself, which is a no-op. It will not forget the underlying
    /// referenced value, which is likely what was intended.
    pub FORGETTING_REFERENCES,
    Warn,
    "calls to `std::mem::forget` with a reference instead of an owned value"
}

declare_lint! {
    /// The `dropping_copy_types` lint checks for calls to `std::mem::drop` with a value
    /// that derives the Copy trait.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x: i32 = 42; // i32 implements Copy
    /// std::mem::drop(x); // A copy of x is passed to the function, leaving the
    ///                    // original unaffected
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Calling `std::mem::drop` [does nothing for types that
    /// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html), since the
    /// value will be copied and moved into the function on invocation.
    pub DROPPING_COPY_TYPES,
    Warn,
    "calls to `std::mem::drop` with a value that implements Copy"
}

declare_lint! {
    /// The `forgetting_copy_types` lint checks for calls to `std::mem::forget` with a value
    /// that derives the Copy trait.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x: i32 = 42; // i32 implements Copy
    /// std::mem::forget(x); // A copy of x is passed to the function, leaving the
    ///                      // original unaffected
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Calling `std::mem::forget` [does nothing for types that
    /// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html) since the
    /// value will be copied and moved into the function on invocation.
    ///
    /// An alternative, but also valid, explanation is that Copy types do not
    /// implement the Drop trait, which means they have no destructors. Without a
    /// destructor, there is nothing for `std::mem::forget` to ignore.
    pub FORGETTING_COPY_TYPES,
    Warn,
    "calls to `std::mem::forget` with a value that implements Copy"
}

declare_lint! {
    /// The `undropped_manually_drops` lint check for calls to `std::mem::drop` with
    /// a value of `std::mem::ManuallyDrop` which doesn't drop.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// struct S;
    /// drop(std::mem::ManuallyDrop::new(S));
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `ManuallyDrop` does not drop it's inner value so calling `std::mem::drop` will
    /// not drop the inner value of the `ManuallyDrop` either.
    pub UNDROPPED_MANUALLY_DROPS,
    Deny,
    "calls to `std::mem::drop` with `std::mem::ManuallyDrop` instead of it's inner value"
}

declare_lint_pass!(DropForgetUseless => [DROPPING_REFERENCES, FORGETTING_REFERENCES, DROPPING_COPY_TYPES, FORGETTING_COPY_TYPES, UNDROPPED_MANUALLY_DROPS]);

impl<'tcx> LateLintPass<'tcx> for DropForgetUseless {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(path, [arg]) = expr.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && let Some(fn_name) = cx.tcx.get_diagnostic_name(def_id)
        {
            let arg_ty = cx.typeck_results().expr_ty(arg);
            let is_copy = arg_ty.is_copy_modulo_regions(cx.tcx, cx.param_env);
            let drop_is_single_call_in_arm = is_single_call_in_arm(cx, arg, expr);
            match fn_name {
                sym::mem_drop if arg_ty.is_ref() && !drop_is_single_call_in_arm => {
                    cx.emit_spanned_lint(DROPPING_REFERENCES, expr.span, DropRefDiag { arg_ty, label: arg.span });
                },
                sym::mem_forget if arg_ty.is_ref() => {
                    cx.emit_spanned_lint(FORGETTING_REFERENCES, expr.span, ForgetRefDiag { arg_ty, label: arg.span });
                },
                sym::mem_drop if is_copy && !drop_is_single_call_in_arm => {
                    cx.emit_spanned_lint(DROPPING_COPY_TYPES, expr.span, DropCopyDiag { arg_ty, label: arg.span });
                }
                sym::mem_forget if is_copy => {
                    cx.emit_spanned_lint(FORGETTING_COPY_TYPES, expr.span, ForgetCopyDiag { arg_ty, label: arg.span });
                }
                sym::mem_drop if let ty::Adt(adt, _) = arg_ty.kind() && adt.is_manually_drop() => {
                    cx.emit_spanned_lint(
                        UNDROPPED_MANUALLY_DROPS,
                        expr.span,
                        UndroppedManuallyDropsDiag {
                            arg_ty,
                            label: arg.span,
                            suggestion: UndroppedManuallyDropsSuggestion {
                                start_span: arg.span.shrink_to_lo(),
                                end_span: arg.span.shrink_to_hi()
                            }
                        }
                    );
                }
                _ => return,
            };
        }
    }
}

// Dropping returned value of a function, as in the following snippet is considered idiomatic, see
// rust-lang/rust-clippy#9482 for examples.
//
// ```
// match <var> {
//     <pat> => drop(fn_with_side_effect_and_returning_some_value()),
//     ..
// }
// ```
fn is_single_call_in_arm<'tcx>(
    cx: &LateContext<'tcx>,
    arg: &'tcx Expr<'_>,
    drop_expr: &'tcx Expr<'_>,
) -> bool {
    if matches!(arg.kind, ExprKind::Call(..) | ExprKind::MethodCall(..)) {
        let parent_node = cx.tcx.hir().find_parent(drop_expr.hir_id);
        if let Some(Node::Arm(Arm { body, .. })) = &parent_node {
            return body.hir_id == drop_expr.hir_id;
        }
    }
    false
}
