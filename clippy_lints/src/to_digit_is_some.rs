use crate::utils::{match_def_path, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `.to_digit(..).is_some()` on `char`s.
    ///
    /// **Why is this bad?** This is a convoluted way of checking if a `char` is a digit. It's
    /// more straight forward to use the dedicated `is_digit` method.
    ///
    /// **Example:**
    /// ```rust
    /// # let c = 'c';
    /// # let radix = 10;
    /// let is_digit = c.to_digit(radix).is_some();
    /// ```
    /// can be written as:
    /// ```
    /// # let c = 'c';
    /// # let radix = 10;
    /// let is_digit = c.is_digit(radix);
    /// ```
    pub TO_DIGIT_IS_SOME,
    style,
    "`char.is_digit()` is clearer"
}

declare_lint_pass!(ToDigitIsSome => [TO_DIGIT_IS_SOME]);

impl<'tcx> LateLintPass<'tcx> for ToDigitIsSome {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            if let hir::ExprKind::MethodCall(is_some_path, _, is_some_args, _) = &expr.kind;
            if is_some_path.ident.name.as_str() == "is_some";
            if let [to_digit_expr] = &**is_some_args;
            then {
                let match_result = match &to_digit_expr.kind {
                    hir::ExprKind::MethodCall(to_digits_path, _, to_digit_args, _) => {
                        if_chain! {
                            if let [char_arg, radix_arg] = &**to_digit_args;
                            if to_digits_path.ident.name.as_str() == "to_digit";
                            let char_arg_ty = cx.typeck_results().expr_ty_adjusted(char_arg);
                            if *char_arg_ty.kind() == ty::Char;
                            then {
                                Some((true, char_arg, radix_arg))
                            } else {
                                None
                            }
                        }
                    }
                    hir::ExprKind::Call(to_digits_call, to_digit_args) => {
                        if_chain! {
                            if let [char_arg, radix_arg] = &**to_digit_args;
                            if let hir::ExprKind::Path(to_digits_path) = &to_digits_call.kind;
                            if let to_digits_call_res = cx.qpath_res(to_digits_path, to_digits_call.hir_id);
                            if let Some(to_digits_def_id) = to_digits_call_res.opt_def_id();
                            if match_def_path(cx, to_digits_def_id, &["core", "char", "methods", "<impl char>", "to_digit"]);
                            then {
                                Some((false, char_arg, radix_arg))
                            } else {
                                None
                            }
                        }
                    }
                    _ => None
                };

                if let Some((is_method_call, char_arg, radix_arg)) = match_result {
                    let mut applicability = Applicability::MachineApplicable;
                    let char_arg_snip = snippet_with_applicability(cx, char_arg.span, "_", &mut applicability);
                    let radix_snip = snippet_with_applicability(cx, radix_arg.span, "_", &mut applicability);

                    span_lint_and_sugg(
                        cx,
                        TO_DIGIT_IS_SOME,
                        expr.span,
                        "use of `.to_digit(..).is_some()`",
                        "try this",
                        if is_method_call {
                            format!("{}.is_digit({})", char_arg_snip, radix_snip)
                        } else {
                            format!("char::is_digit({}, {})", char_arg_snip, radix_snip)
                        },
                        applicability,
                    );
                }
            }
        }
    }
}
