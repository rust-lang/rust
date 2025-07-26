use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_in_const_context, is_path_diagnostic_item, sym};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `.to_digit(..).is_some()` on `char`s.
    ///
    /// ### Why is this bad?
    /// This is a convoluted way of checking if a `char` is a digit. It's
    /// more straight forward to use the dedicated `is_digit` method.
    ///
    /// ### Example
    /// ```no_run
    /// # let c = 'c';
    /// # let radix = 10;
    /// let is_digit = c.to_digit(radix).is_some();
    /// ```
    /// can be written as:
    /// ```no_run
    /// # let c = 'c';
    /// # let radix = 10;
    /// let is_digit = c.is_digit(radix);
    /// ```
    #[clippy::version = "1.41.0"]
    pub TO_DIGIT_IS_SOME,
    style,
    "`char.is_digit()` is clearer"
}

impl_lint_pass!(ToDigitIsSome => [TO_DIGIT_IS_SOME]);

pub(crate) struct ToDigitIsSome {
    msrv: Msrv,
}

impl ToDigitIsSome {
    pub(crate) fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for ToDigitIsSome {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if let hir::ExprKind::MethodCall(is_some_path, to_digit_expr, [], _) = &expr.kind
            && is_some_path.ident.name == sym::is_some
        {
            let match_result = match to_digit_expr.kind {
                hir::ExprKind::MethodCall(to_digits_path, char_arg, [radix_arg], _) => {
                    if to_digits_path.ident.name == sym::to_digit
                        && cx.typeck_results().expr_ty_adjusted(char_arg).is_char()
                    {
                        Some((true, char_arg, radix_arg))
                    } else {
                        None
                    }
                },
                hir::ExprKind::Call(to_digits_call, [char_arg, radix_arg]) => {
                    if is_path_diagnostic_item(cx, to_digits_call, sym::char_to_digit) {
                        Some((false, char_arg, radix_arg))
                    } else {
                        None
                    }
                },
                _ => None,
            };

            if let Some((is_method_call, char_arg, radix_arg)) = match_result
                && (!is_in_const_context(cx) || self.msrv.meets(cx, msrvs::CONST_CHAR_IS_DIGIT))
            {
                let mut applicability = Applicability::MachineApplicable;
                let char_arg_snip = snippet_with_applicability(cx, char_arg.span, "_", &mut applicability);
                let radix_snip = snippet_with_applicability(cx, radix_arg.span, "_", &mut applicability);

                span_lint_and_sugg(
                    cx,
                    TO_DIGIT_IS_SOME,
                    expr.span,
                    "use of `.to_digit(..).is_some()`",
                    "try",
                    if is_method_call {
                        format!("{char_arg_snip}.is_digit({radix_snip})")
                    } else {
                        format!("char::is_digit({char_arg_snip}, {radix_snip})")
                    },
                    applicability,
                );
            }
        }
    }
}
