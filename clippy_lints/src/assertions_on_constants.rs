use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::macros::{find_assert_args, root_macro_call_first_node};
use clippy_utils::msrvs::Msrv;
use clippy_utils::{is_inside_always_const_context, msrvs};
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `assert!(true)` and `assert!(false)` calls.
    ///
    /// ### Why is this bad?
    /// Will be optimized out by the compiler or should probably be replaced by a
    /// `panic!()` or `unreachable!()`
    ///
    /// ### Example
    /// ```rust,ignore
    /// assert!(false)
    /// assert!(true)
    /// const B: bool = false;
    /// assert!(B)
    /// ```
    #[clippy::version = "1.34.0"]
    pub ASSERTIONS_ON_CONSTANTS,
    style,
    "`assert!(true)` / `assert!(false)` will be optimized out by the compiler, and should probably be replaced by a `panic!()` or `unreachable!()`"
}

impl_lint_pass!(AssertionsOnConstants => [ASSERTIONS_ON_CONSTANTS]);
pub struct AssertionsOnConstants {
    msrv: Msrv,
}
impl AssertionsOnConstants {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for AssertionsOnConstants {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let Some(macro_call) = root_macro_call_first_node(cx, e)
            && let is_debug = match cx.tcx.get_diagnostic_name(macro_call.def_id) {
                Some(sym::debug_assert_macro) => true,
                Some(sym::assert_macro) => false,
                _ => return,
            }
            && let Some((condition, _)) = find_assert_args(cx, e, macro_call.expn)
            && let Some((Constant::Bool(assert_val), const_src)) = ConstEvalCtxt::new(cx).eval_with_source(condition)
            && let in_const_context = is_inside_always_const_context(cx.tcx, e.hir_id)
            && (const_src.is_local() || !in_const_context)
            && !(is_debug && as_bool_lit(condition) == Some(false))
        {
            let (msg, help) = if !const_src.is_local() {
                let help = if self.msrv.meets(cx, msrvs::CONST_BLOCKS) {
                    "consider moving this into a const block: `const { assert!(..) }`"
                } else if self.msrv.meets(cx, msrvs::CONST_PANIC) {
                    "consider moving this to an anonymous constant: `const _: () = { assert!(..); }`"
                } else {
                    return;
                };
                ("this assertion has a constant value", help)
            } else if assert_val {
                ("this assertion is always `true`", "remove the assertion")
            } else {
                (
                    "this assertion is always `false`",
                    "replace this with `panic!()` or `unreachable!()`",
                )
            };

            span_lint_and_help(cx, ASSERTIONS_ON_CONSTANTS, macro_call.span, msg, None, help);
        }
    }
}

fn as_bool_lit(e: &Expr<'_>) -> Option<bool> {
    if let ExprKind::Lit(l) = e.kind
        && let LitKind::Bool(b) = l.node
    {
        Some(b)
    } else {
        None
    }
}
