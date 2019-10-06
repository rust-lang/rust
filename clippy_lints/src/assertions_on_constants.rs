use crate::consts::{constant, Constant};
use crate::utils::paths;
use crate::utils::{is_direct_expn_of, is_expn_of, match_def_path, resolve_node, span_help_and_lint};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::LitKind;
use syntax::source_map::symbol::LocalInternedString;

declare_clippy_lint! {
    /// **What it does:** Checks for `assert!(true)` and `assert!(false)` calls.
    ///
    /// **Why is this bad?** Will be optimized out by the compiler or should probably be replaced by a
    /// panic!() or unreachable!()
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```rust,ignore
    /// assert!(false)
    /// // or
    /// assert!(true)
    /// // or
    /// const B: bool = false;
    /// assert!(B)
    /// ```
    pub ASSERTIONS_ON_CONSTANTS,
    style,
    "`assert!(true)` / `assert!(false)` will be optimized out by the compiler, and should probably be replaced by a `panic!()` or `unreachable!()`"
}

declare_lint_pass!(AssertionsOnConstants => [ASSERTIONS_ON_CONSTANTS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AssertionsOnConstants {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        let lint_assert_cb = |is_debug_assert: bool| {
            if let ExprKind::Unary(_, ref lit) = e.kind {
                if let Some((Constant::Bool(is_true), _)) = constant(cx, cx.tables, lit) {
                    if is_true {
                        span_help_and_lint(
                            cx,
                            ASSERTIONS_ON_CONSTANTS,
                            e.span,
                            "`assert!(true)` will be optimized out by the compiler",
                            "remove it",
                        );
                    } else if !is_debug_assert {
                        span_help_and_lint(
                            cx,
                            ASSERTIONS_ON_CONSTANTS,
                            e.span,
                            "`assert!(false)` should probably be replaced",
                            "use `panic!()` or `unreachable!()`",
                        );
                    }
                }
            }
        };
        if let Some(debug_assert_span) = is_expn_of(e.span, "debug_assert") {
            if debug_assert_span.from_expansion() {
                return;
            }
            lint_assert_cb(true);
        } else if let Some(assert_span) = is_direct_expn_of(e.span, "assert") {
            if assert_span.from_expansion() {
                return;
            }
            if let Some((panic_message, is_true)) = assert_with_message(&cx, e) {
                if is_true {
                    span_help_and_lint(
                        cx,
                        ASSERTIONS_ON_CONSTANTS,
                        e.span,
                        "`assert!(true)` will be optimized out by the compiler",
                        "remove it",
                    );
                } else if panic_message.starts_with("assertion failed: ") {
                    span_help_and_lint(
                        cx,
                        ASSERTIONS_ON_CONSTANTS,
                        e.span,
                        "`assert!(false)` should probably be replaced",
                        "use `panic!()` or `unreachable!()`",
                    );
                } else {
                    span_help_and_lint(
                        cx,
                        ASSERTIONS_ON_CONSTANTS,
                        e.span,
                        &format!("`assert!(false, \"{}\")` should probably be replaced", panic_message,),
                        &format!(
                            "use `panic!(\"{}\")` or `unreachable!(\"{}\")`",
                            panic_message, panic_message,
                        ),
                    );
                }
            }
        }
    }
}

/// Check if the expression matches
///
/// ```rust,ignore
/// match { let _t = !c; _t } {
///     true => {
///         {
///             ::std::rt::begin_panic(message, _)
///         }
///     }
///     _ => { }
/// };
/// ```
///
/// where `message` is a string literal and `c` is a constant bool.
///
/// TODO extend this to match anything as message not just string literals
///
/// Returns the `message` argument of `begin_panic` and the value of `c` which is the
/// first argument of `assert!`.
fn assert_with_message<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) -> Option<(LocalInternedString, bool)> {
    if_chain! {
        if let ExprKind::Match(ref expr, ref arms, _) = expr.kind;
        // matches { let _t = expr; _t }
        if let ExprKind::DropTemps(ref expr) = expr.kind;
        if let ExprKind::Unary(UnOp::UnNot, ref expr) = expr.kind;
        // bind the first argument of the `assert!` macro
        if let Some((Constant::Bool(is_true), _)) = constant(cx, cx.tables, expr);
        // arm 1 pattern
        if let PatKind::Lit(ref lit_expr) = arms[0].pat.kind;
        if let ExprKind::Lit(ref lit) = lit_expr.kind;
        if let LitKind::Bool(true) = lit.node;
        // arm 1 block
        if let ExprKind::Block(ref block, _) = arms[0].body.kind;
        if block.stmts.len() == 0;
        if let Some(block_expr) = &block.expr;
        if let ExprKind::Block(ref inner_block, _) = block_expr.kind;
        if let Some(begin_panic_call) = &inner_block.expr;
        // function call
        if let Some(args) = match_function_call(cx, begin_panic_call, &paths::BEGIN_PANIC);
        if args.len() == 2;
        if let ExprKind::Lit(ref lit) = args[0].kind;
        if let LitKind::Str(ref s, _) = lit.node;
        // bind the second argument of the `assert!` macro
        let panic_message = s.as_str();
        // second argument of begin_panic is irrelevant
        // as is the second match arm
        then {
            return Some((panic_message, is_true));
        }
    }
    None
}

/// Matches a function call with the given path and returns the arguments.
///
/// Usage:
///
/// ```rust,ignore
/// if let Some(args) = match_function_call(cx, begin_panic_call, &paths::BEGIN_PANIC);
/// ```
fn match_function_call<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr, path: &[&str]) -> Option<&'a [Expr]> {
    if_chain! {
        if let ExprKind::Call(ref fun, ref args) = expr.kind;
        if let ExprKind::Path(ref qpath) = fun.kind;
        if let Some(fun_def_id) = resolve_node(cx, qpath, fun.hir_id).opt_def_id();
        if match_def_path(cx, fun_def_id, path);
        then {
            return Some(&args)
        }
    };
    None
}
