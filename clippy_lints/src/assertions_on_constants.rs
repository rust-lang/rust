use crate::consts::{constant, Constant};
use crate::utils::{is_direct_expn_of, is_expn_of, match_qpath, span_help_and_lint};
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

// fn get_assert_args(snip: String) -> Option<Vec<String>> {
//
// }

fn assert_with_message<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) -> Option<(LocalInternedString, bool)> {
    if_chain! {
        if let ExprKind::Match(ref expr, ref arms, MatchSource::IfDesugar { contains_else_clause: false }) = expr.kind;
        // match expr
        if let ExprKind::DropTemps(ref expr) = expr.kind;
        if let ExprKind::Unary(UnOp::UnNot, ref expr) = expr.kind;
        //if let ExprKind::Lit(ref lit) = expr.kind;
        if let Some((Constant::Bool(is_true), _)) = constant(cx, cx.tables, expr);
        //if is_true;
        // match arms
        // arm 1 pattern
        if let PatKind::Lit(ref lit_expr) = arms[0].pat.kind;
        if let ExprKind::Lit(ref lit) = lit_expr.kind;
        if let LitKind::Bool(true) = lit.node;
        //if let LitKind::Bool(true) = lit1.node;
        // arm 1 block
        if let ExprKind::Block(ref block1, _) = arms[0].body.kind;
        if let Some(trailing_expr1) = &block1.expr;
        if block1.stmts.len() == 0;
        //
        if let ExprKind::Block(ref actual_block1, _) = trailing_expr1.kind;
        if let Some(block1_expr) = &actual_block1.expr;
        // function call
        if let ExprKind::Call(ref func, ref args) = block1_expr.kind;
        if let ExprKind::Path(ref path) = func.kind;
        // ["{{root}}", "std", "rt", "begin_panic"] does not work
        if match_qpath(path, &["$crate", "rt", "begin_panic"]);
        // arguments
        if args.len() == 2;
        if let ExprKind::Lit(ref lit) = args[0].kind;
        if let LitKind::Str(ref s, _) = lit.node;
        let panic_message = s.as_str(); // bind the panic message
        if let ExprKind::AddrOf(MutImmutable, ref inner) = args[1].kind;
        if let ExprKind::Tup(ref elements) = inner.kind;
        if elements.len() == 3;
        if let ExprKind::Lit(ref lit1) = elements[0].kind;
        if let LitKind::Str(ref s1, _) = lit1.node;
        if let ExprKind::Lit(ref lit2) = elements[1].kind;
        if let LitKind::Int(_, _) = lit2.node;
        if let ExprKind::Lit(ref lit3) = elements[2].kind;
        if let LitKind::Int(_, _) = lit3.node;
        // arm 2 block
        if let PatKind::Wild = arms[1].pat.kind;
        if let ExprKind::Block(ref block2, _) = arms[1].body.kind;
        if let None = &block2.expr;
        if block2.stmts.len() == 0;
        then {
            return Some((panic_message, is_true));
        }
    }
    return None;
}
