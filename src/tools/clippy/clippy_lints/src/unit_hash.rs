use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects `().hash(_)`.
    ///
    /// ### Why is this bad?
    /// Hashing a unit value doesn't do anything as the implementation of `Hash` for `()` is a no-op.
    ///
    /// ### Example
    /// ```rust
    /// # use std::hash::Hash;
    /// # use std::collections::hash_map::DefaultHasher;
    /// # enum Foo { Empty, WithValue(u8) }
    /// # use Foo::*;
    /// # let mut state = DefaultHasher::new();
    /// # let my_enum = Foo::Empty;
    /// match my_enum {
    /// 	Empty => ().hash(&mut state),
    /// 	WithValue(x) => x.hash(&mut state),
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::hash::Hash;
    /// # use std::collections::hash_map::DefaultHasher;
    /// # enum Foo { Empty, WithValue(u8) }
    /// # use Foo::*;
    /// # let mut state = DefaultHasher::new();
    /// # let my_enum = Foo::Empty;
    /// match my_enum {
    /// 	Empty => 0_u8.hash(&mut state),
    /// 	WithValue(x) => x.hash(&mut state),
    /// }
    /// ```
    #[clippy::version = "1.58.0"]
    pub UNIT_HASH,
    correctness,
    "hashing a unit value, which does nothing"
}
declare_lint_pass!(UnitHash => [UNIT_HASH]);

impl<'tcx> LateLintPass<'tcx> for UnitHash {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if_chain! {
            if let ExprKind::MethodCall(name_ident, args, _) = &expr.kind;
            if name_ident.ident.name == sym::hash;
            if let [recv, state_param] = args;
            if cx.typeck_results().expr_ty(recv).is_unit();
            then {
                span_lint_and_then(
                    cx,
                    UNIT_HASH,
                    expr.span,
                    "this call to `hash` on the unit type will do nothing",
                    |diag| {
                        diag.span_suggestion(
                            expr.span,
                            "remove the call to `hash` or consider using",
                            format!(
                                "0_u8.hash({})",
                                snippet(cx, state_param.span, ".."),
                            ),
                            Applicability::MaybeIncorrect,
                        );
                        diag.note("the implementation of `Hash` for `()` is a no-op");
                    }
                );
            }
        }
    }
}
