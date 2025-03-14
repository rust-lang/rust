use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::path_res;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{FnKind, Visitor};
use rustc_hir::{Body, Expr, ExprKind, FnDecl, FnRetTy, Lit, MutTy, Mutability, PrimTy, Ty, TyKind, intravisit};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Detects functions that are written to return `&str` that could return `&'static str` but instead return a `&'a str`.
    ///
    /// ### Why is this bad?
    ///
    /// This leaves the caller unable to use the `&str` as `&'static str`, causing unnecessary allocations or confusion.
    /// This is also most likely what you meant to write.
    ///
    /// ### Example
    /// ```no_run
    /// # struct MyType;
    /// impl MyType {
    ///     fn returns_literal(&self) -> &str {
    ///         "Literal"
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # struct MyType;
    /// impl MyType {
    ///     fn returns_literal(&self) -> &'static str {
    ///         "Literal"
    ///     }
    /// }
    /// ```
    /// Or, in case you may return a non-literal `str` in future:
    /// ```no_run
    /// # struct MyType;
    /// impl MyType {
    ///     fn returns_literal<'a>(&'a self) -> &'a str {
    ///         "Literal"
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.84.0"]
    pub UNNECESSARY_LITERAL_BOUND,
    pedantic,
    "detects &str that could be &'static str in function return types"
}

declare_lint_pass!(UnnecessaryLiteralBound => [UNNECESSARY_LITERAL_BOUND]);

fn extract_anonymous_ref<'tcx>(hir_ty: &Ty<'tcx>) -> Option<&'tcx Ty<'tcx>> {
    let TyKind::Ref(lifetime, MutTy { ty, mutbl }) = hir_ty.kind else {
        return None;
    };

    if !lifetime.is_anonymous() || !matches!(mutbl, Mutability::Not) {
        return None;
    }

    Some(ty)
}

fn is_str_literal(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Lit(Lit {
            node: LitKind::Str(..),
            ..
        }),
    )
}

struct FindNonLiteralReturn;

impl<'hir> Visitor<'hir> for FindNonLiteralReturn {
    type Result = std::ops::ControlFlow<()>;
    type NestedFilter = intravisit::nested_filter::None;

    fn visit_expr(&mut self, expr: &'hir Expr<'hir>) -> Self::Result {
        if let ExprKind::Ret(Some(ret_val_expr)) = expr.kind
            && !is_str_literal(ret_val_expr)
        {
            Self::Result::Break(())
        } else {
            intravisit::walk_expr(self, expr)
        }
    }
}

fn check_implicit_returns_static_str(body: &Body<'_>) -> bool {
    // TODO: Improve this to the same complexity as the Visitor to catch more implicit return cases.
    if let ExprKind::Block(block, _) = body.value.kind
        && let Some(implicit_ret) = block.expr
    {
        return is_str_literal(implicit_ret);
    }

    false
}

fn check_explicit_returns_static_str(expr: &Expr<'_>) -> bool {
    let mut visitor = FindNonLiteralReturn;
    visitor.visit_expr(expr).is_continue()
}

impl<'tcx> LateLintPass<'tcx> for UnnecessaryLiteralBound {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        _: LocalDefId,
    ) {
        if span.from_expansion() {
            return;
        }

        // Checking closures would be a little silly
        if matches!(kind, FnKind::Closure) {
            return;
        }

        // Check for `-> &str`
        let FnRetTy::Return(ret_hir_ty) = decl.output else {
            return;
        };

        let Some(inner_hir_ty) = extract_anonymous_ref(ret_hir_ty) else {
            return;
        };

        if path_res(cx, inner_hir_ty) != Res::PrimTy(PrimTy::Str) {
            return;
        }

        // Check for all return statements returning literals
        if check_explicit_returns_static_str(body.value) && check_implicit_returns_static_str(body) {
            span_lint_and_sugg(
                cx,
                UNNECESSARY_LITERAL_BOUND,
                ret_hir_ty.span,
                "returning a `str` unnecessarily tied to the lifetime of arguments",
                "try",
                "&'static str".into(), // how ironic, a lint about `&'static str` requiring a `String` alloc...
                Applicability::MachineApplicable,
            );
        }
    }
}
