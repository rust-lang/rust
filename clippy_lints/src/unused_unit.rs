use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SpanRangeExt, position_before_rarrow};
use clippy_utils::{is_never_expr, is_unit_expr};
use rustc_ast::{Block, StmtKind};
use rustc_errors::Applicability;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    AssocItemConstraintKind, Body, Expr, ExprKind, FnDecl, FnRetTy, GenericArgsParentheses, PolyTraitRef, Term, Ty,
    TyKind,
};
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::edition::Edition;
use rustc_span::{BytePos, Pos as _, Span, sym};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unit (`()`) expressions that can be removed.
    ///
    /// ### Why is this bad?
    /// Such expressions add no value, but can make the code
    /// less readable. Depending on formatting they can make a `break` or `return`
    /// statement look like a function call.
    ///
    /// ### Example
    /// ```no_run
    /// fn return_unit() -> () {
    ///     ()
    /// }
    /// ```
    /// is equivalent to
    /// ```no_run
    /// fn return_unit() {}
    /// ```
    #[clippy::version = "1.31.0"]
    pub UNUSED_UNIT,
    style,
    "needless unit expression"
}

declare_lint_pass!(UnusedUnit => [UNUSED_UNIT]);

impl<'tcx> LateLintPass<'tcx> for UnusedUnit {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        span: Span,
        _def_id: LocalDefId,
    ) {
        if let FnRetTy::Return(hir_ty) = decl.output
            && is_unit_ty(hir_ty)
            && !hir_ty.span.from_expansion()
            && get_def(span) == get_def(hir_ty.span)
        {
            // The explicit `-> ()` in the closure signature might be necessary for multiple reasons:
            // - Implicit types in closure signatures are forbidden when `for<...>` is present
            // - If the closure body ends with a function call, and that function's return type is generic, the
            //   `-> ()` could be required for it to be inferred
            //
            // There could be more reasons to have it, and, in general, we shouldn't discourage the users from
            // writing more type annotations than strictly necessary, because it can help readability and
            // maintainability
            if let FnKind::Closure = kind {
                return;
            }

            // unit never type fallback is no longer supported since Rust 2024. For more information,
            // see <https://doc.rust-lang.org/nightly/edition-guide/rust-2024/never-type-fallback.html>
            if cx.tcx.sess.edition() >= Edition::Edition2024
                && let ExprKind::Block(block, _) = body.value.kind
                && let Some(expr) = block.expr
                && is_never_expr(cx, expr).is_some()
            {
                return;
            }

            lint_unneeded_unit_return(cx, hir_ty.span, span);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Ret(Some(expr)) | ExprKind::Break(_, Some(expr)) = expr.kind
            && is_unit_expr(expr)
            && !expr.span.from_expansion()
        {
            span_lint_and_sugg(
                cx,
                UNUSED_UNIT,
                expr.span,
                "unneeded `()`",
                "remove the `()`",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }

    fn check_poly_trait_ref(&mut self, cx: &LateContext<'tcx>, poly: &'tcx PolyTraitRef<'tcx>) {
        if let [segment] = &poly.trait_ref.path.segments
            && matches!(segment.ident.name, sym::Fn | sym::FnMut | sym::FnOnce)
            && let Some(args) = segment.args
            && args.parenthesized == GenericArgsParentheses::ParenSugar
            && let [constraint] = &args.constraints
            && constraint.ident.name == sym::Output
            && let AssocItemConstraintKind::Equality { term: Term::Ty(hir_ty) } = constraint.kind
            && args.span_ext.hi() != poly.span.hi()
            && !hir_ty.span.from_expansion()
            && args.span_ext.hi() != hir_ty.span.hi()
            && is_unit_ty(hir_ty)
        {
            lint_unneeded_unit_return(cx, hir_ty.span, poly.span);
        }
    }
}

impl EarlyLintPass for UnusedUnit {
    /// Check for unit expressions in blocks. This is left in the early pass because some macros
    /// expand its inputs as-is, making it invisible to the late pass. See #4076.
    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &Block) {
        if let Some(stmt) = block.stmts.last()
            && let StmtKind::Expr(expr) = &stmt.kind
            && let rustc_ast::ExprKind::Tup(inner) = &expr.kind
            && inner.is_empty()
            && let ctxt = block.span.ctxt()
            && stmt.span.ctxt() == ctxt
            && expr.span.ctxt() == ctxt
            && expr.attrs.is_empty()
        {
            let sp = expr.span;
            span_lint_and_sugg(
                cx,
                UNUSED_UNIT,
                sp,
                "unneeded unit expression",
                "remove the final `()`",
                String::new(),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn is_unit_ty(ty: &Ty<'_>) -> bool {
    matches!(ty.kind, TyKind::Tup([]))
}

// get the def site
#[must_use]
fn get_def(span: Span) -> Option<Span> {
    if span.from_expansion() {
        Some(span.ctxt().outer_expn_data().def_site)
    } else {
        None
    }
}

fn lint_unneeded_unit_return(cx: &LateContext<'_>, ty_span: Span, span: Span) {
    let (ret_span, appl) =
        if let Some(Some(rpos)) = span.with_hi(ty_span.hi()).with_source_text(cx, position_before_rarrow) {
            (
                ty_span.with_lo(span.lo() + BytePos::from_usize(rpos)),
                Applicability::MachineApplicable,
            )
        } else {
            (ty_span, Applicability::MaybeIncorrect)
        };

    span_lint_and_sugg(
        cx,
        UNUSED_UNIT,
        ret_span,
        "unneeded unit return type",
        "remove the `-> ()`",
        String::new(),
        appl,
    );
}
