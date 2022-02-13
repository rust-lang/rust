use clippy_utils::diagnostics::span_lint;
use clippy_utils::visitors::{for_each_expr, Descend};
use clippy_utils::{higher, meets_msrv, msrvs, peel_blocks};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, Pat, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Warn of cases where `let...else` could be used
    ///
    /// ### Why is this bad?
    ///
    /// `let...else` provides a standard construct for this pattern
    /// that people can easily recognize. It's also more compact.
    ///
    /// ### Example
    ///
    /// ```rust
    /// # let w = Some(0);
    /// let v = if let Some(v) = w { v } else { return };
    /// ```
    ///
    /// Could be written:
    ///
    /// ```rust
    /// # #![feature(let_else)]
    /// # fn main () {
    /// # let w = Some(0);
    /// let Some(v) = w else { return };
    /// # }
    /// ```
    #[clippy::version = "1.67.0"]
    pub MANUAL_LET_ELSE,
    pedantic,
    "manual implementation of a let...else statement"
}

pub struct ManualLetElse {
    msrv: Option<RustcVersion>,
}

impl ManualLetElse {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualLetElse => [MANUAL_LET_ELSE]);

impl<'tcx> LateLintPass<'tcx> for ManualLetElse {
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &'tcx Stmt<'tcx>) {
        if !meets_msrv(self.msrv, msrvs::LET_ELSE) {
            return;
        }

        if in_external_macro(cx.sess(), stmt.span) {
            return;
        }

        if_chain! {
            if let StmtKind::Local(local) = stmt.kind;
            if let Some(init) = local.init;
            if let Some(higher::IfLet { let_pat, let_expr: _, if_then, if_else }) = higher::IfLet::hir(cx, init);
            if if_then_simple_identity(let_pat, if_then);
            if let Some(if_else) = if_else;
            if expr_diverges(cx, if_else);
            then {
                span_lint(
                    cx,
                    MANUAL_LET_ELSE,
                    stmt.span,
                    "this could be rewritten as `let else`",
                );
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn expr_diverges(cx: &LateContext<'_>, expr: &'_ Expr<'_>) -> bool {
    fn is_never(cx: &LateContext<'_>, expr: &'_ Expr<'_>) -> bool {
        if let Some(ty) = cx.typeck_results().expr_ty_opt(expr) {
            return ty.is_never();
        }
        false
    }
    // We can't just call is_never on expr and be done, because the type system
    // sometimes coerces the ! type to something different before we can get
    // our hands on it. So instead, we do a manual search. We do fall back to
    // is_never in some places when there is no better alternative.
    for_each_expr(expr, |ex| {
        match ex.kind {
            ExprKind::Continue(_) | ExprKind::Break(_, _) | ExprKind::Ret(_) => ControlFlow::Break(()),
            ExprKind::Call(call, _) => {
                if is_never(cx, ex) || is_never(cx, call) {
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(Descend::Yes)
            },
            ExprKind::MethodCall(..) => {
                if is_never(cx, ex) {
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(Descend::Yes)
            },
            ExprKind::If(if_expr, if_then, if_else) => {
                let else_diverges = if_else.map_or(false, |ex| expr_diverges(cx, ex));
                let diverges = expr_diverges(cx, if_expr) || (else_diverges && expr_diverges(cx, if_then));
                if diverges {
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(Descend::No)
            },
            ExprKind::Match(match_expr, match_arms, _) => {
                let diverges =
                    expr_diverges(cx, match_expr) || match_arms.iter().all(|arm| expr_diverges(cx, arm.body));
                if diverges {
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(Descend::No)
            },

            // Don't continue into loops or labeled blocks, as they are breakable,
            // and we'd have to start checking labels.
            ExprKind::Block(_, Some(_)) | ExprKind::Loop(..) => ControlFlow::Continue(Descend::No),

            // Default: descend
            _ => ControlFlow::Continue(Descend::Yes),
        }
    })
    .is_some()
}

/// Checks if the passed `if_then` is a simple identity
fn if_then_simple_identity(let_pat: &'_ Pat<'_>, if_then: &'_ Expr<'_>) -> bool {
    // TODO support patterns with multiple bindings and tuples, like:
    //   let (foo, bar) = if let (Some(foo), bar) = g() { (foo, bar) } else { ... }
    if_chain! {
        if let ExprKind::Path(QPath::Resolved(_ty, path)) = &peel_blocks(if_then).kind;
        if let [path_seg] = path.segments;
        then {
            let mut pat_bindings = Vec::new();
            let_pat.each_binding(|_ann, _hir_id, _sp, ident| {
                pat_bindings.push(ident);
            });
            if let [binding] = &pat_bindings[..] {
                return path_seg.ident == *binding;
            }
        }
    }
    false
}
