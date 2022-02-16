use clippy_utils::diagnostics::span_lint;
use clippy_utils::higher::IfLetOrMatch;
use clippy_utils::visitors::{for_each_expr, Descend};
use clippy_utils::{meets_msrv, msrvs, peel_blocks};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, MatchSource, Pat, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
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
        let if_let_or_match = if_chain! {
            if meets_msrv(self.msrv, msrvs::LET_ELSE);
            if !in_external_macro(cx.sess(), stmt.span);
            if let StmtKind::Local(local) = stmt.kind;
            if let Some(init) = local.init;
            if !from_different_macros(init.span, stmt.span);
            if let Some(if_let_or_match) = IfLetOrMatch::parse(cx, init);
            then {
                if_let_or_match
            } else {
                return;
            }
        };

        match if_let_or_match {
            IfLetOrMatch::IfLet(_let_expr, let_pat, if_then, if_else) => if_chain! {
                if expr_is_simple_identity(let_pat, if_then);
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
            },
            IfLetOrMatch::Match(_match_expr, arms, source) => {
                if source != MatchSource::Normal {
                    return;
                }
                // Any other number than two arms doesn't (neccessarily)
                // have a trivial mapping to let else.
                if arms.len() != 2 {
                    return;
                }
                // We iterate over both arms, trying to find one that is an identity,
                // one that diverges. Our check needs to work regardless of the order
                // of both arms.
                let mut found_identity_arm = false;
                let mut found_diverging_arm = false;
                for arm in arms {
                    // Guards don't give us an easy mapping to let else
                    if arm.guard.is_some() {
                        return;
                    }
                    if expr_is_simple_identity(arm.pat, arm.body) {
                        found_identity_arm = true;
                    } else if expr_diverges(cx, arm.body) && pat_has_no_bindings(arm.pat) {
                        found_diverging_arm = true;
                    }
                }
                if !(found_identity_arm && found_diverging_arm) {
                    return;
                }
                span_lint(cx, MANUAL_LET_ELSE, stmt.span, "this could be rewritten as `let else`");
            },
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

/// Returns true if the two spans come from different macro sites,
/// or one comes from an invocation and the other is not from a macro at all.
fn from_different_macros(span_a: Span, span_b: Span) -> bool {
    // This pre-check is a speed up so that we don't build outer_expn_data unless needed.
    match (span_a.from_expansion(), span_b.from_expansion()) {
        (false, false) => return false,
        (true, false) | (false, true) => return true,
        // We need to determine if both are from the same macro
        (true, true) => (),
    }
    let data_for_comparison = |sp: Span| {
        let expn_data = sp.ctxt().outer_expn_data();
        (expn_data.kind, expn_data.call_site)
    };
    data_for_comparison(span_a) != data_for_comparison(span_b)
}

fn pat_has_no_bindings(pat: &'_ Pat<'_>) -> bool {
    let mut has_no_bindings = true;
    pat.each_binding_or_first(&mut |_, _, _, _| has_no_bindings = false);
    has_no_bindings
}

/// Checks if the passed block is a simple identity referring to bindings created by the pattern
fn expr_is_simple_identity(pat: &'_ Pat<'_>, expr: &'_ Expr<'_>) -> bool {
    // TODO support patterns with multiple bindings and tuples, like:
    //   let ... = if let (Some(foo), bar) = g() { (foo, bar) } else { ... }
    if_chain! {
        if let ExprKind::Path(QPath::Resolved(_ty, path)) = &peel_blocks(expr).kind;
        if let [path_seg] = path.segments;
        then {
            let mut pat_bindings = Vec::new();
            pat.each_binding_or_first(&mut |_ann, _hir_id, _sp, ident| {
                pat_bindings.push(ident);
            });
            if let [binding] = &pat_bindings[..] {
                return path_seg.ident == *binding;
            }
        }
    }
    false
}
