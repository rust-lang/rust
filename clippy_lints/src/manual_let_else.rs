use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::IfLetOrMatch;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::{for_each_expr, Descend};
use clippy_utils::{meets_msrv, msrvs, peel_blocks};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, MatchSource, Pat, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::sym;
use rustc_span::Span;
use serde::Deserialize;
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
    matches_behaviour: MatchLintBehaviour,
}

impl ManualLetElse {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>, matches_behaviour: MatchLintBehaviour) -> Self {
        Self {
            msrv,
            matches_behaviour,
        }
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
            if local.els.is_none();
            if local.ty.is_none();
            if init.span.ctxt() == stmt.span.ctxt();
            if let Some(if_let_or_match) = IfLetOrMatch::parse(cx, init);
            then {
                if_let_or_match
            } else {
                return;
            }
        };

        match if_let_or_match {
            IfLetOrMatch::IfLet(if_let_expr, let_pat, if_then, if_else) => if_chain! {
                if expr_is_simple_identity(let_pat, if_then);
                if let Some(if_else) = if_else;
                if expr_diverges(cx, if_else);
                then {
                    emit_manual_let_else(cx, stmt.span, if_let_expr, let_pat, if_else);
                }
            },
            IfLetOrMatch::Match(match_expr, arms, source) => {
                if self.matches_behaviour == MatchLintBehaviour::Never {
                    return;
                }
                if source != MatchSource::Normal {
                    return;
                }
                // Any other number than two arms doesn't (neccessarily)
                // have a trivial mapping to let else.
                if arms.len() != 2 {
                    return;
                }
                // Guards don't give us an easy mapping either
                if arms.iter().any(|arm| arm.guard.is_some()) {
                    return;
                }
                let check_types = self.matches_behaviour == MatchLintBehaviour::WellKnownTypes;
                let diverging_arm_opt = arms
                    .iter()
                    .enumerate()
                    .find(|(_, arm)| expr_diverges(cx, arm.body) && pat_allowed_for_else(cx, arm.pat, check_types));
                let Some((idx, diverging_arm)) = diverging_arm_opt else { return; };
                let pat_arm = &arms[1 - idx];
                if !expr_is_simple_identity(pat_arm.pat, pat_arm.body) {
                    return;
                }

                emit_manual_let_else(cx, stmt.span, match_expr, pat_arm.pat, diverging_arm.body);
            },
        }
    }

    extract_msrv_attr!(LateContext);
}

fn emit_manual_let_else(cx: &LateContext<'_>, span: Span, expr: &Expr<'_>, pat: &Pat<'_>, else_body: &Expr<'_>) {
    span_lint_and_then(
        cx,
        MANUAL_LET_ELSE,
        span,
        "this could be rewritten as `let...else`",
        |diag| {
            // This is far from perfect, for example there needs to be:
            // * mut additions for the bindings
            // * renamings of the bindings
            // * unused binding collision detection with existing ones
            // * putting patterns with at the top level | inside ()
            // for this to be machine applicable.
            let app = Applicability::HasPlaceholders;

            if let Some(sn_pat) = snippet_opt(cx, pat.span) &&
                let Some(sn_expr) = snippet_opt(cx, expr.span) &&
                let Some(sn_else) = snippet_opt(cx, else_body.span)
            {
                let else_bl = if matches!(else_body.kind, ExprKind::Block(..)) {
                    sn_else
                } else {
                    format!("{{ {sn_else} }}")
                };
                let sugg = format!("let {sn_pat} = {sn_expr} else {else_bl};");
                diag.span_suggestion(span, "consider writing", sugg, app);
            }
        },
    );
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
                let diverges = expr_diverges(cx, match_expr)
                    || match_arms.iter().all(|arm| {
                        let guard_diverges = arm.guard.as_ref().map_or(false, |g| expr_diverges(cx, g.body()));
                        guard_diverges || expr_diverges(cx, arm.body)
                    });
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

fn pat_allowed_for_else(cx: &LateContext<'_>, pat: &'_ Pat<'_>, check_types: bool) -> bool {
    // Check whether the pattern contains any bindings, as the
    // binding might potentially be used in the body.
    // TODO: only look for *used* bindings.
    let mut has_bindings = false;
    pat.each_binding_or_first(&mut |_, _, _, _| has_bindings = true);
    if has_bindings {
        return false;
    }

    // If we shouldn't check the types, exit early.
    if !check_types {
        return true;
    }

    // Check whether any possibly "unknown" patterns are included,
    // because users might not know which values some enum has.
    // Well-known enums are excepted, as we assume people know them.
    // We do a deep check, to be able to disallow Err(En::Foo(_))
    // for usage of the En::Foo variant, as we disallow En::Foo(_),
    // but we allow Err(_).
    let typeck_results = cx.typeck_results();
    let mut has_disallowed = false;
    pat.walk_always(|pat| {
        // Only do the check if the type is "spelled out" in the pattern
        if !matches!(
            pat.kind,
            PatKind::Struct(..) | PatKind::TupleStruct(..) | PatKind::Path(..)
        ) {
            return;
        };
        let ty = typeck_results.pat_ty(pat);
        // Option and Result are allowed, everything else isn't.
        if !(is_type_diagnostic_item(cx, ty, sym::Option) || is_type_diagnostic_item(cx, ty, sym::Result)) {
            has_disallowed = true;
        }
    });
    !has_disallowed
}

/// Checks if the passed block is a simple identity referring to bindings created by the pattern
fn expr_is_simple_identity(pat: &'_ Pat<'_>, expr: &'_ Expr<'_>) -> bool {
    // We support patterns with multiple bindings and tuples, like:
    //   let ... = if let (Some(foo), bar) = g() { (foo, bar) } else { ... }
    let peeled = peel_blocks(expr);
    let paths = match peeled.kind {
        ExprKind::Tup(exprs) | ExprKind::Array(exprs) => exprs,
        ExprKind::Path(_) => std::slice::from_ref(peeled),
        _ => return false,
    };
    let mut pat_bindings = FxHashSet::default();
    pat.each_binding_or_first(&mut |_ann, _hir_id, _sp, ident| {
        pat_bindings.insert(ident);
    });
    if pat_bindings.len() < paths.len() {
        return false;
    }
    for path in paths {
        if_chain! {
            if let ExprKind::Path(QPath::Resolved(_ty, path)) = path.kind;
            if let [path_seg] = path.segments;
            then {
                if !pat_bindings.remove(&path_seg.ident) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
pub enum MatchLintBehaviour {
    AllTypes,
    WellKnownTypes,
    Never,
}
