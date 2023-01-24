use std::ops::ControlFlow;

use hir::intravisit::Visitor;
use rustc_ast::Recovered;
use rustc_hir as hir;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::Span;

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `if_let_rescope` lint detects cases where a temporary value with
    /// significant drop is generated on the right hand side of `if let`
    /// and suggests a rewrite into `match` when possible.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
    /// #![feature(if_let_rescope)]
    /// #![warn(if_let_rescope)]
    /// #![allow(unused_variables)]
    ///
    /// struct Droppy;
    /// impl Drop for Droppy {
    ///     fn drop(&mut self) {
    ///         // Custom destructor, including this `drop` implementation, is considered
    ///         // significant.
    ///         // Rust does not check whether this destructor emits side-effects that can
    ///         // lead to observable change in program semantics, when the drop order changes.
    ///         // Rust biases to be on the safe side, so that you can apply discretion whether
    ///         // this change indeed breaches any contract or specification that your code needs
    ///         // to honour.
    ///         println!("dropped");
    ///     }
    /// }
    /// impl Droppy {
    ///     fn get(&self) -> Option<u8> {
    ///         None
    ///     }
    /// }
    ///
    /// fn main() {
    ///     if let Some(value) = Droppy.get() {
    ///         // do something
    ///     } else {
    ///         // do something else
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// With Edition 2024, temporaries generated while evaluating `if let`s
    /// will be dropped before the `else` block.
    /// This lint captures a possible change in runtime behaviour due to
    /// a change in sequence of calls to significant `Drop::drop` destructors.
    ///
    /// A significant [`Drop::drop`](https://doc.rust-lang.org/std/ops/trait.Drop.html)
    /// destructor here refers to an explicit, arbitrary implementation of the `Drop` trait on the type
    /// with exceptions including `Vec`, `Box`, `Rc`, `BTreeMap` and `HashMap`
    /// that are marked by the compiler otherwise so long that the generic types have
    /// no significant destructor recursively.
    /// In other words, a type has a significant drop destructor when it has a `Drop` implementation
    /// or its destructor invokes a significant destructor on a type.
    /// Since we cannot completely reason about the change by just inspecting the existence of
    /// a significant destructor, this lint remains only a suggestion and is set to `allow` by default.
    ///
    /// Whenever possible, a rewrite into an equivalent `match` expression that
    /// observe the same order of calls to such destructors is proposed by this lint.
    /// Authors may take their own discretion whether the rewrite suggestion shall be
    /// accepted, or rejected to continue the use of the `if let` expression.
    pub IF_LET_RESCOPE,
    Allow,
    "`if let` assigns a shorter lifetime to temporary values being pattern-matched against in Edition 2024 and \
    rewriting in `match` is an option to preserve the semantics up to Edition 2021",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "issue #124085 <https://github.com/rust-lang/rust/issues/124085>",
    };
}

declare_lint_pass!(
    /// Lint for potential change in program semantics of `if let`s
    IfLetRescope => [IF_LET_RESCOPE]
);

impl<'tcx> LateLintPass<'tcx> for IfLetRescope {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if !expr.span.edition().at_least_rust_2021() || !cx.tcx.features().if_let_rescope {
            return;
        }
        let hir::ExprKind::If(cond, conseq, alt) = expr.kind else { return };
        let hir::ExprKind::Let(&hir::LetExpr {
            span,
            pat,
            init,
            ty: ty_ascription,
            recovered: Recovered::No,
        }) = cond.kind
        else {
            return;
        };
        let source_map = cx.tcx.sess.source_map();
        let expr_end = expr.span.shrink_to_hi();
        let if_let_pat = expr.span.shrink_to_lo().between(init.span);
        let before_conseq = conseq.span.shrink_to_lo();
        let lifetime_end = source_map.end_point(conseq.span);

        if let ControlFlow::Break(significant_dropper) =
            (FindSignificantDropper { cx }).visit_expr(init)
        {
            let lint_without_suggestion = || {
                cx.tcx.emit_node_span_lint(
                    IF_LET_RESCOPE,
                    expr.hir_id,
                    span,
                    IfLetRescopeRewrite { significant_dropper, lifetime_end, sugg: None },
                )
            };
            if ty_ascription.is_some()
                || !expr.span.can_be_used_for_suggestions()
                || !pat.span.can_be_used_for_suggestions()
            {
                // Our `match` rewrites does not support type ascription,
                // so we just bail.
                // Alternatively when the span comes from proc macro expansion,
                // we will also bail.
                // FIXME(#101728): change this when type ascription syntax is stabilized again
                lint_without_suggestion();
            } else {
                let Ok(pat) = source_map.span_to_snippet(pat.span) else {
                    lint_without_suggestion();
                    return;
                };
                if let Some(alt) = alt {
                    let alt_start = conseq.span.between(alt.span);
                    if !alt_start.can_be_used_for_suggestions() {
                        lint_without_suggestion();
                        return;
                    }
                    cx.tcx.emit_node_span_lint(
                        IF_LET_RESCOPE,
                        expr.hir_id,
                        span,
                        IfLetRescopeRewrite {
                            significant_dropper,
                            lifetime_end,
                            sugg: Some(IfLetRescopeRewriteSuggestion::WithElse {
                                if_let_pat,
                                before_conseq,
                                pat,
                                expr_end,
                                alt_start,
                            }),
                        },
                    );
                } else {
                    cx.tcx.emit_node_span_lint(
                        IF_LET_RESCOPE,
                        expr.hir_id,
                        span,
                        IfLetRescopeRewrite {
                            significant_dropper,
                            lifetime_end,
                            sugg: Some(IfLetRescopeRewriteSuggestion::WithoutElse {
                                if_let_pat,
                                before_conseq,
                                pat,
                                expr_end,
                            }),
                        },
                    );
                }
            }
        }
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_if_let_rescope)]
struct IfLetRescopeRewrite {
    #[label]
    significant_dropper: Span,
    #[help]
    lifetime_end: Span,
    #[subdiagnostic]
    sugg: Option<IfLetRescopeRewriteSuggestion>,
}

#[derive(Subdiagnostic)]
enum IfLetRescopeRewriteSuggestion {
    #[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
    WithElse {
        #[suggestion_part(code = "match ")]
        if_let_pat: Span,
        #[suggestion_part(code = " {{ {pat} => ")]
        before_conseq: Span,
        pat: String,
        #[suggestion_part(code = "}}")]
        expr_end: Span,
        #[suggestion_part(code = " _ => ")]
        alt_start: Span,
    },
    #[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
    WithoutElse {
        #[suggestion_part(code = "match ")]
        if_let_pat: Span,
        #[suggestion_part(code = " {{ {pat} => ")]
        before_conseq: Span,
        pat: String,
        #[suggestion_part(code = " _ => {{}} }}")]
        expr_end: Span,
    },
}

struct FindSignificantDropper<'tcx, 'a> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx, 'a> Visitor<'tcx> for FindSignificantDropper<'tcx, 'a> {
    type Result = ControlFlow<Span>;

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) -> Self::Result {
        if self
            .cx
            .typeck_results()
            .expr_ty(expr)
            .has_significant_drop(self.cx.tcx, self.cx.param_env)
        {
            return ControlFlow::Break(expr.span);
        }
        match expr.kind {
            hir::ExprKind::ConstBlock(_)
            | hir::ExprKind::Lit(_)
            | hir::ExprKind::Path(_)
            | hir::ExprKind::Assign(_, _, _)
            | hir::ExprKind::AssignOp(_, _, _)
            | hir::ExprKind::Break(_, _)
            | hir::ExprKind::Continue(_)
            | hir::ExprKind::Ret(_)
            | hir::ExprKind::Become(_)
            | hir::ExprKind::InlineAsm(_)
            | hir::ExprKind::OffsetOf(_, _)
            | hir::ExprKind::Repeat(_, _)
            | hir::ExprKind::Err(_)
            | hir::ExprKind::Struct(_, _, _)
            | hir::ExprKind::Closure(_)
            | hir::ExprKind::Block(_, _)
            | hir::ExprKind::DropTemps(_)
            | hir::ExprKind::Loop(_, _, _, _) => ControlFlow::Continue(()),

            hir::ExprKind::Tup(exprs) | hir::ExprKind::Array(exprs) => {
                for expr in exprs {
                    self.visit_expr(expr)?;
                }
                ControlFlow::Continue(())
            }
            hir::ExprKind::Call(callee, args) => {
                self.visit_expr(callee)?;
                for expr in args {
                    self.visit_expr(expr)?;
                }
                ControlFlow::Continue(())
            }
            hir::ExprKind::MethodCall(_, receiver, args, _) => {
                self.visit_expr(receiver)?;
                for expr in args {
                    self.visit_expr(expr)?;
                }
                ControlFlow::Continue(())
            }
            hir::ExprKind::Index(left, right, _) | hir::ExprKind::Binary(_, left, right) => {
                self.visit_expr(left)?;
                self.visit_expr(right)
            }
            hir::ExprKind::Unary(_, expr)
            | hir::ExprKind::Cast(expr, _)
            | hir::ExprKind::Type(expr, _)
            | hir::ExprKind::Yield(expr, _)
            | hir::ExprKind::AddrOf(_, _, expr)
            | hir::ExprKind::Match(expr, _, _)
            | hir::ExprKind::Field(expr, _)
            | hir::ExprKind::Let(&hir::LetExpr {
                init: expr,
                span: _,
                pat: _,
                ty: _,
                recovered: Recovered::No,
            }) => self.visit_expr(expr),
            hir::ExprKind::Let(_) => ControlFlow::Continue(()),

            hir::ExprKind::If(cond, _, _) => {
                if let hir::ExprKind::Let(hir::LetExpr {
                    init,
                    span: _,
                    pat: _,
                    ty: _,
                    recovered: Recovered::No,
                }) = cond.kind
                {
                    self.visit_expr(init)?;
                }
                ControlFlow::Continue(())
            }
        }
    }
}
