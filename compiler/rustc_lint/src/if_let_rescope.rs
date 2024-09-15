use std::iter::repeat;
use std::ops::ControlFlow;

use hir::intravisit::Visitor;
use rustc_ast::Recovered;
use rustc_errors::{
    Applicability, Diag, EmissionGuarantee, SubdiagMessageOp, Subdiagnostic, SuggestionStyle,
};
use rustc_hir::{self as hir, HirIdSet};
use rustc_macros::LintDiagnostic;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::{FutureIncompatibilityReason, Level};
use rustc_session::{declare_lint, impl_lint_pass};
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
    /// #![cfg_attr(not(bootstrap), feature(if_let_rescope))] // Simplify this in bootstrap bump.
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

/// Lint for potential change in program semantics of `if let`s
#[derive(Default)]
pub(crate) struct IfLetRescope {
    skip: HirIdSet,
}

fn expr_parent_is_else(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    let Some((_, hir::Node::Expr(expr))) = tcx.hir().parent_iter(hir_id).next() else {
        return false;
    };
    let hir::ExprKind::If(_cond, _conseq, Some(alt)) = expr.kind else { return false };
    alt.hir_id == hir_id
}

fn expr_parent_is_stmt(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    let Some((_, hir::Node::Stmt(stmt))) = tcx.hir().parent_iter(hir_id).next() else {
        return false;
    };
    let (hir::StmtKind::Semi(expr) | hir::StmtKind::Expr(expr)) = stmt.kind else { return false };
    expr.hir_id == hir_id
}

fn match_head_needs_bracket(tcx: TyCtxt<'_>, expr: &hir::Expr<'_>) -> bool {
    expr_parent_is_else(tcx, expr.hir_id) && matches!(expr.kind, hir::ExprKind::If(..))
}

impl IfLetRescope {
    fn probe_if_cascade<'tcx>(&mut self, cx: &LateContext<'tcx>, mut expr: &'tcx hir::Expr<'tcx>) {
        if self.skip.contains(&expr.hir_id) {
            return;
        }
        let tcx = cx.tcx;
        let source_map = tcx.sess.source_map();
        let expr_end = expr.span.shrink_to_hi();
        let mut add_bracket_to_match_head = match_head_needs_bracket(tcx, expr);
        let mut significant_droppers = vec![];
        let mut lifetime_ends = vec![];
        let mut closing_brackets = 0;
        let mut alt_heads = vec![];
        let mut match_heads = vec![];
        let mut consequent_heads = vec![];
        let mut first_if_to_lint = None;
        let mut first_if_to_rewrite = false;
        let mut empty_alt = false;
        while let hir::ExprKind::If(cond, conseq, alt) = expr.kind {
            self.skip.insert(expr.hir_id);
            // We are interested in `let` fragment of the condition.
            // Otherwise, we probe into the `else` fragment.
            if let hir::ExprKind::Let(&hir::LetExpr {
                span,
                pat,
                init,
                ty: ty_ascription,
                recovered: Recovered::No,
            }) = cond.kind
            {
                let if_let_pat = expr.span.shrink_to_lo().between(init.span);
                // The consequent fragment is always a block.
                let before_conseq = conseq.span.shrink_to_lo();
                let lifetime_end = source_map.end_point(conseq.span);

                if let ControlFlow::Break(significant_dropper) =
                    (FindSignificantDropper { cx }).visit_expr(init)
                {
                    first_if_to_lint = first_if_to_lint.or_else(|| Some((span, expr.hir_id)));
                    significant_droppers.push(significant_dropper);
                    lifetime_ends.push(lifetime_end);
                    if ty_ascription.is_some()
                        || !expr.span.can_be_used_for_suggestions()
                        || !pat.span.can_be_used_for_suggestions()
                    {
                        // Our `match` rewrites does not support type ascription,
                        // so we just bail.
                        // Alternatively when the span comes from proc macro expansion,
                        // we will also bail.
                        // FIXME(#101728): change this when type ascription syntax is stabilized again
                    } else if let Ok(pat) = source_map.span_to_snippet(pat.span) {
                        let emit_suggestion = |alt_span| {
                            first_if_to_rewrite = true;
                            if add_bracket_to_match_head {
                                closing_brackets += 2;
                                match_heads.push(SingleArmMatchBegin::WithOpenBracket(if_let_pat));
                            } else {
                                // Sometimes, wrapping `match` into a block is undesirable,
                                // because the scrutinee temporary lifetime is shortened and
                                // the proposed fix will not work.
                                closing_brackets += 1;
                                match_heads
                                    .push(SingleArmMatchBegin::WithoutOpenBracket(if_let_pat));
                            }
                            consequent_heads.push(ConsequentRewrite { span: before_conseq, pat });
                            if let Some(alt_span) = alt_span {
                                alt_heads.push(AltHead(alt_span));
                            }
                        };
                        if let Some(alt) = alt {
                            let alt_head = conseq.span.between(alt.span);
                            if alt_head.can_be_used_for_suggestions() {
                                // We lint only when the `else` span is user code, too.
                                emit_suggestion(Some(alt_head));
                            }
                        } else {
                            // This is the end of the `if .. else ..` cascade.
                            // We can stop here.
                            emit_suggestion(None);
                            empty_alt = true;
                            break;
                        }
                    }
                }
            }
            // At this point, any `if let` fragment in the cascade is definitely preceeded by `else`,
            // so a opening bracket is mandatory before each `match`.
            add_bracket_to_match_head = true;
            if let Some(alt) = alt {
                expr = alt;
            } else {
                break;
            }
        }
        if let Some((span, hir_id)) = first_if_to_lint {
            tcx.emit_node_span_lint(
                IF_LET_RESCOPE,
                hir_id,
                span,
                IfLetRescopeLint {
                    significant_droppers,
                    lifetime_ends,
                    rewrite: first_if_to_rewrite.then_some(IfLetRescopeRewrite {
                        match_heads,
                        consequent_heads,
                        closing_brackets: ClosingBrackets {
                            span: expr_end,
                            count: closing_brackets,
                            empty_alt,
                        },
                        alt_heads,
                    }),
                },
            );
        }
    }
}

impl_lint_pass!(
    IfLetRescope => [IF_LET_RESCOPE]
);

impl<'tcx> LateLintPass<'tcx> for IfLetRescope {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if expr.span.edition().at_least_rust_2024() || !cx.tcx.features().if_let_rescope {
            return;
        }
        if let (Level::Allow, _) = cx.tcx.lint_level_at_node(IF_LET_RESCOPE, expr.hir_id) {
            return;
        }
        if expr_parent_is_stmt(cx.tcx, expr.hir_id)
            && matches!(expr.kind, hir::ExprKind::If(_cond, _conseq, None))
        {
            // `if let` statement without an `else` branch has no observable change
            // so we can skip linting it
            return;
        }
        self.probe_if_cascade(cx, expr);
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_if_let_rescope)]
struct IfLetRescopeLint {
    #[label]
    significant_droppers: Vec<Span>,
    #[help]
    lifetime_ends: Vec<Span>,
    #[subdiagnostic]
    rewrite: Option<IfLetRescopeRewrite>,
}

// #[derive(Subdiagnostic)]
struct IfLetRescopeRewrite {
    match_heads: Vec<SingleArmMatchBegin>,
    consequent_heads: Vec<ConsequentRewrite>,
    closing_brackets: ClosingBrackets,
    alt_heads: Vec<AltHead>,
}

impl Subdiagnostic for IfLetRescopeRewrite {
    fn add_to_diag_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        f: &F,
    ) {
        let mut suggestions = vec![];
        for match_head in self.match_heads {
            match match_head {
                SingleArmMatchBegin::WithOpenBracket(span) => {
                    suggestions.push((span, "{ match ".into()))
                }
                SingleArmMatchBegin::WithoutOpenBracket(span) => {
                    suggestions.push((span, "match ".into()))
                }
            }
        }
        for ConsequentRewrite { span, pat } in self.consequent_heads {
            suggestions.push((span, format!("{{ {pat} => ")));
        }
        for AltHead(span) in self.alt_heads {
            suggestions.push((span, " _ => ".into()));
        }
        let closing_brackets = self.closing_brackets;
        suggestions.push((
            closing_brackets.span,
            closing_brackets
                .empty_alt
                .then_some(" _ => {}".chars())
                .into_iter()
                .flatten()
                .chain(repeat('}').take(closing_brackets.count))
                .collect(),
        ));
        let msg = f(diag, crate::fluent_generated::lint_suggestion.into());
        diag.multipart_suggestion_with_style(
            msg,
            suggestions,
            Applicability::MachineApplicable,
            SuggestionStyle::ShowCode,
        );
    }
}

struct AltHead(Span);

struct ConsequentRewrite {
    span: Span,
    pat: String,
}

struct ClosingBrackets {
    span: Span,
    count: usize,
    empty_alt: bool,
}
enum SingleArmMatchBegin {
    WithOpenBracket(Span),
    WithoutOpenBracket(Span),
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
