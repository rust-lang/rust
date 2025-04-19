use std::iter::repeat;
use std::ops::ControlFlow;

use hir::intravisit::{self, Visitor};
use rustc_ast::Recovered;
use rustc_errors::{Applicability, Diag, EmissionGuarantee, Subdiagnostic, SuggestionStyle};
use rustc_hir::{self as hir, HirIdSet};
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::adjustment::Adjust;
use rustc_middle::ty::significant_drop_order::{
    extract_component_with_significant_dtor, ty_dtor_span,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint::{FutureIncompatibilityReason, LintId};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::{DUMMY_SP, Span};
use smallvec::SmallVec;

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `if_let_rescope` lint detects cases where a temporary value with
    /// significant drop is generated on the right hand side of `if let`
    /// and suggests a rewrite into `match` when possible.
    ///
    /// ### Example
    ///
    /// ```rust,edition2021
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
        reference: "<https://doc.rust-lang.org/nightly/edition-guide/rust-2024/temporary-if-let-scope.html>",
    };
}

/// Lint for potential change in program semantics of `if let`s
#[derive(Default)]
pub(crate) struct IfLetRescope {
    skip: HirIdSet,
}

fn expr_parent_is_else(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    let Some((_, hir::Node::Expr(expr))) = tcx.hir_parent_iter(hir_id).next() else {
        return false;
    };
    let hir::ExprKind::If(_cond, _conseq, Some(alt)) = expr.kind else { return false };
    alt.hir_id == hir_id
}

fn expr_parent_is_stmt(tcx: TyCtxt<'_>, hir_id: hir::HirId) -> bool {
    let mut parents = tcx.hir_parent_iter(hir_id);
    let stmt = match parents.next() {
        Some((_, hir::Node::Stmt(stmt))) => stmt,
        Some((_, hir::Node::Block(_) | hir::Node::Arm(_))) => return true,
        _ => return false,
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
        let expr_end = match expr.kind {
            hir::ExprKind::If(_cond, conseq, None) => conseq.span.shrink_to_hi(),
            hir::ExprKind::If(_cond, _conseq, Some(alt)) => alt.span.shrink_to_hi(),
            _ => return,
        };
        let mut seen_dyn = false;
        let mut add_bracket_to_match_head = match_head_needs_bracket(tcx, expr);
        let mut significant_droppers = vec![];
        let mut lifetime_ends = vec![];
        let mut closing_brackets = 0;
        let mut alt_heads = vec![];
        let mut match_heads = vec![];
        let mut consequent_heads = vec![];
        let mut destructors = vec![];
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
                // Peel off round braces
                let if_let_pat = source_map
                    .span_take_while(expr.span, |&ch| ch == '(' || ch.is_whitespace())
                    .between(init.span);
                // The consequent fragment is always a block.
                let before_conseq = conseq.span.shrink_to_lo();
                let lifetime_end = source_map.end_point(conseq.span);

                if let ControlFlow::Break((drop_span, drop_tys)) =
                    (FindSignificantDropper { cx }).check_if_let_scrutinee(init)
                {
                    destructors.extend(drop_tys.into_iter().filter_map(|ty| {
                        if let Some(span) = ty_dtor_span(tcx, ty) {
                            Some(DestructorLabel { span, dtor_kind: "concrete" })
                        } else if matches!(ty.kind(), ty::Dynamic(..)) {
                            if seen_dyn {
                                None
                            } else {
                                seen_dyn = true;
                                Some(DestructorLabel { span: DUMMY_SP, dtor_kind: "dyn" })
                            }
                        } else {
                            None
                        }
                    }));
                    first_if_to_lint = first_if_to_lint.or_else(|| Some((span, expr.hir_id)));
                    significant_droppers.push(drop_span);
                    lifetime_ends.push(lifetime_end);
                    if ty_ascription.is_some()
                        || !expr.span.can_be_used_for_suggestions()
                        || !pat.span.can_be_used_for_suggestions()
                        || !if_let_pat.can_be_used_for_suggestions()
                        || !before_conseq.can_be_used_for_suggestions()
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
                    destructors,
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
        if expr.span.edition().at_least_rust_2024()
            || cx.tcx.lints_that_dont_need_to_run(()).contains(&LintId::of(IF_LET_RESCOPE))
        {
            return;
        }

        if let hir::ExprKind::Loop(block, _label, hir::LoopSource::While, _span) = expr.kind
            && let Some(value) = block.expr
            && let hir::ExprKind::If(cond, _conseq, _alt) = value.kind
            && let hir::ExprKind::Let(..) = cond.kind
        {
            // Recall that `while let` is lowered into this:
            // ```
            // loop {
            //     if let .. { body } else { break; }
            // }
            // ```
            // There is no observable change in drop order on the overall `if let` expression
            // given that the `{ break; }` block is trivial so the edition change
            // means nothing substantial to this `while` statement.
            self.skip.insert(value.hir_id);
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
    #[subdiagnostic]
    destructors: Vec<DestructorLabel>,
    #[label]
    significant_droppers: Vec<Span>,
    #[help]
    lifetime_ends: Vec<Span>,
    #[subdiagnostic]
    rewrite: Option<IfLetRescopeRewrite>,
}

struct IfLetRescopeRewrite {
    match_heads: Vec<SingleArmMatchBegin>,
    consequent_heads: Vec<ConsequentRewrite>,
    closing_brackets: ClosingBrackets,
    alt_heads: Vec<AltHead>,
}

impl Subdiagnostic for IfLetRescopeRewrite {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
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
        let msg = diag.eagerly_translate(crate::fluent_generated::lint_suggestion);
        diag.multipart_suggestion_with_style(
            msg,
            suggestions,
            Applicability::MachineApplicable,
            SuggestionStyle::ShowCode,
        );
    }
}

#[derive(Subdiagnostic)]
#[note(lint_if_let_dtor)]
struct DestructorLabel {
    #[primary_span]
    span: Span,
    dtor_kind: &'static str,
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

struct FindSignificantDropper<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'tcx> FindSignificantDropper<'_, 'tcx> {
    /// Check the scrutinee of an `if let` to see if it promotes any temporary values
    /// that would change drop order in edition 2024. Specifically, it checks the value
    /// of the scrutinee itself, and also recurses into the expression to find any ref
    /// exprs (or autoref) which would promote temporaries that would be scoped to the
    /// end of this `if`.
    fn check_if_let_scrutinee(
        &mut self,
        init: &'tcx hir::Expr<'tcx>,
    ) -> ControlFlow<(Span, SmallVec<[Ty<'tcx>; 4]>)> {
        self.check_promoted_temp_with_drop(init)?;
        self.visit_expr(init)
    }

    /// Check that an expression is not a promoted temporary with a significant
    /// drop impl.
    ///
    /// An expression is a promoted temporary if it has an addr taken (i.e. `&expr` or autoref)
    /// or is the scrutinee of the `if let`, *and* the expression is not a place
    /// expr, and it has a significant drop.
    fn check_promoted_temp_with_drop(
        &self,
        expr: &'tcx hir::Expr<'tcx>,
    ) -> ControlFlow<(Span, SmallVec<[Ty<'tcx>; 4]>)> {
        if expr.is_place_expr(|base| {
            self.cx
                .typeck_results()
                .adjustments()
                .get(base.hir_id)
                .is_some_and(|x| x.iter().any(|adj| matches!(adj.kind, Adjust::Deref(_))))
        }) {
            return ControlFlow::Continue(());
        }

        let drop_tys = extract_component_with_significant_dtor(
            self.cx.tcx,
            self.cx.typing_env(),
            self.cx.typeck_results().expr_ty(expr),
        );
        if drop_tys.is_empty() {
            return ControlFlow::Continue(());
        }

        ControlFlow::Break((expr.span, drop_tys))
    }
}

impl<'tcx> Visitor<'tcx> for FindSignificantDropper<'_, 'tcx> {
    type Result = ControlFlow<(Span, SmallVec<[Ty<'tcx>; 4]>)>;

    fn visit_block(&mut self, b: &'tcx hir::Block<'tcx>) -> Self::Result {
        // Blocks introduce temporary terminating scope for all of its
        // statements, so just visit the tail expr, skipping over any
        // statements. This prevents false positives like `{ let x = &Drop; }`.
        if let Some(expr) = b.expr { self.visit_expr(expr) } else { ControlFlow::Continue(()) }
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) -> Self::Result {
        // Check for promoted temporaries from autoref, e.g.
        // `if let None = TypeWithDrop.as_ref() {} else {}`
        // where `fn as_ref(&self) -> Option<...>`.
        for adj in self.cx.typeck_results().expr_adjustments(expr) {
            match adj.kind {
                // Skip when we hit the first deref expr.
                Adjust::Deref(_) => break,
                Adjust::Borrow(_) => {
                    self.check_promoted_temp_with_drop(expr)?;
                }
                _ => {}
            }
        }

        match expr.kind {
            // Account for cases like `if let None = Some(&Drop) {} else {}`.
            hir::ExprKind::AddrOf(_, _, expr) => {
                self.check_promoted_temp_with_drop(expr)?;
                intravisit::walk_expr(self, expr)
            }
            // `(Drop, ()).1` introduces a temporary and then moves out of
            // part of it, therefore we should check it for temporaries.
            // FIXME: This may have false positives if we move the part
            // that actually has drop, but oh well.
            hir::ExprKind::Index(expr, _, _) | hir::ExprKind::Field(expr, _) => {
                self.check_promoted_temp_with_drop(expr)?;
                intravisit::walk_expr(self, expr)
            }
            // If always introduces a temporary terminating scope for its cond and arms,
            // so don't visit them.
            hir::ExprKind::If(..) => ControlFlow::Continue(()),
            // Match introduces temporary terminating scopes for arms, so don't visit
            // them, and only visit the scrutinee to account for cases like:
            // `if let None = match &Drop { _ => Some(1) } {} else {}`.
            hir::ExprKind::Match(scrut, _, _) => self.visit_expr(scrut),
            // Self explanatory.
            hir::ExprKind::DropTemps(_) => ControlFlow::Continue(()),
            // Otherwise, walk into the expr's parts.
            _ => intravisit::walk_expr(self, expr),
        }
    }
}
