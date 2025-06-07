use rustc_arena::{DroplessArena, TypedArena};
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, ErrorGuaranteed, MultiSpan, struct_span_code_err};
use rustc_hir::def::*;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, BindingMode, ByRef, HirId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::Level;
use rustc_middle::bug;
use rustc_middle::thir::visit::Visitor;
use rustc_middle::thir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_pattern_analysis::errors::Uncovered;
use rustc_pattern_analysis::rustc::{
    Constructor, DeconstructedPat, MatchArm, RedundancyExplanation, RevealedTy,
    RustcPatCtxt as PatCtxt, Usefulness, UsefulnessReport, WitnessPat,
};
use rustc_session::lint::builtin::{
    BINDINGS_WITH_VARIANT_NAME, IRREFUTABLE_LET_PATTERNS, UNREACHABLE_PATTERNS,
};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::{Ident, Span};
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::instrument;

use crate::errors::*;
use crate::fluent_generated as fluent;

pub(crate) fn check_match(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let typeck_results = tcx.typeck(def_id);
    let (thir, expr) = tcx.thir_body(def_id)?;
    let thir = thir.borrow();
    let pattern_arena = TypedArena::default();
    let dropless_arena = DroplessArena::default();
    let mut visitor = MatchVisitor {
        tcx,
        thir: &*thir,
        typeck_results,
        // FIXME(#132279): We're in a body, should handle opaques.
        typing_env: ty::TypingEnv::non_body_analysis(tcx, def_id),
        lint_level: tcx.local_def_id_to_hir_id(def_id),
        let_source: LetSource::None,
        pattern_arena: &pattern_arena,
        dropless_arena: &dropless_arena,
        error: Ok(()),
    };
    visitor.visit_expr(&thir[expr]);

    let origin = match tcx.def_kind(def_id) {
        DefKind::AssocFn | DefKind::Fn => "function argument",
        DefKind::Closure => "closure argument",
        // other types of MIR don't have function parameters, and we don't need to
        // categorize those for the irrefutable check.
        _ if thir.params.is_empty() => "",
        kind => bug!("unexpected function parameters in THIR: {kind:?} {def_id:?}"),
    };

    for param in thir.params.iter() {
        if let Some(box ref pattern) = param.pat {
            visitor.check_binding_is_irrefutable(pattern, origin, None, None);
        }
    }
    visitor.error
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum RefutableFlag {
    Irrefutable,
    Refutable,
}
use RefutableFlag::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LetSource {
    None,
    PlainLet,
    IfLet,
    IfLetGuard,
    LetElse,
    WhileLet,
    Else,
    ElseIfLet,
}

struct MatchVisitor<'p, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
    thir: &'p Thir<'tcx>,
    lint_level: HirId,
    let_source: LetSource,
    pattern_arena: &'p TypedArena<DeconstructedPat<'p, 'tcx>>,
    dropless_arena: &'p DroplessArena,
    /// Tracks if we encountered an error while checking this body. That the first function to
    /// report it stores it here. Some functions return `Result` to allow callers to short-circuit
    /// on error, but callers don't need to store it here again.
    error: Result<(), ErrorGuaranteed>,
}

// Visitor for a thir body. This calls `check_match`, `check_let` and `check_let_chain` as
// appropriate.
impl<'p, 'tcx> Visitor<'p, 'tcx> for MatchVisitor<'p, 'tcx> {
    fn thir(&self) -> &'p Thir<'tcx> {
        self.thir
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_arm(&mut self, arm: &'p Arm<'tcx>) {
        self.with_lint_level(arm.lint_level, |this| {
            if let Some(expr) = arm.guard {
                this.with_let_source(LetSource::IfLetGuard, |this| {
                    this.visit_expr(&this.thir[expr])
                });
            }
            this.visit_pat(&arm.pattern);
            this.visit_expr(&self.thir[arm.body]);
        });
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_expr(&mut self, ex: &'p Expr<'tcx>) {
        match ex.kind {
            ExprKind::Scope { value, lint_level, .. } => {
                self.with_lint_level(lint_level, |this| {
                    this.visit_expr(&this.thir[value]);
                });
                return;
            }
            ExprKind::If { cond, then, else_opt, if_then_scope: _ } => {
                // Give a specific `let_source` for the condition.
                let let_source = match ex.span.desugaring_kind() {
                    Some(DesugaringKind::WhileLoop) => LetSource::WhileLet,
                    _ => match self.let_source {
                        LetSource::Else => LetSource::ElseIfLet,
                        _ => LetSource::IfLet,
                    },
                };
                self.with_let_source(let_source, |this| this.visit_expr(&self.thir[cond]));
                self.with_let_source(LetSource::None, |this| {
                    this.visit_expr(&this.thir[then]);
                });
                if let Some(else_) = else_opt {
                    self.with_let_source(LetSource::Else, |this| {
                        this.visit_expr(&this.thir[else_])
                    });
                }
                return;
            }
            ExprKind::Match { scrutinee, box ref arms, match_source } => {
                self.check_match(scrutinee, arms, match_source, ex.span);
            }
            ExprKind::Let { box ref pat, expr } => {
                self.check_let(pat, Some(expr), ex.span);
            }
            ExprKind::LogicalOp { op: LogicalOp::And, .. }
                if !matches!(self.let_source, LetSource::None) =>
            {
                let mut chain_refutabilities = Vec::new();
                let Ok(()) = self.visit_land(ex, &mut chain_refutabilities) else { return };
                // If at least one of the operands is a `let ... = ...`.
                if chain_refutabilities.iter().any(|x| x.is_some()) {
                    self.check_let_chain(chain_refutabilities, ex.span);
                }
                return;
            }
            _ => {}
        };
        self.with_let_source(LetSource::None, |this| visit::walk_expr(this, ex));
    }

    fn visit_stmt(&mut self, stmt: &'p Stmt<'tcx>) {
        match stmt.kind {
            StmtKind::Let {
                box ref pattern, initializer, else_block, lint_level, span, ..
            } => {
                self.with_lint_level(lint_level, |this| {
                    let let_source =
                        if else_block.is_some() { LetSource::LetElse } else { LetSource::PlainLet };
                    this.with_let_source(let_source, |this| {
                        this.check_let(pattern, initializer, span)
                    });
                    visit::walk_stmt(this, stmt);
                });
            }
            StmtKind::Expr { .. } => {
                visit::walk_stmt(self, stmt);
            }
        }
    }
}

impl<'p, 'tcx> MatchVisitor<'p, 'tcx> {
    #[instrument(level = "trace", skip(self, f))]
    fn with_let_source(&mut self, let_source: LetSource, f: impl FnOnce(&mut Self)) {
        let old_let_source = self.let_source;
        self.let_source = let_source;
        ensure_sufficient_stack(|| f(self));
        self.let_source = old_let_source;
    }

    fn with_lint_level<T>(
        &mut self,
        new_lint_level: LintLevel,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        if let LintLevel::Explicit(hir_id) = new_lint_level {
            let old_lint_level = self.lint_level;
            self.lint_level = hir_id;
            let ret = f(self);
            self.lint_level = old_lint_level;
            ret
        } else {
            f(self)
        }
    }

    /// Visit a nested chain of `&&`. Used for if-let chains. This must call `visit_expr` on the
    /// subexpressions we are not handling ourselves.
    fn visit_land(
        &mut self,
        ex: &'p Expr<'tcx>,
        accumulator: &mut Vec<Option<(Span, RefutableFlag)>>,
    ) -> Result<(), ErrorGuaranteed> {
        match ex.kind {
            ExprKind::Scope { value, lint_level, .. } => self.with_lint_level(lint_level, |this| {
                this.visit_land(&this.thir[value], accumulator)
            }),
            ExprKind::LogicalOp { op: LogicalOp::And, lhs, rhs } => {
                // We recurse into the lhs only, because `&&` chains associate to the left.
                let res_lhs = self.visit_land(&self.thir[lhs], accumulator);
                let res_rhs = self.visit_land_rhs(&self.thir[rhs])?;
                accumulator.push(res_rhs);
                res_lhs
            }
            _ => {
                let res = self.visit_land_rhs(ex)?;
                accumulator.push(res);
                Ok(())
            }
        }
    }

    /// Visit the right-hand-side of a `&&`. Used for if-let chains. Returns `Some` if the
    /// expression was ultimately a `let ... = ...`, and `None` if it was a normal boolean
    /// expression. This must call `visit_expr` on the subexpressions we are not handling ourselves.
    fn visit_land_rhs(
        &mut self,
        ex: &'p Expr<'tcx>,
    ) -> Result<Option<(Span, RefutableFlag)>, ErrorGuaranteed> {
        match ex.kind {
            ExprKind::Scope { value, lint_level, .. } => {
                self.with_lint_level(lint_level, |this| this.visit_land_rhs(&this.thir[value]))
            }
            ExprKind::Let { box ref pat, expr } => {
                let expr = &self.thir()[expr];
                self.with_let_source(LetSource::None, |this| {
                    this.visit_expr(expr);
                });
                Ok(Some((ex.span, self.is_let_irrefutable(pat, Some(expr))?)))
            }
            _ => {
                self.with_let_source(LetSource::None, |this| {
                    this.visit_expr(ex);
                });
                Ok(None)
            }
        }
    }

    fn lower_pattern(
        &mut self,
        cx: &PatCtxt<'p, 'tcx>,
        pat: &'p Pat<'tcx>,
    ) -> Result<&'p DeconstructedPat<'p, 'tcx>, ErrorGuaranteed> {
        if let Err(err) = pat.pat_error_reported() {
            self.error = Err(err);
            Err(err)
        } else {
            // Check the pattern for some things unrelated to exhaustiveness.
            let refutable = if cx.refutable { Refutable } else { Irrefutable };
            let mut err = Ok(());
            pat.walk_always(|pat| {
                check_borrow_conflicts_in_at_patterns(self, pat);
                check_for_bindings_named_same_as_variants(self, pat, refutable);
                err = err.and(check_never_pattern(cx, pat));
            });
            err?;
            Ok(self.pattern_arena.alloc(cx.lower_pat(pat)))
        }
    }

    /// Inspects the match scrutinee expression to determine whether the place it evaluates to may
    /// hold invalid data.
    fn is_known_valid_scrutinee(&self, scrutinee: &Expr<'tcx>) -> bool {
        use ExprKind::*;
        match &scrutinee.kind {
            // Pointers can validly point to a place with invalid data. It is undecided whether
            // references can too, so we conservatively assume they can.
            Deref { .. } => false,
            // Inherit validity of the parent place, unless the parent is an union.
            Field { lhs, .. } => {
                let lhs = &self.thir()[*lhs];
                match lhs.ty.kind() {
                    ty::Adt(def, _) if def.is_union() => false,
                    _ => self.is_known_valid_scrutinee(lhs),
                }
            }
            // Essentially a field access.
            Index { lhs, .. } => {
                let lhs = &self.thir()[*lhs];
                self.is_known_valid_scrutinee(lhs)
            }

            // No-op.
            Scope { value, .. } => self.is_known_valid_scrutinee(&self.thir()[*value]),

            // Casts don't cause a load.
            NeverToAny { source }
            | Cast { source }
            | Use { source }
            | PointerCoercion { source, .. }
            | PlaceTypeAscription { source, .. }
            | ValueTypeAscription { source, .. }
            | PlaceUnwrapUnsafeBinder { source }
            | ValueUnwrapUnsafeBinder { source }
            | WrapUnsafeBinder { source } => self.is_known_valid_scrutinee(&self.thir()[*source]),

            // These diverge.
            Become { .. }
            | Break { .. }
            | Continue { .. }
            | ConstContinue { .. }
            | Return { .. } => true,

            // These are statements that evaluate to `()`.
            Assign { .. } | AssignOp { .. } | InlineAsm { .. } | Let { .. } => true,

            // These evaluate to a value.
            RawBorrow { .. }
            | Adt { .. }
            | Array { .. }
            | Binary { .. }
            | Block { .. }
            | Borrow { .. }
            | Box { .. }
            | Call { .. }
            | ByUse { .. }
            | Closure { .. }
            | ConstBlock { .. }
            | ConstParam { .. }
            | If { .. }
            | Literal { .. }
            | LogicalOp { .. }
            | Loop { .. }
            | LoopMatch { .. }
            | Match { .. }
            | NamedConst { .. }
            | NonHirLiteral { .. }
            | OffsetOf { .. }
            | Repeat { .. }
            | StaticRef { .. }
            | ThreadLocalRef { .. }
            | Tuple { .. }
            | Unary { .. }
            | UpvarRef { .. }
            | VarRef { .. }
            | ZstLiteral { .. }
            | Yield { .. } => true,
        }
    }

    fn new_cx(
        &self,
        refutability: RefutableFlag,
        whole_match_span: Option<Span>,
        scrutinee: Option<&Expr<'tcx>>,
        scrut_span: Span,
    ) -> PatCtxt<'p, 'tcx> {
        let refutable = match refutability {
            Irrefutable => false,
            Refutable => true,
        };
        // If we don't have a scrutinee we're either a function parameter or a `let x;`. Both cases
        // require validity.
        let known_valid_scrutinee =
            scrutinee.map(|scrut| self.is_known_valid_scrutinee(scrut)).unwrap_or(true);
        PatCtxt {
            tcx: self.tcx,
            typeck_results: self.typeck_results,
            typing_env: self.typing_env,
            module: self.tcx.parent_module(self.lint_level).to_def_id(),
            dropless_arena: self.dropless_arena,
            match_lint_level: self.lint_level,
            whole_match_span,
            scrut_span,
            refutable,
            known_valid_scrutinee,
        }
    }

    fn analyze_patterns(
        &mut self,
        cx: &PatCtxt<'p, 'tcx>,
        arms: &[MatchArm<'p, 'tcx>],
        scrut_ty: Ty<'tcx>,
    ) -> Result<UsefulnessReport<'p, 'tcx>, ErrorGuaranteed> {
        let report =
            rustc_pattern_analysis::rustc::analyze_match(&cx, &arms, scrut_ty).map_err(|err| {
                self.error = Err(err);
                err
            })?;

        // Warn unreachable subpatterns.
        for (arm, is_useful) in report.arm_usefulness.iter() {
            if let Usefulness::Useful(redundant_subpats) = is_useful
                && !redundant_subpats.is_empty()
            {
                let mut redundant_subpats = redundant_subpats.clone();
                // Emit lints in the order in which they occur in the file.
                redundant_subpats.sort_unstable_by_key(|(pat, _)| pat.data().span);
                for (pat, explanation) in redundant_subpats {
                    report_unreachable_pattern(cx, arm.arm_data, pat, &explanation, None)
                }
            }
        }
        Ok(report)
    }

    #[instrument(level = "trace", skip(self))]
    fn check_let(&mut self, pat: &'p Pat<'tcx>, scrutinee: Option<ExprId>, span: Span) {
        assert!(self.let_source != LetSource::None);
        let scrut = scrutinee.map(|id| &self.thir[id]);
        if let LetSource::PlainLet = self.let_source {
            self.check_binding_is_irrefutable(pat, "local binding", scrut, Some(span))
        } else {
            let Ok(refutability) = self.is_let_irrefutable(pat, scrut) else { return };
            if matches!(refutability, Irrefutable) {
                report_irrefutable_let_patterns(
                    self.tcx,
                    self.lint_level,
                    self.let_source,
                    1,
                    span,
                );
            }
        }
    }

    fn check_match(
        &mut self,
        scrut: ExprId,
        arms: &[ArmId],
        source: hir::MatchSource,
        expr_span: Span,
    ) {
        let scrut = &self.thir[scrut];
        let cx = self.new_cx(Refutable, Some(expr_span), Some(scrut), scrut.span);

        let mut tarms = Vec::with_capacity(arms.len());
        for &arm in arms {
            let arm = &self.thir.arms[arm];
            let got_error = self.with_lint_level(arm.lint_level, |this| {
                let Ok(pat) = this.lower_pattern(&cx, &arm.pattern) else { return true };
                let arm =
                    MatchArm { pat, arm_data: this.lint_level, has_guard: arm.guard.is_some() };
                tarms.push(arm);
                false
            });
            if got_error {
                return;
            }
        }

        let Ok(report) = self.analyze_patterns(&cx, &tarms, scrut.ty) else { return };

        match source {
            // Don't report arm reachability of desugared `match $iter.into_iter() { iter => .. }`
            // when the iterator is an uninhabited type. unreachable_code will trigger instead.
            hir::MatchSource::ForLoopDesugar if arms.len() == 1 => {}
            hir::MatchSource::ForLoopDesugar
            | hir::MatchSource::Postfix
            | hir::MatchSource::Normal
            | hir::MatchSource::FormatArgs => {
                let is_match_arm =
                    matches!(source, hir::MatchSource::Postfix | hir::MatchSource::Normal);
                report_arm_reachability(&cx, &report, is_match_arm);
            }
            // Unreachable patterns in try and await expressions occur when one of
            // the arms are an uninhabited type. Which is OK.
            hir::MatchSource::AwaitDesugar | hir::MatchSource::TryDesugar(_) => {}
        }

        // Check if the match is exhaustive.
        let witnesses = report.non_exhaustiveness_witnesses;
        if !witnesses.is_empty() {
            if source == hir::MatchSource::ForLoopDesugar
                && let [_, snd_arm] = *arms
            {
                // the for loop pattern is not irrefutable
                let pat = &self.thir[snd_arm].pattern;
                // `pat` should be `Some(<pat_field>)` from a desugared for loop.
                debug_assert_eq!(pat.span.desugaring_kind(), Some(DesugaringKind::ForLoop));
                let PatKind::Variant { ref subpatterns, .. } = pat.kind else { bug!() };
                let [pat_field] = &subpatterns[..] else { bug!() };
                self.check_binding_is_irrefutable(
                    &pat_field.pattern,
                    "`for` loop binding",
                    None,
                    None,
                );
            } else {
                // span after scrutinee, or after `.match`. That is, the braces, arms,
                // and any whitespace preceding the braces.
                let braces_span = match source {
                    hir::MatchSource::Normal => scrut
                        .span
                        .find_ancestor_in_same_ctxt(expr_span)
                        .map(|scrut_span| scrut_span.shrink_to_hi().with_hi(expr_span.hi())),
                    hir::MatchSource::Postfix => {
                        // This is horrendous, and we should deal with it by just
                        // stashing the span of the braces somewhere (like in the match source).
                        scrut.span.find_ancestor_in_same_ctxt(expr_span).and_then(|scrut_span| {
                            let sm = self.tcx.sess.source_map();
                            let brace_span = sm.span_extend_to_next_char(scrut_span, '{', true);
                            if sm.span_to_snippet(sm.next_point(brace_span)).as_deref() == Ok("{") {
                                let sp = brace_span.shrink_to_hi().with_hi(expr_span.hi());
                                // We also need to extend backwards for whitespace
                                sm.span_extend_prev_while(sp, |c| c.is_whitespace()).ok()
                            } else {
                                None
                            }
                        })
                    }
                    hir::MatchSource::ForLoopDesugar
                    | hir::MatchSource::TryDesugar(_)
                    | hir::MatchSource::AwaitDesugar
                    | hir::MatchSource::FormatArgs => None,
                };
                self.error = Err(report_non_exhaustive_match(
                    &cx,
                    self.thir,
                    scrut.ty,
                    scrut.span,
                    witnesses,
                    arms,
                    braces_span,
                ));
            }
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn check_let_chain(
        &mut self,
        chain_refutabilities: Vec<Option<(Span, RefutableFlag)>>,
        whole_chain_span: Span,
    ) {
        assert!(self.let_source != LetSource::None);

        if chain_refutabilities.iter().all(|r| matches!(*r, Some((_, Irrefutable)))) {
            // The entire chain is made up of irrefutable `let` statements
            report_irrefutable_let_patterns(
                self.tcx,
                self.lint_level,
                self.let_source,
                chain_refutabilities.len(),
                whole_chain_span,
            );
            return;
        }

        if let Some(until) =
            chain_refutabilities.iter().position(|r| !matches!(*r, Some((_, Irrefutable))))
            && until > 0
        {
            // The chain has a non-zero prefix of irrefutable `let` statements.

            // Check if the let source is while, for there is no alternative place to put a prefix,
            // and we shouldn't lint.
            // For let guards inside a match, prefixes might use bindings of the match pattern,
            // so can't always be moved out.
            // For `else if let`, an extra indentation level would be required to move the bindings.
            // FIXME: Add checking whether the bindings are actually used in the prefix,
            // and lint if they are not.
            if !matches!(
                self.let_source,
                LetSource::WhileLet | LetSource::IfLetGuard | LetSource::ElseIfLet
            ) {
                // Emit the lint
                let prefix = &chain_refutabilities[..until];
                let span_start = prefix[0].unwrap().0;
                let span_end = prefix.last().unwrap().unwrap().0;
                let span = span_start.to(span_end);
                let count = prefix.len();
                self.tcx.emit_node_span_lint(
                    IRREFUTABLE_LET_PATTERNS,
                    self.lint_level,
                    span,
                    LeadingIrrefutableLetPatterns { count },
                );
            }
        }

        if let Some(from) =
            chain_refutabilities.iter().rposition(|r| !matches!(*r, Some((_, Irrefutable))))
            && from != (chain_refutabilities.len() - 1)
        {
            // The chain has a non-empty suffix of irrefutable `let` statements
            let suffix = &chain_refutabilities[from + 1..];
            let span_start = suffix[0].unwrap().0;
            let span_end = suffix.last().unwrap().unwrap().0;
            let span = span_start.to(span_end);
            let count = suffix.len();
            self.tcx.emit_node_span_lint(
                IRREFUTABLE_LET_PATTERNS,
                self.lint_level,
                span,
                TrailingIrrefutableLetPatterns { count },
            );
        }
    }

    fn analyze_binding(
        &mut self,
        pat: &'p Pat<'tcx>,
        refutability: RefutableFlag,
        scrut: Option<&Expr<'tcx>>,
    ) -> Result<(PatCtxt<'p, 'tcx>, UsefulnessReport<'p, 'tcx>), ErrorGuaranteed> {
        let cx = self.new_cx(refutability, None, scrut, pat.span);
        let pat = self.lower_pattern(&cx, pat)?;
        let arms = [MatchArm { pat, arm_data: self.lint_level, has_guard: false }];
        let report = self.analyze_patterns(&cx, &arms, pat.ty().inner())?;
        Ok((cx, report))
    }

    fn is_let_irrefutable(
        &mut self,
        pat: &'p Pat<'tcx>,
        scrut: Option<&Expr<'tcx>>,
    ) -> Result<RefutableFlag, ErrorGuaranteed> {
        let (cx, report) = self.analyze_binding(pat, Refutable, scrut)?;
        // Report if the pattern is unreachable, which can only occur when the type is uninhabited.
        report_arm_reachability(&cx, &report, false);
        // If the list of witnesses is empty, the match is exhaustive, i.e. the `if let` pattern is
        // irrefutable.
        Ok(if report.non_exhaustiveness_witnesses.is_empty() { Irrefutable } else { Refutable })
    }

    #[instrument(level = "trace", skip(self))]
    fn check_binding_is_irrefutable(
        &mut self,
        pat: &'p Pat<'tcx>,
        origin: &str,
        scrut: Option<&Expr<'tcx>>,
        sp: Option<Span>,
    ) {
        let pattern_ty = pat.ty;

        let Ok((cx, report)) = self.analyze_binding(pat, Irrefutable, scrut) else { return };
        let witnesses = report.non_exhaustiveness_witnesses;
        if witnesses.is_empty() {
            // The pattern is irrefutable.
            return;
        }

        let inform = sp.is_some().then_some(Inform);
        let mut let_suggestion = None;
        let mut misc_suggestion = None;
        let mut interpreted_as_const = None;
        let mut interpreted_as_const_sugg = None;

        // These next few matches want to peek through `AscribeUserType` to see
        // the underlying pattern.
        let mut unpeeled_pat = pat;
        while let PatKind::AscribeUserType { ref subpattern, .. } = unpeeled_pat.kind {
            unpeeled_pat = subpattern;
        }

        if let PatKind::ExpandedConstant { def_id, .. } = unpeeled_pat.kind
            && let DefKind::Const = self.tcx.def_kind(def_id)
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(pat.span)
            // We filter out paths with multiple path::segments.
            && snippet.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            let span = self.tcx.def_span(def_id);
            let variable = self.tcx.item_name(def_id).to_string();
            // When we encounter a constant as the binding name, point at the `const` definition.
            interpreted_as_const = Some(span);
            interpreted_as_const_sugg = Some(InterpretedAsConst { span: pat.span, variable });
        } else if let PatKind::Constant { .. } = unpeeled_pat.kind
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(pat.span)
        {
            // If the pattern to match is an integer literal:
            if snippet.chars().all(|c| c.is_digit(10)) {
                // Then give a suggestion, the user might've meant to create a binding instead.
                misc_suggestion = Some(MiscPatternSuggestion::AttemptedIntegerLiteral {
                    start_span: pat.span.shrink_to_lo(),
                });
            }
        }

        if let Some(span) = sp
            && self.tcx.sess.source_map().is_span_accessible(span)
            && interpreted_as_const.is_none()
            && scrut.is_some()
        {
            let mut bindings = vec![];
            pat.each_binding(|name, _, _, _| bindings.push(name));

            let semi_span = span.shrink_to_hi();
            let start_span = span.shrink_to_lo();
            let end_span = semi_span.shrink_to_lo();
            let count = witnesses.len();

            let_suggestion = Some(if bindings.is_empty() {
                SuggestLet::If { start_span, semi_span, count }
            } else {
                SuggestLet::Else { end_span, count }
            });
        };

        let adt_defined_here = report_adt_defined_here(self.tcx, pattern_ty, &witnesses, false);

        // Emit an extra note if the first uncovered witness would be uninhabited
        // if we disregard visibility.
        let witness_1_is_privately_uninhabited = if let Some(witness_1) = witnesses.get(0)
            && let ty::Adt(adt, args) = witness_1.ty().kind()
            && adt.is_enum()
            && let Constructor::Variant(variant_index) = witness_1.ctor()
        {
            let variant_inhabited = adt
                .variant(*variant_index)
                .inhabited_predicate(self.tcx, *adt)
                .instantiate(self.tcx, args);
            variant_inhabited.apply(self.tcx, cx.typing_env, cx.module)
                && !variant_inhabited.apply_ignore_module(self.tcx, cx.typing_env)
        } else {
            false
        };

        self.error = Err(self.tcx.dcx().emit_err(PatternNotCovered {
            span: pat.span,
            origin,
            uncovered: Uncovered::new(pat.span, &cx, witnesses),
            inform,
            interpreted_as_const,
            interpreted_as_const_sugg,
            witness_1_is_privately_uninhabited,
            _p: (),
            pattern_ty,
            let_suggestion,
            misc_suggestion,
            adt_defined_here,
        }));
    }
}

/// Check if a by-value binding is by-value. That is, check if the binding's type is not `Copy`.
/// Check that there are no borrow or move conflicts in `binding @ subpat` patterns.
///
/// For example, this would reject:
/// - `ref x @ Some(ref mut y)`,
/// - `ref mut x @ Some(ref y)`,
/// - `ref mut x @ Some(ref mut y)`,
/// - `ref mut? x @ Some(y)`, and
/// - `x @ Some(ref mut? y)`.
///
/// This analysis is *not* subsumed by NLL.
fn check_borrow_conflicts_in_at_patterns<'tcx>(cx: &MatchVisitor<'_, 'tcx>, pat: &Pat<'tcx>) {
    // Extract `sub` in `binding @ sub`.
    let PatKind::Binding { name, mode, ty, subpattern: Some(box ref sub), .. } = pat.kind else {
        return;
    };

    let is_binding_by_move = |ty: Ty<'tcx>| !cx.tcx.type_is_copy_modulo_regions(cx.typing_env, ty);

    let sess = cx.tcx.sess;

    // Get the binding move, extract the mutability if by-ref.
    let mut_outer = match mode.0 {
        ByRef::No if is_binding_by_move(ty) => {
            // We have `x @ pat` where `x` is by-move. Reject all borrows in `pat`.
            let mut conflicts_ref = Vec::new();
            sub.each_binding(|_, mode, _, span| {
                if matches!(mode, ByRef::Yes(_)) {
                    conflicts_ref.push(span)
                }
            });
            if !conflicts_ref.is_empty() {
                sess.dcx().emit_err(BorrowOfMovedValue {
                    binding_span: pat.span,
                    conflicts_ref,
                    name: Ident::new(name, pat.span),
                    ty,
                    suggest_borrowing: Some(pat.span.shrink_to_lo()),
                });
            }
            return;
        }
        ByRef::No => return,
        ByRef::Yes(m) => m,
    };

    // We now have `ref $mut_outer binding @ sub` (semantically).
    // Recurse into each binding in `sub` and find mutability or move conflicts.
    let mut conflicts_move = Vec::new();
    let mut conflicts_mut_mut = Vec::new();
    let mut conflicts_mut_ref = Vec::new();
    sub.each_binding(|name, mode, ty, span| {
        match mode {
            ByRef::Yes(mut_inner) => match (mut_outer, mut_inner) {
                // Both sides are `ref`.
                (Mutability::Not, Mutability::Not) => {}
                // 2x `ref mut`.
                (Mutability::Mut, Mutability::Mut) => {
                    conflicts_mut_mut.push(Conflict::Mut { span, name })
                }
                (Mutability::Not, Mutability::Mut) => {
                    conflicts_mut_ref.push(Conflict::Mut { span, name })
                }
                (Mutability::Mut, Mutability::Not) => {
                    conflicts_mut_ref.push(Conflict::Ref { span, name })
                }
            },
            ByRef::No if is_binding_by_move(ty) => {
                conflicts_move.push(Conflict::Moved { span, name }) // `ref mut?` + by-move conflict.
            }
            ByRef::No => {} // `ref mut?` + by-copy is fine.
        }
    });

    let report_mut_mut = !conflicts_mut_mut.is_empty();
    let report_mut_ref = !conflicts_mut_ref.is_empty();
    let report_move_conflict = !conflicts_move.is_empty();

    let mut occurrences = match mut_outer {
        Mutability::Mut => vec![Conflict::Mut { span: pat.span, name }],
        Mutability::Not => vec![Conflict::Ref { span: pat.span, name }],
    };
    occurrences.extend(conflicts_mut_mut);
    occurrences.extend(conflicts_mut_ref);
    occurrences.extend(conflicts_move);

    // Report errors if any.
    if report_mut_mut {
        // Report mutability conflicts for e.g. `ref mut x @ Some(ref mut y)`.
        sess.dcx().emit_err(MultipleMutBorrows { span: pat.span, occurrences });
    } else if report_mut_ref {
        // Report mutability conflicts for e.g. `ref x @ Some(ref mut y)` or the converse.
        match mut_outer {
            Mutability::Mut => {
                sess.dcx().emit_err(AlreadyMutBorrowed { span: pat.span, occurrences });
            }
            Mutability::Not => {
                sess.dcx().emit_err(AlreadyBorrowed { span: pat.span, occurrences });
            }
        };
    } else if report_move_conflict {
        // Report by-ref and by-move conflicts, e.g. `ref x @ y`.
        sess.dcx().emit_err(MovedWhileBorrowed { span: pat.span, occurrences });
    }
}

fn check_for_bindings_named_same_as_variants(
    cx: &MatchVisitor<'_, '_>,
    pat: &Pat<'_>,
    rf: RefutableFlag,
) {
    if let PatKind::Binding {
        name,
        mode: BindingMode(ByRef::No, Mutability::Not),
        subpattern: None,
        ty,
        ..
    } = pat.kind
        && let ty::Adt(edef, _) = ty.peel_refs().kind()
        && edef.is_enum()
        && edef
            .variants()
            .iter()
            .any(|variant| variant.name == name && variant.ctor_kind() == Some(CtorKind::Const))
    {
        let variant_count = edef.variants().len();
        let ty_path = with_no_trimmed_paths!(cx.tcx.def_path_str(edef.did()));
        cx.tcx.emit_node_span_lint(
            BINDINGS_WITH_VARIANT_NAME,
            cx.lint_level,
            pat.span,
            BindingsWithVariantName {
                // If this is an irrefutable pattern, and there's > 1 variant,
                // then we can't actually match on this. Applying the below
                // suggestion would produce code that breaks on `check_binding_is_irrefutable`.
                suggestion: if rf == Refutable || variant_count == 1 {
                    Some(pat.span)
                } else {
                    None
                },
                ty_path,
                name: Ident::new(name, pat.span),
            },
        )
    }
}

/// Check that never patterns are only used on inhabited types.
fn check_never_pattern<'tcx>(
    cx: &PatCtxt<'_, 'tcx>,
    pat: &Pat<'tcx>,
) -> Result<(), ErrorGuaranteed> {
    if let PatKind::Never = pat.kind {
        if !cx.is_uninhabited(pat.ty) {
            return Err(cx.tcx.dcx().emit_err(NonEmptyNeverPattern { span: pat.span, ty: pat.ty }));
        }
    }
    Ok(())
}

fn report_irrefutable_let_patterns(
    tcx: TyCtxt<'_>,
    id: HirId,
    source: LetSource,
    count: usize,
    span: Span,
) {
    macro_rules! emit_diag {
        ($lint:tt) => {{
            tcx.emit_node_span_lint(IRREFUTABLE_LET_PATTERNS, id, span, $lint { count });
        }};
    }

    match source {
        LetSource::None | LetSource::PlainLet | LetSource::Else => bug!(),
        LetSource::IfLet | LetSource::ElseIfLet => emit_diag!(IrrefutableLetPatternsIfLet),
        LetSource::IfLetGuard => emit_diag!(IrrefutableLetPatternsIfLetGuard),
        LetSource::LetElse => emit_diag!(IrrefutableLetPatternsLetElse),
        LetSource::WhileLet => emit_diag!(IrrefutableLetPatternsWhileLet),
    }
}

/// Report unreachable arms, if any.
fn report_unreachable_pattern<'p, 'tcx>(
    cx: &PatCtxt<'p, 'tcx>,
    hir_id: HirId,
    pat: &DeconstructedPat<'p, 'tcx>,
    explanation: &RedundancyExplanation<'p, 'tcx>,
    whole_arm_span: Option<Span>,
) {
    static CAP_COVERED_BY_MANY: usize = 4;
    let pat_span = pat.data().span;
    let mut lint = UnreachablePattern {
        span: Some(pat_span),
        matches_no_values: None,
        matches_no_values_ty: **pat.ty(),
        uninhabited_note: None,
        covered_by_catchall: None,
        covered_by_one: None,
        covered_by_many: None,
        covered_by_many_n_more_count: 0,
        wanted_constant: None,
        accessible_constant: None,
        inaccessible_constant: None,
        pattern_let_binding: None,
        suggest_remove: None,
    };
    match explanation.covered_by.as_slice() {
        [] => {
            // Empty pattern; we report the uninhabited type that caused the emptiness.
            lint.span = None; // Don't label the pattern itself
            lint.uninhabited_note = Some(()); // Give a link about empty types
            lint.matches_no_values = Some(pat_span);
            lint.suggest_remove = whole_arm_span; // Suggest to remove the match arm
            pat.walk(&mut |subpat| {
                let ty = **subpat.ty();
                if cx.is_uninhabited(ty) {
                    lint.matches_no_values_ty = ty;
                    false // No need to dig further.
                } else if matches!(subpat.ctor(), Constructor::Ref | Constructor::UnionField) {
                    false // Don't explore further since they are not by-value.
                } else {
                    true
                }
            });
        }
        [covering_pat] if pat_is_catchall(covering_pat) => {
            // A binding pattern that matches all, a single binding name.
            let pat = covering_pat.data();
            lint.covered_by_catchall = Some(pat.span);
            find_fallback_pattern_typo(cx, hir_id, pat, &mut lint);
        }
        [covering_pat] => {
            lint.covered_by_one = Some(covering_pat.data().span);
        }
        covering_pats => {
            let mut iter = covering_pats.iter();
            let mut multispan = MultiSpan::from_span(pat_span);
            for p in iter.by_ref().take(CAP_COVERED_BY_MANY) {
                multispan.push_span_label(
                    p.data().span,
                    fluent::mir_build_unreachable_matches_same_values,
                );
            }
            let remain = iter.count();
            if remain == 0 {
                multispan.push_span_label(
                    pat_span,
                    fluent::mir_build_unreachable_making_this_unreachable,
                );
            } else {
                lint.covered_by_many_n_more_count = remain;
                multispan.push_span_label(
                    pat_span,
                    fluent::mir_build_unreachable_making_this_unreachable_n_more,
                );
            }
            lint.covered_by_many = Some(multispan);
        }
    }
    cx.tcx.emit_node_span_lint(UNREACHABLE_PATTERNS, hir_id, pat_span, lint);
}

/// Detect typos that were meant to be a `const` but were interpreted as a new pattern binding.
fn find_fallback_pattern_typo<'tcx>(
    cx: &PatCtxt<'_, 'tcx>,
    hir_id: HirId,
    pat: &Pat<'tcx>,
    lint: &mut UnreachablePattern<'_>,
) {
    if let Level::Allow = cx.tcx.lint_level_at_node(UNREACHABLE_PATTERNS, hir_id).level {
        // This is because we use `with_no_trimmed_paths` later, so if we never emit the lint we'd
        // ICE. At the same time, we don't really need to do all of this if we won't emit anything.
        return;
    }
    if let PatKind::Binding { name, subpattern: None, ty, .. } = pat.kind {
        // See if the binding might have been a `const` that was mistyped or out of scope.
        let mut accessible = vec![];
        let mut accessible_path = vec![];
        let mut inaccessible = vec![];
        let mut imported = vec![];
        let mut imported_spans = vec![];
        let (infcx, param_env) = cx.tcx.infer_ctxt().build_with_typing_env(cx.typing_env);
        let parent = cx.tcx.hir_get_parent_item(hir_id);

        for item in cx.tcx.hir_crate_items(()).free_items() {
            if let DefKind::Use = cx.tcx.def_kind(item.owner_id) {
                // Look for consts being re-exported.
                let item = cx.tcx.hir_expect_item(item.owner_id.def_id);
                let hir::ItemKind::Use(path, _) = item.kind else {
                    continue;
                };
                if let Some(value_ns) = path.res.value_ns
                    && let Res::Def(DefKind::Const, id) = value_ns
                    && infcx.can_eq(param_env, ty, cx.tcx.type_of(id).instantiate_identity())
                {
                    if cx.tcx.visibility(id).is_accessible_from(parent, cx.tcx) {
                        // The original const is accessible, suggest using it directly.
                        let item_name = cx.tcx.item_name(id);
                        accessible.push(item_name);
                        accessible_path.push(with_no_trimmed_paths!(cx.tcx.def_path_str(id)));
                    } else if cx.tcx.visibility(item.owner_id).is_accessible_from(parent, cx.tcx) {
                        // The const is accessible only through the re-export, point at
                        // the `use`.
                        let ident = item.kind.ident().unwrap();
                        imported.push(ident.name);
                        imported_spans.push(ident.span);
                    }
                }
            }
            if let DefKind::Const = cx.tcx.def_kind(item.owner_id)
                && infcx.can_eq(param_env, ty, cx.tcx.type_of(item.owner_id).instantiate_identity())
            {
                // Look for local consts.
                let item_name = cx.tcx.item_name(item.owner_id.into());
                let vis = cx.tcx.visibility(item.owner_id);
                if vis.is_accessible_from(parent, cx.tcx) {
                    accessible.push(item_name);
                    // FIXME: the line below from PR #135310 is a workaround for the ICE in issue
                    // #135289, where a macro in a dependency can create unreachable patterns in the
                    // current crate. Path trimming expects diagnostics for a typoed const, but no
                    // diagnostics are emitted and we ICE. See
                    // `tests/ui/resolve/const-with-typo-in-pattern-binding-ice-135289.rs` for a
                    // test that reproduces the ICE if we don't use `with_no_trimmed_paths!`.
                    let path = with_no_trimmed_paths!(cx.tcx.def_path_str(item.owner_id));
                    accessible_path.push(path);
                } else if name == item_name {
                    // The const exists somewhere in this crate, but it can't be imported
                    // from this pattern's scope. We'll just point at its definition.
                    inaccessible.push(cx.tcx.def_span(item.owner_id));
                }
            }
        }
        if let Some((i, &const_name)) =
            accessible.iter().enumerate().find(|&(_, &const_name)| const_name == name)
        {
            // The pattern name is an exact match, so the pattern needed to be imported.
            lint.wanted_constant = Some(WantedConstant {
                span: pat.span,
                is_typo: false,
                const_name: const_name.to_string(),
                const_path: accessible_path[i].clone(),
            });
        } else if let Some(name) = find_best_match_for_name(&accessible, name, None) {
            // The pattern name is likely a typo.
            lint.wanted_constant = Some(WantedConstant {
                span: pat.span,
                is_typo: true,
                const_name: name.to_string(),
                const_path: name.to_string(),
            });
        } else if let Some(i) =
            imported.iter().enumerate().find(|&(_, &const_name)| const_name == name).map(|(i, _)| i)
        {
            // The const with the exact name wasn't re-exported from an import in this
            // crate, we point at the import.
            lint.accessible_constant = Some(imported_spans[i]);
        } else if let Some(name) = find_best_match_for_name(&imported, name, None) {
            // The typoed const wasn't re-exported by an import in this crate, we suggest
            // the right name (which will likely require another follow up suggestion).
            lint.wanted_constant = Some(WantedConstant {
                span: pat.span,
                is_typo: true,
                const_path: name.to_string(),
                const_name: name.to_string(),
            });
        } else if !inaccessible.is_empty() {
            for span in inaccessible {
                // The const with the exact name match isn't accessible, we just point at it.
                lint.inaccessible_constant = Some(span);
            }
        } else {
            // Look for local bindings for people that might have gotten confused with how
            // `let` and `const` works.
            for (_, node) in cx.tcx.hir_parent_iter(hir_id) {
                match node {
                    hir::Node::Stmt(hir::Stmt { kind: hir::StmtKind::Let(let_stmt), .. }) => {
                        if let hir::PatKind::Binding(_, _, binding_name, _) = let_stmt.pat.kind {
                            if name == binding_name.name {
                                lint.pattern_let_binding = Some(binding_name.span);
                            }
                        }
                    }
                    hir::Node::Block(hir::Block { stmts, .. }) => {
                        for stmt in *stmts {
                            if let hir::StmtKind::Let(let_stmt) = stmt.kind {
                                if let hir::PatKind::Binding(_, _, binding_name, _) =
                                    let_stmt.pat.kind
                                {
                                    if name == binding_name.name {
                                        lint.pattern_let_binding = Some(binding_name.span);
                                    }
                                }
                            }
                        }
                    }
                    hir::Node::Item(_) => break,
                    _ => {}
                }
            }
        }
    }
}

/// Report unreachable arms, if any.
fn report_arm_reachability<'p, 'tcx>(
    cx: &PatCtxt<'p, 'tcx>,
    report: &UsefulnessReport<'p, 'tcx>,
    is_match_arm: bool,
) {
    let sm = cx.tcx.sess.source_map();
    for (arm, is_useful) in report.arm_usefulness.iter() {
        if let Usefulness::Redundant(explanation) = is_useful {
            let hir_id = arm.arm_data;
            let arm_span = cx.tcx.hir_span(hir_id);
            let whole_arm_span = if is_match_arm {
                // If the arm is followed by a comma, extend the span to include it.
                let with_whitespace = sm.span_extend_while_whitespace(arm_span);
                if let Some(comma) = sm.span_look_ahead(with_whitespace, ",", Some(1)) {
                    Some(arm_span.to(comma))
                } else {
                    Some(arm_span)
                }
            } else {
                None
            };
            report_unreachable_pattern(cx, hir_id, arm.pat, explanation, whole_arm_span)
        }
    }
}

/// Checks for common cases of "catchall" patterns that may not be intended as such.
fn pat_is_catchall(pat: &DeconstructedPat<'_, '_>) -> bool {
    match pat.ctor() {
        Constructor::Wildcard => true,
        Constructor::Struct | Constructor::Ref => {
            pat.iter_fields().all(|ipat| pat_is_catchall(&ipat.pat))
        }
        _ => false,
    }
}

/// Report that a match is not exhaustive.
fn report_non_exhaustive_match<'p, 'tcx>(
    cx: &PatCtxt<'p, 'tcx>,
    thir: &Thir<'tcx>,
    scrut_ty: Ty<'tcx>,
    sp: Span,
    witnesses: Vec<WitnessPat<'p, 'tcx>>,
    arms: &[ArmId],
    braces_span: Option<Span>,
) -> ErrorGuaranteed {
    let is_empty_match = arms.is_empty();
    let non_empty_enum = match scrut_ty.kind() {
        ty::Adt(def, _) => def.is_enum() && !def.variants().is_empty(),
        _ => false,
    };
    // In the case of an empty match, replace the '`_` not covered' diagnostic with something more
    // informative.
    if is_empty_match && !non_empty_enum {
        return cx.tcx.dcx().emit_err(NonExhaustivePatternsTypeNotEmpty {
            cx,
            scrut_span: sp,
            braces_span,
            ty: scrut_ty,
        });
    }

    // FIXME: migration of this diagnostic will require list support
    let joined_patterns = joined_uncovered_patterns(cx, &witnesses);
    let mut err = struct_span_code_err!(
        cx.tcx.dcx(),
        sp,
        E0004,
        "non-exhaustive patterns: {joined_patterns} not covered"
    );
    err.span_label(
        sp,
        format!(
            "pattern{} {} not covered",
            rustc_errors::pluralize!(witnesses.len()),
            joined_patterns
        ),
    );

    // Point at the definition of non-covered `enum` variants.
    if let Some(AdtDefinedHere { adt_def_span, ty, variants }) =
        report_adt_defined_here(cx.tcx, scrut_ty, &witnesses, true)
    {
        let mut multi_span = MultiSpan::from_span(adt_def_span);
        multi_span.push_span_label(adt_def_span, "");
        for Variant { span } in variants {
            multi_span.push_span_label(span, "not covered");
        }
        err.span_note(multi_span, format!("`{ty}` defined here"));
    }
    err.note(format!("the matched value is of type `{}`", scrut_ty));

    if !is_empty_match {
        let mut special_tys = FxIndexSet::default();
        // Look at the first witness.
        collect_special_tys(cx, &witnesses[0], &mut special_tys);

        for ty in special_tys {
            if ty.is_ptr_sized_integral() {
                if ty.inner() == cx.tcx.types.usize {
                    err.note(format!(
                        "`{ty}` does not have a fixed maximum value, so half-open ranges are \
                         necessary to match exhaustively",
                    ));
                } else if ty.inner() == cx.tcx.types.isize {
                    err.note(format!(
                        "`{ty}` does not have fixed minimum and maximum values, so half-open \
                         ranges are necessary to match exhaustively",
                    ));
                }
            } else if ty.inner() == cx.tcx.types.str_ {
                err.note("`&str` cannot be matched exhaustively, so a wildcard `_` is necessary");
            } else if cx.is_foreign_non_exhaustive_enum(ty) {
                err.note(format!("`{ty}` is marked as non-exhaustive, so a wildcard `_` is necessary to match exhaustively"));
            } else if cx.is_uninhabited(ty.inner()) {
                // The type is uninhabited yet there is a witness: we must be in the `MaybeInvalid`
                // case.
                err.note(format!("`{ty}` is uninhabited but is not being matched by value, so a wildcard `_` is required"));
            }
        }
    }

    if let ty::Ref(_, sub_ty, _) = scrut_ty.kind() {
        if !sub_ty.is_inhabited_from(cx.tcx, cx.module, cx.typing_env) {
            err.note("references are always considered inhabited");
        }
    }

    for &arm in arms {
        let arm = &thir.arms[arm];
        if let PatKind::ExpandedConstant { def_id, .. } = arm.pattern.kind
            && !matches!(cx.tcx.def_kind(def_id), DefKind::InlineConst)
            && let Ok(snippet) = cx.tcx.sess.source_map().span_to_snippet(arm.pattern.span)
            // We filter out paths with multiple path::segments.
            && snippet.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            let const_name = cx.tcx.item_name(def_id);
            err.span_label(
                arm.pattern.span,
                format!(
                    "this pattern doesn't introduce a new catch-all binding, but rather pattern \
                     matches against the value of constant `{const_name}`",
                ),
            );
            err.span_note(cx.tcx.def_span(def_id), format!("constant `{const_name}` defined here"));
            err.span_suggestion_verbose(
                arm.pattern.span.shrink_to_hi(),
                "if you meant to introduce a binding, use a different name",
                "_var".to_string(),
                Applicability::MaybeIncorrect,
            );
        }
    }

    // Whether we suggest the actual missing patterns or `_`.
    let suggest_the_witnesses = witnesses.len() < 4;
    let suggested_arm = if suggest_the_witnesses {
        let pattern = witnesses
            .iter()
            .map(|witness| cx.print_witness_pat(witness))
            .collect::<Vec<String>>()
            .join(" | ");
        if witnesses.iter().all(|p| p.is_never_pattern()) && cx.tcx.features().never_patterns() {
            // Arms with a never pattern don't take a body.
            pattern
        } else {
            format!("{pattern} => todo!()")
        }
    } else {
        format!("_ => todo!()")
    };
    let mut suggestion = None;
    let sm = cx.tcx.sess.source_map();
    match arms {
        [] if let Some(braces_span) = braces_span => {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(sp) {
                (format!("\n{snippet}"), "    ")
            } else {
                (" ".to_string(), "")
            };
            suggestion = Some((
                braces_span,
                format!(" {{{indentation}{more}{suggested_arm},{indentation}}}",),
            ));
        }
        [only] => {
            let only = &thir[*only];
            let (pre_indentation, is_multiline) = if let Some(snippet) =
                sm.indentation_before(only.span)
                && let Ok(with_trailing) =
                    sm.span_extend_while(only.span, |c| c.is_whitespace() || c == ',')
                && sm.is_multiline(with_trailing)
            {
                (format!("\n{snippet}"), true)
            } else {
                (" ".to_string(), false)
            };
            let only_body = &thir[only.body];
            let comma = if matches!(only_body.kind, ExprKind::Block { .. })
                && only.span.eq_ctxt(only_body.span)
                && is_multiline
            {
                ""
            } else {
                ","
            };
            suggestion = Some((
                only.span.shrink_to_hi(),
                format!("{comma}{pre_indentation}{suggested_arm}"),
            ));
        }
        [.., prev, last] => {
            let prev = &thir[*prev];
            let last = &thir[*last];
            if prev.span.eq_ctxt(last.span) {
                let last_body = &thir[last.body];
                let comma = if matches!(last_body.kind, ExprKind::Block { .. })
                    && last.span.eq_ctxt(last_body.span)
                {
                    ""
                } else {
                    ","
                };
                let spacing = if sm.is_multiline(prev.span.between(last.span)) {
                    sm.indentation_before(last.span).map(|indent| format!("\n{indent}"))
                } else {
                    Some(" ".to_string())
                };
                if let Some(spacing) = spacing {
                    suggestion = Some((
                        last.span.shrink_to_hi(),
                        format!("{comma}{spacing}{suggested_arm}"),
                    ));
                }
            }
        }
        _ => {}
    }

    let msg = format!(
        "ensure that all possible cases are being handled by adding a match arm with a wildcard \
         pattern{}{}",
        if witnesses.len() > 1 && suggest_the_witnesses && suggestion.is_some() {
            ", a match arm with multiple or-patterns"
        } else {
            // we are either not suggesting anything, or suggesting `_`
            ""
        },
        match witnesses.len() {
            // non-exhaustive enum case
            0 if suggestion.is_some() => " as shown",
            0 => "",
            1 if suggestion.is_some() => " or an explicit pattern as shown",
            1 => " or an explicit pattern",
            _ if suggestion.is_some() => " as shown, or multiple match arms",
            _ => " or multiple match arms",
        },
    );

    let all_arms_have_guards = arms.iter().all(|arm_id| thir[*arm_id].guard.is_some());
    if !is_empty_match && all_arms_have_guards {
        err.subdiagnostic(NonExhaustiveMatchAllArmsGuarded);
    }
    if let Some((span, sugg)) = suggestion {
        err.span_suggestion_verbose(span, msg, sugg, Applicability::HasPlaceholders);
    } else {
        err.help(msg);
    }
    err.emit()
}

fn joined_uncovered_patterns<'p, 'tcx>(
    cx: &PatCtxt<'p, 'tcx>,
    witnesses: &[WitnessPat<'p, 'tcx>],
) -> String {
    const LIMIT: usize = 3;
    let pat_to_str = |pat: &WitnessPat<'p, 'tcx>| cx.print_witness_pat(pat);
    match witnesses {
        [] => bug!(),
        [witness] => format!("`{}`", cx.print_witness_pat(witness)),
        [head @ .., tail] if head.len() < LIMIT => {
            let head: Vec<_> = head.iter().map(pat_to_str).collect();
            format!("`{}` and `{}`", head.join("`, `"), cx.print_witness_pat(tail))
        }
        _ => {
            let (head, tail) = witnesses.split_at(LIMIT);
            let head: Vec<_> = head.iter().map(pat_to_str).collect();
            format!("`{}` and {} more", head.join("`, `"), tail.len())
        }
    }
}

/// Collect types that require specific explanations when they show up in witnesses.
fn collect_special_tys<'tcx>(
    cx: &PatCtxt<'_, 'tcx>,
    pat: &WitnessPat<'_, 'tcx>,
    special_tys: &mut FxIndexSet<RevealedTy<'tcx>>,
) {
    if matches!(pat.ctor(), Constructor::NonExhaustive | Constructor::Never) {
        special_tys.insert(*pat.ty());
    }
    if let Constructor::IntRange(range) = pat.ctor() {
        if cx.is_range_beyond_boundaries(range, *pat.ty()) {
            // The range denotes the values before `isize::MIN` or the values after `usize::MAX`/`isize::MAX`.
            special_tys.insert(*pat.ty());
        }
    }
    pat.iter_fields().for_each(|field_pat| collect_special_tys(cx, field_pat, special_tys))
}

fn report_adt_defined_here<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    witnesses: &[WitnessPat<'_, 'tcx>],
    point_at_non_local_ty: bool,
) -> Option<AdtDefinedHere<'tcx>> {
    let ty = ty.peel_refs();
    let ty::Adt(def, _) = ty.kind() else {
        return None;
    };
    let adt_def_span =
        tcx.hir_get_if_local(def.did()).and_then(|node| node.ident()).map(|ident| ident.span);
    let adt_def_span = if point_at_non_local_ty {
        adt_def_span.unwrap_or_else(|| tcx.def_span(def.did()))
    } else {
        adt_def_span?
    };

    let mut variants = vec![];
    for span in maybe_point_at_variant(tcx, *def, witnesses.iter().take(5)) {
        variants.push(Variant { span });
    }
    Some(AdtDefinedHere { adt_def_span, ty, variants })
}

fn maybe_point_at_variant<'a, 'p: 'a, 'tcx: 'p>(
    tcx: TyCtxt<'tcx>,
    def: AdtDef<'tcx>,
    patterns: impl Iterator<Item = &'a WitnessPat<'p, 'tcx>>,
) -> Vec<Span> {
    let mut covered = vec![];
    for pattern in patterns {
        if let Constructor::Variant(variant_index) = pattern.ctor() {
            if let ty::Adt(this_def, _) = pattern.ty().kind()
                && this_def.did() != def.did()
            {
                continue;
            }
            let sp = def.variant(*variant_index).ident(tcx).span;
            if covered.contains(&sp) {
                // Don't point at variants that have already been covered due to other patterns to avoid
                // visual clutter.
                continue;
            }
            covered.push(sp);
        }
        covered.extend(maybe_point_at_variant(tcx, def, pattern.iter_fields()));
    }
    covered
}
