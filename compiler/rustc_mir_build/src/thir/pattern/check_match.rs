use rustc_pattern_analysis::errors::Uncovered;
use rustc_pattern_analysis::rustc::{
    Constructor, DeconstructedPat, MatchArm, RustcPatCtxt as PatCtxt, Usefulness, UsefulnessReport,
    WitnessPat,
};

use crate::errors::*;

use rustc_arena::{DroplessArena, TypedArena};
use rustc_ast::Mutability;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{
    codes::*, struct_span_code_err, Applicability, Diag, ErrorGuaranteed, MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::*;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::HirId;
use rustc_middle::middle::limits::get_limit_size;
use rustc_middle::thir::visit::Visitor;
use rustc_middle::thir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_session::lint::builtin::{
    BINDINGS_WITH_VARIANT_NAME, IRREFUTABLE_LET_PATTERNS, UNREACHABLE_PATTERNS,
};
use rustc_session::Session;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::{sym, Span};

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
        param_env: tcx.param_env(def_id),
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

fn create_e0004(sess: &Session, sp: Span, error_message: String) -> Diag<'_> {
    struct_span_code_err!(sess.dcx(), sp, E0004, "{}", &error_message)
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
}

struct MatchVisitor<'p, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
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
                    _ => LetSource::IfLet,
                };
                self.with_let_source(let_source, |this| this.visit_expr(&self.thir[cond]));
                self.with_let_source(LetSource::None, |this| {
                    this.visit_expr(&this.thir[then]);
                    if let Some(else_) = else_opt {
                        this.visit_expr(&this.thir[else_]);
                    }
                });
                return;
            }
            ExprKind::Match { scrutinee, scrutinee_hir_id, box ref arms } => {
                let source = match ex.span.desugaring_kind() {
                    Some(DesugaringKind::ForLoop) => hir::MatchSource::ForLoopDesugar,
                    Some(DesugaringKind::QuestionMark) => {
                        hir::MatchSource::TryDesugar(scrutinee_hir_id)
                    }
                    Some(DesugaringKind::Await) => hir::MatchSource::AwaitDesugar,
                    _ => hir::MatchSource::Normal,
                };
                self.check_match(scrutinee, arms, source, ex.span);
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
            | ValueTypeAscription { source, .. } => {
                self.is_known_valid_scrutinee(&self.thir()[*source])
            }

            // These diverge.
            Become { .. } | Break { .. } | Continue { .. } | Return { .. } => true,

            // These are statements that evaluate to `()`.
            Assign { .. } | AssignOp { .. } | InlineAsm { .. } | Let { .. } => true,

            // These evaluate to a value.
            AddressOf { .. }
            | Adt { .. }
            | Array { .. }
            | Binary { .. }
            | Block { .. }
            | Borrow { .. }
            | Box { .. }
            | Call { .. }
            | Closure { .. }
            | ConstBlock { .. }
            | ConstParam { .. }
            | If { .. }
            | Literal { .. }
            | LogicalOp { .. }
            | Loop { .. }
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
            param_env: self.param_env,
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
        let pattern_complexity_limit =
            get_limit_size(cx.tcx.hir().krate_attrs(), cx.tcx.sess, sym::pattern_complexity);
        let report =
            rustc_pattern_analysis::analyze_match(&cx, &arms, scrut_ty, pattern_complexity_limit)
                .map_err(|err| {
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
                redundant_subpats.sort_unstable_by_key(|pat| pat.data().span);
                for pat in redundant_subpats {
                    report_unreachable_pattern(cx, arm.arm_data, pat.data().span, None)
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
            | hir::MatchSource::Normal
            | hir::MatchSource::FormatArgs => report_arm_reachability(&cx, &report),
            // Unreachable patterns in try and await expressions occur when one of
            // the arms are an uninhabited type. Which is OK.
            hir::MatchSource::AwaitDesugar | hir::MatchSource::TryDesugar(_) => {}
        }

        // Check if the match is exhaustive.
        let witnesses = report.non_exhaustiveness_witnesses;
        if !witnesses.is_empty() {
            if source == hir::MatchSource::ForLoopDesugar && arms.len() == 2 {
                // the for loop pattern is not irrefutable
                let pat = &self.thir[arms[1]].pattern;
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
                self.error = Err(report_non_exhaustive_match(
                    &cx, self.thir, scrut.ty, scrut.span, witnesses, arms, expr_span,
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
            // FIXME: Add checking whether the bindings are actually used in the prefix,
            // and lint if they are not.
            if !matches!(self.let_source, LetSource::WhileLet | LetSource::IfLetGuard) {
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
        report_arm_reachability(&cx, &report);
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

        if let PatKind::Constant { .. }
        | PatKind::AscribeUserType {
            subpattern: box Pat { kind: PatKind::Constant { .. }, .. },
            ..
        } = pat.kind
            && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(pat.span)
        {
            // If the pattern to match is an integer literal:
            if snippet.chars().all(|c| c.is_digit(10)) {
                // Then give a suggestion, the user might've meant to create a binding instead.
                misc_suggestion = Some(MiscPatternSuggestion::AttemptedIntegerLiteral {
                    start_span: pat.span.shrink_to_lo(),
                });
            } else if snippet.chars().all(|c| c.is_alphanumeric() || c == '_') {
                interpreted_as_const =
                    Some(InterpretedAsConst { span: pat.span, variable: snippet });
            }
        }

        if let Some(span) = sp
            && self.tcx.sess.source_map().is_span_accessible(span)
            && interpreted_as_const.is_none()
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
        let witness_1_is_privately_uninhabited = if (self.tcx.features().exhaustive_patterns
            || self.tcx.features().min_exhaustive_patterns)
            && let Some(witness_1) = witnesses.get(0)
            && let ty::Adt(adt, args) = witness_1.ty().kind()
            && adt.is_enum()
            && let Constructor::Variant(variant_index) = witness_1.ctor()
        {
            let variant = adt.variant(*variant_index);
            let inhabited = variant.inhabited_predicate(self.tcx, *adt).instantiate(self.tcx, args);
            assert!(inhabited.apply(self.tcx, cx.param_env, cx.module));
            !inhabited.apply_ignore_module(self.tcx, cx.param_env)
        } else {
            false
        };

        self.error = Err(self.tcx.dcx().emit_err(PatternNotCovered {
            span: pat.span,
            origin,
            uncovered: Uncovered::new(pat.span, &cx, witnesses),
            inform,
            interpreted_as_const,
            witness_1_is_privately_uninhabited: witness_1_is_privately_uninhabited.then_some(()),
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

    let is_binding_by_move = |ty: Ty<'tcx>| !ty.is_copy_modulo_regions(cx.tcx, cx.param_env);

    let sess = cx.tcx.sess;

    // Get the binding move, extract the mutability if by-ref.
    let mut_outer = match mode {
        BindingMode::ByValue if is_binding_by_move(ty) => {
            // We have `x @ pat` where `x` is by-move. Reject all borrows in `pat`.
            let mut conflicts_ref = Vec::new();
            sub.each_binding(|_, mode, _, span| match mode {
                BindingMode::ByValue => {}
                BindingMode::ByRef(_) => conflicts_ref.push(span),
            });
            if !conflicts_ref.is_empty() {
                sess.dcx().emit_err(BorrowOfMovedValue {
                    binding_span: pat.span,
                    conflicts_ref,
                    name,
                    ty,
                    suggest_borrowing: Some(pat.span.shrink_to_lo()),
                });
            }
            return;
        }
        BindingMode::ByValue => return,
        BindingMode::ByRef(m) => m.mutability(),
    };

    // We now have `ref $mut_outer binding @ sub` (semantically).
    // Recurse into each binding in `sub` and find mutability or move conflicts.
    let mut conflicts_move = Vec::new();
    let mut conflicts_mut_mut = Vec::new();
    let mut conflicts_mut_ref = Vec::new();
    sub.each_binding(|name, mode, ty, span| {
        match mode {
            BindingMode::ByRef(mut_inner) => match (mut_outer, mut_inner.mutability()) {
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
            BindingMode::ByValue if is_binding_by_move(ty) => {
                conflicts_move.push(Conflict::Moved { span, name }) // `ref mut?` + by-move conflict.
            }
            BindingMode::ByValue => {} // `ref mut?` + by-copy is fine.
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
        mode: BindingMode::ByValue,
        mutability: Mutability::Not,
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
                name,
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
        LetSource::None | LetSource::PlainLet => bug!(),
        LetSource::IfLet => emit_diag!(IrrefutableLetPatternsIfLet),
        LetSource::IfLetGuard => emit_diag!(IrrefutableLetPatternsIfLetGuard),
        LetSource::LetElse => emit_diag!(IrrefutableLetPatternsLetElse),
        LetSource::WhileLet => emit_diag!(IrrefutableLetPatternsWhileLet),
    }
}

/// Report unreachable arms, if any.
fn report_unreachable_pattern<'p, 'tcx>(
    cx: &PatCtxt<'p, 'tcx>,
    hir_id: HirId,
    span: Span,
    catchall: Option<Span>,
) {
    cx.tcx.emit_node_span_lint(
        UNREACHABLE_PATTERNS,
        hir_id,
        span,
        UnreachablePattern { span: if catchall.is_some() { Some(span) } else { None }, catchall },
    );
}

/// Report unreachable arms, if any.
fn report_arm_reachability<'p, 'tcx>(cx: &PatCtxt<'p, 'tcx>, report: &UsefulnessReport<'p, 'tcx>) {
    let mut catchall = None;
    for (arm, is_useful) in report.arm_usefulness.iter() {
        if matches!(is_useful, Usefulness::Redundant) {
            report_unreachable_pattern(cx, arm.arm_data, arm.pat.data().span, catchall)
        }
        if !arm.has_guard && catchall.is_none() && pat_is_catchall(arm.pat) {
            catchall = Some(arm.pat.data().span);
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
    expr_span: Span,
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
            expr_span,
            span: sp,
            ty: scrut_ty,
        });
    }

    // FIXME: migration of this diagnostic will require list support
    let joined_patterns = joined_uncovered_patterns(cx, &witnesses);
    let mut err = create_e0004(
        cx.tcx.sess,
        sp,
        format!("non-exhaustive patterns: {joined_patterns} not covered"),
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
        let mut non_exhaustive_tys = FxIndexSet::default();
        // Look at the first witness.
        collect_non_exhaustive_tys(cx, &witnesses[0], &mut non_exhaustive_tys);

        for ty in non_exhaustive_tys {
            if ty.is_ptr_sized_integral() {
                if ty == cx.tcx.types.usize {
                    err.note(format!(
                        "`{ty}` does not have a fixed maximum value, so half-open ranges are necessary to match \
                             exhaustively",
                    ));
                } else if ty == cx.tcx.types.isize {
                    err.note(format!(
                        "`{ty}` does not have fixed minimum and maximum values, so half-open ranges are necessary to match \
                             exhaustively",
                    ));
                }
            } else if ty == cx.tcx.types.str_ {
                err.note("`&str` cannot be matched exhaustively, so a wildcard `_` is necessary");
            } else if cx.is_foreign_non_exhaustive_enum(cx.reveal_opaque_ty(ty)) {
                err.note(format!("`{ty}` is marked as non-exhaustive, so a wildcard `_` is necessary to match exhaustively"));
            }
        }
    }

    if let ty::Ref(_, sub_ty, _) = scrut_ty.kind() {
        if !sub_ty.is_inhabited_from(cx.tcx, cx.module, cx.param_env) {
            err.note("references are always considered inhabited");
        }
    }

    // Whether we suggest the actual missing patterns or `_`.
    let suggest_the_witnesses = witnesses.len() < 4;
    let suggested_arm = if suggest_the_witnesses {
        let pattern = witnesses
            .iter()
            .map(|witness| cx.hoist_witness_pat(witness).to_string())
            .collect::<Vec<String>>()
            .join(" | ");
        if witnesses.iter().all(|p| p.is_never_pattern()) && cx.tcx.features().never_patterns {
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
        [] if sp.eq_ctxt(expr_span) => {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(sp) {
                (format!("\n{snippet}"), "    ")
            } else {
                (" ".to_string(), "")
            };
            suggestion = Some((
                sp.shrink_to_hi().with_hi(expr_span.hi()),
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
        err.subdiagnostic(cx.tcx.dcx(), NonExhaustiveMatchAllArmsGuarded);
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
    let pat_to_str = |pat: &WitnessPat<'p, 'tcx>| cx.hoist_witness_pat(pat).to_string();
    match witnesses {
        [] => bug!(),
        [witness] => format!("`{}`", cx.hoist_witness_pat(witness)),
        [head @ .., tail] if head.len() < LIMIT => {
            let head: Vec<_> = head.iter().map(pat_to_str).collect();
            format!("`{}` and `{}`", head.join("`, `"), cx.hoist_witness_pat(tail))
        }
        _ => {
            let (head, tail) = witnesses.split_at(LIMIT);
            let head: Vec<_> = head.iter().map(pat_to_str).collect();
            format!("`{}` and {} more", head.join("`, `"), tail.len())
        }
    }
}

fn collect_non_exhaustive_tys<'tcx>(
    cx: &PatCtxt<'_, 'tcx>,
    pat: &WitnessPat<'_, 'tcx>,
    non_exhaustive_tys: &mut FxIndexSet<Ty<'tcx>>,
) {
    if matches!(pat.ctor(), Constructor::NonExhaustive) {
        non_exhaustive_tys.insert(pat.ty().inner());
    }
    if let Constructor::IntRange(range) = pat.ctor() {
        if cx.is_range_beyond_boundaries(range, *pat.ty()) {
            // The range denotes the values before `isize::MIN` or the values after `usize::MAX`/`isize::MAX`.
            non_exhaustive_tys.insert(pat.ty().inner());
        }
    }
    pat.iter_fields()
        .for_each(|field_pat| collect_non_exhaustive_tys(cx, field_pat, non_exhaustive_tys))
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
        tcx.hir().get_if_local(def.did()).and_then(|node| node.ident()).map(|ident| ident.span);
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
