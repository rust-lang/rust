use super::deconstruct_pat::{Constructor, DeconstructedPat};
use super::usefulness::{
    compute_match_usefulness, MatchArm, MatchCheckCtxt, Reachability, UsefulnessReport,
};

use crate::errors::*;

use rustc_arena::TypedArena;
use rustc_ast::Mutability;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_errors::{
    struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed, MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::*;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::HirId;
use rustc_middle::thir::visit::{self, Visitor};
use rustc_middle::thir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_session::lint::builtin::{
    BINDINGS_WITH_VARIANT_NAME, IRREFUTABLE_LET_PATTERNS, UNREACHABLE_PATTERNS,
};
use rustc_session::Session;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::Span;

pub(crate) fn check_match(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let (thir, expr) = tcx.thir_body(def_id)?;
    let thir = thir.borrow();
    let pattern_arena = TypedArena::default();
    let mut visitor = MatchVisitor {
        tcx,
        thir: &*thir,
        param_env: tcx.param_env(def_id),
        lint_level: tcx.hir().local_def_id_to_hir_id(def_id),
        let_source: LetSource::None,
        pattern_arena: &pattern_arena,
        error: Ok(()),
    };
    visitor.visit_expr(&thir[expr]);

    for param in thir.params.iter() {
        if let Some(box ref pattern) = param.pat {
            visitor.check_irrefutable(pattern, "function argument", None);
        }
    }
    visitor.error
}

fn create_e0004(
    sess: &Session,
    sp: Span,
    error_message: String,
) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
    struct_span_err!(sess, sp, E0004, "{}", &error_message)
}

#[derive(PartialEq)]
enum RefutableFlag {
    Irrefutable,
    Refutable,
}
use RefutableFlag::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LetSource {
    None,
    IfLet,
    IfLetGuard,
    LetElse,
    WhileLet,
}

struct MatchVisitor<'a, 'p, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    thir: &'a Thir<'tcx>,
    lint_level: HirId,
    let_source: LetSource,
    pattern_arena: &'p TypedArena<DeconstructedPat<'p, 'tcx>>,
    error: Result<(), ErrorGuaranteed>,
}

impl<'a, 'tcx> Visitor<'a, 'tcx> for MatchVisitor<'a, '_, 'tcx> {
    fn thir(&self) -> &'a Thir<'tcx> {
        self.thir
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_arm(&mut self, arm: &Arm<'tcx>) {
        self.with_lint_level(arm.lint_level, |this| {
            match arm.guard {
                Some(Guard::If(expr)) => {
                    this.with_let_source(LetSource::IfLetGuard, |this| {
                        this.visit_expr(&this.thir[expr])
                    });
                }
                Some(Guard::IfLet(ref pat, expr)) => {
                    this.with_let_source(LetSource::IfLetGuard, |this| {
                        this.check_let(pat, expr, LetSource::IfLetGuard, pat.span);
                        this.visit_pat(pat);
                        this.visit_expr(&this.thir[expr]);
                    });
                }
                None => {}
            }
            this.visit_pat(&arm.pattern);
            this.visit_expr(&self.thir[arm.body]);
        });
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_expr(&mut self, ex: &Expr<'tcx>) {
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
            ExprKind::Match { scrutinee, box ref arms } => {
                let source = match ex.span.desugaring_kind() {
                    Some(DesugaringKind::ForLoop) => hir::MatchSource::ForLoopDesugar,
                    Some(DesugaringKind::QuestionMark) => hir::MatchSource::TryDesugar,
                    Some(DesugaringKind::Await) => hir::MatchSource::AwaitDesugar,
                    _ => hir::MatchSource::Normal,
                };
                self.check_match(scrutinee, arms, source, ex.span);
            }
            ExprKind::Let { box ref pat, expr } => {
                self.check_let(pat, expr, self.let_source, ex.span);
            }
            ExprKind::LogicalOp { op: LogicalOp::And, lhs, rhs } => {
                self.check_let_chain(self.let_source, ex.span, lhs, rhs);
            }
            _ => {}
        };
        self.with_let_source(LetSource::None, |this| visit::walk_expr(this, ex));
    }

    fn visit_stmt(&mut self, stmt: &Stmt<'tcx>) {
        let old_lint_level = self.lint_level;
        match stmt.kind {
            StmtKind::Let {
                box ref pattern, initializer, else_block, lint_level, span, ..
            } => {
                if let LintLevel::Explicit(lint_level) = lint_level {
                    self.lint_level = lint_level;
                }

                if let Some(initializer) = initializer && else_block.is_some() {
                    self.check_let(pattern, initializer, LetSource::LetElse, span);
                }

                if else_block.is_none() {
                    self.check_irrefutable(pattern, "local binding", Some(span));
                }
            }
            _ => {}
        }
        visit::walk_stmt(self, stmt);
        self.lint_level = old_lint_level;
    }
}

impl<'p, 'tcx> MatchVisitor<'_, 'p, 'tcx> {
    #[instrument(level = "trace", skip(self, f))]
    fn with_let_source(&mut self, let_source: LetSource, f: impl FnOnce(&mut Self)) {
        let old_let_source = self.let_source;
        self.let_source = let_source;
        ensure_sufficient_stack(|| f(self));
        self.let_source = old_let_source;
    }

    fn with_lint_level(&mut self, new_lint_level: LintLevel, f: impl FnOnce(&mut Self)) {
        if let LintLevel::Explicit(hir_id) = new_lint_level {
            let old_lint_level = self.lint_level;
            self.lint_level = hir_id;
            f(self);
            self.lint_level = old_lint_level;
        } else {
            f(self);
        }
    }

    fn check_patterns(&self, pat: &Pat<'tcx>, rf: RefutableFlag) {
        pat.walk_always(|pat| check_borrow_conflicts_in_at_patterns(self, pat));
        check_for_bindings_named_same_as_variants(self, pat, rf);
    }

    fn lower_pattern(
        &self,
        cx: &mut MatchCheckCtxt<'p, 'tcx>,
        pattern: &Pat<'tcx>,
    ) -> &'p DeconstructedPat<'p, 'tcx> {
        cx.pattern_arena.alloc(DeconstructedPat::from_pat(cx, &pattern))
    }

    fn new_cx(&self, hir_id: HirId, refutable: bool) -> MatchCheckCtxt<'p, 'tcx> {
        MatchCheckCtxt {
            tcx: self.tcx,
            param_env: self.param_env,
            module: self.tcx.parent_module(hir_id).to_def_id(),
            pattern_arena: &self.pattern_arena,
            refutable,
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn check_let(&mut self, pat: &Pat<'tcx>, scrutinee: ExprId, source: LetSource, span: Span) {
        if let LetSource::None = source {
            return;
        }
        self.check_patterns(pat, Refutable);
        let mut cx = self.new_cx(self.lint_level, true);
        let tpat = self.lower_pattern(&mut cx, pat);
        self.check_let_reachability(&mut cx, self.lint_level, source, tpat, span);
    }

    fn check_match(
        &mut self,
        scrut: ExprId,
        arms: &[ArmId],
        source: hir::MatchSource,
        expr_span: Span,
    ) {
        let mut cx = self.new_cx(self.lint_level, true);

        for &arm in arms {
            // Check the arm for some things unrelated to exhaustiveness.
            let arm = &self.thir.arms[arm];
            self.with_lint_level(arm.lint_level, |this| {
                this.check_patterns(&arm.pattern, Refutable);
            });
        }

        let tarms: Vec<_> = arms
            .iter()
            .map(|&arm| {
                let arm = &self.thir.arms[arm];
                let hir_id = match arm.lint_level {
                    LintLevel::Explicit(hir_id) => hir_id,
                    LintLevel::Inherited => self.lint_level,
                };
                let pat = self.lower_pattern(&mut cx, &arm.pattern);
                MatchArm { pat, hir_id, has_guard: arm.guard.is_some() }
            })
            .collect();

        let scrut = &self.thir[scrut];
        let scrut_ty = scrut.ty;
        let report = compute_match_usefulness(&cx, &tarms, self.lint_level, scrut_ty);

        match source {
            // Don't report arm reachability of desugared `match $iter.into_iter() { iter => .. }`
            // when the iterator is an uninhabited type. unreachable_code will trigger instead.
            hir::MatchSource::ForLoopDesugar if arms.len() == 1 => {}
            hir::MatchSource::ForLoopDesugar
            | hir::MatchSource::Normal
            | hir::MatchSource::FormatArgs => report_arm_reachability(&cx, &report),
            // Unreachable patterns in try and await expressions occur when one of
            // the arms are an uninhabited type. Which is OK.
            hir::MatchSource::AwaitDesugar | hir::MatchSource::TryDesugar => {}
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
                self.check_irrefutable(&pat_field.pattern, "`for` loop binding", None);
            } else {
                self.error = Err(non_exhaustive_match(
                    &cx, self.thir, scrut_ty, scrut.span, witnesses, arms, expr_span,
                ));
            }
        }
    }

    fn check_let_reachability(
        &mut self,
        cx: &mut MatchCheckCtxt<'p, 'tcx>,
        pat_id: HirId,
        source: LetSource,
        pat: &'p DeconstructedPat<'p, 'tcx>,
        span: Span,
    ) {
        if is_let_irrefutable(cx, pat_id, pat) {
            irrefutable_let_patterns(cx.tcx, pat_id, source, 1, span);
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn check_let_chain(
        &mut self,
        let_source: LetSource,
        top_expr_span: Span,
        mut lhs: ExprId,
        rhs: ExprId,
    ) {
        if let LetSource::None = let_source {
            return;
        }

        // Lint level enclosing the next `lhs`.
        let mut cur_lint_level = self.lint_level;

        // Obtain the refutabilities of all exprs in the chain,
        // and record chain members that aren't let exprs.
        let mut chain_refutabilities = Vec::new();

        let add = |expr: ExprId, mut local_lint_level| {
            // `local_lint_level` is the lint level enclosing the pattern inside `expr`.
            let mut expr = &self.thir[expr];
            debug!(?expr, ?local_lint_level, "add");
            // Fast-forward through scopes.
            while let ExprKind::Scope { value, lint_level, .. } = expr.kind {
                if let LintLevel::Explicit(hir_id) = lint_level {
                    local_lint_level = hir_id
                }
                expr = &self.thir[value];
            }
            debug!(?expr, ?local_lint_level, "after scopes");
            match expr.kind {
                ExprKind::Let { box ref pat, expr: _ } => {
                    let mut ncx = self.new_cx(local_lint_level, true);
                    let tpat = self.lower_pattern(&mut ncx, pat);
                    let refutable = !is_let_irrefutable(&mut ncx, local_lint_level, tpat);
                    Some((expr.span, refutable))
                }
                _ => None,
            }
        };

        // Let chains recurse on the left, so we start by adding the rightmost.
        chain_refutabilities.push(add(rhs, cur_lint_level));

        loop {
            while let ExprKind::Scope { value, lint_level, .. } = self.thir[lhs].kind {
                if let LintLevel::Explicit(hir_id) = lint_level {
                    cur_lint_level = hir_id
                }
                lhs = value;
            }
            if let ExprKind::LogicalOp { op: LogicalOp::And, lhs: new_lhs, rhs: expr } =
                self.thir[lhs].kind
            {
                chain_refutabilities.push(add(expr, cur_lint_level));
                lhs = new_lhs;
            } else {
                chain_refutabilities.push(add(lhs, cur_lint_level));
                break;
            }
        }
        debug!(?chain_refutabilities);
        chain_refutabilities.reverse();

        // Third, emit the actual warnings.
        if chain_refutabilities.iter().all(|r| matches!(*r, Some((_, false)))) {
            // The entire chain is made up of irrefutable `let` statements
            irrefutable_let_patterns(
                self.tcx,
                self.lint_level,
                let_source,
                chain_refutabilities.len(),
                top_expr_span,
            );
            return;
        }

        if let Some(until) = chain_refutabilities.iter().position(|r| !matches!(*r, Some((_, false)))) && until > 0 {
            // The chain has a non-zero prefix of irrefutable `let` statements.

            // Check if the let source is while, for there is no alternative place to put a prefix,
            // and we shouldn't lint.
            // For let guards inside a match, prefixes might use bindings of the match pattern,
            // so can't always be moved out.
            // FIXME: Add checking whether the bindings are actually used in the prefix,
            // and lint if they are not.
            if !matches!(let_source, LetSource::WhileLet | LetSource::IfLetGuard) {
                // Emit the lint
                let prefix = &chain_refutabilities[..until];
                let span_start = prefix[0].unwrap().0;
                let span_end = prefix.last().unwrap().unwrap().0;
                let span = span_start.to(span_end);
                let count = prefix.len();
                self.tcx.emit_spanned_lint(IRREFUTABLE_LET_PATTERNS, self.lint_level, span, LeadingIrrefutableLetPatterns { count });
            }
        }

        if let Some(from) = chain_refutabilities.iter().rposition(|r| !matches!(*r, Some((_, false)))) && from != (chain_refutabilities.len() - 1) {
            // The chain has a non-empty suffix of irrefutable `let` statements
            let suffix = &chain_refutabilities[from + 1..];
            let span_start = suffix[0].unwrap().0;
            let span_end = suffix.last().unwrap().unwrap().0;
            let span = span_start.to(span_end);
            let count = suffix.len();
            self.tcx.emit_spanned_lint(IRREFUTABLE_LET_PATTERNS, self.lint_level, span, TrailingIrrefutableLetPatterns { count });
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn check_irrefutable(&mut self, pat: &Pat<'tcx>, origin: &str, sp: Option<Span>) {
        let mut cx = self.new_cx(self.lint_level, false);

        let pattern = self.lower_pattern(&mut cx, pat);
        let pattern_ty = pattern.ty();
        let arm = MatchArm { pat: pattern, hir_id: self.lint_level, has_guard: false };
        let report = compute_match_usefulness(&cx, &[arm], self.lint_level, pattern_ty);

        // Note: we ignore whether the pattern is unreachable (i.e. whether the type is empty). We
        // only care about exhaustiveness here.
        let witnesses = report.non_exhaustiveness_witnesses;
        if witnesses.is_empty() {
            // The pattern is irrefutable.
            self.check_patterns(pat, Irrefutable);
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
                    start_span: pat.span.shrink_to_lo()
                });
            } else if snippet.chars().all(|c| c.is_alphanumeric() || c == '_') {
                interpreted_as_const = Some(InterpretedAsConst {
                    span: pat.span,
                    variable: snippet,
                });
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

        let adt_defined_here = try {
            let ty = pattern_ty.peel_refs();
            let ty::Adt(def, _) = ty.kind() else { None? };
            let adt_def_span = cx.tcx.hir().get_if_local(def.did())?.ident()?.span;
            let mut variants = vec![];

            for span in maybe_point_at_variant(&cx, *def, witnesses.iter().take(5)) {
                variants.push(Variant { span });
            }
            AdtDefinedHere { adt_def_span, ty, variants }
        };

        // Emit an extra note if the first uncovered witness would be uninhabited
        // if we disregard visibility.
        let witness_1_is_privately_uninhabited =
            if cx.tcx.features().exhaustive_patterns
                && let Some(witness_1) = witnesses.get(0)
                && let ty::Adt(adt, substs) = witness_1.ty().kind()
                && adt.is_enum()
                && let Constructor::Variant(variant_index) = witness_1.ctor()
            {
                let variant = adt.variant(*variant_index);
                let inhabited = variant.inhabited_predicate(cx.tcx, *adt).subst(cx.tcx, substs);
                assert!(inhabited.apply(cx.tcx, cx.param_env, cx.module));
                !inhabited.apply_ignore_module(cx.tcx, cx.param_env)
            } else {
                false
            };

        self.error = Err(self.tcx.sess.emit_err(PatternNotCovered {
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

fn check_for_bindings_named_same_as_variants(
    cx: &MatchVisitor<'_, '_, '_>,
    pat: &Pat<'_>,
    rf: RefutableFlag,
) {
    pat.walk_always(|p| {
        if let PatKind::Binding {
                name,
                mode: BindingMode::ByValue,
                mutability: Mutability::Not,
                subpattern: None,
                ty,
                ..
            } = p.kind
            && let ty::Adt(edef, _) = ty.peel_refs().kind()
            && edef.is_enum()
            && edef.variants().iter().any(|variant| {
                variant.name == name && variant.ctor_kind() == Some(CtorKind::Const)
            })
        {
            let variant_count = edef.variants().len();
            let ty_path = with_no_trimmed_paths!({
                cx.tcx.def_path_str(edef.did())
            });
            cx.tcx.emit_spanned_lint(
                BINDINGS_WITH_VARIANT_NAME,
                cx.lint_level,
                p.span,
                BindingsWithVariantName {
                    // If this is an irrefutable pattern, and there's > 1 variant,
                    // then we can't actually match on this. Applying the below
                    // suggestion would produce code that breaks on `check_irrefutable`.
                    suggestion: if rf == Refutable || variant_count == 1 {
                        Some(p.span)
                    } else { None },
                    ty_path,
                    name,
                },
            )
        }
    });
}

/// Checks for common cases of "catchall" patterns that may not be intended as such.
fn pat_is_catchall(pat: &DeconstructedPat<'_, '_>) -> bool {
    use Constructor::*;
    match pat.ctor() {
        Wildcard => true,
        Single => pat.iter_fields().all(|pat| pat_is_catchall(pat)),
        _ => false,
    }
}

fn unreachable_pattern(tcx: TyCtxt<'_>, span: Span, id: HirId, catchall: Option<Span>) {
    tcx.emit_spanned_lint(
        UNREACHABLE_PATTERNS,
        id,
        span,
        UnreachablePattern { span: if catchall.is_some() { Some(span) } else { None }, catchall },
    );
}

fn irrefutable_let_patterns(
    tcx: TyCtxt<'_>,
    id: HirId,
    source: LetSource,
    count: usize,
    span: Span,
) {
    macro_rules! emit_diag {
        ($lint:tt) => {{
            tcx.emit_spanned_lint(IRREFUTABLE_LET_PATTERNS, id, span, $lint { count });
        }};
    }

    match source {
        LetSource::None => bug!(),
        LetSource::IfLet => emit_diag!(IrrefutableLetPatternsIfLet),
        LetSource::IfLetGuard => emit_diag!(IrrefutableLetPatternsIfLetGuard),
        LetSource::LetElse => emit_diag!(IrrefutableLetPatternsLetElse),
        LetSource::WhileLet => emit_diag!(IrrefutableLetPatternsWhileLet),
    }
}

fn is_let_irrefutable<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    pat_id: HirId,
    pat: &'p DeconstructedPat<'p, 'tcx>,
) -> bool {
    let arms = [MatchArm { pat, hir_id: pat_id, has_guard: false }];
    let report = compute_match_usefulness(&cx, &arms, pat_id, pat.ty());

    // Report if the pattern is unreachable, which can only occur when the type is uninhabited.
    // This also reports unreachable sub-patterns though, so we can't just replace it with an
    // `is_uninhabited` check.
    report_arm_reachability(&cx, &report);

    // If the list of witnesses is empty, the match is exhaustive,
    // i.e. the `if let` pattern is irrefutable.
    report.non_exhaustiveness_witnesses.is_empty()
}

/// Report unreachable arms, if any.
fn report_arm_reachability<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    report: &UsefulnessReport<'p, 'tcx>,
) {
    use Reachability::*;
    let mut catchall = None;
    for (arm, is_useful) in report.arm_usefulness.iter() {
        match is_useful {
            Unreachable => unreachable_pattern(cx.tcx, arm.pat.span(), arm.hir_id, catchall),
            Reachable(unreachables) if unreachables.is_empty() => {}
            // The arm is reachable, but contains unreachable subpatterns (from or-patterns).
            Reachable(unreachables) => {
                let mut unreachables = unreachables.clone();
                // Emit lints in the order in which they occur in the file.
                unreachables.sort_unstable();
                for span in unreachables {
                    unreachable_pattern(cx.tcx, span, arm.hir_id, None);
                }
            }
        }
        if !arm.has_guard && catchall.is_none() && pat_is_catchall(arm.pat) {
            catchall = Some(arm.pat.span());
        }
    }
}

/// Report that a match is not exhaustive.
fn non_exhaustive_match<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    thir: &Thir<'tcx>,
    scrut_ty: Ty<'tcx>,
    sp: Span,
    witnesses: Vec<DeconstructedPat<'p, 'tcx>>,
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
    let mut err;
    let pattern;
    let patterns_len;
    if is_empty_match && !non_empty_enum {
        return cx.tcx.sess.emit_err(NonExhaustivePatternsTypeNotEmpty {
            cx,
            expr_span,
            span: sp,
            ty: scrut_ty,
        });
    } else {
        // FIXME: migration of this diagnostic will require list support
        let joined_patterns = joined_uncovered_patterns(cx, &witnesses);
        err = create_e0004(
            cx.tcx.sess,
            sp,
            format!("non-exhaustive patterns: {} not covered", joined_patterns),
        );
        err.span_label(sp, pattern_not_covered_label(&witnesses, &joined_patterns));
        patterns_len = witnesses.len();
        pattern = if witnesses.len() < 4 {
            witnesses
                .iter()
                .map(|witness| witness.to_pat(cx).to_string())
                .collect::<Vec<String>>()
                .join(" | ")
        } else {
            "_".to_string()
        };
    };

    let is_variant_list_non_exhaustive = matches!(scrut_ty.kind(),
        ty::Adt(def, _) if def.is_variant_list_non_exhaustive() && !def.did().is_local());

    adt_defined_here(cx, &mut err, scrut_ty, &witnesses);
    err.note(format!(
        "the matched value is of type `{}`{}",
        scrut_ty,
        if is_variant_list_non_exhaustive { ", which is marked as non-exhaustive" } else { "" }
    ));
    if (scrut_ty == cx.tcx.types.usize || scrut_ty == cx.tcx.types.isize)
        && !is_empty_match
        && witnesses.len() == 1
        && matches!(witnesses[0].ctor(), Constructor::NonExhaustive)
    {
        err.note(format!(
            "`{}` does not have a fixed maximum value, so a wildcard `_` is necessary to match \
             exhaustively",
            scrut_ty,
        ));
        if cx.tcx.sess.is_nightly_build() {
            err.help(format!(
                "add `#![feature(precise_pointer_size_matching)]` to the crate attributes to \
                 enable precise `{}` matching",
                scrut_ty,
            ));
        }
    }
    if let ty::Ref(_, sub_ty, _) = scrut_ty.kind() {
        if !sub_ty.is_inhabited_from(cx.tcx, cx.module, cx.param_env) {
            err.note("references are always considered inhabited");
        }
    }

    let mut suggestion = None;
    let sm = cx.tcx.sess.source_map();
    match arms {
        [] if sp.eq_ctxt(expr_span) => {
            // Get the span for the empty match body `{}`.
            let (indentation, more) = if let Some(snippet) = sm.indentation_before(sp) {
                (format!("\n{}", snippet), "    ")
            } else {
                (" ".to_string(), "")
            };
            suggestion = Some((
                sp.shrink_to_hi().with_hi(expr_span.hi()),
                format!(
                    " {{{indentation}{more}{pattern} => todo!(),{indentation}}}",
                    indentation = indentation,
                    more = more,
                    pattern = pattern,
                ),
            ));
        }
        [only] => {
            let only = &thir[*only];
            let (pre_indentation, is_multiline) = if let Some(snippet) = sm.indentation_before(only.span)
                && let Ok(with_trailing) = sm.span_extend_while(only.span, |c| c.is_whitespace() || c == ',')
                && sm.is_multiline(with_trailing)
            {
                (format!("\n{}", snippet), true)
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
                format!("{}{}{} => todo!()", comma, pre_indentation, pattern),
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
                        format!("{}{}{} => todo!()", comma, spacing, pattern),
                    ));
                }
            }
        }
        _ => {}
    }

    let msg = format!(
        "ensure that all possible cases are being handled by adding a match arm with a wildcard \
         pattern{}{}",
        if patterns_len > 1 && patterns_len < 4 && suggestion.is_some() {
            ", a match arm with multiple or-patterns"
        } else {
            // we are either not suggesting anything, or suggesting `_`
            ""
        },
        match patterns_len {
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

pub(crate) fn joined_uncovered_patterns<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    witnesses: &[DeconstructedPat<'p, 'tcx>],
) -> String {
    const LIMIT: usize = 3;
    let pat_to_str = |pat: &DeconstructedPat<'p, 'tcx>| pat.to_pat(cx).to_string();
    match witnesses {
        [] => bug!(),
        [witness] => format!("`{}`", witness.to_pat(cx)),
        [head @ .., tail] if head.len() < LIMIT => {
            let head: Vec<_> = head.iter().map(pat_to_str).collect();
            format!("`{}` and `{}`", head.join("`, `"), tail.to_pat(cx))
        }
        _ => {
            let (head, tail) = witnesses.split_at(LIMIT);
            let head: Vec<_> = head.iter().map(pat_to_str).collect();
            format!("`{}` and {} more", head.join("`, `"), tail.len())
        }
    }
}

pub(crate) fn pattern_not_covered_label(
    witnesses: &[DeconstructedPat<'_, '_>],
    joined_patterns: &str,
) -> String {
    format!("pattern{} {} not covered", rustc_errors::pluralize!(witnesses.len()), joined_patterns)
}

/// Point at the definition of non-covered `enum` variants.
fn adt_defined_here<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    err: &mut Diagnostic,
    ty: Ty<'tcx>,
    witnesses: &[DeconstructedPat<'p, 'tcx>],
) {
    let ty = ty.peel_refs();
    if let ty::Adt(def, _) = ty.kind() {
        let mut spans = vec![];
        if witnesses.len() < 5 {
            for sp in maybe_point_at_variant(cx, *def, witnesses.iter()) {
                spans.push(sp);
            }
        }
        let def_span = cx
            .tcx
            .hir()
            .get_if_local(def.did())
            .and_then(|node| node.ident())
            .map(|ident| ident.span)
            .unwrap_or_else(|| cx.tcx.def_span(def.did()));
        let mut span: MultiSpan =
            if spans.is_empty() { def_span.into() } else { spans.clone().into() };

        span.push_span_label(def_span, "");
        for pat in spans {
            span.push_span_label(pat, "not covered");
        }
        err.span_note(span, format!("`{}` defined here", ty));
    }
}

fn maybe_point_at_variant<'a, 'p: 'a, 'tcx: 'a>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    def: AdtDef<'tcx>,
    patterns: impl Iterator<Item = &'a DeconstructedPat<'p, 'tcx>>,
) -> Vec<Span> {
    use Constructor::*;
    let mut covered = vec![];
    for pattern in patterns {
        if let Variant(variant_index) = pattern.ctor() {
            if let ty::Adt(this_def, _) = pattern.ty().kind() && this_def.did() != def.did() {
                continue;
            }
            let sp = def.variant(*variant_index).ident(cx.tcx).span;
            if covered.contains(&sp) {
                // Don't point at variants that have already been covered due to other patterns to avoid
                // visual clutter.
                continue;
            }
            covered.push(sp);
        }
        covered.extend(maybe_point_at_variant(cx, def, pattern.iter_fields()));
    }
    covered
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
fn check_borrow_conflicts_in_at_patterns<'tcx>(cx: &MatchVisitor<'_, '_, 'tcx>, pat: &Pat<'tcx>) {
    // Extract `sub` in `binding @ sub`.
    let PatKind::Binding { name, mode, ty, subpattern: Some(box ref sub), .. } = pat.kind else { return };

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
                sess.emit_err(BorrowOfMovedValue {
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
        sess.emit_err(MultipleMutBorrows { span: pat.span, occurrences });
    } else if report_mut_ref {
        // Report mutability conflicts for e.g. `ref x @ Some(ref mut y)` or the converse.
        match mut_outer {
            Mutability::Mut => {
                sess.emit_err(AlreadyMutBorrowed { span: pat.span, occurrences });
            }
            Mutability::Not => {
                sess.emit_err(AlreadyBorrowed { span: pat.span, occurrences });
            }
        };
    } else if report_move_conflict {
        // Report by-ref and by-move conflicts, e.g. `ref x @ y`.
        sess.emit_err(MovedWhileBorrowed { span: pat.span, occurrences });
    }
}
