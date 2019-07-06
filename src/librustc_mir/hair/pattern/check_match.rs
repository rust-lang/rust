use super::_match::{MatchCheckCtxt, Matrix, expand_pattern, is_useful};
use super::_match::Usefulness::*;
use super::_match::WitnessPreference::*;

use super::{Pattern, PatternContext, PatternError, PatternKind};

use rustc::middle::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor};
use rustc::middle::expr_use_visitor::{LoanCause, MutateMode};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization::cmt_;
use rustc::middle::region;
use rustc::session::Session;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{InternalSubsts, SubstsRef};
use rustc::lint;
use rustc_errors::{Applicability, DiagnosticBuilder};

use rustc::hir::def::*;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::ptr::P;
use rustc::hir::{self, Pat, PatKind};

use smallvec::smallvec;
use std::slice;

use syntax_pos::{Span, DUMMY_SP, MultiSpan};

pub(crate) fn check_match(tcx: TyCtxt<'_>, def_id: DefId) {
    let body_id = if let Some(id) = tcx.hir().as_local_hir_id(def_id) {
        tcx.hir().body_owned_by(id)
    } else {
        return;
    };

    MatchVisitor {
        tcx,
        body_owner: def_id,
        tables: tcx.body_tables(body_id),
        region_scope_tree: &tcx.region_scope_tree(def_id),
        param_env: tcx.param_env(def_id),
        identity_substs: InternalSubsts::identity_for_item(tcx, def_id),
    }.visit_body(tcx.hir().body(body_id));
}

fn create_e0004(sess: &Session, sp: Span, error_message: String) -> DiagnosticBuilder<'_> {
    struct_span_err!(sess, sp, E0004, "{}", &error_message)
}

struct MatchVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body_owner: DefId,
    tables: &'a ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    identity_substs: SubstsRef<'tcx>,
    region_scope_tree: &'a region::ScopeTree,
}

impl<'a, 'tcx> Visitor<'tcx> for MatchVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        intravisit::walk_expr(self, ex);

        match ex.node {
            hir::ExprKind::Match(ref scrut, ref arms, source) => {
                self.check_match(scrut, arms, source);
            }
            _ => {}
        }
    }

    fn visit_local(&mut self, loc: &'tcx hir::Local) {
        intravisit::walk_local(self, loc);

        self.check_irrefutable(&loc.pat, match loc.source {
            hir::LocalSource::Normal => "local binding",
            hir::LocalSource::ForLoopDesugar => "`for` loop binding",
            hir::LocalSource::AsyncFn => "async fn binding",
            hir::LocalSource::AwaitDesugar => "`await` future binding",
        });

        // Check legality of move bindings and `@` patterns.
        self.check_patterns(false, slice::from_ref(&loc.pat));
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        intravisit::walk_body(self, body);

        for arg in &body.arguments {
            self.check_irrefutable(&arg.pat, "function argument");
            self.check_patterns(false, slice::from_ref(&arg.pat));
        }
    }
}


impl<'a, 'tcx> PatternContext<'a, 'tcx> {
    fn report_inlining_errors(&self, pat_span: Span) {
        for error in &self.errors {
            match *error {
                PatternError::StaticInPattern(span) => {
                    self.span_e0158(span, "statics cannot be referenced in patterns")
                }
                PatternError::AssocConstInPattern(span) => {
                    self.span_e0158(span, "associated consts cannot be referenced in patterns")
                }
                PatternError::FloatBug => {
                    // FIXME(#31407) this is only necessary because float parsing is buggy
                    ::rustc::mir::interpret::struct_error(
                        self.tcx.at(pat_span),
                        "could not evaluate float literal (see issue #31407)",
                    ).emit();
                }
                PatternError::NonConstPath(span) => {
                    ::rustc::mir::interpret::struct_error(
                        self.tcx.at(span),
                        "runtime values cannot be referenced in patterns",
                    ).emit();
                }
            }
        }
    }

    fn span_e0158(&self, span: Span, text: &str) {
        span_err!(self.tcx.sess, span, E0158, "{}", text)
    }
}

impl<'a, 'tcx> MatchVisitor<'a, 'tcx> {
    fn check_patterns(&self, has_guard: bool, pats: &[P<Pat>]) {
        check_legality_of_move_bindings(self, has_guard, pats);
        for pat in pats {
            check_legality_of_bindings_in_at_patterns(self, pat);
        }
    }

    fn check_match(
        &self,
        scrut: &hir::Expr,
        arms: &'tcx [hir::Arm],
        source: hir::MatchSource)
    {
        for arm in arms {
            // First, check legality of move bindings.
            self.check_patterns(arm.guard.is_some(), &arm.pats);

            // Second, if there is a guard on each arm, make sure it isn't
            // assigning or borrowing anything mutably.
            if let Some(ref guard) = arm.guard {
                if !self.tcx.features().bind_by_move_pattern_guards {
                    check_for_mutation_in_guard(self, &guard);
                }
            }

            // Third, perform some lints.
            for pat in &arm.pats {
                check_for_bindings_named_same_as_variants(self, pat);
            }
        }

        let module = self.tcx.hir().get_module_parent(scrut.hir_id);
        MatchCheckCtxt::create_and_enter(self.tcx, self.param_env, module, |ref mut cx| {
            let mut have_errors = false;

            let inlined_arms : Vec<(Vec<_>, _)> = arms.iter().map(|arm| (
                arm.pats.iter().map(|pat| {
                    let mut patcx = PatternContext::new(self.tcx,
                                                        self.param_env.and(self.identity_substs),
                                                        self.tables);
                    let pattern = expand_pattern(cx, patcx.lower_pattern(&pat));
                    if !patcx.errors.is_empty() {
                        patcx.report_inlining_errors(pat.span);
                        have_errors = true;
                    }
                    (pattern, &**pat)
                }).collect(),
                arm.guard.as_ref().map(|g| match g {
                    hir::Guard::If(ref e) => &**e,
                })
            )).collect();

            // Bail out early if inlining failed.
            if have_errors {
                return;
            }

            // Fourth, check for unreachable arms.
            check_arms(cx, &inlined_arms, source);

            // Then, if the match has no arms, check whether the scrutinee
            // is uninhabited.
            let pat_ty = self.tables.node_type(scrut.hir_id);
            let module = self.tcx.hir().get_module_parent(scrut.hir_id);
            let mut def_span = None;
            let mut missing_variants = vec![];
            if inlined_arms.is_empty() {
                let scrutinee_is_uninhabited = if self.tcx.features().exhaustive_patterns {
                    self.tcx.is_ty_uninhabited_from(module, pat_ty)
                } else {
                    match pat_ty.sty {
                        ty::Never => true,
                        ty::Adt(def, _) => {
                            def_span = self.tcx.hir().span_if_local(def.did);
                            if def.variants.len() < 4 && !def.variants.is_empty() {
                                // keep around to point at the definition of non-covered variants
                                missing_variants = def.variants.iter()
                                    .map(|variant| variant.ident)
                                    .collect();
                            }

                            let is_non_exhaustive_and_non_local =
                                def.is_variant_list_non_exhaustive() && !def.did.is_local();

                            !(is_non_exhaustive_and_non_local) && def.variants.is_empty()
                        },
                        _ => false
                    }
                };
                if !scrutinee_is_uninhabited {
                    // We know the type is inhabited, so this must be wrong
                    let mut err = create_e0004(
                        self.tcx.sess,
                        scrut.span,
                        format!("non-exhaustive patterns: {}", match missing_variants.len() {
                            0 => format!("type `{}` is non-empty", pat_ty),
                            1 => format!(
                                "pattern `{}` of type `{}` is not handled",
                                missing_variants[0].name,
                                pat_ty,
                            ),
                            _ => format!("multiple patterns of type `{}` are not handled", pat_ty),
                        }),
                    );
                    err.help("ensure that all possible cases are being handled, \
                              possibly by adding wildcards or more match arms");
                    if let Some(sp) = def_span {
                        err.span_label(sp, format!("`{}` defined here", pat_ty));
                    }
                    // point at the definition of non-covered enum variants
                    for variant in &missing_variants {
                        err.span_label(variant.span, "variant not covered");
                    }
                    err.emit();
                }
                // If the type *is* uninhabited, it's vacuously exhaustive
                return;
            }

            let matrix: Matrix<'_, '_> = inlined_arms
                .iter()
                .filter(|&&(_, guard)| guard.is_none())
                .flat_map(|arm| &arm.0)
                .map(|pat| smallvec![pat.0])
                .collect();
            let scrut_ty = self.tables.node_type(scrut.hir_id);
            check_exhaustive(cx, scrut_ty, scrut.span, &matrix);
        })
    }

    fn check_irrefutable(&self, pat: &'tcx Pat, origin: &str) {
        let module = self.tcx.hir().get_module_parent(pat.hir_id);
        MatchCheckCtxt::create_and_enter(self.tcx, self.param_env, module, |ref mut cx| {
            let mut patcx = PatternContext::new(self.tcx,
                                                self.param_env.and(self.identity_substs),
                                                self.tables);
            let pattern = patcx.lower_pattern(pat);
            let pattern_ty = pattern.ty;
            let pats: Matrix<'_, '_> = vec![smallvec![
                expand_pattern(cx, pattern)
            ]].into_iter().collect();

            let wild_pattern = Pattern {
                ty: pattern_ty,
                span: DUMMY_SP,
                kind: box PatternKind::Wild,
            };
            let witness = match is_useful(cx, &pats, &[&wild_pattern], ConstructWitness) {
                UsefulWithWitness(witness) => witness,
                NotUseful => return,
                Useful => bug!()
            };

            let pattern_string = witness[0].single_pattern().to_string();
            let mut err = struct_span_err!(
                self.tcx.sess, pat.span, E0005,
                "refutable pattern in {}: `{}` not covered",
                origin, pattern_string
            );
            let label_msg = match pat.node {
                PatKind::Path(hir::QPath::Resolved(None, ref path))
                        if path.segments.len() == 1 && path.segments[0].args.is_none() => {
                    format!("interpreted as {} {} pattern, not new variable",
                            path.res.article(), path.res.descr())
                }
                _ => format!("pattern `{}` not covered", pattern_string),
            };
            err.span_label(pat.span, label_msg);
            if let ty::Adt(def, _) = pattern_ty.sty {
                if let Some(sp) = self.tcx.hir().span_if_local(def.did){
                    err.span_label(sp, format!("`{}` defined here", pattern_ty));
                }
            }
            err.emit();
        });
    }
}

fn check_for_bindings_named_same_as_variants(cx: &MatchVisitor<'_, '_>, pat: &Pat) {
    pat.walk(|p| {
        if let PatKind::Binding(_, _, ident, None) = p.node {
            if let Some(&bm) = cx.tables.pat_binding_modes().get(p.hir_id) {
                if bm != ty::BindByValue(hir::MutImmutable) {
                    // Nothing to check.
                    return true;
                }
                let pat_ty = cx.tables.pat_ty(p);
                if let ty::Adt(edef, _) = pat_ty.sty {
                    if edef.is_enum() && edef.variants.iter().any(|variant| {
                        variant.ident == ident && variant.ctor_kind == CtorKind::Const
                    }) {
                        let ty_path = cx.tcx.def_path_str(edef.did);
                        let mut err = struct_span_warn!(cx.tcx.sess, p.span, E0170,
                            "pattern binding `{}` is named the same as one \
                            of the variants of the type `{}`",
                            ident, ty_path);
                        err.span_suggestion(
                            p.span,
                            "to match on the variant, qualify the path",
                            format!("{}::{}", ty_path, ident),
                            Applicability::MachineApplicable
                        );
                        err.emit();
                    }
                }
            } else {
                cx.tcx.sess.delay_span_bug(p.span, "missing binding mode");
            }
        }
        true
    });
}

/// Checks for common cases of "catchall" patterns that may not be intended as such.
fn pat_is_catchall(pat: &Pat) -> bool {
    match pat.node {
        PatKind::Binding(.., None) => true,
        PatKind::Binding(.., Some(ref s)) => pat_is_catchall(s),
        PatKind::Ref(ref s, _) => pat_is_catchall(s),
        PatKind::Tuple(ref v, _) => v.iter().all(|p| {
            pat_is_catchall(&p)
        }),
        _ => false
    }
}

// Check for unreachable patterns
fn check_arms<'a, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    arms: &[(Vec<(&'a Pattern<'tcx>, &hir::Pat)>, Option<&hir::Expr>)],
    source: hir::MatchSource,
) {
    let mut seen = Matrix::empty();
    let mut catchall = None;
    for (arm_index, &(ref pats, guard)) in arms.iter().enumerate() {
        for &(pat, hir_pat) in pats {
            let v = smallvec![pat];

            match is_useful(cx, &seen, &v, LeaveOutWitness) {
                NotUseful => {
                    match source {
                        hir::MatchSource::IfDesugar { .. } => bug!(),
                        hir::MatchSource::IfLetDesugar { .. } => {
                            cx.tcx.lint_hir(
                                lint::builtin::IRREFUTABLE_LET_PATTERNS,
                                hir_pat.hir_id,
                                pat.span,
                                "irrefutable if-let pattern",
                            );
                        }

                        hir::MatchSource::WhileLetDesugar => {
                            // check which arm we're on.
                            match arm_index {
                                // The arm with the user-specified pattern.
                                0 => {
                                    cx.tcx.lint_hir(
                                        lint::builtin::UNREACHABLE_PATTERNS,
                                        hir_pat.hir_id, pat.span,
                                        "unreachable pattern");
                                },
                                // The arm with the wildcard pattern.
                                1 => {
                                    cx.tcx.lint_hir(
                                        lint::builtin::IRREFUTABLE_LET_PATTERNS,
                                        hir_pat.hir_id,
                                        pat.span,
                                        "irrefutable while-let pattern",
                                    );
                                },
                                _ => bug!(),
                            }
                        }

                        hir::MatchSource::ForLoopDesugar |
                        hir::MatchSource::Normal => {
                            let mut err = cx.tcx.struct_span_lint_hir(
                                lint::builtin::UNREACHABLE_PATTERNS,
                                hir_pat.hir_id,
                                pat.span,
                                "unreachable pattern",
                            );
                            // if we had a catchall pattern, hint at that
                            if let Some(catchall) = catchall {
                                err.span_label(pat.span, "unreachable pattern");
                                err.span_label(catchall, "matches any value");
                            }
                            err.emit();
                        }

                        // Unreachable patterns in try and await expressions occur when one of
                        // the arms are an uninhabited type. Which is OK.
                        hir::MatchSource::AwaitDesugar |
                        hir::MatchSource::TryDesugar => {}
                    }
                }
                Useful => (),
                UsefulWithWitness(_) => bug!()
            }
            if guard.is_none() {
                seen.push(v);
                if catchall.is_none() && pat_is_catchall(hir_pat) {
                    catchall = Some(pat.span);
                }
            }
        }
    }
}

fn check_exhaustive<'p, 'a, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    scrut_ty: Ty<'tcx>,
    sp: Span,
    matrix: &Matrix<'p, 'tcx>,
) {
    let wild_pattern = Pattern {
        ty: scrut_ty,
        span: DUMMY_SP,
        kind: box PatternKind::Wild,
    };
    match is_useful(cx, matrix, &[&wild_pattern], ConstructWitness) {
        UsefulWithWitness(pats) => {
            let witnesses = if pats.is_empty() {
                vec![&wild_pattern]
            } else {
                pats.iter().map(|w| w.single_pattern()).collect()
            };

            const LIMIT: usize = 3;
            let joined_patterns = match witnesses.len() {
                0 => bug!(),
                1 => format!("`{}`", witnesses[0]),
                2..=LIMIT => {
                    let (tail, head) = witnesses.split_last().unwrap();
                    let head: Vec<_> = head.iter().map(|w| w.to_string()).collect();
                    format!("`{}` and `{}`", head.join("`, `"), tail)
                }
                _ => {
                    let (head, tail) = witnesses.split_at(LIMIT);
                    let head: Vec<_> = head.iter().map(|w| w.to_string()).collect();
                    format!("`{}` and {} more", head.join("`, `"), tail.len())
                }
            };

            let label_text = match witnesses.len() {
                1 => format!("pattern {} not covered", joined_patterns),
                _ => format!("patterns {} not covered", joined_patterns),
            };
            let mut err = create_e0004(cx.tcx.sess, sp, format!(
                "non-exhaustive patterns: {} not covered",
                joined_patterns,
            ));
            err.span_label(sp, label_text);
            // point at the definition of non-covered enum variants
            if let ty::Adt(def, _) = scrut_ty.sty {
                if let Some(sp) = cx.tcx.hir().span_if_local(def.did){
                    err.span_label(sp, format!("`{}` defined here", scrut_ty));
                }
            }
            let patterns = witnesses.iter().map(|p| (**p).clone()).collect::<Vec<Pattern<'_>>>();
            if patterns.len() < 4 {
                for sp in maybe_point_at_variant(cx, scrut_ty, patterns.as_slice()) {
                    err.span_label(sp, "not covered");
                }
            }
            err.help("ensure that all possible cases are being handled, \
                      possibly by adding wildcards or more match arms");
            err.emit();
        }
        NotUseful => {
            // This is good, wildcard pattern isn't reachable
        }
        _ => bug!()
    }
}

fn maybe_point_at_variant(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    ty: Ty<'tcx>,
    patterns: &[Pattern<'_>],
) -> Vec<Span> {
    let mut covered = vec![];
    if let ty::Adt(def, _) = ty.sty {
        // Don't point at variants that have already been covered due to other patterns to avoid
        // visual clutter
        for pattern in patterns {
            let pk: &PatternKind<'_> = &pattern.kind;
            if let PatternKind::Variant { adt_def, variant_index, subpatterns, .. } = pk {
                if adt_def.did == def.did {
                    let sp = def.variants[*variant_index].ident.span;
                    if covered.contains(&sp) {
                        continue;
                    }
                    covered.push(sp);
                    let subpatterns = subpatterns.iter()
                        .map(|field_pattern| field_pattern.pattern.clone())
                        .collect::<Vec<_>>();
                    covered.extend(
                        maybe_point_at_variant(cx, ty, subpatterns.as_slice()),
                    );
                }
            }
            if let PatternKind::Leaf { subpatterns } = pk {
                let subpatterns = subpatterns.iter()
                    .map(|field_pattern| field_pattern.pattern.clone())
                    .collect::<Vec<_>>();
                covered.extend(maybe_point_at_variant(cx, ty, subpatterns.as_slice()));
            }
        }
    }
    covered
}

// Legality of move bindings checking
fn check_legality_of_move_bindings(
    cx: &MatchVisitor<'_, '_>,
    has_guard: bool,
    pats: &[P<Pat>],
) {
    let mut by_ref_span = None;
    for pat in pats {
        pat.each_binding(|_, hir_id, span, _path| {
            if let Some(&bm) = cx.tables.pat_binding_modes().get(hir_id) {
                if let ty::BindByReference(..) = bm {
                    by_ref_span = Some(span);
                }
            } else {
                cx.tcx.sess.delay_span_bug(pat.span, "missing binding mode");
            }
        })
    }
    let span_vec = &mut Vec::new();
    let check_move = |p: &Pat, sub: Option<&Pat>, span_vec: &mut Vec<Span>| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| p.contains_bindings()) {
            struct_span_err!(cx.tcx.sess, p.span, E0007,
                             "cannot bind by-move with sub-bindings")
                .span_label(p.span, "binds an already bound by-move value by moving it")
                .emit();
        } else if has_guard && !cx.tcx.features().bind_by_move_pattern_guards {
            let mut err = struct_span_err!(cx.tcx.sess, p.span, E0008,
                                           "cannot bind by-move into a pattern guard");
            err.span_label(p.span, "moves value into pattern guard");
            if cx.tcx.sess.opts.unstable_features.is_nightly_build() {
                err.help("add #![feature(bind_by_move_pattern_guards)] to the \
                          crate attributes to enable");
            }
            err.emit();
        } else if let Some(_by_ref_span) = by_ref_span {
            span_vec.push(p.span);
        }
    };

    for pat in pats {
        pat.walk(|p| {
            if let PatKind::Binding(_, _, _, ref sub) = p.node {
                if let Some(&bm) = cx.tables.pat_binding_modes().get(p.hir_id) {
                    match bm {
                        ty::BindByValue(..) => {
                            let pat_ty = cx.tables.node_type(p.hir_id);
                            if !pat_ty.is_copy_modulo_regions(cx.tcx, cx.param_env, pat.span) {
                                check_move(p, sub.as_ref().map(|p| &**p), span_vec);
                            }
                        }
                        _ => {}
                    }
                } else {
                    cx.tcx.sess.delay_span_bug(pat.span, "missing binding mode");
                }
            }
            true
        });
    }
    if !span_vec.is_empty(){
        let span = MultiSpan::from_spans(span_vec.clone());
        let mut err = struct_span_err!(
            cx.tcx.sess,
            span,
            E0009,
            "cannot bind by-move and by-ref in the same pattern",
        );
        if let Some(by_ref_span) = by_ref_span {
            err.span_label(by_ref_span, "both by-ref and by-move used");
        }
        for span in span_vec.iter(){
            err.span_label(*span, "by-move pattern here");
        }
        err.emit();
    }
}

/// Ensures that a pattern guard doesn't borrow by mutable reference or assign.
//
// FIXME: this should be done by borrowck.
fn check_for_mutation_in_guard(cx: &MatchVisitor<'_, '_>, guard: &hir::Guard) {
    let mut checker = MutationChecker {
        cx,
    };
    match guard {
        hir::Guard::If(expr) =>
            ExprUseVisitor::new(&mut checker,
                                cx.tcx,
                                cx.body_owner,
                                cx.param_env,
                                cx.region_scope_tree,
                                cx.tables,
                                None).walk_expr(expr),
    };
}

struct MutationChecker<'a, 'tcx> {
    cx: &'a MatchVisitor<'a, 'tcx>,
}

impl<'a, 'tcx> Delegate<'tcx> for MutationChecker<'a, 'tcx> {
    fn matched_pat(&mut self, _: &Pat, _: &cmt_<'_>, _: euv::MatchMode) {}
    fn consume(&mut self, _: hir::HirId, _: Span, _: &cmt_<'_>, _: ConsumeMode) {}
    fn consume_pat(&mut self, _: &Pat, _: &cmt_<'_>, _: ConsumeMode) {}
    fn borrow(&mut self,
              _: hir::HirId,
              span: Span,
              _: &cmt_<'_>,
              _: ty::Region<'tcx>,
              kind:ty:: BorrowKind,
              _: LoanCause) {
        match kind {
            ty::MutBorrow => {
                let mut err = struct_span_err!(self.cx.tcx.sess, span, E0301,
                          "cannot mutably borrow in a pattern guard");
                err.span_label(span, "borrowed mutably in pattern guard");
                if self.cx.tcx.sess.opts.unstable_features.is_nightly_build() {
                    err.help("add #![feature(bind_by_move_pattern_guards)] to the \
                              crate attributes to enable");
                }
                err.emit();
            }
            ty::ImmBorrow | ty::UniqueImmBorrow => {}
        }
    }
    fn decl_without_init(&mut self, _: hir::HirId, _: Span) {}
    fn mutate(&mut self, _: hir::HirId, span: Span, _: &cmt_<'_>, mode: MutateMode) {
        match mode {
            MutateMode::JustWrite | MutateMode::WriteAndRead => {
                struct_span_err!(self.cx.tcx.sess, span, E0302, "cannot assign in a pattern guard")
                    .span_label(span, "assignment in pattern guard")
                    .emit();
            }
            MutateMode::Init => {}
        }
    }
}

/// Forbids bindings in `@` patterns. This is necessary for memory safety,
/// because of the way rvalues are handled in the borrow check. (See issue
/// #14587.)
fn check_legality_of_bindings_in_at_patterns(cx: &MatchVisitor<'_, '_>, pat: &Pat) {
    AtBindingPatternVisitor { cx: cx, bindings_allowed: true }.visit_pat(pat);
}

struct AtBindingPatternVisitor<'a, 'b, 'tcx> {
    cx: &'a MatchVisitor<'b, 'tcx>,
    bindings_allowed: bool
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for AtBindingPatternVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &Pat) {
        match pat.node {
            PatKind::Binding(.., ref subpat) => {
                if !self.bindings_allowed {
                    struct_span_err!(self.cx.tcx.sess, pat.span, E0303,
                                     "pattern bindings are not allowed after an `@`")
                        .span_label(pat.span,  "not allowed after `@`")
                        .emit();
                }

                if subpat.is_some() {
                    let bindings_were_allowed = self.bindings_allowed;
                    self.bindings_allowed = false;
                    intravisit::walk_pat(self, pat);
                    self.bindings_allowed = bindings_were_allowed;
                }
            }
            _ => intravisit::walk_pat(self, pat),
        }
    }
}
