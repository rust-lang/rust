use super::_match::{MatchCheckCtxt, Matrix, expand_pattern, is_useful};
use super::_match::Usefulness::*;
use super::_match::WitnessPreference::*;

use super::{PatCtxt, PatternError, PatKind};

use rustc::session::Session;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{InternalSubsts, SubstsRef};
use rustc::lint;
use rustc_errors::{Applicability, DiagnosticBuilder};

use rustc::hir::HirId;
use rustc::hir::def::*;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::hir::{self, Pat};

use smallvec::smallvec;
use std::slice;

use syntax_pos::{Span, DUMMY_SP, MultiSpan};

crate fn check_match(tcx: TyCtxt<'_>, def_id: DefId) {
    let body_id = match tcx.hir().as_local_hir_id(def_id) {
        None => return,
        Some(id) => tcx.hir().body_owned_by(id),
    };

    let mut visitor = MatchVisitor {
        tcx,
        tables: tcx.body_tables(body_id),
        param_env: tcx.param_env(def_id),
        identity_substs: InternalSubsts::identity_for_item(tcx, def_id),
    };
    visitor.visit_body(tcx.hir().body(body_id));
}

fn create_e0004(sess: &Session, sp: Span, error_message: String) -> DiagnosticBuilder<'_> {
    struct_span_err!(sess, sp, E0004, "{}", &error_message)
}

struct MatchVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    tables: &'a ty::TypeckTables<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    identity_substs: SubstsRef<'tcx>,
}

impl<'tcx> Visitor<'tcx> for MatchVisitor<'_, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        intravisit::walk_expr(self, ex);

        if let hir::ExprKind::Match(ref scrut, ref arms, source) = ex.kind {
            self.check_match(scrut, arms, source);
        }
    }

    fn visit_local(&mut self, loc: &'tcx hir::Local) {
        intravisit::walk_local(self, loc);

        let (msg, sp) = match loc.source {
            hir::LocalSource::Normal => ("local binding", Some(loc.span)),
            hir::LocalSource::ForLoopDesugar => ("`for` loop binding", None),
            hir::LocalSource::AsyncFn => ("async fn binding", None),
            hir::LocalSource::AwaitDesugar => ("`await` future binding", None),
        };
        self.check_irrefutable(&loc.pat, msg, sp);

        // Check legality of move bindings and `@` patterns.
        self.check_patterns(false, &loc.pat);
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        intravisit::walk_body(self, body);

        for param in &body.params {
            self.check_irrefutable(&param.pat, "function argument", None);
            self.check_patterns(false, &param.pat);
        }
    }
}

impl PatCtxt<'_, '_> {
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

impl<'tcx> MatchVisitor<'_, 'tcx> {
    fn check_patterns(&mut self, has_guard: bool, pat: &Pat) {
        check_legality_of_move_bindings(self, has_guard, pat);
        check_legality_of_bindings_in_at_patterns(self, pat);
    }

    fn check_match(
        &mut self,
        scrut: &hir::Expr,
        arms: &'tcx [hir::Arm],
        source: hir::MatchSource
    ) {
        for arm in arms {
            // First, check legality of move bindings.
            self.check_patterns(arm.guard.is_some(), &arm.pat);

            // Second, perform some lints.
            check_for_bindings_named_same_as_variants(self, &arm.pat);
        }

        let module = self.tcx.hir().get_module_parent(scrut.hir_id);
        MatchCheckCtxt::create_and_enter(self.tcx, self.param_env, module, |ref mut cx| {
            let mut have_errors = false;

            let inlined_arms : Vec<(Vec<_>, _)> = arms.iter().map(|arm| (
                // HACK(or_patterns; Centril | dlrobertson): Remove this and
                // correctly handle exhaustiveness checking for nested or-patterns.
                match &arm.pat.kind {
                    hir::PatKind::Or(pats) => pats,
                    _ => std::slice::from_ref(&arm.pat),
                }.iter().map(|pat| {
                    let mut patcx = PatCtxt::new(
                        self.tcx,
                        self.param_env.and(self.identity_substs),
                        self.tables
                    );
                    patcx.include_lint_checks();
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
                    match pat_ty.kind {
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
            check_exhaustive(cx, scrut_ty, scrut.span, &matrix, scrut.hir_id);
        })
    }

    fn check_irrefutable(&self, pat: &'tcx Pat, origin: &str, sp: Option<Span>) {
        let module = self.tcx.hir().get_module_parent(pat.hir_id);
        MatchCheckCtxt::create_and_enter(self.tcx, self.param_env, module, |ref mut cx| {
            let mut patcx = PatCtxt::new(self.tcx,
                                                self.param_env.and(self.identity_substs),
                                                self.tables);
            patcx.include_lint_checks();
            let pattern = patcx.lower_pattern(pat);
            let pattern_ty = pattern.ty;
            let pats: Matrix<'_, '_> = vec![smallvec![
                expand_pattern(cx, pattern)
            ]].into_iter().collect();

            let witnesses = match check_not_useful(cx, pattern_ty, &pats, pat.hir_id) {
                Ok(_) => return,
                Err(err) => err,
            };

            let joined_patterns = joined_uncovered_patterns(&witnesses);
            let mut err = struct_span_err!(
                self.tcx.sess, pat.span, E0005,
                "refutable pattern in {}: {} not covered",
                origin, joined_patterns
            );
            let suggest_if_let = match &pat.kind {
                hir::PatKind::Path(hir::QPath::Resolved(None, path))
                    if path.segments.len() == 1 && path.segments[0].args.is_none() =>
                {
                    const_not_var(&mut err, cx.tcx, pat, path);
                    false
                }
                _ => {
                    err.span_label(
                        pat.span,
                        pattern_not_covered_label(&witnesses, &joined_patterns),
                    );
                    true
                }
            };

            if let (Some(span), true) = (sp, suggest_if_let) {
                err.note("`let` bindings require an \"irrefutable pattern\", like a `struct` or \
                          an `enum` with only one variant");
                if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                    err.span_suggestion(
                        span,
                        "you might want to use `if let` to ignore the variant that isn't matched",
                        format!("if {} {{ /* */ }}", &snippet[..snippet.len() - 1]),
                        Applicability::HasPlaceholders,
                    );
                }
                err.note("for more information, visit \
                          https://doc.rust-lang.org/book/ch18-02-refutability.html");
            }

            adt_defined_here(cx, &mut err, pattern_ty, &witnesses);
            err.emit();
        });
    }
}

/// A path pattern was interpreted as a constant, not a new variable.
/// This caused an irrefutable match failure in e.g. `let`.
fn const_not_var(err: &mut DiagnosticBuilder<'_>, tcx: TyCtxt<'_>, pat: &Pat, path: &hir::Path) {
    let descr = path.res.descr();
    err.span_label(pat.span, format!(
        "interpreted as {} {} pattern, not a new variable",
        path.res.article(),
        descr,
    ));

    err.span_suggestion(
        pat.span,
        "introduce a variable instead",
        format!("{}_var", path.segments[0].ident).to_lowercase(),
        // Cannot use `MachineApplicable` as it's not really *always* correct
        // because there may be such an identifier in scope or the user maybe
        // really wanted to match against the constant. This is quite unlikely however.
        Applicability::MaybeIncorrect,
    );

    if let Some(span) = tcx.hir().res_span(path.res) {
        err.span_label(span, format!("{} defined here", descr));
    }
}

fn check_for_bindings_named_same_as_variants(cx: &MatchVisitor<'_, '_>, pat: &Pat) {
    pat.walk(|p| {
        if let hir::PatKind::Binding(_, _, ident, None) = p.kind {
            if let Some(&bm) = cx.tables.pat_binding_modes().get(p.hir_id) {
                if bm != ty::BindByValue(hir::MutImmutable) {
                    // Nothing to check.
                    return true;
                }
                let pat_ty = cx.tables.pat_ty(p);
                if let ty::Adt(edef, _) = pat_ty.kind {
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
    match pat.kind {
        hir::PatKind::Binding(.., None) => true,
        hir::PatKind::Binding(.., Some(ref s)) => pat_is_catchall(s),
        hir::PatKind::Ref(ref s, _) => pat_is_catchall(s),
        hir::PatKind::Tuple(ref v, _) => v.iter().all(|p| {
            pat_is_catchall(&p)
        }),
        _ => false
    }
}

// Check for unreachable patterns
fn check_arms<'tcx>(
    cx: &mut MatchCheckCtxt<'_, 'tcx>,
    arms: &[(Vec<(&super::Pat<'tcx>, &hir::Pat)>, Option<&hir::Expr>)],
    source: hir::MatchSource,
) {
    let mut seen = Matrix::empty();
    let mut catchall = None;
    for (arm_index, &(ref pats, guard)) in arms.iter().enumerate() {
        for &(pat, hir_pat) in pats {
            let v = smallvec![pat];

            match is_useful(cx, &seen, &v, LeaveOutWitness, hir_pat.hir_id) {
                NotUseful => {
                    match source {
                        hir::MatchSource::IfDesugar { .. } |
                        hir::MatchSource::WhileDesugar => bug!(),
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

fn check_not_useful(
    cx: &mut MatchCheckCtxt<'_, 'tcx>,
    ty: Ty<'tcx>,
    matrix: &Matrix<'_, 'tcx>,
    hir_id: HirId,
) -> Result<(), Vec<super::Pat<'tcx>>> {
    let wild_pattern = super::Pat { ty, span: DUMMY_SP, kind: box PatKind::Wild };
    match is_useful(cx, matrix, &[&wild_pattern], ConstructWitness, hir_id) {
        NotUseful => Ok(()), // This is good, wildcard pattern isn't reachable.
        UsefulWithWitness(pats) => Err(if pats.is_empty() {
            vec![wild_pattern]
        } else {
            pats.into_iter().map(|w| w.single_pattern()).collect()
        }),
        Useful => bug!(),
    }
}

fn check_exhaustive<'tcx>(
    cx: &mut MatchCheckCtxt<'_, 'tcx>,
    scrut_ty: Ty<'tcx>,
    sp: Span,
    matrix: &Matrix<'_, 'tcx>,
    hir_id: HirId,
) {
    let witnesses = match check_not_useful(cx, scrut_ty, matrix, hir_id) {
        Ok(_) => return,
        Err(err) => err,
    };

    let joined_patterns = joined_uncovered_patterns(&witnesses);
    let mut err = create_e0004(
        cx.tcx.sess, sp,
        format!("non-exhaustive patterns: {} not covered", joined_patterns),
    );
    err.span_label(sp, pattern_not_covered_label(&witnesses, &joined_patterns));
    adt_defined_here(cx, &mut err, scrut_ty, &witnesses);
    err.help(
        "ensure that all possible cases are being handled, \
        possibly by adding wildcards or more match arms"
    )
    .emit();
}

fn joined_uncovered_patterns(witnesses: &[super::Pat<'_>]) -> String {
    const LIMIT: usize = 3;
    match witnesses {
        [] => bug!(),
        [witness] => format!("`{}`", witness),
        [head @ .., tail] if head.len() < LIMIT => {
            let head: Vec<_> = head.iter().map(<_>::to_string).collect();
            format!("`{}` and `{}`", head.join("`, `"), tail)
        }
        _ => {
            let (head, tail) = witnesses.split_at(LIMIT);
            let head: Vec<_> = head.iter().map(<_>::to_string).collect();
            format!("`{}` and {} more", head.join("`, `"), tail.len())
        }
    }
}

fn pattern_not_covered_label(witnesses: &[super::Pat<'_>], joined_patterns: &str) -> String {
    format!("pattern{} {} not covered", rustc_errors::pluralise!(witnesses.len()), joined_patterns)
}

/// Point at the definition of non-covered `enum` variants.
fn adt_defined_here(
    cx: &MatchCheckCtxt<'_, '_>,
    err: &mut DiagnosticBuilder<'_>,
    ty: Ty<'_>,
    witnesses: &[super::Pat<'_>],
) {
    let ty = ty.peel_refs();
    if let ty::Adt(def, _) = ty.kind {
        if let Some(sp) = cx.tcx.hir().span_if_local(def.did) {
            err.span_label(sp, format!("`{}` defined here", ty));
        }

        if witnesses.len() < 4 {
            for sp in maybe_point_at_variant(ty, &witnesses) {
                err.span_label(sp, "not covered");
            }
        }
    }
}

fn maybe_point_at_variant(ty: Ty<'_>, patterns: &[super::Pat<'_>]) -> Vec<Span> {
    let mut covered = vec![];
    if let ty::Adt(def, _) = ty.kind {
        // Don't point at variants that have already been covered due to other patterns to avoid
        // visual clutter.
        for pattern in patterns {
            use PatKind::{AscribeUserType, Deref, Variant, Or, Leaf};
            match &*pattern.kind {
                AscribeUserType { subpattern, .. } | Deref { subpattern } => {
                    covered.extend(maybe_point_at_variant(ty, slice::from_ref(&subpattern)));
                }
                Variant { adt_def, variant_index, subpatterns, .. } if adt_def.did == def.did => {
                    let sp = def.variants[*variant_index].ident.span;
                    if covered.contains(&sp) {
                        continue;
                    }
                    covered.push(sp);

                    let pats = subpatterns.iter()
                        .map(|field_pattern| field_pattern.pattern.clone())
                        .collect::<Box<[_]>>();
                    covered.extend(maybe_point_at_variant(ty, &pats));
                }
                Leaf { subpatterns } => {
                    let pats = subpatterns.iter()
                        .map(|field_pattern| field_pattern.pattern.clone())
                        .collect::<Box<[_]>>();
                    covered.extend(maybe_point_at_variant(ty, &pats));
                }
                Or { pats } => {
                    let pats = pats.iter().cloned().collect::<Box<[_]>>();
                    covered.extend(maybe_point_at_variant(ty, &pats));
                }
                _ => {}
            }
        }
    }
    covered
}

// Check the legality of legality of by-move bindings.
fn check_legality_of_move_bindings(cx: &mut MatchVisitor<'_, '_>, has_guard: bool, pat: &Pat) {
    let mut by_ref_span = None;
    pat.each_binding(|_, hir_id, span, _| {
        if let Some(&bm) = cx.tables.pat_binding_modes().get(hir_id) {
            if let ty::BindByReference(..) = bm {
                by_ref_span = Some(span);
            }
        } else {
            cx.tcx.sess.delay_span_bug(pat.span, "missing binding mode");
        }
    });

    let span_vec = &mut Vec::new();
    let mut check_move = |p: &Pat, sub: Option<&Pat>| {
        // Check legality of moving out of the enum.
        //
        // `x @ Foo(..)` is legal, but `x @ Foo(y)` isn't.
        if sub.map_or(false, |p| p.contains_bindings()) {
            struct_span_err!(cx.tcx.sess, p.span, E0007, "cannot bind by-move with sub-bindings")
                .span_label(p.span, "binds an already bound by-move value by moving it")
                .emit();
        } else if !has_guard && by_ref_span.is_some() {
            span_vec.push(p.span);
        }
    };

    pat.walk(|p| {
        if let hir::PatKind::Binding(.., sub) = &p.kind {
            if let Some(&bm) = cx.tables.pat_binding_modes().get(p.hir_id) {
                if let ty::BindByValue(..) = bm {
                    let pat_ty = cx.tables.node_type(p.hir_id);
                    if !pat_ty.is_copy_modulo_regions(cx.tcx, cx.param_env, pat.span) {
                        check_move(p, sub.as_deref());
                    }
                }
            } else {
                cx.tcx.sess.delay_span_bug(pat.span, "missing binding mode");
            }
        }
        true
    });

    if !span_vec.is_empty() {
        let mut err = struct_span_err!(
            cx.tcx.sess,
            MultiSpan::from_spans(span_vec.clone()),
            E0009,
            "cannot bind by-move and by-ref in the same pattern",
        );
        if let Some(by_ref_span) = by_ref_span {
            err.span_label(by_ref_span, "both by-ref and by-move used");
        }
        for span in span_vec.iter() {
            err.span_label(*span, "by-move pattern here");
        }
        err.emit();
    }
}

/// Forbids bindings in `@` patterns. This is necessary for memory safety,
/// because of the way rvalues are handled in the borrow check. (See issue
/// #14587.)
fn check_legality_of_bindings_in_at_patterns(cx: &MatchVisitor<'_, '_>, pat: &Pat) {
    AtBindingPatternVisitor { cx, bindings_allowed: true }.visit_pat(pat);
}

struct AtBindingPatternVisitor<'a, 'b, 'tcx> {
    cx: &'a MatchVisitor<'b, 'tcx>,
    bindings_allowed: bool
}

impl<'v> Visitor<'v> for AtBindingPatternVisitor<'_, '_, '_> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

    fn visit_pat(&mut self, pat: &Pat) {
        match pat.kind {
            hir::PatKind::Binding(.., ref subpat) => {
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
