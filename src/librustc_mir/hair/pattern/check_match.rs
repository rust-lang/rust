use super::_match::Usefulness::*;
use super::_match::WitnessPreference::*;
use super::_match::{expand_pattern, is_useful, MatchCheckCtxt, Matrix, PatStack};

use super::{PatCtxt, PatKind, PatternError};

use rustc::hir::def::*;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc::hir::HirId;
use rustc::hir::{self, Pat};
use rustc::lint;
use rustc::session::Session;
use rustc::ty::subst::{InternalSubsts, SubstsRef};
use rustc::ty::{self, Ty, TyCtxt};
use rustc_error_codes::*;
use rustc_errors::{Applicability, DiagnosticBuilder};
use syntax::ast::Mutability;
use syntax::feature_gate::feature_err;
use syntax_pos::symbol::sym;
use syntax_pos::{MultiSpan, Span};

use std::slice;

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

    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        intravisit::walk_expr(self, ex);

        if let hir::ExprKind::Match(ref scrut, ref arms, source) = ex.kind {
            self.check_match(scrut, arms, source);
        }
    }

    fn visit_local(&mut self, loc: &'tcx hir::Local<'tcx>) {
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

    fn visit_body(&mut self, body: &'tcx hir::Body<'tcx>) {
        intravisit::walk_body(self, body);

        for param in body.params {
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
                    )
                    .emit();
                }
                PatternError::NonConstPath(span) => {
                    ::rustc::mir::interpret::struct_error(
                        self.tcx.at(span),
                        "runtime values cannot be referenced in patterns",
                    )
                    .emit();
                }
            }
        }
    }

    fn span_e0158(&self, span: Span, text: &str) {
        span_err!(self.tcx.sess, span, E0158, "{}", text)
    }
}

impl<'tcx> MatchVisitor<'_, 'tcx> {
    fn check_patterns(&mut self, has_guard: bool, pat: &Pat<'_>) {
        check_legality_of_move_bindings(self, has_guard, pat);
        check_borrow_conflicts_in_at_patterns(self, pat);
        if !self.tcx.features().bindings_after_at {
            check_legality_of_bindings_in_at_patterns(self, pat);
        }
    }

    fn check_match(
        &mut self,
        scrut: &hir::Expr<'_>,
        arms: &'tcx [hir::Arm<'tcx>],
        source: hir::MatchSource,
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

            let inlined_arms: Vec<_> = arms
                .iter()
                .map(|arm| {
                    let mut patcx = PatCtxt::new(
                        self.tcx,
                        self.param_env.and(self.identity_substs),
                        self.tables,
                    );
                    patcx.include_lint_checks();
                    let pattern = patcx.lower_pattern(&arm.pat);
                    let pattern: &_ = cx.pattern_arena.alloc(expand_pattern(cx, pattern));
                    if !patcx.errors.is_empty() {
                        patcx.report_inlining_errors(arm.pat.span);
                        have_errors = true;
                    }
                    (pattern, &*arm.pat, arm.guard.is_some())
                })
                .collect();

            // Bail out early if inlining failed.
            if have_errors {
                return;
            }

            // Fourth, check for unreachable arms.
            let matrix = check_arms(cx, &inlined_arms, source);

            // Fifth, check if the match is exhaustive.
            let scrut_ty = self.tables.node_type(scrut.hir_id);
            // Note: An empty match isn't the same as an empty matrix for diagnostics purposes,
            // since an empty matrix can occur when there are arms, if those arms all have guards.
            let is_empty_match = inlined_arms.is_empty();
            check_exhaustive(cx, scrut_ty, scrut.span, &matrix, scrut.hir_id, is_empty_match);
        })
    }

    fn check_irrefutable(&self, pat: &'tcx Pat<'tcx>, origin: &str, sp: Option<Span>) {
        let module = self.tcx.hir().get_module_parent(pat.hir_id);
        MatchCheckCtxt::create_and_enter(self.tcx, self.param_env, module, |ref mut cx| {
            let mut patcx =
                PatCtxt::new(self.tcx, self.param_env.and(self.identity_substs), self.tables);
            patcx.include_lint_checks();
            let pattern = patcx.lower_pattern(pat);
            let pattern_ty = pattern.ty;
            let pattern = cx.pattern_arena.alloc(expand_pattern(cx, pattern));
            let pats: Matrix<'_, '_> = vec![PatStack::from_pattern(pattern)].into_iter().collect();

            let witnesses = match check_not_useful(cx, pattern_ty, &pats, pat.hir_id) {
                Ok(_) => return,
                Err(err) => err,
            };

            let joined_patterns = joined_uncovered_patterns(&witnesses);
            let mut err = struct_span_err!(
                self.tcx.sess,
                pat.span,
                E0005,
                "refutable pattern in {}: {} not covered",
                origin,
                joined_patterns
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
                err.note(
                    "`let` bindings require an \"irrefutable pattern\", like a `struct` or \
                     an `enum` with only one variant",
                );
                if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                    err.span_suggestion(
                        span,
                        "you might want to use `if let` to ignore the variant that isn't matched",
                        format!("if {} {{ /* */ }}", &snippet[..snippet.len() - 1]),
                        Applicability::HasPlaceholders,
                    );
                }
                err.note(
                    "for more information, visit \
                     https://doc.rust-lang.org/book/ch18-02-refutability.html",
                );
            }

            adt_defined_here(cx, &mut err, pattern_ty, &witnesses);
            err.emit();
        });
    }
}

/// A path pattern was interpreted as a constant, not a new variable.
/// This caused an irrefutable match failure in e.g. `let`.
fn const_not_var(
    err: &mut DiagnosticBuilder<'_>,
    tcx: TyCtxt<'_>,
    pat: &Pat<'_>,
    path: &hir::Path<'_>,
) {
    let descr = path.res.descr();
    err.span_label(
        pat.span,
        format!("interpreted as {} {} pattern, not a new variable", path.res.article(), descr,),
    );

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

fn check_for_bindings_named_same_as_variants(cx: &MatchVisitor<'_, '_>, pat: &Pat<'_>) {
    pat.walk_always(|p| {
        if let hir::PatKind::Binding(_, _, ident, None) = p.kind {
            if let Some(ty::BindByValue(hir::Mutability::Not)) =
                cx.tables.extract_binding_mode(cx.tcx.sess, p.hir_id, p.span)
            {
                let pat_ty = cx.tables.pat_ty(p);
                if let ty::Adt(edef, _) = pat_ty.kind {
                    if edef.is_enum()
                        && edef.variants.iter().any(|variant| {
                            variant.ident == ident && variant.ctor_kind == CtorKind::Const
                        })
                    {
                        // FIXME(Centril): Should be a lint?
                        let ty_path = cx.tcx.def_path_str(edef.did);
                        let mut err = struct_span_warn!(
                            cx.tcx.sess,
                            p.span,
                            E0170,
                            "pattern binding `{}` is named the same as one \
                             of the variants of the type `{}`",
                            ident,
                            ty_path
                        );
                        err.span_suggestion(
                            p.span,
                            "to match on the variant, qualify the path",
                            format!("{}::{}", ty_path, ident),
                            Applicability::MachineApplicable,
                        );
                        err.emit();
                    }
                }
            }
        }
    });
}

/// Checks for common cases of "catchall" patterns that may not be intended as such.
fn pat_is_catchall(pat: &Pat<'_>) -> bool {
    match pat.kind {
        hir::PatKind::Binding(.., None) => true,
        hir::PatKind::Binding(.., Some(ref s)) => pat_is_catchall(s),
        hir::PatKind::Ref(ref s, _) => pat_is_catchall(s),
        hir::PatKind::Tuple(ref v, _) => v.iter().all(|p| pat_is_catchall(&p)),
        _ => false,
    }
}

/// Check for unreachable patterns.
fn check_arms<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    arms: &[(&'p super::Pat<'tcx>, &hir::Pat<'_>, bool)],
    source: hir::MatchSource,
) -> Matrix<'p, 'tcx> {
    let mut seen = Matrix::empty();
    let mut catchall = None;
    for (arm_index, (pat, hir_pat, has_guard)) in arms.iter().enumerate() {
        let v = PatStack::from_pattern(pat);

        match is_useful(cx, &seen, &v, LeaveOutWitness, hir_pat.hir_id, true) {
            NotUseful => {
                match source {
                    hir::MatchSource::IfDesugar { .. } | hir::MatchSource::WhileDesugar => bug!(),

                    hir::MatchSource::IfLetDesugar { .. } | hir::MatchSource::WhileLetDesugar => {
                        // check which arm we're on.
                        match arm_index {
                            // The arm with the user-specified pattern.
                            0 => {
                                cx.tcx.lint_hir(
                                    lint::builtin::UNREACHABLE_PATTERNS,
                                    hir_pat.hir_id,
                                    pat.span,
                                    "unreachable pattern",
                                );
                            }
                            // The arm with the wildcard pattern.
                            1 => {
                                let msg = match source {
                                    hir::MatchSource::IfLetDesugar { .. } => {
                                        "irrefutable if-let pattern"
                                    }
                                    hir::MatchSource::WhileLetDesugar => {
                                        "irrefutable while-let pattern"
                                    }
                                    _ => bug!(),
                                };
                                cx.tcx.lint_hir(
                                    lint::builtin::IRREFUTABLE_LET_PATTERNS,
                                    hir_pat.hir_id,
                                    pat.span,
                                    msg,
                                );
                            }
                            _ => bug!(),
                        }
                    }

                    hir::MatchSource::ForLoopDesugar | hir::MatchSource::Normal => {
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
                    hir::MatchSource::AwaitDesugar | hir::MatchSource::TryDesugar => {}
                }
            }
            Useful(unreachable_subpatterns) => {
                for pat in unreachable_subpatterns {
                    cx.tcx.lint_hir(
                        lint::builtin::UNREACHABLE_PATTERNS,
                        hir_pat.hir_id,
                        pat.span,
                        "unreachable pattern",
                    );
                }
            }
            UsefulWithWitness(_) => bug!(),
        }
        if !has_guard {
            seen.push(v);
            if catchall.is_none() && pat_is_catchall(hir_pat) {
                catchall = Some(pat.span);
            }
        }
    }
    seen
}

fn check_not_useful<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    ty: Ty<'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    hir_id: HirId,
) -> Result<(), Vec<super::Pat<'tcx>>> {
    let wild_pattern = cx.pattern_arena.alloc(super::Pat::wildcard_from_ty(ty));
    let v = PatStack::from_pattern(wild_pattern);
    match is_useful(cx, matrix, &v, ConstructWitness, hir_id, true) {
        NotUseful => Ok(()), // This is good, wildcard pattern isn't reachable.
        UsefulWithWitness(pats) => Err(if pats.is_empty() {
            bug!("Exhaustiveness check returned no witnesses")
        } else {
            pats.into_iter().map(|w| w.single_pattern()).collect()
        }),
        Useful(_) => bug!(),
    }
}

fn check_exhaustive<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    scrut_ty: Ty<'tcx>,
    sp: Span,
    matrix: &Matrix<'p, 'tcx>,
    hir_id: HirId,
    is_empty_match: bool,
) {
    // In the absence of the `exhaustive_patterns` feature, empty matches are not detected by
    // `is_useful` to exhaustively match uninhabited types, so we manually check here.
    if is_empty_match && !cx.tcx.features().exhaustive_patterns {
        let scrutinee_is_visibly_uninhabited = match scrut_ty.kind {
            ty::Never => true,
            ty::Adt(def, _) => {
                def.is_enum()
                    && def.variants.is_empty()
                    && !cx.is_foreign_non_exhaustive_enum(scrut_ty)
            }
            _ => false,
        };
        if scrutinee_is_visibly_uninhabited {
            // If the type *is* uninhabited, an empty match is vacuously exhaustive.
            return;
        }
    }

    let witnesses = match check_not_useful(cx, scrut_ty, matrix, hir_id) {
        Ok(_) => return,
        Err(err) => err,
    };

    let non_empty_enum = match scrut_ty.kind {
        ty::Adt(def, _) => def.is_enum() && !def.variants.is_empty(),
        _ => false,
    };
    // In the case of an empty match, replace the '`_` not covered' diagnostic with something more
    // informative.
    let mut err;
    if is_empty_match && !non_empty_enum {
        err = create_e0004(
            cx.tcx.sess,
            sp,
            format!("non-exhaustive patterns: type `{}` is non-empty", scrut_ty),
        );
    } else {
        let joined_patterns = joined_uncovered_patterns(&witnesses);
        err = create_e0004(
            cx.tcx.sess,
            sp,
            format!("non-exhaustive patterns: {} not covered", joined_patterns),
        );
        err.span_label(sp, pattern_not_covered_label(&witnesses, &joined_patterns));
    };

    adt_defined_here(cx, &mut err, scrut_ty, &witnesses);
    err.help(
        "ensure that all possible cases are being handled, \
         possibly by adding wildcards or more match arms",
    );
    err.emit();
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
    format!("pattern{} {} not covered", rustc_errors::pluralize!(witnesses.len()), joined_patterns)
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
            use PatKind::{AscribeUserType, Deref, Leaf, Or, Variant};
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

                    let pats = subpatterns
                        .iter()
                        .map(|field_pattern| field_pattern.pattern.clone())
                        .collect::<Box<[_]>>();
                    covered.extend(maybe_point_at_variant(ty, &pats));
                }
                Leaf { subpatterns } => {
                    let pats = subpatterns
                        .iter()
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

/// Check the legality of legality of by-move bindings.
fn check_legality_of_move_bindings(cx: &mut MatchVisitor<'_, '_>, has_guard: bool, pat: &Pat<'_>) {
    let sess = cx.tcx.sess;
    let tables = cx.tables;

    // Find all by-ref spans.
    let mut by_ref_spans = Vec::new();
    pat.each_binding(|_, hir_id, span, _| {
        if let Some(ty::BindByReference(_)) = tables.extract_binding_mode(sess, hir_id, span) {
            by_ref_spans.push(span);
        }
    });

    // Find bad by-move spans:
    let by_move_spans = &mut Vec::new();
    let mut check_move = |p: &Pat<'_>, sub: Option<&Pat<'_>>| {
        // Check legality of moving out of the enum.
        //
        // `x @ Foo(..)` is legal, but `x @ Foo(y)` isn't.
        if sub.map_or(false, |p| p.contains_bindings()) {
            struct_span_err!(sess, p.span, E0007, "cannot bind by-move with sub-bindings")
                .span_label(p.span, "binds an already bound by-move value by moving it")
                .emit();
        } else if !has_guard && !by_ref_spans.is_empty() {
            by_move_spans.push(p.span);
        }
    };
    pat.walk_always(|p| {
        if let hir::PatKind::Binding(.., sub) = &p.kind {
            if let Some(ty::BindByValue(_)) = tables.extract_binding_mode(sess, p.hir_id, p.span) {
                let pat_ty = tables.node_type(p.hir_id);
                if !pat_ty.is_copy_modulo_regions(cx.tcx, cx.param_env, pat.span) {
                    check_move(p, sub.as_deref());
                }
            }
        }
    });

    // Found some bad by-move spans, error!
    if !by_move_spans.is_empty() {
        let mut err = struct_span_err!(
            sess,
            MultiSpan::from_spans(by_move_spans.clone()),
            E0009,
            "cannot bind by-move and by-ref in the same pattern",
        );
        for span in by_ref_spans.iter() {
            err.span_label(*span, "by-ref pattern here");
        }
        for span in by_move_spans.iter() {
            err.span_label(*span, "by-move pattern here");
        }
        err.emit();
    }
}

/// Check that there are no borrow conflicts in `binding @ subpat` patterns.
///
/// For example, this would reject:
/// - `ref x @ Some(ref mut y)`,
/// - `ref mut x @ Some(ref y)`
/// - `ref mut x @ Some(ref mut y)`.
///
/// This analysis is *not* subsumed by NLL.
fn check_borrow_conflicts_in_at_patterns(cx: &MatchVisitor<'_, '_>, pat: &Pat<'_>) {
    let tab = cx.tables;
    let sess = cx.tcx.sess;
    // Get the mutability of `p` if it's by-ref.
    let extract_binding_mut = |hir_id, span| match tab.extract_binding_mode(sess, hir_id, span)? {
        ty::BindByValue(_) => None,
        ty::BindByReference(m) => Some(m),
    };
    pat.walk_always(|pat| {
        // Extract `sub` in `binding @ sub`.
        let (name, sub) = match &pat.kind {
            hir::PatKind::Binding(.., name, Some(sub)) => (*name, sub),
            _ => return,
        };

        // Extract the mutability.
        let mut_outer = match extract_binding_mut(pat.hir_id, pat.span) {
            None => return,
            Some(m) => m,
        };

        // We now have `ref $mut_outer binding @ sub` (semantically).
        // Recurse into each binding in `sub` and find mutability conflicts.
        let mut conflicts_mut_mut = Vec::new();
        let mut conflicts_mut_ref = Vec::new();
        sub.each_binding(|_, hir_id, span, _| {
            if let Some(mut_inner) = extract_binding_mut(hir_id, span) {
                match (mut_outer, mut_inner) {
                    (Mutability::Not, Mutability::Not) => {}
                    (Mutability::Mut, Mutability::Mut) => conflicts_mut_mut.push(span),
                    _ => conflicts_mut_ref.push(span),
                }
            }
        });

        // Report errors if any.
        let binding_span = pat.span.with_hi(name.span.hi());
        if !conflicts_mut_mut.is_empty() {
            // Report mutability conflicts for e.g. `ref mut x @ Some(ref mut y)`.
            let msg = &format!("cannot borrow `{}` as mutable more than once at a time", name);
            let mut err = sess.struct_span_err(pat.span, msg);
            err.span_label(binding_span, "first mutable borrow occurs here");
            for sp in conflicts_mut_mut {
                err.span_label(sp, "another mutable borrow occurs here");
            }
            for sp in conflicts_mut_ref {
                err.span_label(sp, "also borrowed as immutable here");
            }
            err.emit();
        } else if !conflicts_mut_ref.is_empty() {
            // Report mutability conflicts for e.g. `ref x @ Some(ref mut y)` or the converse.
            let (primary, also) = match mut_outer {
                Mutability::Mut => ("mutable", "immutable"),
                Mutability::Not => ("immutable", "mutable"),
            };
            let msg = &format!(
                "cannot borrow `{}` as {} because it is also borrowed as {}",
                name, also, primary,
            );
            let mut err = sess.struct_span_err(pat.span, msg);
            err.span_label(binding_span, &format!("{} borrow occurs here", primary));
            for sp in conflicts_mut_ref {
                err.span_label(sp, &format!("{} borrow occurs here", also));
            }
            err.emit();
        }
    });
}

/// Forbids bindings in `@` patterns. This used to be is necessary for memory safety,
/// because of the way rvalues were handled in the borrow check. (See issue #14587.)
fn check_legality_of_bindings_in_at_patterns(cx: &MatchVisitor<'_, '_>, pat: &Pat<'_>) {
    AtBindingPatternVisitor { cx, bindings_allowed: true }.visit_pat(pat);

    struct AtBindingPatternVisitor<'a, 'b, 'tcx> {
        cx: &'a MatchVisitor<'b, 'tcx>,
        bindings_allowed: bool,
    }

    impl<'v> Visitor<'v> for AtBindingPatternVisitor<'_, '_, '_> {
        fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
            NestedVisitorMap::None
        }

        fn visit_pat(&mut self, pat: &Pat<'_>) {
            match pat.kind {
                hir::PatKind::Binding(.., ref subpat) => {
                    if !self.bindings_allowed {
                        feature_err(
                            &self.cx.tcx.sess.parse_sess,
                            sym::bindings_after_at,
                            pat.span,
                            "pattern bindings after an `@` are unstable",
                        )
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
}
