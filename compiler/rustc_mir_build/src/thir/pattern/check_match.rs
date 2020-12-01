use super::usefulness::Usefulness::*;
use super::usefulness::{
    compute_match_usefulness, expand_pattern, MatchArm, MatchCheckCtxt, UsefulnessReport,
};
use super::{PatCtxt, PatKind, PatternError};

use rustc_arena::TypedArena;
use rustc_ast::Mutability;
use rustc_errors::{error_code, struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::*;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{HirId, Pat};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint::builtin::BINDINGS_WITH_VARIANT_NAME;
use rustc_session::lint::builtin::{IRREFUTABLE_LET_PATTERNS, UNREACHABLE_PATTERNS};
use rustc_session::parse::feature_err;
use rustc_session::Session;
use rustc_span::{sym, Span};
use std::slice;

crate fn check_match(tcx: TyCtxt<'_>, def_id: DefId) {
    let body_id = match def_id.as_local() {
        None => return,
        Some(id) => tcx.hir().body_owned_by(tcx.hir().local_def_id_to_hir_id(id)),
    };

    let mut visitor = MatchVisitor {
        tcx,
        typeck_results: tcx.typeck_body(body_id),
        param_env: tcx.param_env(def_id),
        pattern_arena: TypedArena::default(),
    };
    visitor.visit_body(tcx.hir().body(body_id));
}

fn create_e0004(sess: &Session, sp: Span, error_message: String) -> DiagnosticBuilder<'_> {
    struct_span_err!(sess, sp, E0004, "{}", &error_message)
}

struct MatchVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typeck_results: &'a ty::TypeckResults<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pattern_arena: TypedArena<super::Pat<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for MatchVisitor<'_, 'tcx> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
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
            hir::LocalSource::AssignDesugar(_) => ("destructuring assignment binding", None),
        };
        self.check_irrefutable(&loc.pat, msg, sp);
        self.check_patterns(&loc.pat);
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        intravisit::walk_param(self, param);
        self.check_irrefutable(&param.pat, "function argument", None);
        self.check_patterns(&param.pat);
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
                PatternError::ConstParamInPattern(span) => {
                    self.span_e0158(span, "const parameters cannot be referenced in patterns")
                }
                PatternError::FloatBug => {
                    // FIXME(#31407) this is only necessary because float parsing is buggy
                    rustc_middle::mir::interpret::struct_error(
                        self.tcx.at(pat_span),
                        "could not evaluate float literal (see issue #31407)",
                    )
                    .emit();
                }
                PatternError::NonConstPath(span) => {
                    rustc_middle::mir::interpret::struct_error(
                        self.tcx.at(span),
                        "runtime values cannot be referenced in patterns",
                    )
                    .emit();
                }
            }
        }
    }

    fn span_e0158(&self, span: Span, text: &str) {
        struct_span_err!(self.tcx.sess, span, E0158, "{}", text).emit();
    }
}

impl<'tcx> MatchVisitor<'_, 'tcx> {
    fn check_patterns(&mut self, pat: &Pat<'_>) {
        pat.walk_always(|pat| check_borrow_conflicts_in_at_patterns(self, pat));
        if !self.tcx.features().bindings_after_at {
            check_legality_of_bindings_in_at_patterns(self, pat);
        }
        check_for_bindings_named_same_as_variants(self, pat);
    }

    fn lower_pattern<'p>(
        &self,
        cx: &mut MatchCheckCtxt<'p, 'tcx>,
        pat: &'tcx hir::Pat<'tcx>,
        have_errors: &mut bool,
    ) -> (&'p super::Pat<'tcx>, Ty<'tcx>) {
        let mut patcx = PatCtxt::new(self.tcx, self.param_env, self.typeck_results);
        patcx.include_lint_checks();
        let pattern = patcx.lower_pattern(pat);
        let pattern_ty = pattern.ty;
        let pattern: &_ = cx.pattern_arena.alloc(expand_pattern(pattern));
        if !patcx.errors.is_empty() {
            *have_errors = true;
            patcx.report_inlining_errors(pat.span);
        }
        (pattern, pattern_ty)
    }

    fn new_cx(&self, hir_id: HirId) -> MatchCheckCtxt<'_, 'tcx> {
        MatchCheckCtxt {
            tcx: self.tcx,
            param_env: self.param_env,
            module: self.tcx.parent_module(hir_id).to_def_id(),
            pattern_arena: &self.pattern_arena,
        }
    }

    fn check_match(
        &mut self,
        scrut: &hir::Expr<'_>,
        arms: &'tcx [hir::Arm<'tcx>],
        source: hir::MatchSource,
    ) {
        for arm in arms {
            // Check the arm for some things unrelated to exhaustiveness.
            self.check_patterns(&arm.pat);
        }

        let mut cx = self.new_cx(scrut.hir_id);

        let mut have_errors = false;

        let arms: Vec<_> = arms
            .iter()
            .map(|hir::Arm { pat, guard, .. }| MatchArm {
                pat: self.lower_pattern(&mut cx, pat, &mut have_errors).0,
                hir_id: pat.hir_id,
                has_guard: guard.is_some(),
            })
            .collect();

        // Bail out early if lowering failed.
        if have_errors {
            return;
        }

        let scrut_ty = self.typeck_results.expr_ty_adjusted(scrut);
        let report = compute_match_usefulness(&cx, &arms, scrut.hir_id, scrut_ty);

        // Report unreachable arms.
        report_arm_reachability(&cx, &report, source);

        // Check if the match is exhaustive.
        // Note: An empty match isn't the same as an empty matrix for diagnostics purposes,
        // since an empty matrix can occur when there are arms, if those arms all have guards.
        let is_empty_match = arms.is_empty();
        let witnesses = report.non_exhaustiveness_witnesses;
        if !witnesses.is_empty() {
            non_exhaustive_match(&cx, scrut_ty, scrut.span, witnesses, is_empty_match);
        }
    }

    fn check_irrefutable(&self, pat: &'tcx Pat<'tcx>, origin: &str, sp: Option<Span>) {
        let mut cx = self.new_cx(pat.hir_id);

        let (pattern, pattern_ty) = self.lower_pattern(&mut cx, pat, &mut false);
        let arms = vec![MatchArm { pat: pattern, hir_id: pat.hir_id, has_guard: false }];
        let report = compute_match_usefulness(&cx, &arms, pat.hir_id, pattern_ty);

        // Note: we ignore whether the pattern is unreachable (i.e. whether the type is empty). We
        // only care about exhaustiveness here.
        let witnesses = report.non_exhaustiveness_witnesses;
        if witnesses.is_empty() {
            // The pattern is irrefutable.
            return;
        }

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
                err.span_label(pat.span, pattern_not_covered_label(&witnesses, &joined_patterns));
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

        adt_defined_here(&cx, &mut err, pattern_ty, &witnesses);
        err.note(&format!("the matched value is of type `{}`", pattern_ty));
        err.emit();
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
                cx.typeck_results.extract_binding_mode(cx.tcx.sess, p.hir_id, p.span)
            {
                let pat_ty = cx.typeck_results.pat_ty(p).peel_refs();
                if let ty::Adt(edef, _) = pat_ty.kind() {
                    if edef.is_enum()
                        && edef.variants.iter().any(|variant| {
                            variant.ident == ident && variant.ctor_kind == CtorKind::Const
                        })
                    {
                        cx.tcx.struct_span_lint_hir(
                            BINDINGS_WITH_VARIANT_NAME,
                            p.hir_id,
                            p.span,
                            |lint| {
                                let ty_path = cx.tcx.def_path_str(edef.did);
                                lint.build(&format!(
                                    "pattern binding `{}` is named the same as one \
                                                of the variants of the type `{}`",
                                    ident, ty_path
                                ))
                                .code(error_code!(E0170))
                                .span_suggestion(
                                    p.span,
                                    "to match on the variant, qualify the path",
                                    format!("{}::{}", ty_path, ident),
                                    Applicability::MachineApplicable,
                                )
                                .emit();
                            },
                        )
                    }
                }
            }
        }
    });
}

/// Checks for common cases of "catchall" patterns that may not be intended as such.
fn pat_is_catchall(pat: &super::Pat<'_>) -> bool {
    use super::PatKind::*;
    match &*pat.kind {
        Binding { subpattern: None, .. } => true,
        Binding { subpattern: Some(s), .. } | Deref { subpattern: s } => pat_is_catchall(s),
        Leaf { subpatterns: s } => s.iter().all(|p| pat_is_catchall(&p.pattern)),
        _ => false,
    }
}

fn unreachable_pattern(tcx: TyCtxt<'_>, span: Span, id: HirId, catchall: Option<Span>) {
    tcx.struct_span_lint_hir(UNREACHABLE_PATTERNS, id, span, |lint| {
        let mut err = lint.build("unreachable pattern");
        if let Some(catchall) = catchall {
            // We had a catchall pattern, hint at that.
            err.span_label(span, "unreachable pattern");
            err.span_label(catchall, "matches any value");
        }
        err.emit();
    });
}

fn irrefutable_let_pattern(tcx: TyCtxt<'_>, span: Span, id: HirId, source: hir::MatchSource) {
    tcx.struct_span_lint_hir(IRREFUTABLE_LET_PATTERNS, id, span, |lint| {
        let msg = match source {
            hir::MatchSource::IfLetDesugar { .. } => "irrefutable if-let pattern",
            hir::MatchSource::WhileLetDesugar => "irrefutable while-let pattern",
            _ => bug!(),
        };
        lint.build(msg).emit()
    });
}

/// Report unreachable arms, if any.
fn report_arm_reachability<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    report: &UsefulnessReport<'p, 'tcx>,
    source: hir::MatchSource,
) {
    let mut catchall = None;
    for (arm_index, (arm, is_useful)) in report.arm_usefulness.iter().enumerate() {
        match is_useful {
            NotUseful => {
                match source {
                    hir::MatchSource::IfDesugar { .. } | hir::MatchSource::WhileDesugar => bug!(),

                    hir::MatchSource::IfLetDesugar { .. } | hir::MatchSource::WhileLetDesugar => {
                        // Check which arm we're on.
                        match arm_index {
                            // The arm with the user-specified pattern.
                            0 => unreachable_pattern(cx.tcx, arm.pat.span, arm.hir_id, None),
                            // The arm with the wildcard pattern.
                            1 => irrefutable_let_pattern(cx.tcx, arm.pat.span, arm.hir_id, source),
                            _ => bug!(),
                        }
                    }

                    hir::MatchSource::ForLoopDesugar | hir::MatchSource::Normal => {
                        unreachable_pattern(cx.tcx, arm.pat.span, arm.hir_id, catchall);
                    }

                    // Unreachable patterns in try and await expressions occur when one of
                    // the arms are an uninhabited type. Which is OK.
                    hir::MatchSource::AwaitDesugar | hir::MatchSource::TryDesugar => {}
                }
            }
            Useful(unreachables) if unreachables.is_empty() => {}
            // The arm is reachable, but contains unreachable subpatterns (from or-patterns).
            Useful(unreachables) => {
                let mut unreachables: Vec<_> = unreachables.iter().flatten().copied().collect();
                // Emit lints in the order in which they occur in the file.
                unreachables.sort_unstable();
                for span in unreachables {
                    unreachable_pattern(cx.tcx, span, arm.hir_id, None);
                }
            }
            UsefulWithWitness(_) => bug!(),
        }
        if !arm.has_guard && catchall.is_none() && pat_is_catchall(arm.pat) {
            catchall = Some(arm.pat.span);
        }
    }
}

/// Report that a match is not exhaustive.
fn non_exhaustive_match<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    scrut_ty: Ty<'tcx>,
    sp: Span,
    witnesses: Vec<super::Pat<'tcx>>,
    is_empty_match: bool,
) {
    let non_empty_enum = match scrut_ty.kind() {
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
    err.note(&format!("the matched value is of type `{}`", scrut_ty));
    if (scrut_ty == cx.tcx.types.usize || scrut_ty == cx.tcx.types.isize)
        && !is_empty_match
        && witnesses.len() == 1
        && witnesses[0].is_wildcard()
    {
        err.note(&format!(
            "`{}` does not have a fixed maximum value, \
                so a wildcard `_` is necessary to match exhaustively",
            scrut_ty,
        ));
        if cx.tcx.sess.is_nightly_build() {
            err.help(&format!(
                "add `#![feature(precise_pointer_size_matching)]` \
                    to the crate attributes to enable precise `{}` matching",
                scrut_ty,
            ));
        }
    }
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
    if let ty::Adt(def, _) = ty.kind() {
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
    if let ty::Adt(def, _) = ty.kind() {
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

/// Check if a by-value binding is by-value. That is, check if the binding's type is not `Copy`.
fn is_binding_by_move(cx: &MatchVisitor<'_, '_>, hir_id: HirId, span: Span) -> bool {
    !cx.typeck_results.node_type(hir_id).is_copy_modulo_regions(cx.tcx.at(span), cx.param_env)
}

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
fn check_borrow_conflicts_in_at_patterns(cx: &MatchVisitor<'_, '_>, pat: &Pat<'_>) {
    // Extract `sub` in `binding @ sub`.
    let (name, sub) = match &pat.kind {
        hir::PatKind::Binding(.., name, Some(sub)) => (*name, sub),
        _ => return,
    };
    let binding_span = pat.span.with_hi(name.span.hi());

    let typeck_results = cx.typeck_results;
    let sess = cx.tcx.sess;

    // Get the binding move, extract the mutability if by-ref.
    let mut_outer = match typeck_results.extract_binding_mode(sess, pat.hir_id, pat.span) {
        Some(ty::BindByValue(_)) if is_binding_by_move(cx, pat.hir_id, pat.span) => {
            // We have `x @ pat` where `x` is by-move. Reject all borrows in `pat`.
            let mut conflicts_ref = Vec::new();
            sub.each_binding(|_, hir_id, span, _| {
                match typeck_results.extract_binding_mode(sess, hir_id, span) {
                    Some(ty::BindByValue(_)) | None => {}
                    Some(ty::BindByReference(_)) => conflicts_ref.push(span),
                }
            });
            if !conflicts_ref.is_empty() {
                let occurs_because = format!(
                    "move occurs because `{}` has type `{}` which does not implement the `Copy` trait",
                    name,
                    typeck_results.node_type(pat.hir_id),
                );
                sess.struct_span_err(pat.span, "borrow of moved value")
                    .span_label(binding_span, format!("value moved into `{}` here", name))
                    .span_label(binding_span, occurs_because)
                    .span_labels(conflicts_ref, "value borrowed here after move")
                    .emit();
            }
            return;
        }
        Some(ty::BindByValue(_)) | None => return,
        Some(ty::BindByReference(m)) => m,
    };

    // We now have `ref $mut_outer binding @ sub` (semantically).
    // Recurse into each binding in `sub` and find mutability or move conflicts.
    let mut conflicts_move = Vec::new();
    let mut conflicts_mut_mut = Vec::new();
    let mut conflicts_mut_ref = Vec::new();
    sub.each_binding(|_, hir_id, span, name| {
        match typeck_results.extract_binding_mode(sess, hir_id, span) {
            Some(ty::BindByReference(mut_inner)) => match (mut_outer, mut_inner) {
                (Mutability::Not, Mutability::Not) => {} // Both sides are `ref`.
                (Mutability::Mut, Mutability::Mut) => conflicts_mut_mut.push((span, name)), // 2x `ref mut`.
                _ => conflicts_mut_ref.push((span, name)), // `ref` + `ref mut` in either direction.
            },
            Some(ty::BindByValue(_)) if is_binding_by_move(cx, hir_id, span) => {
                conflicts_move.push((span, name)) // `ref mut?` + by-move conflict.
            }
            Some(ty::BindByValue(_)) | None => {} // `ref mut?` + by-copy is fine.
        }
    });

    // Report errors if any.
    if !conflicts_mut_mut.is_empty() {
        // Report mutability conflicts for e.g. `ref mut x @ Some(ref mut y)`.
        let mut err = sess
            .struct_span_err(pat.span, "cannot borrow value as mutable more than once at a time");
        err.span_label(binding_span, format!("first mutable borrow, by `{}`, occurs here", name));
        for (span, name) in conflicts_mut_mut {
            err.span_label(span, format!("another mutable borrow, by `{}`, occurs here", name));
        }
        for (span, name) in conflicts_mut_ref {
            err.span_label(span, format!("also borrowed as immutable, by `{}`, here", name));
        }
        for (span, name) in conflicts_move {
            err.span_label(span, format!("also moved into `{}` here", name));
        }
        err.emit();
    } else if !conflicts_mut_ref.is_empty() {
        // Report mutability conflicts for e.g. `ref x @ Some(ref mut y)` or the converse.
        let (primary, also) = match mut_outer {
            Mutability::Mut => ("mutable", "immutable"),
            Mutability::Not => ("immutable", "mutable"),
        };
        let msg =
            format!("cannot borrow value as {} because it is also borrowed as {}", also, primary);
        let mut err = sess.struct_span_err(pat.span, &msg);
        err.span_label(binding_span, format!("{} borrow, by `{}`, occurs here", primary, name));
        for (span, name) in conflicts_mut_ref {
            err.span_label(span, format!("{} borrow, by `{}`, occurs here", also, name));
        }
        for (span, name) in conflicts_move {
            err.span_label(span, format!("also moved into `{}` here", name));
        }
        err.emit();
    } else if !conflicts_move.is_empty() {
        // Report by-ref and by-move conflicts, e.g. `ref x @ y`.
        let mut err =
            sess.struct_span_err(pat.span, "cannot move out of value because it is borrowed");
        err.span_label(binding_span, format!("value borrowed, by `{}`, here", name));
        for (span, name) in conflicts_move {
            err.span_label(span, format!("value moved into `{}` here", name));
        }
        err.emit();
    }
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
        type Map = intravisit::ErasedMap<'v>;

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
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
