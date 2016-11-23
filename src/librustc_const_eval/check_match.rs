// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use _match::{MatchCheckCtxt, Matrix, expand_pattern, is_useful};
use _match::{DUMMY_WILD_PAT};
use _match::Usefulness::*;
use _match::WitnessPreference::*;

use pattern::{Pattern, PatternContext, PatternError};

use eval::report_const_eval_err;

use rustc::dep_graph::DepNode;

use rustc::hir::pat_util::{pat_bindings, pat_contains_bindings};

use rustc::middle::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor};
use rustc::middle::expr_use_visitor::{LoanCause, MutateMode};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization::{cmt};
use rustc::session::Session;
use rustc::traits::Reveal;
use rustc::ty::{self, TyCtxt};
use rustc_errors::DiagnosticBuilder;

use rustc::hir::def::*;
use rustc::hir::intravisit::{self, Visitor, FnKind};
use rustc::hir::print::pat_to_string;
use rustc::hir::{self, Pat, PatKind};

use rustc_back::slice;

use syntax::ast;
use syntax::ptr::P;
use syntax_pos::Span;

struct OuterVisitor<'a, 'tcx: 'a> { tcx: TyCtxt<'a, 'tcx, 'tcx> }

impl<'a, 'v, 'tcx> Visitor<'v> for OuterVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, _expr: &hir::Expr) {
        return // const, static and N in [T; N] - shouldn't contain anything
    }

    fn visit_trait_item(&mut self, item: &hir::TraitItem) {
        if let hir::ConstTraitItem(..) = item.node {
            return // nothing worth match checking in a constant
        } else {
            intravisit::walk_trait_item(self, item);
        }
    }

    fn visit_impl_item(&mut self, item: &hir::ImplItem) {
        if let hir::ImplItemKind::Const(..) = item.node {
            return // nothing worth match checking in a constant
        } else {
            intravisit::walk_impl_item(self, item);
        }
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Expr, s: Span, id: ast::NodeId) {
        if let FnKind::Closure(..) = fk {
            span_bug!(s, "check_match: closure outside of function")
        }

        MatchVisitor {
            tcx: self.tcx,
            param_env: &ty::ParameterEnvironment::for_item(self.tcx, id)
        }.visit_fn(fk, fd, b, s, id);
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.visit_all_item_likes_in_krate(DepNode::MatchCheck,
                                      &mut OuterVisitor { tcx: tcx }.as_deep_visitor());
    tcx.sess.abort_if_errors();
}

fn create_e0004<'a>(sess: &'a Session, sp: Span, error_message: String) -> DiagnosticBuilder<'a> {
    struct_span_err!(sess, sp, E0004, "{}", &error_message)
}

struct MatchVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: &'a ty::ParameterEnvironment<'tcx>
}

impl<'a, 'tcx, 'v> Visitor<'v> for MatchVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &hir::Expr) {
        intravisit::walk_expr(self, ex);

        match ex.node {
            hir::ExprMatch(ref scrut, ref arms, source) => {
                self.check_match(scrut, arms, source, ex.span);
            }
            _ => {}
        }
    }

    fn visit_local(&mut self, loc: &hir::Local) {
        intravisit::walk_local(self, loc);

        self.check_irrefutable(&loc.pat, false);

        // Check legality of move bindings and `@` patterns.
        self.check_patterns(false, slice::ref_slice(&loc.pat));
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Expr, s: Span, n: ast::NodeId) {
        intravisit::walk_fn(self, fk, fd, b, s, n);

        for input in &fd.inputs {
            self.check_irrefutable(&input.pat, true);
            self.check_patterns(false, slice::ref_slice(&input.pat));
        }
    }
}

impl<'a, 'tcx> MatchVisitor<'a, 'tcx> {
    fn check_patterns(&self, has_guard: bool, pats: &[P<Pat>]) {
        check_legality_of_move_bindings(self, has_guard, pats);
        for pat in pats {
            check_legality_of_bindings_in_at_patterns(self, pat);
        }
    }

    fn report_inlining_errors(&self, patcx: PatternContext, pat_span: Span) {
        for error in patcx.errors {
            match error {
                PatternError::BadConstInPattern(span, def_id) => {
                    self.tcx.sess.span_err(
                        span,
                        &format!("constants of the type `{}` \
                                  cannot be used in patterns",
                                 self.tcx.item_path_str(def_id)));
                }
                PatternError::StaticInPattern(span) => {
                    span_err!(self.tcx.sess, span, E0158,
                              "statics cannot be referenced in patterns");
                }
                PatternError::ConstEval(err) => {
                    report_const_eval_err(self.tcx, &err, pat_span, "pattern").emit();
                }
            }
        }
    }

    fn check_match(
        &self,
        scrut: &hir::Expr,
        arms: &[hir::Arm],
        source: hir::MatchSource,
        span: Span)
    {
        for arm in arms {
            // First, check legality of move bindings.
            self.check_patterns(arm.guard.is_some(), &arm.pats);

            // Second, if there is a guard on each arm, make sure it isn't
            // assigning or borrowing anything mutably.
            if let Some(ref guard) = arm.guard {
                check_for_mutation_in_guard(self, &guard);
            }

            // Third, perform some lints.
            for pat in &arm.pats {
                check_for_bindings_named_the_same_as_variants(self, pat);
            }
        }

        MatchCheckCtxt::create_and_enter(self.tcx, |ref mut cx| {
            let mut have_errors = false;

            let inlined_arms : Vec<(Vec<_>, _)> = arms.iter().map(|arm| (
                arm.pats.iter().map(|pat| {
                    let mut patcx = PatternContext::new(self.tcx);
                    let pattern = expand_pattern(cx, patcx.lower_pattern(&pat));
                    if !patcx.errors.is_empty() {
                        self.report_inlining_errors(patcx, pat.span);
                        have_errors = true;
                    }
                    (pattern, &**pat)
                }).collect(),
                arm.guard.as_ref().map(|e| &**e)
            )).collect();

            // Bail out early if inlining failed.
            if have_errors {
                return;
            }

            // Fourth, check for unreachable arms.
            check_arms(cx, &inlined_arms, source);

            // Finally, check if the whole match expression is exhaustive.
            // Check for empty enum, because is_useful only works on inhabited types.
            let pat_ty = self.tcx.tables().node_id_to_type(scrut.id);
            if inlined_arms.is_empty() {
                if !pat_ty.is_uninhabited(Some(scrut.id), self.tcx) {
                    // We know the type is inhabited, so this must be wrong
                    let mut err = create_e0004(self.tcx.sess, span,
                                               format!("non-exhaustive patterns: type {} \
                                                        is non-empty",
                                                       pat_ty));
                    span_help!(&mut err, span,
                               "Please ensure that all possible cases are being handled; \
                                possibly adding wildcards or more match arms.");
                    err.emit();
                }
                // If the type *is* uninhabited, it's vacuously exhaustive
                return;
            }

            let matrix: Matrix = inlined_arms
                .iter()
                .filter(|&&(_, guard)| guard.is_none())
                .flat_map(|arm| &arm.0)
                .map(|pat| vec![pat.0])
                .collect();
            check_exhaustive(cx, scrut.span, &matrix, source);
        })
    }

    fn check_irrefutable(&self, pat: &Pat, is_fn_arg: bool) {
        let origin = if is_fn_arg {
            "function argument"
        } else {
            "local binding"
        };

        MatchCheckCtxt::create_and_enter(self.tcx, |ref mut cx| {
            let mut patcx = PatternContext::new(self.tcx);
            let pats : Matrix = vec![vec![
                expand_pattern(cx, patcx.lower_pattern(pat))
            ]].into_iter().collect();

            let witness = match is_useful(cx, &pats, &[cx.wild_pattern], ConstructWitness) {
                UsefulWithWitness(witness) => witness,
                NotUseful => return,
                Useful => bug!()
            };

            let pattern_string = pat_to_string(witness[0].single_pattern());
            let mut diag = struct_span_err!(
                self.tcx.sess, pat.span, E0005,
                "refutable pattern in {}: `{}` not covered",
                origin, pattern_string
            );
            diag.span_label(pat.span, &format!("pattern `{}` not covered", pattern_string));
            diag.emit();
        });
    }
}

fn check_for_bindings_named_the_same_as_variants(cx: &MatchVisitor, pat: &Pat) {
    pat.walk(|p| {
        if let PatKind::Binding(hir::BindByValue(hir::MutImmutable), name, None) = p.node {
            let pat_ty = cx.tcx.tables().pat_ty(p);
            if let ty::TyAdt(edef, _) = pat_ty.sty {
                if edef.is_enum() {
                    if let Def::Local(..) = cx.tcx.expect_def(p.id) {
                        if edef.variants.iter().any(|variant| {
                            variant.name == name.node && variant.ctor_kind == CtorKind::Const
                        }) {
                            let ty_path = cx.tcx.item_path_str(edef.did);
                            let mut err = struct_span_warn!(cx.tcx.sess, p.span, E0170,
                                "pattern binding `{}` is named the same as one \
                                of the variants of the type `{}`",
                                name.node, ty_path);
                            help!(err,
                                "if you meant to match on a variant, \
                                consider making the path in the pattern qualified: `{}::{}`",
                                ty_path, name.node);
                            err.emit();
                        }
                    }
                }
            }
        }
        true
    });
}

/// Checks for common cases of "catchall" patterns that may not be intended as such.
fn pat_is_catchall(dm: &DefMap, pat: &Pat) -> bool {
    match pat.node {
        PatKind::Binding(.., None) => true,
        PatKind::Binding(.., Some(ref s)) => pat_is_catchall(dm, s),
        PatKind::Ref(ref s, _) => pat_is_catchall(dm, s),
        PatKind::Tuple(ref v, _) => v.iter().all(|p| {
            pat_is_catchall(dm, &p)
        }),
        _ => false
    }
}

// Check for unreachable patterns
fn check_arms<'a, 'tcx>(cx: &mut MatchCheckCtxt<'a, 'tcx>,
                        arms: &[(Vec<(&'a Pattern<'tcx>, &hir::Pat)>, Option<&hir::Expr>)],
                        source: hir::MatchSource)
{
    let mut seen = Matrix::empty();
    let mut catchall = None;
    let mut printed_if_let_err = false;
    for &(ref pats, guard) in arms {
        for &(pat, hir_pat) in pats {
            let v = vec![pat];

            match is_useful(cx, &seen, &v[..], LeaveOutWitness) {
                NotUseful => {
                    match source {
                        hir::MatchSource::IfLetDesugar { .. } => {
                            if printed_if_let_err {
                                // we already printed an irrefutable if-let pattern error.
                                // We don't want two, that's just confusing.
                            } else {
                                // find the first arm pattern so we can use its span
                                let &(ref first_arm_pats, _) = &arms[0];
                                let first_pat = &first_arm_pats[0];
                                let span = first_pat.0.span;
                                struct_span_err!(cx.tcx.sess, span, E0162,
                                                "irrefutable if-let pattern")
                                    .span_label(span, &format!("irrefutable pattern"))
                                    .emit();
                                printed_if_let_err = true;
                            }
                        },

                        hir::MatchSource::WhileLetDesugar => {
                            // find the first arm pattern so we can use its span
                            let &(ref first_arm_pats, _) = &arms[0];
                            let first_pat = &first_arm_pats[0];
                            let span = first_pat.0.span;
                            struct_span_err!(cx.tcx.sess, span, E0165,
                                             "irrefutable while-let pattern")
                                .span_label(span, &format!("irrefutable pattern"))
                                .emit();
                        },

                        hir::MatchSource::ForLoopDesugar => {
                            // this is a bug, because on `match iter.next()` we cover
                            // `Some(<head>)` and `None`. It's impossible to have an unreachable
                            // pattern
                            // (see libsyntax/ext/expand.rs for the full expansion of a for loop)
                            span_bug!(pat.span, "unreachable for-loop pattern")
                        },

                        hir::MatchSource::Normal => {
                            let mut err = struct_span_err!(cx.tcx.sess, pat.span, E0001,
                                                           "unreachable pattern");
                            err.span_label(pat.span, &"this is an unreachable pattern");
                            // if we had a catchall pattern, hint at that
                            if let Some(catchall) = catchall {
                                err.span_note(catchall, "this pattern matches any value");
                            }
                            err.emit();
                        },

                        hir::MatchSource::TryDesugar => {
                            span_bug!(pat.span, "unreachable try pattern")
                        },
                    }
                }
                Useful => (),
                UsefulWithWitness(_) => bug!()
            }
            if guard.is_none() {
                seen.push(v);
                if catchall.is_none() && pat_is_catchall(&cx.tcx.def_map.borrow(), hir_pat) {
                    catchall = Some(pat.span);
                }
            }
        }
    }
}

fn check_exhaustive<'a, 'tcx>(cx: &mut MatchCheckCtxt<'a, 'tcx>,
                              sp: Span,
                              matrix: &Matrix<'a, 'tcx>,
                              source: hir::MatchSource) {
    match is_useful(cx, matrix, &[cx.wild_pattern], ConstructWitness) {
        UsefulWithWitness(pats) => {
            let witnesses = if pats.is_empty() {
                vec![DUMMY_WILD_PAT]
            } else {
                pats.iter().map(|w| w.single_pattern()).collect()
            };
            match source {
                hir::MatchSource::ForLoopDesugar => {
                    // `witnesses[0]` has the form `Some(<head>)`, peel off the `Some`
                    let witness = match witnesses[0].node {
                        PatKind::TupleStruct(_, ref pats, _) => match &pats[..] {
                            &[ref pat] => &**pat,
                            _ => bug!(),
                        },
                        _ => bug!(),
                    };
                    let pattern_string = pat_to_string(witness);
                    struct_span_err!(cx.tcx.sess, sp, E0297,
                        "refutable pattern in `for` loop binding: \
                                `{}` not covered",
                                pattern_string)
                        .span_label(sp, &format!("pattern `{}` not covered", pattern_string))
                        .emit();
                },
                _ => {
                    let pattern_strings: Vec<_> = witnesses.iter().map(|w| {
                        pat_to_string(w)
                    }).collect();
                    const LIMIT: usize = 3;
                    let joined_patterns = match pattern_strings.len() {
                        0 => bug!(),
                        1 => format!("`{}`", pattern_strings[0]),
                        2...LIMIT => {
                            let (tail, head) = pattern_strings.split_last().unwrap();
                            format!("`{}`", head.join("`, `") + "` and `" + tail)
                        },
                        _ => {
                            let (head, tail) = pattern_strings.split_at(LIMIT);
                            format!("`{}` and {} more", head.join("`, `"), tail.len())
                        }
                    };

                    let label_text = match pattern_strings.len(){
                        1 => format!("pattern {} not covered", joined_patterns),
                        _ => format!("patterns {} not covered", joined_patterns)
                    };
                    create_e0004(cx.tcx.sess, sp,
                                 format!("non-exhaustive patterns: {} not covered",
                                         joined_patterns))
                        .span_label(sp, &label_text)
                        .emit();
                },
            }
        }
        NotUseful => {
            // This is good, wildcard pattern isn't reachable
        },
        _ => bug!()
    }
}

// Legality of move bindings checking
fn check_legality_of_move_bindings(cx: &MatchVisitor,
                                   has_guard: bool,
                                   pats: &[P<Pat>]) {
    let mut by_ref_span = None;
    for pat in pats {
        pat_bindings(&pat, |bm, _, span, _path| {
            if let hir::BindByRef(..) = bm {
                by_ref_span = Some(span);
            }
        })
    }

    let check_move = |p: &Pat, sub: Option<&Pat>| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| pat_contains_bindings(&p)) {
            struct_span_err!(cx.tcx.sess, p.span, E0007,
                             "cannot bind by-move with sub-bindings")
                .span_label(p.span, &format!("binds an already bound by-move value by moving it"))
                .emit();
        } else if has_guard {
            struct_span_err!(cx.tcx.sess, p.span, E0008,
                      "cannot bind by-move into a pattern guard")
                .span_label(p.span, &format!("moves value into pattern guard"))
                .emit();
        } else if by_ref_span.is_some() {
            struct_span_err!(cx.tcx.sess, p.span, E0009,
                            "cannot bind by-move and by-ref in the same pattern")
                    .span_label(p.span, &format!("by-move pattern here"))
                    .span_label(by_ref_span.unwrap(), &format!("both by-ref and by-move used"))
                    .emit();
        }
    };

    for pat in pats {
        pat.walk(|p| {
            if let PatKind::Binding(hir::BindByValue(..), _, ref sub) = p.node {
                let pat_ty = cx.tcx.tables().node_id_to_type(p.id);
                if pat_ty.moves_by_default(cx.tcx, cx.param_env, pat.span) {
                    check_move(p, sub.as_ref().map(|p| &**p));
                }
            }
            true
        });
    }
}

/// Ensures that a pattern guard doesn't borrow by mutable reference or
/// assign.
///
/// FIXME: this should be done by borrowck.
fn check_for_mutation_in_guard(cx: &MatchVisitor, guard: &hir::Expr) {
    cx.tcx.infer_ctxt(None, Some(cx.param_env.clone()),
                      Reveal::NotSpecializable).enter(|infcx| {
        let mut checker = MutationChecker {
            cx: cx,
        };
        let mut visitor = ExprUseVisitor::new(&mut checker, &infcx);
        visitor.walk_expr(guard);
    });
}

struct MutationChecker<'a, 'gcx: 'a> {
    cx: &'a MatchVisitor<'a, 'gcx>,
}

impl<'a, 'gcx, 'tcx> Delegate<'tcx> for MutationChecker<'a, 'gcx> {
    fn matched_pat(&mut self, _: &Pat, _: cmt, _: euv::MatchMode) {}
    fn consume(&mut self, _: ast::NodeId, _: Span, _: cmt, _: ConsumeMode) {}
    fn consume_pat(&mut self, _: &Pat, _: cmt, _: ConsumeMode) {}
    fn borrow(&mut self,
              _: ast::NodeId,
              span: Span,
              _: cmt,
              _: &'tcx ty::Region,
              kind:ty:: BorrowKind,
              _: LoanCause) {
        match kind {
            ty::MutBorrow => {
                struct_span_err!(self.cx.tcx.sess, span, E0301,
                          "cannot mutably borrow in a pattern guard")
                    .span_label(span, &format!("borrowed mutably in pattern guard"))
                    .emit();
            }
            ty::ImmBorrow | ty::UniqueImmBorrow => {}
        }
    }
    fn decl_without_init(&mut self, _: ast::NodeId, _: Span) {}
    fn mutate(&mut self, _: ast::NodeId, span: Span, _: cmt, mode: MutateMode) {
        match mode {
            MutateMode::JustWrite | MutateMode::WriteAndRead => {
                struct_span_err!(self.cx.tcx.sess, span, E0302, "cannot assign in a pattern guard")
                    .span_label(span, &format!("assignment in pattern guard"))
                    .emit();
            }
            MutateMode::Init => {}
        }
    }
}

/// Forbids bindings in `@` patterns. This is necessary for memory safety,
/// because of the way rvalues are handled in the borrow check. (See issue
/// #14587.)
fn check_legality_of_bindings_in_at_patterns(cx: &MatchVisitor, pat: &Pat) {
    AtBindingPatternVisitor { cx: cx, bindings_allowed: true }.visit_pat(pat);
}

struct AtBindingPatternVisitor<'a, 'b:'a, 'tcx:'b> {
    cx: &'a MatchVisitor<'b, 'tcx>,
    bindings_allowed: bool
}

impl<'a, 'b, 'tcx, 'v> Visitor<'v> for AtBindingPatternVisitor<'a, 'b, 'tcx> {
    fn visit_pat(&mut self, pat: &Pat) {
        match pat.node {
            PatKind::Binding(.., ref subpat) => {
                if !self.bindings_allowed {
                    struct_span_err!(self.cx.tcx.sess, pat.span, E0303,
                                     "pattern bindings are not allowed after an `@`")
                        .span_label(pat.span,  &format!("not allowed after `@`"))
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
