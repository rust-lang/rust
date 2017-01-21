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
use _match::Usefulness::*;
use _match::WitnessPreference::*;

use pattern::{Pattern, PatternContext, PatternError, PatternKind};

use eval::report_const_eval_err;

use rustc::dep_graph::DepNode;

use rustc::middle::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor};
use rustc::middle::expr_use_visitor::{LoanCause, MutateMode};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization::{cmt};
use rustc::session::Session;
use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::lint;
use rustc_errors::{Diagnostic, Level, DiagnosticBuilder};

use rustc::hir::def::*;
use rustc::hir::intravisit::{self, Visitor, FnKind, NestedVisitorMap};
use rustc::hir::{self, Pat, PatKind};

use rustc_back::slice;

use syntax::ast;
use syntax::ptr::P;
use syntax_pos::{Span, DUMMY_SP};

struct OuterVisitor<'a, 'tcx: 'a> { tcx: TyCtxt<'a, 'tcx, 'tcx> }

impl<'a, 'tcx> Visitor<'tcx> for OuterVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.tcx.map)
    }

    fn visit_fn(&mut self, fk: FnKind<'tcx>, fd: &'tcx hir::FnDecl,
                b: hir::BodyId, s: Span, id: ast::NodeId) {
        intravisit::walk_fn(self, fk, fd, b, s, id);

        MatchVisitor {
            tcx: self.tcx,
            tables: self.tcx.body_tables(b),
            param_env: &ty::ParameterEnvironment::for_item(self.tcx, id)
        }.visit_body(self.tcx.map.body(b));
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
    tables: &'a ty::Tables<'tcx>,
    param_env: &'a ty::ParameterEnvironment<'tcx>
}

impl<'a, 'tcx> Visitor<'tcx> for MatchVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        intravisit::walk_expr(self, ex);

        match ex.node {
            hir::ExprMatch(ref scrut, ref arms, source) => {
                self.check_match(scrut, arms, source);
            }
            _ => {}
        }
    }

    fn visit_local(&mut self, loc: &'tcx hir::Local) {
        intravisit::walk_local(self, loc);

        self.check_irrefutable(&loc.pat, false);

        // Check legality of move bindings and `@` patterns.
        self.check_patterns(false, slice::ref_slice(&loc.pat));
    }

    fn visit_body(&mut self, body: &'tcx hir::Body) {
        intravisit::walk_body(self, body);

        for arg in &body.arguments {
            self.check_irrefutable(&arg.pat, true);
            self.check_patterns(false, slice::ref_slice(&arg.pat));
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
        source: hir::MatchSource)
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

        let module = self.tcx.map.local_def_id(self.tcx.map.get_module_parent(scrut.id));
        MatchCheckCtxt::create_and_enter(self.tcx, module, |ref mut cx| {
            let mut have_errors = false;

            let inlined_arms : Vec<(Vec<_>, _)> = arms.iter().map(|arm| (
                arm.pats.iter().map(|pat| {
                    let mut patcx = PatternContext::new(self.tcx, self.tables);
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

            let matrix: Matrix = inlined_arms
                .iter()
                .filter(|&&(_, guard)| guard.is_none())
                .flat_map(|arm| &arm.0)
                .map(|pat| vec![pat.0])
                .collect();
            let scrut_ty = self.tables.node_id_to_type(scrut.id);
            check_exhaustive(cx, scrut_ty, scrut.span, &matrix, source);
        })
    }

    fn check_irrefutable(&self, pat: &Pat, is_fn_arg: bool) {
        let origin = if is_fn_arg {
            "function argument"
        } else {
            "local binding"
        };

        let module = self.tcx.map.local_def_id(self.tcx.map.get_module_parent(pat.id));
        MatchCheckCtxt::create_and_enter(self.tcx, module, |ref mut cx| {
            let mut patcx = PatternContext::new(self.tcx, self.tables);
            let pattern = patcx.lower_pattern(pat);
            let pattern_ty = pattern.ty;
            let pats : Matrix = vec![vec![
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
        if let PatKind::Binding(hir::BindByValue(hir::MutImmutable), _, name, None) = p.node {
            let pat_ty = cx.tables.pat_ty(p);
            if let ty::TyAdt(edef, _) = pat_ty.sty {
                if edef.is_enum() && edef.variants.iter().any(|variant| {
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

                        hir::MatchSource::ForLoopDesugar |
                        hir::MatchSource::Normal => {
                            let mut diagnostic = Diagnostic::new(Level::Warning,
                                                                 "unreachable pattern");
                            diagnostic.set_span(pat.span);
                            // if we had a catchall pattern, hint at that
                            if let Some(catchall) = catchall {
                                diagnostic.span_label(pat.span, &"this is an unreachable pattern");
                                diagnostic.span_note(catchall, "this pattern matches any value");
                            }
                            cx.tcx.sess.add_lint_diagnostic(lint::builtin::UNREACHABLE_PATTERNS,
                                                            hir_pat.id, diagnostic);
                        },

                        // Unreachable patterns in try expressions occur when one of the arms
                        // are an uninhabited type. Which is OK.
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

fn check_exhaustive<'a, 'tcx>(cx: &mut MatchCheckCtxt<'a, 'tcx>,
                              scrut_ty: Ty<'tcx>,
                              sp: Span,
                              matrix: &Matrix<'a, 'tcx>,
                              source: hir::MatchSource) {
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
            match source {
                hir::MatchSource::ForLoopDesugar => {
                    // `witnesses[0]` has the form `Some(<head>)`, peel off the `Some`
                    let witness = match *witnesses[0].kind {
                        PatternKind::Variant { ref subpatterns, .. } => match &subpatterns[..] {
                            &[ref pat] => &pat.pattern,
                            _ => bug!(),
                        },
                        _ => bug!(),
                    };
                    let pattern_string = witness.to_string();
                    struct_span_err!(cx.tcx.sess, sp, E0297,
                        "refutable pattern in `for` loop binding: \
                                `{}` not covered",
                                pattern_string)
                        .span_label(sp, &format!("pattern `{}` not covered", pattern_string))
                        .emit();
                },
                _ => {
                    const LIMIT: usize = 3;
                    let joined_patterns = match witnesses.len() {
                        0 => bug!(),
                        1 => format!("`{}`", witnesses[0]),
                        2...LIMIT => {
                            let (tail, head) = witnesses.split_last().unwrap();
                            let head: Vec<_> = head.iter().map(|w| w.to_string()).collect();
                            format!("`{}` and `{}`", head.join("`, `"), tail)
                        },
                        _ => {
                            let (head, tail) = witnesses.split_at(LIMIT);
                            let head: Vec<_> = head.iter().map(|w| w.to_string()).collect();
                            format!("`{}` and {} more", head.join("`, `"), tail.len())
                        }
                    };

                    let label_text = match witnesses.len() {
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
        pat.each_binding(|bm, _, span, _path| {
            if let hir::BindByRef(..) = bm {
                by_ref_span = Some(span);
            }
        })
    }

    let check_move = |p: &Pat, sub: Option<&Pat>| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| p.contains_bindings()) {
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
            if let PatKind::Binding(hir::BindByValue(..), _, _, ref sub) = p.node {
                let pat_ty = cx.tables.node_id_to_type(p.id);
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
    cx.tcx.infer_ctxt((cx.tables, cx.param_env.clone()), Reveal::NotSpecializable).enter(|infcx| {
        let mut checker = MutationChecker {
            cx: cx,
        };
        ExprUseVisitor::new(&mut checker, &infcx).walk_expr(guard);
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
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'v> {
        NestedVisitorMap::None
    }

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
