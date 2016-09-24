// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use _match::{MatchCheckCtxt, Matrix, wrap_pat, is_refutable, is_useful};
use _match::{DUMMY_WILD_PATTERN, DUMMY_WILD_PAT};
use _match::Usefulness::*;
use _match::WitnessPreference::*;

use eval::report_const_eval_err;
use eval::{eval_const_expr_partial, const_expr_to_pat, lookup_const_by_id};
use eval::EvalHint::ExprTypeChecked;

use rustc::dep_graph::DepNode;

use rustc::hir::pat_util::{pat_bindings, pat_contains_bindings};

use rustc::middle::const_val::ConstVal;
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
use syntax::codemap::Spanned;
use syntax::ptr::P;
use syntax::util::move_map::MoveMap;
use syntax_pos::Span;

impl<'a, 'tcx, 'v> Visitor<'v> for MatchCheckCtxt<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &hir::Expr) {
        check_expr(self, ex);
    }
    fn visit_local(&mut self, l: &hir::Local) {
        check_local(self, l);
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, s: Span, n: ast::NodeId) {
        check_fn(self, fk, fd, b, s, n);
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.visit_all_items_in_krate(DepNode::MatchCheck, &mut MatchCheckCtxt {
        tcx: tcx,
        param_env: tcx.empty_parameter_environment(),
    });
    tcx.sess.abort_if_errors();
}

fn create_e0004<'a>(sess: &'a Session, sp: Span, error_message: String) -> DiagnosticBuilder<'a> {
    struct_span_err!(sess, sp, E0004, "{}", &error_message)
}

fn check_expr(cx: &mut MatchCheckCtxt, ex: &hir::Expr) {
    intravisit::walk_expr(cx, ex);
    match ex.node {
        hir::ExprMatch(ref scrut, ref arms, source) => {
            for arm in arms {
                // First, check legality of move bindings.
                check_legality_of_move_bindings(cx,
                                                arm.guard.is_some(),
                                                &arm.pats);

                // Second, if there is a guard on each arm, make sure it isn't
                // assigning or borrowing anything mutably.
                if let Some(ref guard) = arm.guard {
                    check_for_mutation_in_guard(cx, &guard);
                }
            }

            let mut static_inliner = StaticInliner::new(cx.tcx);
            let inlined_arms = arms.iter().map(|arm| {
                (arm.pats.iter().map(|pat| {
                    static_inliner.fold_pat((*pat).clone())
                }).collect(), arm.guard.as_ref().map(|e| &**e))
            }).collect::<Vec<(Vec<P<Pat>>, Option<&hir::Expr>)>>();

            // Bail out early if inlining failed.
            if static_inliner.failed {
                return;
            }

            for pat in inlined_arms
                .iter()
                .flat_map(|&(ref pats, _)| pats) {
                // Third, check legality of move bindings.
                check_legality_of_bindings_in_at_patterns(cx, &pat);

                // Fourth, check if there are any references to NaN that we should warn about.
                check_for_static_nan(cx, &pat);

                // Fifth, check if for any of the patterns that match an enumerated type
                // are bindings with the same name as one of the variants of said type.
                check_for_bindings_named_the_same_as_variants(cx, &pat);
            }

            // Fourth, check for unreachable arms.
            check_arms(cx, &inlined_arms[..], source);

            // Finally, check if the whole match expression is exhaustive.
            // Check for empty enum, because is_useful only works on inhabited types.
            let pat_ty = cx.tcx.node_id_to_type(scrut.id);
            if inlined_arms.is_empty() {
                if !pat_ty.is_uninhabited(cx.tcx) {
                    // We know the type is inhabited, so this must be wrong
                    let mut err = create_e0004(cx.tcx.sess, ex.span,
                                               format!("non-exhaustive patterns: type {} \
                                                        is non-empty",
                                                       pat_ty));
                    span_help!(&mut err, ex.span,
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
                .map(|pat| vec![wrap_pat(cx, &pat)])
                .collect();
            check_exhaustive(cx, scrut.span, &matrix, source);
        },
        _ => ()
    }
}

fn check_for_bindings_named_the_same_as_variants(cx: &MatchCheckCtxt, pat: &Pat) {
    pat.walk(|p| {
        if let PatKind::Binding(hir::BindByValue(hir::MutImmutable), name, None) = p.node {
            let pat_ty = cx.tcx.pat_ty(p);
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

// Check that we do not match against a static NaN (#6804)
fn check_for_static_nan(cx: &MatchCheckCtxt, pat: &Pat) {
    pat.walk(|p| {
        if let PatKind::Lit(ref expr) = p.node {
            match eval_const_expr_partial(cx.tcx, &expr, ExprTypeChecked, None) {
                Ok(ConstVal::Float(f)) if f.is_nan() => {
                    span_warn!(cx.tcx.sess, p.span, E0003,
                               "unmatchable NaN in pattern, \
                                use the is_nan method in a guard instead");
                }
                Ok(_) => {}

                Err(err) => {
                    report_const_eval_err(cx.tcx, &err, p.span, "pattern").emit();
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
fn check_arms(cx: &MatchCheckCtxt,
              arms: &[(Vec<P<Pat>>, Option<&hir::Expr>)],
              source: hir::MatchSource) {
    let mut seen = Matrix::empty();
    let mut catchall = None;
    let mut printed_if_let_err = false;
    for &(ref pats, guard) in arms {
        for pat in pats {
            let v = vec![wrap_pat(cx, &pat)];

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
                                let span = first_pat.span;
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
                            let span = first_pat.span;
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
                if catchall.is_none() && pat_is_catchall(&cx.tcx.def_map.borrow(), pat) {
                    catchall = Some(pat.span);
                }
            }
        }
    }
}

fn check_exhaustive<'a, 'tcx>(cx: &MatchCheckCtxt<'a, 'tcx>,
                              sp: Span,
                              matrix: &Matrix<'a, 'tcx>,
                              source: hir::MatchSource) {
    match is_useful(cx, matrix, &[DUMMY_WILD_PATTERN], ConstructWitness) {
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


struct StaticInliner<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    failed: bool
}

impl<'a, 'tcx> StaticInliner<'a, 'tcx> {
    pub fn new<'b>(tcx: TyCtxt<'b, 'tcx, 'tcx>) -> StaticInliner<'b, 'tcx> {
        StaticInliner {
            tcx: tcx,
            failed: false
        }
    }
}

impl<'a, 'tcx> StaticInliner<'a, 'tcx> {
    fn fold_pat(&mut self, pat: P<Pat>) -> P<Pat> {
        match pat.node {
            PatKind::Path(..) => {
                match self.tcx.expect_def(pat.id) {
                    Def::AssociatedConst(did) | Def::Const(did) => {
                        let substs = Some(self.tcx.node_id_item_substs(pat.id).substs);
                        if let Some((const_expr, _)) = lookup_const_by_id(self.tcx, did, substs) {
                            match const_expr_to_pat(self.tcx, const_expr, pat.id, pat.span) {
                                Ok(new_pat) => return new_pat,
                                Err(def_id) => {
                                    self.failed = true;
                                    self.tcx.sess.span_err(
                                        pat.span,
                                        &format!("constants of the type `{}` \
                                                  cannot be used in patterns",
                                                 self.tcx.item_path_str(def_id)));
                                }
                            }
                        } else {
                            self.failed = true;
                            span_err!(self.tcx.sess, pat.span, E0158,
                                "statics cannot be referenced in patterns");
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        pat.map(|Pat { id, node, span }| {
            let node = match node {
                PatKind::Binding(binding_mode, pth1, sub) => {
                    PatKind::Binding(binding_mode, pth1, sub.map(|x| self.fold_pat(x)))
                }
                PatKind::TupleStruct(pth, pats, ddpos) => {
                    PatKind::TupleStruct(pth, pats.move_map(|x| self.fold_pat(x)), ddpos)
                }
                PatKind::Struct(pth, fields, etc) => {
                    let fs = fields.move_map(|f| {
                        Spanned {
                            span: f.span,
                            node: hir::FieldPat {
                                name: f.node.name,
                                pat: self.fold_pat(f.node.pat),
                                is_shorthand: f.node.is_shorthand,
                            },
                        }
                    });
                    PatKind::Struct(pth, fs, etc)
                }
                PatKind::Tuple(elts, ddpos) => {
                    PatKind::Tuple(elts.move_map(|x| self.fold_pat(x)), ddpos)
                }
                PatKind::Box(inner) => PatKind::Box(self.fold_pat(inner)),
                PatKind::Ref(inner, mutbl) => PatKind::Ref(self.fold_pat(inner), mutbl),
                PatKind::Slice(before, slice, after) => {
                    PatKind::Slice(before.move_map(|x| self.fold_pat(x)),
                                   slice.map(|x| self.fold_pat(x)),
                                   after.move_map(|x| self.fold_pat(x)))
                }
                PatKind::Wild |
                PatKind::Lit(_) |
                PatKind::Range(..) |
                PatKind::Path(..) => node
            };
            Pat {
                id: id,
                node: node,
                span: span
            }
        })
    }
}

fn check_local(cx: &mut MatchCheckCtxt, loc: &hir::Local) {
    intravisit::walk_local(cx, loc);

    let pat = StaticInliner::new(cx.tcx).fold_pat(loc.pat.clone());
    check_irrefutable(cx, &pat, false);

    // Check legality of move bindings and `@` patterns.
    check_legality_of_move_bindings(cx, false, slice::ref_slice(&loc.pat));
    check_legality_of_bindings_in_at_patterns(cx, &loc.pat);
}

fn check_fn(cx: &mut MatchCheckCtxt,
            kind: FnKind,
            decl: &hir::FnDecl,
            body: &hir::Block,
            sp: Span,
            fn_id: ast::NodeId) {
    match kind {
        FnKind::Closure(_) => {}
        _ => cx.param_env = ty::ParameterEnvironment::for_item(cx.tcx, fn_id),
    }

    intravisit::walk_fn(cx, kind, decl, body, sp, fn_id);

    for input in &decl.inputs {
        check_irrefutable(cx, &input.pat, true);
        check_legality_of_move_bindings(cx, false, slice::ref_slice(&input.pat));
        check_legality_of_bindings_in_at_patterns(cx, &input.pat);
    }
}

fn check_irrefutable(cx: &MatchCheckCtxt, pat: &Pat, is_fn_arg: bool) {
    let origin = if is_fn_arg {
        "function argument"
    } else {
        "local binding"
    };

    is_refutable(cx, pat, |uncovered_pat| {
        let pattern_string = pat_to_string(uncovered_pat.single_pattern());
        struct_span_err!(cx.tcx.sess, pat.span, E0005,
            "refutable pattern in {}: `{}` not covered",
            origin,
            pattern_string,
        ).span_label(pat.span, &format!("pattern `{}` not covered", pattern_string)).emit();
    });
}

// Legality of move bindings checking
fn check_legality_of_move_bindings(cx: &MatchCheckCtxt,
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
                let pat_ty = cx.tcx.node_id_to_type(p.id);
                //FIXME: (@jroesch) this code should be floated up as well
                cx.tcx.infer_ctxt(None, Some(cx.param_env.clone()),
                                  Reveal::NotSpecializable).enter(|infcx| {
                    if infcx.type_moves_by_default(pat_ty, pat.span) {
                        check_move(p, sub.as_ref().map(|p| &**p));
                    }
                });
            }
            true
        });
    }
}

/// Ensures that a pattern guard doesn't borrow by mutable reference or
/// assign.
fn check_for_mutation_in_guard<'a, 'tcx>(cx: &'a MatchCheckCtxt<'a, 'tcx>,
                                         guard: &hir::Expr) {
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
    cx: &'a MatchCheckCtxt<'a, 'gcx>,
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
fn check_legality_of_bindings_in_at_patterns(cx: &MatchCheckCtxt, pat: &Pat) {
    AtBindingPatternVisitor { cx: cx, bindings_allowed: true }.visit_pat(pat);
}

struct AtBindingPatternVisitor<'a, 'b:'a, 'tcx:'b> {
    cx: &'a MatchCheckCtxt<'b, 'tcx>,
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
