// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::Constructor::*;
use self::Usefulness::*;
use self::WitnessPreference::*;

use rustc::dep_graph::DepNode;
use rustc::middle::const_val::ConstVal;
use ::{eval_const_expr, eval_const_expr_partial, compare_const_vals};
use ::{const_expr_to_pat, lookup_const_by_id};
use ::EvalHint::ExprTypeChecked;
use rustc::hir::def::*;
use rustc::hir::def_id::{DefId};
use rustc::middle::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor};
use rustc::middle::expr_use_visitor::{LoanCause, MutateMode};
use rustc::middle::expr_use_visitor as euv;
use rustc::infer;
use rustc::middle::mem_categorization::{cmt};
use rustc::hir::pat_util::*;
use rustc::traits::ProjectionMode;
use rustc::ty::*;
use rustc::ty;
use std::cmp::Ordering;
use std::fmt;
use std::iter::{FromIterator, IntoIterator, repeat};

use rustc::hir;
use rustc::hir::{Pat, PatKind};
use rustc::hir::intravisit::{self, IdVisitor, IdVisitingOperation, Visitor, FnKind};
use rustc_back::slice;

use syntax::ast::{self, DUMMY_NODE_ID, NodeId};
use syntax::codemap::{Span, Spanned, DUMMY_SP};
use rustc::hir::fold::{Folder, noop_fold_pat};
use rustc::hir::print::pat_to_string;
use syntax::ptr::P;
use rustc::util::nodemap::FnvHashMap;

pub const DUMMY_WILD_PAT: &'static Pat = &Pat {
    id: DUMMY_NODE_ID,
    node: PatKind::Wild,
    span: DUMMY_SP
};

struct Matrix<'a>(Vec<Vec<&'a Pat>>);

/// Pretty-printer for matrices of patterns, example:
/// ++++++++++++++++++++++++++
/// + _     + []             +
/// ++++++++++++++++++++++++++
/// + true  + [First]        +
/// ++++++++++++++++++++++++++
/// + true  + [Second(true)] +
/// ++++++++++++++++++++++++++
/// + false + [_]            +
/// ++++++++++++++++++++++++++
/// + _     + [_, _, ..tail] +
/// ++++++++++++++++++++++++++
impl<'a> fmt::Debug for Matrix<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n")?;

        let &Matrix(ref m) = self;
        let pretty_printed_matrix: Vec<Vec<String>> = m.iter().map(|row| {
            row.iter()
               .map(|&pat| pat_to_string(&pat))
               .collect::<Vec<String>>()
        }).collect();

        let column_count = m.iter().map(|row| row.len()).max().unwrap_or(0);
        assert!(m.iter().all(|row| row.len() == column_count));
        let column_widths: Vec<usize> = (0..column_count).map(|col| {
            pretty_printed_matrix.iter().map(|row| row[col].len()).max().unwrap_or(0)
        }).collect();

        let total_width = column_widths.iter().cloned().sum::<usize>() + column_count * 3 + 1;
        let br = repeat('+').take(total_width).collect::<String>();
        write!(f, "{}\n", br)?;
        for row in pretty_printed_matrix {
            write!(f, "+")?;
            for (column, pat_str) in row.into_iter().enumerate() {
                write!(f, " ")?;
                write!(f, "{:1$}", pat_str, column_widths[column])?;
                write!(f, " +")?;
            }
            write!(f, "\n")?;
            write!(f, "{}\n", br)?;
        }
        Ok(())
    }
}

impl<'a> FromIterator<Vec<&'a Pat>> for Matrix<'a> {
    fn from_iter<T: IntoIterator<Item=Vec<&'a Pat>>>(iter: T) -> Matrix<'a> {
        Matrix(iter.into_iter().collect())
    }
}

//NOTE: appears to be the only place other then InferCtxt to contain a ParamEnv
pub struct MatchCheckCtxt<'a, 'tcx: 'a> {
    pub tcx: &'a TyCtxt<'tcx>,
    pub param_env: ParameterEnvironment<'a, 'tcx>,
}

#[derive(Clone, PartialEq)]
pub enum Constructor {
    /// The constructor of all patterns that don't vary by constructor,
    /// e.g. struct patterns and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(DefId),
    /// Literal values.
    ConstantValue(ConstVal),
    /// Ranges of literal values (2..5).
    ConstantRange(ConstVal, ConstVal),
    /// Array patterns of length n.
    Slice(usize),
    /// Array patterns with a subslice.
    SliceWithSubslice(usize, usize)
}

#[derive(Clone, PartialEq)]
enum Usefulness {
    Useful,
    UsefulWithWitness(Vec<P<Pat>>),
    NotUseful
}

#[derive(Copy, Clone)]
enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness
}

impl<'a, 'tcx, 'v> Visitor<'v> for MatchCheckCtxt<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &hir::Expr) {
        check_expr(self, ex);
    }
    fn visit_local(&mut self, l: &hir::Local) {
        check_local(self, l);
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, s: Span, n: NodeId) {
        check_fn(self, fk, fd, b, s, n);
    }
}

pub fn check_crate(tcx: &TyCtxt) {
    tcx.visit_all_items_in_krate(DepNode::MatchCheck, &mut MatchCheckCtxt {
        tcx: tcx,
        param_env: tcx.empty_parameter_environment(),
    });
    tcx.sess.abort_if_errors();
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
                match arm.guard {
                    Some(ref guard) => check_for_mutation_in_guard(cx, &guard),
                    None => {}
                }
            }

            let mut static_inliner = StaticInliner::new(cx.tcx, None);
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
                if !pat_ty.is_empty(cx.tcx) {
                    // We know the type is inhabited, so this must be wrong
                    let mut err = struct_span_err!(cx.tcx.sess, ex.span, E0002,
                                                   "non-exhaustive patterns: type {} is non-empty",
                                                   pat_ty);
                    span_help!(&mut err, ex.span,
                        "Please ensure that all possible cases are being handled; \
                         possibly adding wildcards or more match arms.");
                    err.emit();
                }
                // If the type *is* empty, it's vacuously exhaustive
                return;
            }

            let matrix: Matrix = inlined_arms
                .iter()
                .filter(|&&(_, guard)| guard.is_none())
                .flat_map(|arm| &arm.0)
                .map(|pat| vec![&**pat])
                .collect();
            check_exhaustive(cx, ex.span, &matrix, source);
        },
        _ => ()
    }
}

fn check_for_bindings_named_the_same_as_variants(cx: &MatchCheckCtxt, pat: &Pat) {
    pat.walk(|p| {
        match p.node {
            PatKind::Ident(hir::BindByValue(hir::MutImmutable), ident, None) => {
                let pat_ty = cx.tcx.pat_ty(p);
                if let ty::TyEnum(edef, _) = pat_ty.sty {
                    let def = cx.tcx.def_map.borrow().get(&p.id).map(|d| d.full_def());
                    if let Some(Def::Local(..)) = def {
                        if edef.variants.iter().any(|variant|
                            variant.name == ident.node.unhygienic_name
                                && variant.kind() == VariantKind::Unit
                        ) {
                            let ty_path = cx.tcx.item_path_str(edef.did);
                            let mut err = struct_span_warn!(cx.tcx.sess, p.span, E0170,
                                "pattern binding `{}` is named the same as one \
                                 of the variants of the type `{}`",
                                ident.node, ty_path);
                            help!(err,
                                "if you meant to match on a variant, \
                                 consider making the path in the pattern qualified: `{}::{}`",
                                ty_path, ident.node);
                            err.emit();
                        }
                    }
                }
            }
            _ => ()
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
                    let mut diag = struct_span_err!(cx.tcx.sess, err.span, E0471,
                                                    "constant evaluation error: {}",
                                                    err.description());
                    if !p.span.contains(err.span) {
                        diag.span_note(p.span, "in pattern here");
                    }
                    diag.emit();
                }
            }
        }
        true
    });
}

// Check for unreachable patterns
fn check_arms(cx: &MatchCheckCtxt,
              arms: &[(Vec<P<Pat>>, Option<&hir::Expr>)],
              source: hir::MatchSource) {
    let mut seen = Matrix(vec![]);
    let mut printed_if_let_err = false;
    for &(ref pats, guard) in arms {
        for pat in pats {
            let v = vec![&**pat];

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
                                span_err!(cx.tcx.sess, span, E0162, "irrefutable if-let pattern");
                                printed_if_let_err = true;
                            }
                        },

                        hir::MatchSource::WhileLetDesugar => {
                            // find the first arm pattern so we can use its span
                            let &(ref first_arm_pats, _) = &arms[0];
                            let first_pat = &first_arm_pats[0];
                            let span = first_pat.span;
                            span_err!(cx.tcx.sess, span, E0165, "irrefutable while-let pattern");
                        },

                        hir::MatchSource::ForLoopDesugar => {
                            // this is a bug, because on `match iter.next()` we cover
                            // `Some(<head>)` and `None`. It's impossible to have an unreachable
                            // pattern
                            // (see libsyntax/ext/expand.rs for the full expansion of a for loop)
                            span_bug!(pat.span, "unreachable for-loop pattern")
                        },

                        hir::MatchSource::Normal => {
                            span_err!(cx.tcx.sess, pat.span, E0001, "unreachable pattern")
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
                let Matrix(mut rows) = seen;
                rows.push(v);
                seen = Matrix(rows);
            }
        }
    }
}

fn raw_pat<'a>(p: &'a Pat) -> &'a Pat {
    match p.node {
        PatKind::Ident(_, _, Some(ref s)) => raw_pat(&s),
        _ => p
    }
}

fn check_exhaustive(cx: &MatchCheckCtxt, sp: Span, matrix: &Matrix, source: hir::MatchSource) {
    match is_useful(cx, matrix, &[DUMMY_WILD_PAT], ConstructWitness) {
        UsefulWithWitness(pats) => {
            let witnesses = if pats.is_empty() {
                vec![DUMMY_WILD_PAT]
            } else {
                pats.iter().map(|w| &**w ).collect()
            };
            match source {
                hir::MatchSource::ForLoopDesugar => {
                    // `witnesses[0]` has the form `Some(<head>)`, peel off the `Some`
                    let witness = match witnesses[0].node {
                        PatKind::TupleStruct(_, Some(ref pats)) => match &pats[..] {
                            [ref pat] => &**pat,
                            _ => bug!(),
                        },
                        _ => bug!(),
                    };
                    span_err!(cx.tcx.sess, sp, E0297,
                        "refutable pattern in `for` loop binding: \
                                `{}` not covered",
                                pat_to_string(witness));
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
                    span_err!(cx.tcx.sess, sp, E0004,
                        "non-exhaustive patterns: {} not covered",
                        joined_patterns
                    );
                },
            }
        }
        NotUseful => {
            // This is good, wildcard pattern isn't reachable
        },
        _ => bug!()
    }
}

fn const_val_to_expr(value: &ConstVal) -> P<hir::Expr> {
    let node = match value {
        &ConstVal::Bool(b) => ast::LitKind::Bool(b),
        _ => bug!()
    };
    P(hir::Expr {
        id: 0,
        node: hir::ExprLit(P(Spanned { node: node, span: DUMMY_SP })),
        span: DUMMY_SP,
        attrs: None,
    })
}

pub struct StaticInliner<'a, 'tcx: 'a> {
    pub tcx: &'a TyCtxt<'tcx>,
    pub failed: bool,
    pub renaming_map: Option<&'a mut FnvHashMap<(NodeId, Span), NodeId>>,
}

impl<'a, 'tcx> StaticInliner<'a, 'tcx> {
    pub fn new<'b>(tcx: &'b TyCtxt<'tcx>,
                   renaming_map: Option<&'b mut FnvHashMap<(NodeId, Span), NodeId>>)
                   -> StaticInliner<'b, 'tcx> {
        StaticInliner {
            tcx: tcx,
            failed: false,
            renaming_map: renaming_map
        }
    }
}

struct RenamingRecorder<'map> {
    substituted_node_id: NodeId,
    origin_span: Span,
    renaming_map: &'map mut FnvHashMap<(NodeId, Span), NodeId>
}

impl<'map> IdVisitingOperation for RenamingRecorder<'map> {
    fn visit_id(&mut self, node_id: NodeId) {
        let key = (node_id, self.origin_span);
        self.renaming_map.insert(key, self.substituted_node_id);
    }
}

impl<'a, 'tcx> Folder for StaticInliner<'a, 'tcx> {
    fn fold_pat(&mut self, pat: P<Pat>) -> P<Pat> {
        return match pat.node {
            PatKind::Ident(..) | PatKind::Path(..) | PatKind::QPath(..) => {
                let def = self.tcx.def_map.borrow().get(&pat.id).map(|d| d.full_def());
                match def {
                    Some(Def::AssociatedConst(did)) |
                    Some(Def::Const(did)) => {
                        let substs = Some(self.tcx.node_id_item_substs(pat.id).substs);
                        if let Some((const_expr, _)) = lookup_const_by_id(self.tcx, did, substs) {
                            match const_expr_to_pat(self.tcx, const_expr, pat.id, pat.span) {
                                Ok(new_pat) => {
                                    if let Some(ref mut map) = self.renaming_map {
                                        // Record any renamings we do here
                                        record_renamings(const_expr, &pat, map);
                                    }
                                    new_pat
                                }
                                Err(def_id) => {
                                    self.failed = true;
                                    self.tcx.sess.span_err(
                                        pat.span,
                                        &format!("constants of the type `{}` \
                                                  cannot be used in patterns",
                                                 self.tcx.item_path_str(def_id)));
                                    pat
                                }
                            }
                        } else {
                            self.failed = true;
                            span_err!(self.tcx.sess, pat.span, E0158,
                                "statics cannot be referenced in patterns");
                            pat
                        }
                    }
                    _ => noop_fold_pat(pat, self)
                }
            }
            _ => noop_fold_pat(pat, self)
        };

        fn record_renamings(const_expr: &hir::Expr,
                            substituted_pat: &hir::Pat,
                            renaming_map: &mut FnvHashMap<(NodeId, Span), NodeId>) {
            let mut renaming_recorder = RenamingRecorder {
                substituted_node_id: substituted_pat.id,
                origin_span: substituted_pat.span,
                renaming_map: renaming_map,
            };

            let mut id_visitor = IdVisitor::new(&mut renaming_recorder);

            id_visitor.visit_expr(const_expr);
        }
    }
}

/// Constructs a partial witness for a pattern given a list of
/// patterns expanded by the specialization step.
///
/// When a pattern P is discovered to be useful, this function is used bottom-up
/// to reconstruct a complete witness, e.g. a pattern P' that covers a subset
/// of values, V, where each value in that set is not covered by any previously
/// used patterns and is covered by the pattern P'. Examples:
///
/// left_ty: tuple of 3 elements
/// pats: [10, 20, _]           => (10, 20, _)
///
/// left_ty: struct X { a: (bool, &'static str), b: usize}
/// pats: [(false, "foo"), 42]  => X { a: (false, "foo"), b: 42 }
fn construct_witness<'a,'tcx>(cx: &MatchCheckCtxt<'a,'tcx>, ctor: &Constructor,
                              pats: Vec<&Pat>, left_ty: Ty<'tcx>) -> P<Pat> {
    let pats_len = pats.len();
    let mut pats = pats.into_iter().map(|p| P((*p).clone()));
    let pat = match left_ty.sty {
        ty::TyTuple(_) => PatKind::Tup(pats.collect()),

        ty::TyEnum(adt, _) | ty::TyStruct(adt, _)  => {
            let v = ctor.variant_for_adt(adt);
            match v.kind() {
                VariantKind::Struct => {
                    let field_pats: hir::HirVec<_> = v.fields.iter()
                        .zip(pats)
                        .filter(|&(_, ref pat)| pat.node != PatKind::Wild)
                        .map(|(field, pat)| Spanned {
                            span: DUMMY_SP,
                            node: hir::FieldPat {
                                name: field.name,
                                pat: pat,
                                is_shorthand: false,
                            }
                        }).collect();
                    let has_more_fields = field_pats.len() < pats_len;
                    PatKind::Struct(def_to_path(cx.tcx, v.did), field_pats, has_more_fields)
                }
                VariantKind::Tuple => {
                    PatKind::TupleStruct(def_to_path(cx.tcx, v.did), Some(pats.collect()))
                }
                VariantKind::Unit => {
                    PatKind::Path(def_to_path(cx.tcx, v.did))
                }
            }
        }

        ty::TyRef(_, ty::TypeAndMut { ty, mutbl }) => {
            match ty.sty {
               ty::TyArray(_, n) => match ctor {
                    &Single => {
                        assert_eq!(pats_len, n);
                        PatKind::Vec(pats.collect(), None, hir::HirVec::new())
                    },
                    _ => bug!()
                },
                ty::TySlice(_) => match ctor {
                    &Slice(n) => {
                        assert_eq!(pats_len, n);
                        PatKind::Vec(pats.collect(), None, hir::HirVec::new())
                    },
                    _ => bug!()
                },
                ty::TyStr => PatKind::Wild,

                _ => {
                    assert_eq!(pats_len, 1);
                    PatKind::Ref(pats.nth(0).unwrap(), mutbl)
                }
            }
        }

        ty::TyArray(_, len) => {
            assert_eq!(pats_len, len);
            PatKind::Vec(pats.collect(), None, hir::HirVec::new())
        }

        _ => {
            match *ctor {
                ConstantValue(ref v) => PatKind::Lit(const_val_to_expr(v)),
                _ => PatKind::Wild,
            }
        }
    };

    P(hir::Pat {
        id: 0,
        node: pat,
        span: DUMMY_SP
    })
}

impl Constructor {
    fn variant_for_adt<'tcx, 'container, 'a>(&self,
                                             adt: &'a ty::AdtDefData<'tcx, 'container>)
                                             -> &'a VariantDefData<'tcx, 'container> {
        match self {
            &Variant(vid) => adt.variant_with_id(vid),
            _ => adt.struct_variant()
        }
    }
}

fn missing_constructors(cx: &MatchCheckCtxt, &Matrix(ref rows): &Matrix,
                       left_ty: Ty, max_slice_length: usize) -> Vec<Constructor> {
    let used_constructors: Vec<Constructor> = rows.iter()
        .flat_map(|row| pat_constructors(cx, row[0], left_ty, max_slice_length))
        .collect();
    all_constructors(cx, left_ty, max_slice_length)
        .into_iter()
        .filter(|c| !used_constructors.contains(c))
        .collect()
}

/// This determines the set of all possible constructors of a pattern matching
/// values of type `left_ty`. For vectors, this would normally be an infinite set
/// but is instead bounded by the maximum fixed length of slice patterns in
/// the column of patterns being analyzed.
fn all_constructors(_cx: &MatchCheckCtxt, left_ty: Ty,
                    max_slice_length: usize) -> Vec<Constructor> {
    match left_ty.sty {
        ty::TyBool =>
            [true, false].iter().map(|b| ConstantValue(ConstVal::Bool(*b))).collect(),

        ty::TyRef(_, ty::TypeAndMut { ty, .. }) => match ty.sty {
            ty::TySlice(_) =>
                (0..max_slice_length+1).map(|length| Slice(length)).collect(),
            _ => vec![Single]
        },

        ty::TyEnum(def, _) => def.variants.iter().map(|v| Variant(v.did)).collect(),
        _ => vec![Single]
    }
}

// Algorithm from http://moscova.inria.fr/~maranget/papers/warn/index.html
//
// Whether a vector `v` of patterns is 'useful' in relation to a set of such
// vectors `m` is defined as there being a set of inputs that will match `v`
// but not any of the sets in `m`.
//
// This is used both for reachability checking (if a pattern isn't useful in
// relation to preceding patterns, it is not reachable) and exhaustiveness
// checking (if a wildcard pattern is useful in relation to a matrix, the
// matrix isn't exhaustive).

// Note: is_useful doesn't work on empty types, as the paper notes.
// So it assumes that v is non-empty.
fn is_useful(cx: &MatchCheckCtxt,
             matrix: &Matrix,
             v: &[&Pat],
             witness: WitnessPreference)
             -> Usefulness {
    let &Matrix(ref rows) = matrix;
    debug!("{:?}", matrix);
    if rows.is_empty() {
        return match witness {
            ConstructWitness => UsefulWithWitness(vec!()),
            LeaveOutWitness => Useful
        };
    }
    if rows[0].is_empty() {
        return NotUseful;
    }
    assert!(rows.iter().all(|r| r.len() == v.len()));
    let real_pat = match rows.iter().find(|r| (*r)[0].id != DUMMY_NODE_ID) {
        Some(r) => raw_pat(r[0]),
        None if v.is_empty() => return NotUseful,
        None => v[0]
    };
    let left_ty = if real_pat.id == DUMMY_NODE_ID {
        cx.tcx.mk_nil()
    } else {
        let left_ty = cx.tcx.pat_ty(&real_pat);

        match real_pat.node {
            PatKind::Ident(hir::BindByRef(..), _, _) => {
                left_ty.builtin_deref(false, NoPreference).unwrap().ty
            }
            _ => left_ty,
        }
    };

    let max_slice_length = rows.iter().filter_map(|row| match row[0].node {
        PatKind::Vec(ref before, _, ref after) => Some(before.len() + after.len()),
        _ => None
    }).max().map_or(0, |v| v + 1);

    let constructors = pat_constructors(cx, v[0], left_ty, max_slice_length);
    if constructors.is_empty() {
        let constructors = missing_constructors(cx, matrix, left_ty, max_slice_length);
        if constructors.is_empty() {
            all_constructors(cx, left_ty, max_slice_length).into_iter().map(|c| {
                match is_useful_specialized(cx, matrix, v, c.clone(), left_ty, witness) {
                    UsefulWithWitness(pats) => UsefulWithWitness({
                        let arity = constructor_arity(cx, &c, left_ty);
                        let mut result = {
                            let pat_slice = &pats[..];
                            let subpats: Vec<_> = (0..arity).map(|i| {
                                pat_slice.get(i).map_or(DUMMY_WILD_PAT, |p| &**p)
                            }).collect();
                            vec![construct_witness(cx, &c, subpats, left_ty)]
                        };
                        result.extend(pats.into_iter().skip(arity));
                        result
                    }),
                    result => result
                }
            }).find(|result| result != &NotUseful).unwrap_or(NotUseful)
        } else {
            let matrix = rows.iter().filter_map(|r| {
                if pat_is_binding_or_wild(&cx.tcx.def_map.borrow(), raw_pat(r[0])) {
                    Some(r[1..].to_vec())
                } else {
                    None
                }
            }).collect();
            match is_useful(cx, &matrix, &v[1..], witness) {
                UsefulWithWitness(pats) => {
                    let mut new_pats: Vec<_> = constructors.into_iter().map(|constructor| {
                        let arity = constructor_arity(cx, &constructor, left_ty);
                        let wild_pats = vec![DUMMY_WILD_PAT; arity];
                        construct_witness(cx, &constructor, wild_pats, left_ty)
                    }).collect();
                    new_pats.extend(pats);
                    UsefulWithWitness(new_pats)
                },
                result => result
            }
        }
    } else {
        constructors.into_iter().map(|c|
            is_useful_specialized(cx, matrix, v, c.clone(), left_ty, witness)
        ).find(|result| result != &NotUseful).unwrap_or(NotUseful)
    }
}

fn is_useful_specialized(cx: &MatchCheckCtxt, &Matrix(ref m): &Matrix,
                         v: &[&Pat], ctor: Constructor, lty: Ty,
                         witness: WitnessPreference) -> Usefulness {
    let arity = constructor_arity(cx, &ctor, lty);
    let matrix = Matrix(m.iter().filter_map(|r| {
        specialize(cx, &r[..], &ctor, 0, arity)
    }).collect());
    match specialize(cx, v, &ctor, 0, arity) {
        Some(v) => is_useful(cx, &matrix, &v[..], witness),
        None => NotUseful
    }
}

/// Determines the constructors that the given pattern can be specialized to.
///
/// In most cases, there's only one constructor that a specific pattern
/// represents, such as a specific enum variant or a specific literal value.
/// Slice patterns, however, can match slices of different lengths. For instance,
/// `[a, b, ..tail]` can match a slice of length 2, 3, 4 and so on.
///
/// On the other hand, a wild pattern and an identifier pattern cannot be
/// specialized in any way.
fn pat_constructors(cx: &MatchCheckCtxt, p: &Pat,
                    left_ty: Ty, max_slice_length: usize) -> Vec<Constructor> {
    let pat = raw_pat(p);
    match pat.node {
        PatKind::Struct(..) | PatKind::TupleStruct(..) | PatKind::Path(..) | PatKind::Ident(..) =>
            match cx.tcx.def_map.borrow().get(&pat.id).unwrap().full_def() {
                Def::Const(..) | Def::AssociatedConst(..) =>
                    span_bug!(pat.span, "const pattern should've \
                                         been rewritten"),
                Def::Struct(..) | Def::TyAlias(..) => vec![Single],
                Def::Variant(_, id) => vec![Variant(id)],
                Def::Local(..) => vec![],
                def => span_bug!(pat.span, "pat_constructors: unexpected \
                                            definition {:?}", def),
            },
        PatKind::QPath(..) =>
            span_bug!(pat.span, "const pattern should've been rewritten"),
        PatKind::Lit(ref expr) =>
            vec!(ConstantValue(eval_const_expr(cx.tcx, &expr))),
        PatKind::Range(ref lo, ref hi) =>
            vec!(ConstantRange(eval_const_expr(cx.tcx, &lo), eval_const_expr(cx.tcx, &hi))),
        PatKind::Vec(ref before, ref slice, ref after) =>
            match left_ty.sty {
                ty::TyArray(_, _) => vec!(Single),
                _                      => if slice.is_some() {
                    (before.len() + after.len()..max_slice_length+1)
                        .map(|length| Slice(length))
                        .collect()
                } else {
                    vec!(Slice(before.len() + after.len()))
                }
            },
        PatKind::Box(_) | PatKind::Tup(_) | PatKind::Ref(..) =>
            vec!(Single),
        PatKind::Wild =>
            vec!(),
    }
}

/// This computes the arity of a constructor. The arity of a constructor
/// is how many subpattern patterns of that constructor should be expanded to.
///
/// For instance, a tuple pattern (_, 42, Some([])) has the arity of 3.
/// A struct pattern's arity is the number of fields it contains, etc.
pub fn constructor_arity(_cx: &MatchCheckCtxt, ctor: &Constructor, ty: Ty) -> usize {
    match ty.sty {
        ty::TyTuple(ref fs) => fs.len(),
        ty::TyBox(_) => 1,
        ty::TyRef(_, ty::TypeAndMut { ty, .. }) => match ty.sty {
            ty::TySlice(_) => match *ctor {
                Slice(length) => length,
                ConstantValue(_) => 0,
                _ => bug!()
            },
            ty::TyStr => 0,
            _ => 1
        },
        ty::TyEnum(adt, _) | ty::TyStruct(adt, _) => {
            ctor.variant_for_adt(adt).fields.len()
        }
        ty::TyArray(_, n) => n,
        _ => 0
    }
}

fn range_covered_by_constructor(ctor: &Constructor,
                                from: &ConstVal, to: &ConstVal) -> Option<bool> {
    let (c_from, c_to) = match *ctor {
        ConstantValue(ref value)        => (value, value),
        ConstantRange(ref from, ref to) => (from, to),
        Single                          => return Some(true),
        _                               => bug!()
    };
    let cmp_from = compare_const_vals(c_from, from);
    let cmp_to = compare_const_vals(c_to, to);
    match (cmp_from, cmp_to) {
        (Some(cmp_from), Some(cmp_to)) => {
            Some(cmp_from != Ordering::Less && cmp_to != Ordering::Greater)
        }
        _ => None
    }
}

/// This is the main specialization step. It expands the first pattern in the given row
/// into `arity` patterns based on the constructor. For most patterns, the step is trivial,
/// for instance tuple patterns are flattened and box patterns expand into their inner pattern.
///
/// OTOH, slice patterns with a subslice pattern (..tail) can be expanded into multiple
/// different patterns.
/// Structure patterns with a partial wild pattern (Foo { a: 42, .. }) have their missing
/// fields filled with wild patterns.
pub fn specialize<'a>(cx: &MatchCheckCtxt, r: &[&'a Pat],
                      constructor: &Constructor, col: usize, arity: usize) -> Option<Vec<&'a Pat>> {
    let &Pat {
        id: pat_id, ref node, span: pat_span
    } = raw_pat(r[col]);
    let head: Option<Vec<&Pat>> = match *node {
        PatKind::Wild =>
            Some(vec![DUMMY_WILD_PAT; arity]),

        PatKind::Path(..) | PatKind::Ident(..) => {
            let def = cx.tcx.def_map.borrow().get(&pat_id).unwrap().full_def();
            match def {
                Def::Const(..) | Def::AssociatedConst(..) =>
                    span_bug!(pat_span, "const pattern should've \
                                         been rewritten"),
                Def::Variant(_, id) if *constructor != Variant(id) => None,
                Def::Variant(..) | Def::Struct(..) => Some(Vec::new()),
                Def::Local(..) => Some(vec![DUMMY_WILD_PAT; arity]),
                _ => span_bug!(pat_span, "specialize: unexpected \
                                          definition {:?}", def),
            }
        }

        PatKind::TupleStruct(_, ref args) => {
            let def = cx.tcx.def_map.borrow().get(&pat_id).unwrap().full_def();
            match def {
                Def::Const(..) | Def::AssociatedConst(..) =>
                    span_bug!(pat_span, "const pattern should've \
                                         been rewritten"),
                Def::Variant(_, id) if *constructor != Variant(id) => None,
                Def::Variant(..) | Def::Struct(..) => {
                    Some(match args {
                        &Some(ref args) => args.iter().map(|p| &**p).collect(),
                        &None => vec![DUMMY_WILD_PAT; arity],
                    })
                }
                _ => None
            }
        }

        PatKind::QPath(_, _) => {
            span_bug!(pat_span, "const pattern should've been rewritten")
        }

        PatKind::Struct(_, ref pattern_fields, _) => {
            let def = cx.tcx.def_map.borrow().get(&pat_id).unwrap().full_def();
            let adt = cx.tcx.node_id_to_type(pat_id).ty_adt_def().unwrap();
            let variant = constructor.variant_for_adt(adt);
            let def_variant = adt.variant_of_def(def);
            if variant.did == def_variant.did {
                Some(variant.fields.iter().map(|sf| {
                    match pattern_fields.iter().find(|f| f.node.name == sf.name) {
                        Some(ref f) => &*f.node.pat,
                        _ => DUMMY_WILD_PAT
                    }
                }).collect())
            } else {
                None
            }
        }

        PatKind::Tup(ref args) =>
            Some(args.iter().map(|p| &**p).collect()),

        PatKind::Box(ref inner) | PatKind::Ref(ref inner, _) =>
            Some(vec![&**inner]),

        PatKind::Lit(ref expr) => {
            let expr_value = eval_const_expr(cx.tcx, &expr);
            match range_covered_by_constructor(constructor, &expr_value, &expr_value) {
                Some(true) => Some(vec![]),
                Some(false) => None,
                None => {
                    span_err!(cx.tcx.sess, pat_span, E0298, "mismatched types between arms");
                    None
                }
            }
        }

        PatKind::Range(ref from, ref to) => {
            let from_value = eval_const_expr(cx.tcx, &from);
            let to_value = eval_const_expr(cx.tcx, &to);
            match range_covered_by_constructor(constructor, &from_value, &to_value) {
                Some(true) => Some(vec![]),
                Some(false) => None,
                None => {
                    span_err!(cx.tcx.sess, pat_span, E0299, "mismatched types between arms");
                    None
                }
            }
        }

        PatKind::Vec(ref before, ref slice, ref after) => {
            match *constructor {
                // Fixed-length vectors.
                Single => {
                    let mut pats: Vec<&Pat> = before.iter().map(|p| &**p).collect();
                    pats.extend(repeat(DUMMY_WILD_PAT).take(arity - before.len() - after.len()));
                    pats.extend(after.iter().map(|p| &**p));
                    Some(pats)
                },
                Slice(length) if before.len() + after.len() <= length && slice.is_some() => {
                    let mut pats: Vec<&Pat> = before.iter().map(|p| &**p).collect();
                    pats.extend(repeat(DUMMY_WILD_PAT).take(arity - before.len() - after.len()));
                    pats.extend(after.iter().map(|p| &**p));
                    Some(pats)
                },
                Slice(length) if before.len() + after.len() == length => {
                    let mut pats: Vec<&Pat> = before.iter().map(|p| &**p).collect();
                    pats.extend(after.iter().map(|p| &**p));
                    Some(pats)
                },
                SliceWithSubslice(prefix, suffix)
                    if before.len() == prefix
                        && after.len() == suffix
                        && slice.is_some() => {
                    let mut pats: Vec<&Pat> = before.iter().map(|p| &**p).collect();
                    pats.extend(after.iter().map(|p| &**p));
                    Some(pats)
                }
                _ => None
            }
        }
    };
    head.map(|mut head| {
        head.extend_from_slice(&r[..col]);
        head.extend_from_slice(&r[col + 1..]);
        head
    })
}

fn check_local(cx: &mut MatchCheckCtxt, loc: &hir::Local) {
    intravisit::walk_local(cx, loc);

    let pat = StaticInliner::new(cx.tcx, None).fold_pat(loc.pat.clone());
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
            fn_id: NodeId) {
    match kind {
        FnKind::Closure(_) => {}
        _ => cx.param_env = ParameterEnvironment::for_item(cx.tcx, fn_id),
    }

    intravisit::walk_fn(cx, kind, decl, body, sp);

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
        span_err!(cx.tcx.sess, pat.span, E0005,
            "refutable pattern in {}: `{}` not covered",
            origin,
            pat_to_string(uncovered_pat),
        );
    });
}

fn is_refutable<A, F>(cx: &MatchCheckCtxt, pat: &Pat, refutable: F) -> Option<A> where
    F: FnOnce(&Pat) -> A,
{
    let pats = Matrix(vec!(vec!(pat)));
    match is_useful(cx, &pats, &[DUMMY_WILD_PAT], ConstructWitness) {
        UsefulWithWitness(pats) => Some(refutable(&pats[0])),
        NotUseful => None,
        Useful => bug!()
    }
}

// Legality of move bindings checking
fn check_legality_of_move_bindings(cx: &MatchCheckCtxt,
                                   has_guard: bool,
                                   pats: &[P<Pat>]) {
    let tcx = cx.tcx;
    let def_map = &tcx.def_map;
    let mut by_ref_span = None;
    for pat in pats {
        pat_bindings(def_map, &pat, |bm, _, span, _path| {
            match bm {
                hir::BindByRef(_) => {
                    by_ref_span = Some(span);
                }
                hir::BindByValue(_) => {
                }
            }
        })
    }

    let check_move = |p: &Pat, sub: Option<&Pat>| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| pat_contains_bindings(&def_map.borrow(), &p)) {
            span_err!(cx.tcx.sess, p.span, E0007, "cannot bind by-move with sub-bindings");
        } else if has_guard {
            span_err!(cx.tcx.sess, p.span, E0008, "cannot bind by-move into a pattern guard");
        } else if by_ref_span.is_some() {
            let mut err = struct_span_err!(cx.tcx.sess, p.span, E0009,
                                           "cannot bind by-move and by-ref in the same pattern");
            span_note!(&mut err, by_ref_span.unwrap(), "by-ref binding occurs here");
            err.emit();
        }
    };

    for pat in pats {
        pat.walk(|p| {
            if pat_is_binding(&def_map.borrow(), &p) {
                match p.node {
                    PatKind::Ident(hir::BindByValue(_), _, ref sub) => {
                        let pat_ty = tcx.node_id_to_type(p.id);
                        //FIXME: (@jroesch) this code should be floated up as well
                        let infcx = infer::new_infer_ctxt(cx.tcx,
                                                          &cx.tcx.tables,
                                                          Some(cx.param_env.clone()),
                                                          ProjectionMode::AnyFinal);
                        if infcx.type_moves_by_default(pat_ty, pat.span) {
                            check_move(p, sub.as_ref().map(|p| &**p));
                        }
                    }
                    PatKind::Ident(hir::BindByRef(_), _, _) => {
                    }
                    _ => {
                        span_bug!(
                            p.span,
                            "binding pattern {} is not an identifier: {:?}",
                            p.id,
                            p.node);
                    }
                }
            }
            true
        });
    }
}

/// Ensures that a pattern guard doesn't borrow by mutable reference or
/// assign.
fn check_for_mutation_in_guard<'a, 'tcx>(cx: &'a MatchCheckCtxt<'a, 'tcx>,
                                         guard: &hir::Expr) {
    let mut checker = MutationChecker {
        cx: cx,
    };

    let infcx = infer::new_infer_ctxt(cx.tcx,
                                      &cx.tcx.tables,
                                      Some(checker.cx.param_env.clone()),
                                      ProjectionMode::AnyFinal);

    let mut visitor = ExprUseVisitor::new(&mut checker, &infcx);
    visitor.walk_expr(guard);
}

struct MutationChecker<'a, 'tcx: 'a> {
    cx: &'a MatchCheckCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> Delegate<'tcx> for MutationChecker<'a, 'tcx> {
    fn matched_pat(&mut self, _: &Pat, _: cmt, _: euv::MatchMode) {}
    fn consume(&mut self, _: NodeId, _: Span, _: cmt, _: ConsumeMode) {}
    fn consume_pat(&mut self, _: &Pat, _: cmt, _: ConsumeMode) {}
    fn borrow(&mut self,
              _: NodeId,
              span: Span,
              _: cmt,
              _: Region,
              kind: BorrowKind,
              _: LoanCause) {
        match kind {
            MutBorrow => {
                span_err!(self.cx.tcx.sess, span, E0301,
                          "cannot mutably borrow in a pattern guard")
            }
            ImmBorrow | UniqueImmBorrow => {}
        }
    }
    fn decl_without_init(&mut self, _: NodeId, _: Span) {}
    fn mutate(&mut self, _: NodeId, span: Span, _: cmt, mode: MutateMode) {
        match mode {
            MutateMode::JustWrite | MutateMode::WriteAndRead => {
                span_err!(self.cx.tcx.sess, span, E0302, "cannot assign in a pattern guard")
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
        if !self.bindings_allowed && pat_is_binding(&self.cx.tcx.def_map.borrow(), pat) {
            span_err!(self.cx.tcx.sess, pat.span, E0303,
                                      "pattern bindings are not allowed \
                                       after an `@`");
        }

        match pat.node {
            PatKind::Ident(_, _, Some(_)) => {
                let bindings_were_allowed = self.bindings_allowed;
                self.bindings_allowed = false;
                intravisit::walk_pat(self, pat);
                self.bindings_allowed = bindings_were_allowed;
            }
            _ => intravisit::walk_pat(self, pat),
        }
    }
}
