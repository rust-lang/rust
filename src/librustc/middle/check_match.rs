// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Constructor::*;
use self::Usefulness::*;
use self::WitnessPreference::*;

use middle::const_eval::{compare_const_vals, const_bool, const_float, const_val};
use middle::const_eval::{const_expr_to_pat, eval_const_expr, lookup_const_by_id};
use middle::def::*;
use middle::expr_use_visitor::{ConsumeMode, Delegate, ExprUseVisitor, Init};
use middle::expr_use_visitor::{JustWrite, LoanCause, MutateMode};
use middle::expr_use_visitor::{WriteAndRead};
use middle::expr_use_visitor as euv;
use middle::mem_categorization::cmt;
use middle::pat_util::*;
use middle::ty::*;
use middle::ty;
use std::cmp::Ordering;
use std::fmt;
use std::iter::{range_inclusive, AdditiveIterator, FromIterator, IntoIterator, repeat};
use std::num::Float;
use std::slice;
use syntax::ast::{self, DUMMY_NODE_ID, NodeId, Pat};
use syntax::ast_util;
use syntax::codemap::{Span, Spanned, DUMMY_SP};
use syntax::fold::{Folder, noop_fold_pat};
use syntax::print::pprust::pat_to_string;
use syntax::parse::token;
use syntax::ptr::P;
use syntax::visit::{self, Visitor, FnKind};
use util::ppaux::ty_to_string;
use util::nodemap::FnvHashMap;

pub const DUMMY_WILD_PAT: &'static Pat = &Pat {
    id: DUMMY_NODE_ID,
    node: ast::PatWild(ast::PatWildSingle),
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
        try!(write!(f, "\n"));

        let &Matrix(ref m) = self;
        let pretty_printed_matrix: Vec<Vec<String>> = m.iter().map(|row| {
            row.iter()
               .map(|&pat| pat_to_string(&*pat))
               .collect::<Vec<String>>()
        }).collect();

        let column_count = m.iter().map(|row| row.len()).max().unwrap_or(0);
        assert!(m.iter().all(|row| row.len() == column_count));
        let column_widths: Vec<uint> = (0..column_count).map(|col| {
            pretty_printed_matrix.iter().map(|row| row[col].len()).max().unwrap_or(0)
        }).collect();

        let total_width = column_widths.iter().cloned().sum() + column_count * 3 + 1;
        let br = repeat('+').take(total_width).collect::<String>();
        try!(write!(f, "{}\n", br));
        for row in pretty_printed_matrix {
            try!(write!(f, "+"));
            for (column, pat_str) in row.into_iter().enumerate() {
                try!(write!(f, " "));
                try!(write!(f, "{:1$}", pat_str, column_widths[column]));
                try!(write!(f, " +"));
            }
            try!(write!(f, "\n"));
            try!(write!(f, "{}\n", br));
        }
        Ok(())
    }
}

impl<'a> FromIterator<Vec<&'a Pat>> for Matrix<'a> {
    fn from_iter<T: IntoIterator<Item=Vec<&'a Pat>>>(iter: T) -> Matrix<'a> {
        Matrix(iter.into_iter().collect())
    }
}

pub struct MatchCheckCtxt<'a, 'tcx: 'a> {
    pub tcx: &'a ty::ctxt<'tcx>,
    pub param_env: ParameterEnvironment<'a, 'tcx>,
}

#[derive(Clone, PartialEq)]
pub enum Constructor {
    /// The constructor of all patterns that don't vary by constructor,
    /// e.g. struct patterns and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(ast::DefId),
    /// Literal values.
    ConstantValue(const_val),
    /// Ranges of literal values (2..5).
    ConstantRange(const_val, const_val),
    /// Array patterns of length n.
    Slice(uint),
    /// Array patterns with a subslice.
    SliceWithSubslice(uint, uint)
}

#[derive(Clone, PartialEq)]
enum Usefulness {
    Useful,
    UsefulWithWitness(Vec<P<Pat>>),
    NotUseful
}

#[derive(Copy)]
enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness
}

impl<'a, 'tcx, 'v> Visitor<'v> for MatchCheckCtxt<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &ast::Expr) {
        check_expr(self, ex);
    }
    fn visit_local(&mut self, l: &ast::Local) {
        check_local(self, l);
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v ast::FnDecl,
                b: &'v ast::Block, s: Span, n: NodeId) {
        check_fn(self, fk, fd, b, s, n);
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    visit::walk_crate(&mut MatchCheckCtxt {
        tcx: tcx,
        param_env: ty::empty_parameter_environment(tcx),
    }, tcx.map.krate());
    tcx.sess.abort_if_errors();
}

fn check_expr(cx: &mut MatchCheckCtxt, ex: &ast::Expr) {
    visit::walk_expr(cx, ex);
    match ex.node {
        ast::ExprMatch(ref scrut, ref arms, source) => {
            for arm in arms {
                // First, check legality of move bindings.
                check_legality_of_move_bindings(cx,
                                                arm.guard.is_some(),
                                                &arm.pats[]);

                // Second, if there is a guard on each arm, make sure it isn't
                // assigning or borrowing anything mutably.
                match arm.guard {
                    Some(ref guard) => check_for_mutation_in_guard(cx, &**guard),
                    None => {}
                }
            }

            let mut static_inliner = StaticInliner::new(cx.tcx, None);
            let inlined_arms = arms.iter().map(|arm| {
                (arm.pats.iter().map(|pat| {
                    static_inliner.fold_pat((*pat).clone())
                }).collect(), arm.guard.as_ref().map(|e| &**e))
            }).collect::<Vec<(Vec<P<Pat>>, Option<&ast::Expr>)>>();

            // Bail out early if inlining failed.
            if static_inliner.failed {
                return;
            }

            for pat in inlined_arms
                .iter()
                .flat_map(|&(ref pats, _)| pats.iter()) {
                // Third, check legality of move bindings.
                check_legality_of_bindings_in_at_patterns(cx, &**pat);

                // Fourth, check if there are any references to NaN that we should warn about.
                check_for_static_nan(cx, &**pat);

                // Fifth, check if for any of the patterns that match an enumerated type
                // are bindings with the same name as one of the variants of said type.
                check_for_bindings_named_the_same_as_variants(cx, &**pat);
            }

            // Fourth, check for unreachable arms.
            check_arms(cx, &inlined_arms[..], source);

            // Finally, check if the whole match expression is exhaustive.
            // Check for empty enum, because is_useful only works on inhabited types.
            let pat_ty = node_id_to_type(cx.tcx, scrut.id);
            if inlined_arms.is_empty() {
                if !type_is_empty(cx.tcx, pat_ty) {
                    // We know the type is inhabited, so this must be wrong
                    span_err!(cx.tcx.sess, ex.span, E0002,
                        "non-exhaustive patterns: type {} is non-empty",
                        ty_to_string(cx.tcx, pat_ty)
                    );
                }
                // If the type *is* empty, it's vacuously exhaustive
                return;
            }

            let matrix: Matrix = inlined_arms
                .iter()
                .filter(|&&(_, guard)| guard.is_none())
                .flat_map(|arm| arm.0.iter())
                .map(|pat| vec![&**pat])
                .collect();
            check_exhaustive(cx, ex.span, &matrix, source);
        },
        _ => ()
    }
}

fn is_expr_const_nan(tcx: &ty::ctxt, expr: &ast::Expr) -> bool {
    match eval_const_expr(tcx, expr) {
        const_float(f) => f.is_nan(),
        _ => false
    }
}

fn check_for_bindings_named_the_same_as_variants(cx: &MatchCheckCtxt, pat: &Pat) {
    ast_util::walk_pat(pat, |p| {
        match p.node {
            ast::PatIdent(ast::BindByValue(ast::MutImmutable), ident, None) => {
                let pat_ty = ty::pat_ty(cx.tcx, p);
                if let ty::ty_enum(def_id, _) = pat_ty.sty {
                    let def = cx.tcx.def_map.borrow().get(&p.id).cloned();
                    if let Some(DefLocal(_)) = def {
                        if ty::enum_variants(cx.tcx, def_id).iter().any(|variant|
                            token::get_name(variant.name) == token::get_name(ident.node.name)
                                && variant.args.len() == 0
                        ) {
                            span_warn!(cx.tcx.sess, p.span, E0170,
                                "pattern binding `{}` is named the same as one \
                                 of the variants of the type `{}`",
                                &token::get_ident(ident.node), ty_to_string(cx.tcx, pat_ty));
                            span_help!(cx.tcx.sess, p.span,
                                "if you meant to match on a variant, \
                                 consider making the path in the pattern qualified: `{}::{}`",
                                ty_to_string(cx.tcx, pat_ty), &token::get_ident(ident.node));
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
    ast_util::walk_pat(pat, |p| {
        match p.node {
            ast::PatLit(ref expr) if is_expr_const_nan(cx.tcx, &**expr) => {
                span_warn!(cx.tcx.sess, p.span, E0003,
                    "unmatchable NaN in pattern, \
                        use the is_nan method in a guard instead");
            }
            _ => ()
        }
        true
    });
}

// Check for unreachable patterns
fn check_arms(cx: &MatchCheckCtxt,
              arms: &[(Vec<P<Pat>>, Option<&ast::Expr>)],
              source: ast::MatchSource) {
    let mut seen = Matrix(vec![]);
    let mut printed_if_let_err = false;
    for &(ref pats, guard) in arms {
        for pat in pats {
            let v = vec![&**pat];

            match is_useful(cx, &seen, &v[..], LeaveOutWitness) {
                NotUseful => {
                    match source {
                        ast::MatchSource::IfLetDesugar { .. } => {
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

                        ast::MatchSource::WhileLetDesugar => {
                            // find the first arm pattern so we can use its span
                            let &(ref first_arm_pats, _) = &arms[0];
                            let first_pat = &first_arm_pats[0];
                            let span = first_pat.span;
                            span_err!(cx.tcx.sess, span, E0165, "irrefutable while-let pattern");
                        },

                        ast::MatchSource::ForLoopDesugar => {
                            // this is a bug, because on `match iter.next()` we cover
                            // `Some(<head>)` and `None`. It's impossible to have an unreachable
                            // pattern
                            // (see libsyntax/ext/expand.rs for the full expansion of a for loop)
                            cx.tcx.sess.span_bug(pat.span, "unreachable for-loop pattern")
                        },

                        ast::MatchSource::Normal => {
                            span_err!(cx.tcx.sess, pat.span, E0001, "unreachable pattern")
                        },
                    }
                }
                Useful => (),
                UsefulWithWitness(_) => unreachable!()
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
        ast::PatIdent(_, _, Some(ref s)) => raw_pat(&**s),
        _ => p
    }
}

fn check_exhaustive(cx: &MatchCheckCtxt, sp: Span, matrix: &Matrix, source: ast::MatchSource) {
    match is_useful(cx, matrix, &[DUMMY_WILD_PAT], ConstructWitness) {
        UsefulWithWitness(pats) => {
            let witness = match &pats[..] {
                [ref witness] => &**witness,
                [] => DUMMY_WILD_PAT,
                _ => unreachable!()
            };
            match source {
                ast::MatchSource::ForLoopDesugar => {
                    // `witness` has the form `Some(<head>)`, peel off the `Some`
                    let witness = match witness.node {
                        ast::PatEnum(_, Some(ref pats)) => match &pats[..] {
                            [ref pat] => &**pat,
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };

                    span_err!(cx.tcx.sess, sp, E0297,
                        "refutable pattern in `for` loop binding: \
                                `{}` not covered",
                                pat_to_string(witness));
                },
                _ => {
                    span_err!(cx.tcx.sess, sp, E0004,
                        "non-exhaustive patterns: `{}` not covered",
                        pat_to_string(witness)
                    );
                },
            }
        }
        NotUseful => {
            // This is good, wildcard pattern isn't reachable
        },
        _ => unreachable!()
    }
}

fn const_val_to_expr(value: &const_val) -> P<ast::Expr> {
    let node = match value {
        &const_bool(b) => ast::LitBool(b),
        _ => unreachable!()
    };
    P(ast::Expr {
        id: 0,
        node: ast::ExprLit(P(Spanned { node: node, span: DUMMY_SP })),
        span: DUMMY_SP
    })
}

pub struct StaticInliner<'a, 'tcx: 'a> {
    pub tcx: &'a ty::ctxt<'tcx>,
    pub failed: bool,
    pub renaming_map: Option<&'a mut FnvHashMap<(NodeId, Span), NodeId>>,
}

impl<'a, 'tcx> StaticInliner<'a, 'tcx> {
    pub fn new<'b>(tcx: &'b ty::ctxt<'tcx>,
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

impl<'map> ast_util::IdVisitingOperation for RenamingRecorder<'map> {
    fn visit_id(&mut self, node_id: NodeId) {
        let key = (node_id, self.origin_span);
        self.renaming_map.insert(key, self.substituted_node_id);
    }
}

impl<'a, 'tcx> Folder for StaticInliner<'a, 'tcx> {
    fn fold_pat(&mut self, pat: P<Pat>) -> P<Pat> {
        return match pat.node {
            ast::PatIdent(..) | ast::PatEnum(..) => {
                let def = self.tcx.def_map.borrow().get(&pat.id).cloned();
                match def {
                    Some(DefConst(did)) => match lookup_const_by_id(self.tcx, did) {
                        Some(const_expr) => {
                            const_expr_to_pat(self.tcx, const_expr, pat.span).map(|new_pat| {

                                if let Some(ref mut renaming_map) = self.renaming_map {
                                    // Record any renamings we do here
                                    record_renamings(const_expr, &pat, renaming_map);
                                }

                                new_pat
                            })
                        }
                        None => {
                            self.failed = true;
                            span_err!(self.tcx.sess, pat.span, E0158,
                                "statics cannot be referenced in patterns");
                            pat
                        }
                    },
                    _ => noop_fold_pat(pat, self)
                }
            }
            _ => noop_fold_pat(pat, self)
        };

        fn record_renamings(const_expr: &ast::Expr,
                            substituted_pat: &ast::Pat,
                            renaming_map: &mut FnvHashMap<(NodeId, Span), NodeId>) {
            let mut renaming_recorder = RenamingRecorder {
                substituted_node_id: substituted_pat.id,
                origin_span: substituted_pat.span,
                renaming_map: renaming_map,
            };

            let mut id_visitor = ast_util::IdVisitor {
                operation: &mut renaming_recorder,
                pass_through_items: true,
                visited_outermost: false,
            };

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
/// left_ty: struct X { a: (bool, &'static str), b: uint}
/// pats: [(false, "foo"), 42]  => X { a: (false, "foo"), b: 42 }
fn construct_witness(cx: &MatchCheckCtxt, ctor: &Constructor,
                     pats: Vec<&Pat>, left_ty: Ty) -> P<Pat> {
    let pats_len = pats.len();
    let mut pats = pats.into_iter().map(|p| P((*p).clone()));
    let pat = match left_ty.sty {
        ty::ty_tup(_) => ast::PatTup(pats.collect()),

        ty::ty_enum(cid, _) | ty::ty_struct(cid, _)  => {
            let (vid, is_structure) = match ctor {
                &Variant(vid) =>
                    (vid, ty::enum_variant_with_id(cx.tcx, cid, vid).arg_names.is_some()),
                _ =>
                    (cid, !ty::is_tuple_struct(cx.tcx, cid))
            };
            if is_structure {
                let fields = ty::lookup_struct_fields(cx.tcx, vid);
                let field_pats: Vec<_> = fields.into_iter()
                    .zip(pats)
                    .filter(|&(_, ref pat)| pat.node != ast::PatWild(ast::PatWildSingle))
                    .map(|(field, pat)| Spanned {
                        span: DUMMY_SP,
                        node: ast::FieldPat {
                            ident: ast::Ident::new(field.name),
                            pat: pat,
                            is_shorthand: false,
                        }
                    }).collect();
                let has_more_fields = field_pats.len() < pats_len;
                ast::PatStruct(def_to_path(cx.tcx, vid), field_pats, has_more_fields)
            } else {
                ast::PatEnum(def_to_path(cx.tcx, vid), Some(pats.collect()))
            }
        }

        ty::ty_rptr(_, ty::mt { ty, mutbl }) => {
            match ty.sty {
               ty::ty_vec(_, Some(n)) => match ctor {
                    &Single => {
                        assert_eq!(pats_len, n);
                        ast::PatVec(pats.collect(), None, vec!())
                    },
                    _ => unreachable!()
                },
                ty::ty_vec(_, None) => match ctor {
                    &Slice(n) => {
                        assert_eq!(pats_len, n);
                        ast::PatVec(pats.collect(), None, vec!())
                    },
                    _ => unreachable!()
                },
                ty::ty_str => ast::PatWild(ast::PatWildSingle),

                _ => {
                    assert_eq!(pats_len, 1);
                    ast::PatRegion(pats.nth(0).unwrap(), mutbl)
                }
            }
        }

        ty::ty_vec(_, Some(len)) => {
            assert_eq!(pats_len, len);
            ast::PatVec(pats.collect(), None, vec![])
        }

        _ => {
            match *ctor {
                ConstantValue(ref v) => ast::PatLit(const_val_to_expr(v)),
                _ => ast::PatWild(ast::PatWildSingle),
            }
        }
    };

    P(ast::Pat {
        id: 0,
        node: pat,
        span: DUMMY_SP
    })
}

fn missing_constructor(cx: &MatchCheckCtxt, &Matrix(ref rows): &Matrix,
                       left_ty: Ty, max_slice_length: uint) -> Option<Constructor> {
    let used_constructors: Vec<Constructor> = rows.iter()
        .flat_map(|row| pat_constructors(cx, row[0], left_ty, max_slice_length).into_iter())
        .collect();
    all_constructors(cx, left_ty, max_slice_length)
        .into_iter()
        .find(|c| !used_constructors.contains(c))
}

/// This determines the set of all possible constructors of a pattern matching
/// values of type `left_ty`. For vectors, this would normally be an infinite set
/// but is instead bounded by the maximum fixed length of slice patterns in
/// the column of patterns being analyzed.
fn all_constructors(cx: &MatchCheckCtxt, left_ty: Ty,
                    max_slice_length: uint) -> Vec<Constructor> {
    match left_ty.sty {
        ty::ty_bool =>
            [true, false].iter().map(|b| ConstantValue(const_bool(*b))).collect(),

        ty::ty_rptr(_, ty::mt { ty, .. }) => match ty.sty {
            ty::ty_vec(_, None) =>
                range_inclusive(0, max_slice_length).map(|length| Slice(length)).collect(),
            _ => vec!(Single)
        },

        ty::ty_enum(eid, _) =>
            ty::enum_variants(cx.tcx, eid)
                .iter()
                .map(|va| Variant(va.id))
                .collect(),

        _ =>
            vec!(Single)
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
    if rows.len() == 0 {
        return match witness {
            ConstructWitness => UsefulWithWitness(vec!()),
            LeaveOutWitness => Useful
        };
    }
    if rows[0].len() == 0 {
        return NotUseful;
    }
    let real_pat = match rows.iter().find(|r| (*r)[0].id != DUMMY_NODE_ID) {
        Some(r) => raw_pat(r[0]),
        None if v.len() == 0 => return NotUseful,
        None => v[0]
    };
    let left_ty = if real_pat.id == DUMMY_NODE_ID {
        ty::mk_nil(cx.tcx)
    } else {
        ty::pat_ty(cx.tcx, &*real_pat)
    };

    let max_slice_length = rows.iter().filter_map(|row| match row[0].node {
        ast::PatVec(ref before, _, ref after) => Some(before.len() + after.len()),
        _ => None
    }).max().map_or(0, |v| v + 1);

    let constructors = pat_constructors(cx, v[0], left_ty, max_slice_length);
    if constructors.is_empty() {
        match missing_constructor(cx, matrix, left_ty, max_slice_length) {
            None => {
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
            },

            Some(constructor) => {
                let matrix = rows.iter().filter_map(|r| {
                    if pat_is_binding_or_wild(&cx.tcx.def_map, raw_pat(r[0])) {
                        Some(r.tail().to_vec())
                    } else {
                        None
                    }
                }).collect();
                match is_useful(cx, &matrix, v.tail(), witness) {
                    UsefulWithWitness(pats) => {
                        let arity = constructor_arity(cx, &constructor, left_ty);
                        let wild_pats: Vec<_> = repeat(DUMMY_WILD_PAT).take(arity).collect();
                        let enum_pat = construct_witness(cx, &constructor, wild_pats, left_ty);
                        let mut new_pats = vec![enum_pat];
                        new_pats.extend(pats.into_iter());
                        UsefulWithWitness(new_pats)
                    },
                    result => result
                }
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
                    left_ty: Ty, max_slice_length: uint) -> Vec<Constructor> {
    let pat = raw_pat(p);
    match pat.node {
        ast::PatIdent(..) =>
            match cx.tcx.def_map.borrow().get(&pat.id) {
                Some(&DefConst(..)) =>
                    cx.tcx.sess.span_bug(pat.span, "const pattern should've \
                                                    been rewritten"),
                Some(&DefStruct(_)) => vec!(Single),
                Some(&DefVariant(_, id, _)) => vec!(Variant(id)),
                _ => vec!()
            },
        ast::PatEnum(..) =>
            match cx.tcx.def_map.borrow().get(&pat.id) {
                Some(&DefConst(..)) =>
                    cx.tcx.sess.span_bug(pat.span, "const pattern should've \
                                                    been rewritten"),
                Some(&DefVariant(_, id, _)) => vec!(Variant(id)),
                _ => vec!(Single)
            },
        ast::PatStruct(..) =>
            match cx.tcx.def_map.borrow().get(&pat.id) {
                Some(&DefConst(..)) =>
                    cx.tcx.sess.span_bug(pat.span, "const pattern should've \
                                                    been rewritten"),
                Some(&DefVariant(_, id, _)) => vec!(Variant(id)),
                _ => vec!(Single)
            },
        ast::PatLit(ref expr) =>
            vec!(ConstantValue(eval_const_expr(cx.tcx, &**expr))),
        ast::PatRange(ref lo, ref hi) =>
            vec!(ConstantRange(eval_const_expr(cx.tcx, &**lo), eval_const_expr(cx.tcx, &**hi))),
        ast::PatVec(ref before, ref slice, ref after) =>
            match left_ty.sty {
                ty::ty_vec(_, Some(_)) => vec!(Single),
                _                      => if slice.is_some() {
                    range_inclusive(before.len() + after.len(), max_slice_length)
                        .map(|length| Slice(length))
                        .collect()
                } else {
                    vec!(Slice(before.len() + after.len()))
                }
            },
        ast::PatBox(_) | ast::PatTup(_) | ast::PatRegion(..) =>
            vec!(Single),
        ast::PatWild(_) =>
            vec!(),
        ast::PatMac(_) =>
            cx.tcx.sess.bug("unexpanded macro")
    }
}

/// This computes the arity of a constructor. The arity of a constructor
/// is how many subpattern patterns of that constructor should be expanded to.
///
/// For instance, a tuple pattern (_, 42, Some([])) has the arity of 3.
/// A struct pattern's arity is the number of fields it contains, etc.
pub fn constructor_arity(cx: &MatchCheckCtxt, ctor: &Constructor, ty: Ty) -> uint {
    match ty.sty {
        ty::ty_tup(ref fs) => fs.len(),
        ty::ty_uniq(_) => 1,
        ty::ty_rptr(_, ty::mt { ty, .. }) => match ty.sty {
            ty::ty_vec(_, None) => match *ctor {
                Slice(length) => length,
                ConstantValue(_) => 0,
                _ => unreachable!()
            },
            ty::ty_str => 0,
            _ => 1
        },
        ty::ty_enum(eid, _) => {
            match *ctor {
                Variant(id) => enum_variant_with_id(cx.tcx, eid, id).args.len(),
                _ => unreachable!()
            }
        }
        ty::ty_struct(cid, _) => ty::lookup_struct_fields(cx.tcx, cid).len(),
        ty::ty_vec(_, Some(n)) => n,
        _ => 0
    }
}

fn range_covered_by_constructor(ctor: &Constructor,
                                from: &const_val, to: &const_val) -> Option<bool> {
    let (c_from, c_to) = match *ctor {
        ConstantValue(ref value)        => (value, value),
        ConstantRange(ref from, ref to) => (from, to),
        Single                          => return Some(true),
        _                               => unreachable!()
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
                      constructor: &Constructor, col: uint, arity: uint) -> Option<Vec<&'a Pat>> {
    let &Pat {
        id: pat_id, ref node, span: pat_span
    } = raw_pat(r[col]);
    let head: Option<Vec<&Pat>> = match *node {
        ast::PatWild(_) =>
            Some(repeat(DUMMY_WILD_PAT).take(arity).collect()),

        ast::PatIdent(_, _, _) => {
            let opt_def = cx.tcx.def_map.borrow().get(&pat_id).cloned();
            match opt_def {
                Some(DefConst(..)) =>
                    cx.tcx.sess.span_bug(pat_span, "const pattern should've \
                                                    been rewritten"),
                Some(DefVariant(_, id, _)) => if *constructor == Variant(id) {
                    Some(vec!())
                } else {
                    None
                },
                _ => Some(repeat(DUMMY_WILD_PAT).take(arity).collect())
            }
        }

        ast::PatEnum(_, ref args) => {
            let def = cx.tcx.def_map.borrow()[pat_id].clone();
            match def {
                DefConst(..) =>
                    cx.tcx.sess.span_bug(pat_span, "const pattern should've \
                                                    been rewritten"),
                DefVariant(_, id, _) if *constructor != Variant(id) => None,
                DefVariant(..) | DefStruct(..) => {
                    Some(match args {
                        &Some(ref args) => args.iter().map(|p| &**p).collect(),
                        &None => repeat(DUMMY_WILD_PAT).take(arity).collect(),
                    })
                }
                _ => None
            }
        }

        ast::PatStruct(_, ref pattern_fields, _) => {
            // Is this a struct or an enum variant?
            let def = cx.tcx.def_map.borrow()[pat_id].clone();
            let class_id = match def {
                DefConst(..) =>
                    cx.tcx.sess.span_bug(pat_span, "const pattern should've \
                                                    been rewritten"),
                DefVariant(_, variant_id, _) => if *constructor == Variant(variant_id) {
                    Some(variant_id)
                } else {
                    None
                },
                _ => {
                    // Assume this is a struct.
                    match ty::ty_to_def_id(node_id_to_type(cx.tcx, pat_id)) {
                        None => {
                            cx.tcx.sess.span_bug(pat_span,
                                                 "struct pattern wasn't of a \
                                                  type with a def ID?!")
                        }
                        Some(def_id) => Some(def_id),
                    }
                }
            };
            class_id.map(|variant_id| {
                let struct_fields = ty::lookup_struct_fields(cx.tcx, variant_id);
                let args = struct_fields.iter().map(|sf| {
                    match pattern_fields.iter().find(|f| f.node.ident.name == sf.name) {
                        Some(ref f) => &*f.node.pat,
                        _ => DUMMY_WILD_PAT
                    }
                }).collect();
                args
            })
        }

        ast::PatTup(ref args) =>
            Some(args.iter().map(|p| &**p).collect()),

        ast::PatBox(ref inner) | ast::PatRegion(ref inner, _) =>
            Some(vec![&**inner]),

        ast::PatLit(ref expr) => {
            let expr_value = eval_const_expr(cx.tcx, &**expr);
            match range_covered_by_constructor(constructor, &expr_value, &expr_value) {
                Some(true) => Some(vec![]),
                Some(false) => None,
                None => {
                    span_err!(cx.tcx.sess, pat_span, E0298, "mismatched types between arms");
                    None
                }
            }
        }

        ast::PatRange(ref from, ref to) => {
            let from_value = eval_const_expr(cx.tcx, &**from);
            let to_value = eval_const_expr(cx.tcx, &**to);
            match range_covered_by_constructor(constructor, &from_value, &to_value) {
                Some(true) => Some(vec![]),
                Some(false) => None,
                None => {
                    span_err!(cx.tcx.sess, pat_span, E0299, "mismatched types between arms");
                    None
                }
            }
        }

        ast::PatVec(ref before, ref slice, ref after) => {
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

        ast::PatMac(_) => {
            span_err!(cx.tcx.sess, pat_span, E0300, "unexpanded macro");
            None
        }
    };
    head.map(|mut head| {
        head.push_all(&r[..col]);
        head.push_all(&r[col + 1..]);
        head
    })
}

fn check_local(cx: &mut MatchCheckCtxt, loc: &ast::Local) {
    visit::walk_local(cx, loc);

    let name = match loc.source {
        ast::LocalLet => "local",
        ast::LocalFor => "`for` loop"
    };

    let mut static_inliner = StaticInliner::new(cx.tcx, None);
    is_refutable(cx, &*static_inliner.fold_pat(loc.pat.clone()), |pat| {
        span_err!(cx.tcx.sess, loc.pat.span, E0005,
            "refutable pattern in {} binding: `{}` not covered",
            name, pat_to_string(pat)
        );
    });

    // Check legality of move bindings and `@` patterns.
    check_legality_of_move_bindings(cx, false, slice::ref_slice(&loc.pat));
    check_legality_of_bindings_in_at_patterns(cx, &*loc.pat);
}

fn check_fn(cx: &mut MatchCheckCtxt,
            kind: FnKind,
            decl: &ast::FnDecl,
            body: &ast::Block,
            sp: Span,
            fn_id: NodeId) {
    match kind {
        visit::FkFnBlock => {}
        _ => cx.param_env = ParameterEnvironment::for_item(cx.tcx, fn_id),
    }

    visit::walk_fn(cx, kind, decl, body, sp);

    for input in &decl.inputs {
        is_refutable(cx, &*input.pat, |pat| {
            span_err!(cx.tcx.sess, input.pat.span, E0006,
                "refutable pattern in function argument: `{}` not covered",
                pat_to_string(pat)
            );
        });
        check_legality_of_move_bindings(cx, false, slice::ref_slice(&input.pat));
        check_legality_of_bindings_in_at_patterns(cx, &*input.pat);
    }
}

fn is_refutable<A, F>(cx: &MatchCheckCtxt, pat: &Pat, refutable: F) -> Option<A> where
    F: FnOnce(&Pat) -> A,
{
    let pats = Matrix(vec!(vec!(pat)));
    match is_useful(cx, &pats, &[DUMMY_WILD_PAT], ConstructWitness) {
        UsefulWithWitness(pats) => {
            assert_eq!(pats.len(), 1);
            Some(refutable(&*pats[0]))
        },
        NotUseful => None,
        Useful => unreachable!()
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
        pat_bindings(def_map, &**pat, |bm, _, span, _path| {
            match bm {
                ast::BindByRef(_) => {
                    by_ref_span = Some(span);
                }
                ast::BindByValue(_) => {
                }
            }
        })
    }

    let check_move = |p: &Pat, sub: Option<&Pat>| {
        // check legality of moving out of the enum

        // x @ Foo(..) is legal, but x @ Foo(y) isn't.
        if sub.map_or(false, |p| pat_contains_bindings(def_map, &*p)) {
            span_err!(cx.tcx.sess, p.span, E0007, "cannot bind by-move with sub-bindings");
        } else if has_guard {
            span_err!(cx.tcx.sess, p.span, E0008, "cannot bind by-move into a pattern guard");
        } else if by_ref_span.is_some() {
            span_err!(cx.tcx.sess, p.span, E0009,
                "cannot bind by-move and by-ref in the same pattern");
            span_note!(cx.tcx.sess, by_ref_span.unwrap(), "by-ref binding occurs here");
        }
    };

    for pat in pats {
        ast_util::walk_pat(&**pat, |p| {
            if pat_is_binding(def_map, &*p) {
                match p.node {
                    ast::PatIdent(ast::BindByValue(_), _, ref sub) => {
                        let pat_ty = ty::node_id_to_type(tcx, p.id);
                        if ty::type_moves_by_default(&cx.param_env, pat.span, pat_ty) {
                            check_move(p, sub.as_ref().map(|p| &**p));
                        }
                    }
                    ast::PatIdent(ast::BindByRef(_), _, _) => {
                    }
                    _ => {
                        cx.tcx.sess.span_bug(
                            p.span,
                            &format!("binding pattern {} is not an \
                                     identifier: {:?}",
                                    p.id,
                                    p.node)[]);
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
                                         guard: &ast::Expr) {
    let mut checker = MutationChecker {
        cx: cx,
    };
    let mut visitor = ExprUseVisitor::new(&mut checker,
                                          &checker.cx.param_env);
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
            JustWrite | WriteAndRead => {
                span_err!(self.cx.tcx.sess, span, E0302, "cannot assign in a pattern guard")
            }
            Init => {}
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
        if !self.bindings_allowed && pat_is_binding(&self.cx.tcx.def_map, pat) {
            span_err!(self.cx.tcx.sess, pat.span, E0303,
                                      "pattern bindings are not allowed \
                                       after an `@`");
        }

        match pat.node {
            ast::PatIdent(_, _, Some(_)) => {
                let bindings_were_allowed = self.bindings_allowed;
                self.bindings_allowed = false;
                visit::walk_pat(self, pat);
                self.bindings_allowed = bindings_were_allowed;
            }
            _ => visit::walk_pat(self, pat),
        }
    }
}
