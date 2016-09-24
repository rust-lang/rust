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

use rustc::middle::const_val::ConstVal;
use eval::{eval_const_expr, compare_const_vals};

use rustc::hir::def::*;
use rustc::hir::def_id::{DefId};
use rustc::hir::pat_util::def_to_path;
use rustc::ty::{self, Ty, TyCtxt};

use std::cmp::Ordering;
use std::fmt;
use std::iter::{FromIterator, IntoIterator, repeat};

use rustc::hir;
use rustc::hir::{Pat, PatKind};
use rustc::hir::print::pat_to_string;
use rustc::util::common::ErrorReported;

use syntax::ast::{self, DUMMY_NODE_ID};
use syntax::codemap::Spanned;
use syntax::ptr::P;
use syntax_pos::{Span, DUMMY_SP};

pub const DUMMY_WILD_PAT: &'static Pat = &Pat {
    id: DUMMY_NODE_ID,
    node: PatKind::Wild,
    span: DUMMY_SP
};

pub const DUMMY_WILD_PATTERN : Pattern<'static, 'static> = Pattern {
    pat: DUMMY_WILD_PAT,
    pattern_ty: None
};

#[derive(Copy, Clone)]
pub struct Pattern<'a, 'tcx> {
    pat: &'a Pat,
    pattern_ty: Option<Ty<'tcx>>
}

impl<'a, 'tcx> Pattern<'a, 'tcx> {
    fn as_raw(self) -> &'a Pat {
        let mut pat = self.pat;

        while let PatKind::Binding(.., Some(ref s)) = pat.node {
            pat = s;
        }

        return pat;
    }

    pub fn span(self) -> Span {
        self.pat.span
    }
}

impl<'a, 'tcx> fmt::Debug for Pattern<'a, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {:?}", pat_to_string(self.pat), self.pattern_ty)
    }
}

pub struct Matrix<'a, 'tcx>(Vec<Vec<Pattern<'a, 'tcx>>>);

impl<'a, 'tcx> Matrix<'a, 'tcx> {
    pub fn empty() -> Self {
        Matrix(vec![])
    }

    pub fn push(&mut self, row: Vec<Pattern<'a, 'tcx>>) {
        self.0.push(row)
    }
}

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
impl<'a, 'tcx> fmt::Debug for Matrix<'a, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n")?;

        let &Matrix(ref m) = self;
        let pretty_printed_matrix: Vec<Vec<String>> = m.iter().map(|row| {
            row.iter().map(|pat| format!("{:?}", pat)).collect()
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

impl<'a, 'tcx> FromIterator<Vec<Pattern<'a, 'tcx>>> for Matrix<'a, 'tcx> {
    fn from_iter<T: IntoIterator<Item=Vec<Pattern<'a, 'tcx>>>>(iter: T) -> Self
    {
        Matrix(iter.into_iter().collect())
    }
}

//NOTE: appears to be the only place other then InferCtxt to contain a ParamEnv
pub struct MatchCheckCtxt<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub param_env: ty::ParameterEnvironment<'tcx>,
}

#[derive(Clone, Debug, PartialEq)]
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
}

impl Constructor {
    fn variant_for_adt<'tcx, 'container, 'a>(&self,
                                             adt: &'a ty::AdtDefData<'tcx, 'container>)
                                             -> &'a ty::VariantDefData<'tcx, 'container> {
        match self {
            &Variant(vid) => adt.variant_with_id(vid),
            _ => adt.struct_variant()
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Usefulness {
    Useful,
    UsefulWithWitness(Vec<P<Pat>>),
    NotUseful
}

#[derive(Copy, Clone)]
pub enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness
}

fn const_val_to_expr(value: &ConstVal) -> P<hir::Expr> {
    let node = match value {
        &ConstVal::Bool(b) => ast::LitKind::Bool(b),
        _ => bug!()
    };
    P(hir::Expr {
        id: DUMMY_NODE_ID,
        node: hir::ExprLit(P(Spanned { node: node, span: DUMMY_SP })),
        span: DUMMY_SP,
        attrs: ast::ThinVec::new(),
    })
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
        ty::TyTuple(..) => PatKind::Tuple(pats.collect(), None),

        ty::TyAdt(adt, _) => {
            let v = ctor.variant_for_adt(adt);
            match v.ctor_kind {
                CtorKind::Fictive => {
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
                CtorKind::Fn => {
                    PatKind::TupleStruct(def_to_path(cx.tcx, v.did), pats.collect(), None)
                }
                CtorKind::Const => {
                    PatKind::Path(None, def_to_path(cx.tcx, v.did))
                }
            }
        }

        ty::TyRef(_, ty::TypeAndMut { mutbl, .. }) => {
            assert_eq!(pats_len, 1);
            PatKind::Ref(pats.nth(0).unwrap(), mutbl)
        }

        ty::TySlice(_) => match ctor {
            &Slice(n) => {
                assert_eq!(pats_len, n);
                PatKind::Slice(pats.collect(), None, hir::HirVec::new())
            },
            _ => unreachable!()
        },

        ty::TyArray(_, len) => {
            assert_eq!(pats_len, len);
            PatKind::Slice(pats.collect(), None, hir::HirVec::new())
        }

        _ => {
            match *ctor {
                ConstantValue(ref v) => PatKind::Lit(const_val_to_expr(v)),
                _ => PatKind::Wild,
            }
        }
    };

    P(hir::Pat {
        id: DUMMY_NODE_ID,
        node: pat,
        span: DUMMY_SP
    })
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
        ty::TySlice(_) =>
            (0..max_slice_length+1).map(|length| Slice(length)).collect(),
        ty::TyAdt(def, _) if def.is_enum() =>
            def.variants.iter().map(|v| Variant(v.did)).collect(),
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
pub fn is_useful<'a, 'tcx>(cx: &MatchCheckCtxt<'a, 'tcx>,
                           matrix: &Matrix<'a, 'tcx>,
                           v: &[Pattern<'a, 'tcx>],
                           witness: WitnessPreference)
                           -> Usefulness {
    let &Matrix(ref rows) = matrix;
    debug!("is_useful({:?}, {:?})", matrix, v);
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
    let left_ty = match rows.iter().filter_map(|r| r[0].pattern_ty).next()
        .or_else(|| v[0].pattern_ty)
    {
        Some(ty) => ty,
        None => {
            // all patterns are wildcards - we can pick any type we want
            cx.tcx.types.bool
        }
    };

    let max_slice_length = rows.iter().filter_map(|row| match row[0].pat.node {
        PatKind::Slice(ref before, _, ref after) => Some(before.len() + after.len()),
        _ => None
    }).max().map_or(0, |v| v + 1);

    let constructors = pat_constructors(cx, v[0], left_ty, max_slice_length);
    debug!("is_useful - pat_constructors = {:?} left_ty = {:?}", constructors,
           left_ty);
    if constructors.is_empty() {
        let constructors = missing_constructors(cx, matrix, left_ty, max_slice_length);
        debug!("is_useful - missing_constructors = {:?}", constructors);
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
                match r[0].as_raw().node {
                    PatKind::Binding(..) | PatKind::Wild => Some(r[1..].to_vec()),
                    _ => None,
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

fn is_useful_specialized<'a, 'tcx>(
    cx: &MatchCheckCtxt<'a, 'tcx>,
    &Matrix(ref m): &Matrix<'a, 'tcx>,
    v: &[Pattern<'a, 'tcx>],
    ctor: Constructor,
    lty: Ty<'tcx>,
    witness: WitnessPreference) -> Usefulness
{
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
fn pat_constructors(cx: &MatchCheckCtxt, p: Pattern,
                    left_ty: Ty, max_slice_length: usize) -> Vec<Constructor> {
    let pat = p.as_raw();
    match pat.node {
        PatKind::Struct(..) | PatKind::TupleStruct(..) | PatKind::Path(..) =>
            match cx.tcx.expect_def(pat.id) {
                Def::Variant(id) | Def::VariantCtor(id, _) => vec![Variant(id)],
                Def::Struct(..) | Def::StructCtor(..) | Def::Union(..) |
                Def::TyAlias(..) | Def::AssociatedTy(..) => vec![Single],
                Def::Const(..) | Def::AssociatedConst(..) =>
                    span_bug!(p.span(), "const pattern should've been rewritten"),
                def => span_bug!(p.span(), "pat_constructors: unexpected definition {:?}", def),
            },
        PatKind::Lit(ref expr) =>
            vec![ConstantValue(eval_const_expr(cx.tcx, &expr))],
        PatKind::Range(ref lo, ref hi) =>
            vec![ConstantRange(eval_const_expr(cx.tcx, &lo), eval_const_expr(cx.tcx, &hi))],
        PatKind::Slice(ref before, ref slice, ref after) =>
            match left_ty.sty {
                ty::TyArray(..) => vec![Single],
                ty::TySlice(_) if slice.is_some() => {
                    (before.len() + after.len()..max_slice_length+1)
                        .map(|length| Slice(length))
                        .collect()
                }
                ty::TySlice(_) => vec!(Slice(before.len() + after.len())),
                _ => span_bug!(pat.span, "pat_constructors: unexpected \
                                          slice pattern type {:?}", left_ty)
            },
        PatKind::Box(..) | PatKind::Tuple(..) | PatKind::Ref(..) =>
            vec![Single],
        PatKind::Binding(..) | PatKind::Wild =>
            vec![],
    }
}

/// This computes the arity of a constructor. The arity of a constructor
/// is how many subpattern patterns of that constructor should be expanded to.
///
/// For instance, a tuple pattern (_, 42, Some([])) has the arity of 3.
/// A struct pattern's arity is the number of fields it contains, etc.
pub fn constructor_arity(_cx: &MatchCheckCtxt, ctor: &Constructor, ty: Ty) -> usize {
    debug!("constructor_arity({:?}, {:?})", ctor, ty);
    match ty.sty {
        ty::TyTuple(ref fs) => fs.len(),
        ty::TyBox(_) => 1,
        ty::TySlice(_) => match *ctor {
            Slice(length) => length,
            ConstantValue(_) => {
                // TODO: this is utterly wrong, but required for byte arrays
                0
            }
            _ => bug!("bad slice pattern {:?} {:?}", ctor, ty)
        },
        ty::TyRef(..) => 1,
        ty::TyAdt(adt, _) => {
            ctor.variant_for_adt(adt).fields.len()
        }
        ty::TyArray(_, n) => n,
        _ => 0
    }
}

fn range_covered_by_constructor(tcx: TyCtxt, span: Span,
                                ctor: &Constructor,
                                from: &ConstVal, to: &ConstVal)
                                -> Result<bool, ErrorReported> {
    let (c_from, c_to) = match *ctor {
        ConstantValue(ref value)        => (value, value),
        ConstantRange(ref from, ref to) => (from, to),
        Single                          => return Ok(true),
        _                               => bug!()
    };
    let cmp_from = compare_const_vals(tcx, span, c_from, from)?;
    let cmp_to = compare_const_vals(tcx, span, c_to, to)?;
    Ok(cmp_from != Ordering::Less && cmp_to != Ordering::Greater)
}

pub fn wrap_pat<'a, 'b, 'tcx>(cx: &MatchCheckCtxt<'b, 'tcx>,
                              pat: &'a Pat)
                              -> Pattern<'a, 'tcx>
{
    let pat_ty = cx.tcx.pat_ty(pat);
    Pattern {
        pat: pat,
        pattern_ty: Some(match pat.node {
            PatKind::Binding(hir::BindByRef(..), ..) => {
                pat_ty.builtin_deref(false, ty::NoPreference).unwrap().ty
            }
            _ => pat_ty
        })
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
fn specialize<'a, 'b, 'tcx>(
    cx: &MatchCheckCtxt<'b, 'tcx>,
    r: &[Pattern<'a, 'tcx>],
    constructor: &Constructor, col: usize, arity: usize)
    -> Option<Vec<Pattern<'a, 'tcx>>>
{
    let pat = r[col].as_raw();
    let &Pat {
        id: pat_id, ref node, span: pat_span
    } = pat;
    let wpat = |pat: &'a Pat| wrap_pat(cx, pat);

    let head: Option<Vec<Pattern>> = match *node {
        PatKind::Binding(..) | PatKind::Wild =>
            Some(vec![DUMMY_WILD_PATTERN; arity]),

        PatKind::Path(..) => {
            match cx.tcx.expect_def(pat_id) {
                Def::Const(..) | Def::AssociatedConst(..) =>
                    span_bug!(pat_span, "const pattern should've \
                                         been rewritten"),
                Def::VariantCtor(id, CtorKind::Const) if *constructor != Variant(id) => None,
                Def::VariantCtor(_, CtorKind::Const) |
                Def::StructCtor(_, CtorKind::Const) => Some(Vec::new()),
                def => span_bug!(pat_span, "specialize: unexpected \
                                          definition {:?}", def),
            }
        }

        PatKind::TupleStruct(_, ref args, ddpos) => {
            match cx.tcx.expect_def(pat_id) {
                Def::Const(..) | Def::AssociatedConst(..) =>
                    span_bug!(pat_span, "const pattern should've \
                                         been rewritten"),
                Def::VariantCtor(id, CtorKind::Fn) if *constructor != Variant(id) => None,
                Def::VariantCtor(_, CtorKind::Fn) |
                Def::StructCtor(_, CtorKind::Fn) => {
                    match ddpos {
                        Some(ddpos) => {
                            let mut pats: Vec<_> = args[..ddpos].iter().map(|p| {
                                wpat(p)
                            }).collect();
                            pats.extend(repeat(DUMMY_WILD_PATTERN).take(arity - args.len()));
                            pats.extend(args[ddpos..].iter().map(|p| wpat(p)));
                            Some(pats)
                        }
                        None => Some(args.iter().map(|p| wpat(p)).collect())
                    }
                }
                def => span_bug!(pat_span, "specialize: unexpected definition: {:?}", def)
            }
        }

        PatKind::Struct(_, ref pattern_fields, _) => {
            let adt = cx.tcx.node_id_to_type(pat_id).ty_adt_def().unwrap();
            let variant = constructor.variant_for_adt(adt);
            let def_variant = adt.variant_of_def(cx.tcx.expect_def(pat_id));
            if variant.did == def_variant.did {
                Some(variant.fields.iter().map(|sf| {
                    match pattern_fields.iter().find(|f| f.node.name == sf.name) {
                        Some(ref f) => wpat(&f.node.pat),
                        _ => DUMMY_WILD_PATTERN
                    }
                }).collect())
            } else {
                None
            }
        }

        PatKind::Tuple(ref args, Some(ddpos)) => {
            let mut pats: Vec<_> = args[..ddpos].iter().map(|p| wpat(p)).collect();
            pats.extend(repeat(DUMMY_WILD_PATTERN).take(arity - args.len()));
            pats.extend(args[ddpos..].iter().map(|p| wpat(p)));
            Some(pats)
        }
        PatKind::Tuple(ref args, None) =>
            Some(args.iter().map(|p| wpat(&**p)).collect()),

        PatKind::Box(ref inner) | PatKind::Ref(ref inner, _) =>
            Some(vec![wpat(&**inner)]),

        PatKind::Lit(ref expr) => {
            match r[col].pattern_ty {
                Some(&ty::TyS { sty: ty::TyRef(_, mt), .. }) => {
                    // HACK: handle string literals. A string literal pattern
                    // serves both as an unary reference pattern and as a
                    // nullary value pattern, depending on the type.
                    Some(vec![Pattern {
                        pat: pat,
                        pattern_ty: Some(mt.ty)
                    }])
                }
                Some(ty) => {
                    assert_eq!(constructor_arity(cx, constructor, ty), 0);
                    let expr_value = eval_const_expr(cx.tcx, &expr);
                    match range_covered_by_constructor(
                        cx.tcx, expr.span, constructor, &expr_value, &expr_value
                            ) {
                        Ok(true) => Some(vec![]),
                        Ok(false) => None,
                        Err(ErrorReported) => None,
                    }
                }
                None => span_bug!(pat.span, "literal pattern {:?} has no type", pat)
            }
        }

        PatKind::Range(ref from, ref to) => {
            let from_value = eval_const_expr(cx.tcx, &from);
            let to_value = eval_const_expr(cx.tcx, &to);
            match range_covered_by_constructor(
                cx.tcx, pat_span, constructor, &from_value, &to_value
            ) {
                Ok(true) => Some(vec![]),
                Ok(false) => None,
                Err(ErrorReported) => None,
            }
        }

        PatKind::Slice(ref before, ref slice, ref after) => {
            let pat_len = before.len() + after.len();
            match *constructor {
                Single => {
                    // Fixed-length vectors.
                    Some(
                        before.iter().map(|p| wpat(p)).chain(
                        repeat(DUMMY_WILD_PATTERN).take(arity - pat_len).chain(
                        after.iter().map(|p| wpat(p))
                    )).collect())
                },
                Slice(length) if pat_len <= length && slice.is_some() => {
                    Some(
                        before.iter().map(|p| wpat(p)).chain(
                        repeat(DUMMY_WILD_PATTERN).take(arity - pat_len).chain(
                        after.iter().map(|p| wpat(p))
                    )).collect())
                }
                Slice(length) if pat_len == length => {
                    Some(
                        before.iter().map(|p| wpat(p)).chain(
                        after.iter().map(|p| wpat(p))
                    ).collect())
                }
                _ => None
            }
        }
    };
    debug!("specialize({:?}, {:?}) = {:?}", r[col], arity, head);

    head.map(|mut head| {
        head.extend_from_slice(&r[..col]);
        head.extend_from_slice(&r[col + 1..]);
        head
    })
}

pub fn is_refutable<A, F>(cx: &MatchCheckCtxt, pat: &Pat, refutable: F)
                          -> Option<A> where
    F: FnOnce(&Pat) -> A,
{
    let pats = Matrix(vec!(vec!(wrap_pat(cx, pat))));
    match is_useful(cx, &pats, &[DUMMY_WILD_PATTERN], ConstructWitness) {
        UsefulWithWitness(pats) => Some(refutable(&pats[0])),
        NotUseful => None,
        Useful => bug!()
    }
}
