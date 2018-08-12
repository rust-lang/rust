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

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;

use super::{FieldPattern, Pattern, PatternKind};
use super::{PatternFoldable, PatternFolder, compare_const_vals};

use rustc::hir::def_id::DefId;
use rustc::hir::RangeEnd;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc::ty::layout::{Integer, IntegerExt};

use rustc::mir::Field;
use rustc::mir::interpret::ConstValue;
use rustc::util::common::ErrorReported;

use syntax::attr::{SignedInt, UnsignedInt};
use syntax_pos::{Span, DUMMY_SP};

use arena::TypedArena;

use std::cmp::{self, Ordering};
use std::fmt;
use std::iter::{FromIterator, IntoIterator};
use std::ops::RangeInclusive;

pub fn expand_pattern<'a, 'tcx>(cx: &MatchCheckCtxt<'a, 'tcx>, pat: Pattern<'tcx>)
                                -> &'a Pattern<'tcx>
{
    cx.pattern_arena.alloc(LiteralExpander.fold_pattern(&pat))
}

struct LiteralExpander;
impl<'tcx> PatternFolder<'tcx> for LiteralExpander {
    fn fold_pattern(&mut self, pat: &Pattern<'tcx>) -> Pattern<'tcx> {
        match (&pat.ty.sty, &*pat.kind) {
            (&ty::TyRef(_, rty, _), &PatternKind::Constant { ref value }) => {
                Pattern {
                    ty: pat.ty,
                    span: pat.span,
                    kind: box PatternKind::Deref {
                        subpattern: Pattern {
                            ty: rty,
                            span: pat.span,
                            kind: box PatternKind::Constant { value: value.clone() },
                        }
                    }
                }
            }
            (_, &PatternKind::Binding { subpattern: Some(ref s), .. }) => {
                s.fold_with(self)
            }
            _ => pat.super_fold_with(self)
        }
    }
}

impl<'tcx> Pattern<'tcx> {
    fn is_wildcard(&self) -> bool {
        match *self.kind {
            PatternKind::Binding { subpattern: None, .. } | PatternKind::Wild =>
                true,
            _ => false
        }
    }
}

pub struct Matrix<'a, 'tcx: 'a>(Vec<Vec<&'a Pattern<'tcx>>>);

impl<'a, 'tcx> Matrix<'a, 'tcx> {
    pub fn empty() -> Self {
        Matrix(vec![])
    }

    pub fn push(&mut self, row: Vec<&'a Pattern<'tcx>>) {
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
        let br = "+".repeat(total_width);
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

impl<'a, 'tcx> FromIterator<Vec<&'a Pattern<'tcx>>> for Matrix<'a, 'tcx> {
    fn from_iter<T: IntoIterator<Item=Vec<&'a Pattern<'tcx>>>>(iter: T) -> Self
    {
        Matrix(iter.into_iter().collect())
    }
}

pub struct MatchCheckCtxt<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    /// The module in which the match occurs. This is necessary for
    /// checking inhabited-ness of types because whether a type is (visibly)
    /// inhabited can depend on whether it was defined in the current module or
    /// not. eg. `struct Foo { _private: ! }` cannot be seen to be empty
    /// outside it's module and should not be matchable with an empty match
    /// statement.
    pub module: DefId,
    pub pattern_arena: &'a TypedArena<Pattern<'tcx>>,
    pub byte_array_map: FxHashMap<*const Pattern<'tcx>, Vec<&'a Pattern<'tcx>>>,
}

impl<'a, 'tcx> MatchCheckCtxt<'a, 'tcx> {
    pub fn create_and_enter<F, R>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        module: DefId,
        f: F) -> R
        where F: for<'b> FnOnce(MatchCheckCtxt<'b, 'tcx>) -> R
    {
        let pattern_arena = TypedArena::new();

        f(MatchCheckCtxt {
            tcx,
            module,
            pattern_arena: &pattern_arena,
            byte_array_map: FxHashMap(),
        })
    }

    // convert a byte-string pattern to a list of u8 patterns.
    fn lower_byte_str_pattern<'p>(&mut self, pat: &'p Pattern<'tcx>) -> Vec<&'p Pattern<'tcx>>
            where 'a: 'p
    {
        let pattern_arena = &*self.pattern_arena;
        let tcx = self.tcx;
        self.byte_array_map.entry(pat).or_insert_with(|| {
            match pat.kind {
                box PatternKind::Constant {
                    value: const_val
                } => {
                    if let Some(ptr) = const_val.to_ptr() {
                        let is_array_ptr = const_val.ty
                            .builtin_deref(true)
                            .and_then(|t| t.ty.builtin_index())
                            .map_or(false, |t| t == tcx.types.u8);
                        assert!(is_array_ptr);
                        let alloc = tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id);
                        assert_eq!(ptr.offset.bytes(), 0);
                        // FIXME: check length
                        alloc.bytes.iter().map(|b| {
                            &*pattern_arena.alloc(Pattern {
                                ty: tcx.types.u8,
                                span: pat.span,
                                kind: box PatternKind::Constant {
                                    value: ty::Const::from_bits(
                                        tcx,
                                        *b as u128,
                                        ty::ParamEnv::empty().and(tcx.types.u8))
                                }
                            })
                        }).collect()
                    } else {
                        bug!("not a byte str: {:?}", const_val)
                    }
                }
                _ => span_bug!(pat.span, "unexpected byte array pattern {:?}", pat)
            }
        }).clone()
    }

    fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        if self.tcx.features().exhaustive_patterns {
            self.tcx.is_ty_uninhabited_from(self.module, ty)
        } else {
            false
        }
    }

    fn is_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyAdt(adt_def, ..) => adt_def.is_enum() && adt_def.is_non_exhaustive(),
            _ => false,
        }
    }

    fn is_local(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::TyAdt(adt_def, ..) => adt_def.did.is_local(),
            _ => false,
        }
    }

    fn is_variant_uninhabited(&self,
                              variant: &'tcx ty::VariantDef,
                              substs: &'tcx ty::subst::Substs<'tcx>)
                              -> bool
    {
        if self.tcx.features().exhaustive_patterns {
            self.tcx.is_enum_variant_uninhabited_from(self.module, variant, substs)
        } else {
            false
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constructor<'tcx> {
    /// The constructor of all patterns that don't vary by constructor,
    /// e.g. struct patterns and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(DefId),
    /// Literal values.
    ConstantValue(&'tcx ty::Const<'tcx>),
    /// Ranges of literal values (`2...5` and `2..5`).
    ConstantRange(&'tcx ty::Const<'tcx>, &'tcx ty::Const<'tcx>, RangeEnd),
    /// Array patterns of length n.
    Slice(u64),
}

impl<'tcx> Constructor<'tcx> {
    fn variant_index_for_adt(&self, adt: &'tcx ty::AdtDef) -> usize {
        match self {
            &Variant(vid) => adt.variant_index_with_id(vid),
            &Single => {
                assert!(!adt.is_enum());
                0
            }
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt)
        }
    }
}

#[derive(Clone, Debug)]
pub enum Usefulness<'tcx> {
    Useful,
    UsefulWithWitness(Vec<Witness<'tcx>>),
    NotUseful
}

impl<'tcx> Usefulness<'tcx> {
    fn is_useful(&self) -> bool {
        match *self {
            NotUseful => false,
            _ => true
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness
}

#[derive(Copy, Clone, Debug)]
struct PatternContext<'tcx> {
    ty: Ty<'tcx>,
    max_slice_length: u64,
}

/// A witness of non-exhaustiveness for error reporting, represented
/// as a list of patterns (in reverse order of construction) with
/// wildcards inside to represent elements that can take any inhabitant
/// of the type as a value.
///
/// A witness against a list of patterns should have the same types
/// and length as the pattern matched against. Because Rust `match`
/// is always against a single pattern, at the end the witness will
/// have length 1, but in the middle of the algorithm, it can contain
/// multiple patterns.
///
/// For example, if we are constructing a witness for the match against
/// ```
/// struct Pair(Option<(u32, u32)>, bool);
///
/// match (p: Pair) {
///    Pair(None, _) => {}
///    Pair(_, false) => {}
/// }
/// ```
///
/// We'll perform the following steps:
/// 1. Start with an empty witness
///     `Witness(vec![])`
/// 2. Push a witness `Some(_)` against the `None`
///     `Witness(vec![Some(_)])`
/// 3. Push a witness `true` against the `false`
///     `Witness(vec![Some(_), true])`
/// 4. Apply the `Pair` constructor to the witnesses
///     `Witness(vec![Pair(Some(_), true)])`
///
/// The final `Pair(Some(_), true)` is then the resulting witness.
#[derive(Clone, Debug)]
pub struct Witness<'tcx>(Vec<Pattern<'tcx>>);

impl<'tcx> Witness<'tcx> {
    pub fn single_pattern(&self) -> &Pattern<'tcx> {
        assert_eq!(self.0.len(), 1);
        &self.0[0]
    }

    fn push_wild_constructor<'a>(
        mut self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        ctor: &Constructor<'tcx>,
        ty: Ty<'tcx>)
        -> Self
    {
        let sub_pattern_tys = constructor_sub_pattern_tys(cx, ctor, ty);
        self.0.extend(sub_pattern_tys.into_iter().map(|ty| {
            Pattern {
                ty,
                span: DUMMY_SP,
                kind: box PatternKind::Wild,
            }
        }));
        self.apply_constructor(cx, ctor, ty)
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
    fn apply_constructor<'a>(
        mut self,
        cx: &MatchCheckCtxt<'a,'tcx>,
        ctor: &Constructor<'tcx>,
        ty: Ty<'tcx>)
        -> Self
    {
        let arity = constructor_arity(cx, ctor, ty);
        let pat = {
            let len = self.0.len() as u64;
            let mut pats = self.0.drain((len - arity) as usize..).rev();

            match ty.sty {
                ty::TyAdt(..) |
                ty::TyTuple(..) => {
                    let pats = pats.enumerate().map(|(i, p)| {
                        FieldPattern {
                            field: Field::new(i),
                            pattern: p
                        }
                    }).collect();

                    if let ty::TyAdt(adt, substs) = ty.sty {
                        if adt.is_enum() {
                            PatternKind::Variant {
                                adt_def: adt,
                                substs,
                                variant_index: ctor.variant_index_for_adt(adt),
                                subpatterns: pats
                            }
                        } else {
                            PatternKind::Leaf { subpatterns: pats }
                        }
                    } else {
                        PatternKind::Leaf { subpatterns: pats }
                    }
                }

                ty::TyRef(..) => {
                    PatternKind::Deref { subpattern: pats.nth(0).unwrap() }
                }

                ty::TySlice(_) | ty::TyArray(..) => {
                    PatternKind::Slice {
                        prefix: pats.collect(),
                        slice: None,
                        suffix: vec![]
                    }
                }

                _ => {
                    match *ctor {
                        ConstantValue(value) => PatternKind::Constant { value },
                        ConstantRange(lo, hi, end) => PatternKind::Range { lo, hi, end },
                        _ => PatternKind::Wild,
                    }
                }
            }
        };

        self.0.push(Pattern {
            ty,
            span: DUMMY_SP,
            kind: Box::new(pat),
        });

        self
    }
}

/// This determines the set of all possible constructors of a pattern matching
/// values of type `left_ty`. For vectors, this would normally be an infinite set
/// but is instead bounded by the maximum fixed length of slice patterns in
/// the column of patterns being analyzed.
///
/// We make sure to omit constructors that are statically impossible. eg for
/// Option<!> we do not include Some(_) in the returned list of constructors.
fn all_constructors<'a, 'tcx: 'a>(cx: &mut MatchCheckCtxt<'a, 'tcx>,
                                  pcx: PatternContext<'tcx>)
                                  -> Vec<Constructor<'tcx>>
{
    debug!("all_constructors({:?})", pcx.ty);
    let exhaustive_integer_patterns = cx.tcx.features().exhaustive_integer_patterns;
    let ctors = match pcx.ty.sty {
        ty::TyBool => {
            [true, false].iter().map(|&b| {
                ConstantValue(ty::Const::from_bool(cx.tcx, b))
            }).collect()
        }
        ty::TyArray(ref sub_ty, len) if len.assert_usize(cx.tcx).is_some() => {
            let len = len.unwrap_usize(cx.tcx);
            if len != 0 && cx.is_uninhabited(sub_ty) {
                vec![]
            } else {
                vec![Slice(len)]
            }
        }
        // Treat arrays of a constant but unknown length like slices.
        ty::TyArray(ref sub_ty, _) |
        ty::TySlice(ref sub_ty) => {
            if cx.is_uninhabited(sub_ty) {
                vec![Slice(0)]
            } else {
                (0..pcx.max_slice_length+1).map(|length| Slice(length)).collect()
            }
        }
        ty::TyAdt(def, substs) if def.is_enum() => {
            def.variants.iter()
                .filter(|v| !cx.is_variant_uninhabited(v, substs))
                .map(|v| Variant(v.did))
                .collect()
        }
        ty::TyChar if exhaustive_integer_patterns => {
            let endpoint = |c: char| {
                let ty = ty::ParamEnv::empty().and(cx.tcx.types.char);
                ty::Const::from_bits(cx.tcx, c as u128, ty)
            };
            vec![
                // The valid Unicode Scalar Value ranges.
                ConstantRange(endpoint('\u{0000}'), endpoint('\u{D7FF}'), RangeEnd::Included),
                ConstantRange(endpoint('\u{E000}'), endpoint('\u{10FFFF}'), RangeEnd::Included),
            ]
        }
        ty::TyInt(ity) if exhaustive_integer_patterns => {
            // FIXME(49937): refactor these bit manipulations into interpret.
            let bits = Integer::from_attr(cx.tcx, SignedInt(ity)).size().bits() as u128;
            let min = 1u128 << (bits - 1);
            let max = (1u128 << (bits - 1)) - 1;
            let ty = ty::ParamEnv::empty().and(pcx.ty);
            vec![ConstantRange(ty::Const::from_bits(cx.tcx, min as u128, ty),
                               ty::Const::from_bits(cx.tcx, max as u128, ty),
                               RangeEnd::Included)]
        }
        ty::TyUint(uty) if exhaustive_integer_patterns => {
            // FIXME(49937): refactor these bit manipulations into interpret.
            let bits = Integer::from_attr(cx.tcx, UnsignedInt(uty)).size().bits() as u128;
            let max = !0u128 >> (128 - bits);
            let ty = ty::ParamEnv::empty().and(pcx.ty);
            vec![ConstantRange(ty::Const::from_bits(cx.tcx, 0, ty),
                               ty::Const::from_bits(cx.tcx, max, ty),
                               RangeEnd::Included)]
        }
        _ => {
            if cx.is_uninhabited(pcx.ty) {
                vec![]
            } else {
                vec![Single]
            }
        }
    };
    ctors
}

fn max_slice_length<'p, 'a: 'p, 'tcx: 'a, I>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    patterns: I) -> u64
    where I: Iterator<Item=&'p Pattern<'tcx>>
{
    // The exhaustiveness-checking paper does not include any details on
    // checking variable-length slice patterns. However, they are matched
    // by an infinite collection of fixed-length array patterns.
    //
    // Checking the infinite set directly would take an infinite amount
    // of time. However, it turns out that for each finite set of
    // patterns `P`, all sufficiently large array lengths are equivalent:
    //
    // Each slice `s` with a "sufficiently-large" length `l ≥ L` that applies
    // to exactly the subset `Pₜ` of `P` can be transformed to a slice
    // `sₘ` for each sufficiently-large length `m` that applies to exactly
    // the same subset of `P`.
    //
    // Because of that, each witness for reachability-checking from one
    // of the sufficiently-large lengths can be transformed to an
    // equally-valid witness from any other length, so we only have
    // to check slice lengths from the "minimal sufficiently-large length"
    // and below.
    //
    // Note that the fact that there is a *single* `sₘ` for each `m`
    // not depending on the specific pattern in `P` is important: if
    // you look at the pair of patterns
    //     `[true, ..]`
    //     `[.., false]`
    // Then any slice of length ≥1 that matches one of these two
    // patterns can be trivially turned to a slice of any
    // other length ≥1 that matches them and vice-versa - for
    // but the slice from length 2 `[false, true]` that matches neither
    // of these patterns can't be turned to a slice from length 1 that
    // matches neither of these patterns, so we have to consider
    // slices from length 2 there.
    //
    // Now, to see that that length exists and find it, observe that slice
    // patterns are either "fixed-length" patterns (`[_, _, _]`) or
    // "variable-length" patterns (`[_, .., _]`).
    //
    // For fixed-length patterns, all slices with lengths *longer* than
    // the pattern's length have the same outcome (of not matching), so
    // as long as `L` is greater than the pattern's length we can pick
    // any `sₘ` from that length and get the same result.
    //
    // For variable-length patterns, the situation is more complicated,
    // because as seen above the precise value of `sₘ` matters.
    //
    // However, for each variable-length pattern `p` with a prefix of length
    // `plₚ` and suffix of length `slₚ`, only the first `plₚ` and the last
    // `slₚ` elements are examined.
    //
    // Therefore, as long as `L` is positive (to avoid concerns about empty
    // types), all elements after the maximum prefix length and before
    // the maximum suffix length are not examined by any variable-length
    // pattern, and therefore can be added/removed without affecting
    // them - creating equivalent patterns from any sufficiently-large
    // length.
    //
    // Of course, if fixed-length patterns exist, we must be sure
    // that our length is large enough to miss them all, so
    // we can pick `L = max(FIXED_LEN+1 ∪ {max(PREFIX_LEN) + max(SUFFIX_LEN)})`
    //
    // for example, with the above pair of patterns, all elements
    // but the first and last can be added/removed, so any
    // witness of length ≥2 (say, `[false, false, true]`) can be
    // turned to a witness from any other length ≥2.

    let mut max_prefix_len = 0;
    let mut max_suffix_len = 0;
    let mut max_fixed_len = 0;

    for row in patterns {
        match *row.kind {
            PatternKind::Constant { value } => {
                if let Some(ptr) = value.to_ptr() {
                    let is_array_ptr = value.ty
                        .builtin_deref(true)
                        .and_then(|t| t.ty.builtin_index())
                        .map_or(false, |t| t == cx.tcx.types.u8);
                    if is_array_ptr {
                        let alloc = cx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id);
                        max_fixed_len = cmp::max(max_fixed_len, alloc.bytes.len() as u64);
                    }
                }
            }
            PatternKind::Slice { ref prefix, slice: None, ref suffix } => {
                let fixed_len = prefix.len() as u64 + suffix.len() as u64;
                max_fixed_len = cmp::max(max_fixed_len, fixed_len);
            }
            PatternKind::Slice { ref prefix, slice: Some(_), ref suffix } => {
                max_prefix_len = cmp::max(max_prefix_len, prefix.len() as u64);
                max_suffix_len = cmp::max(max_suffix_len, suffix.len() as u64);
            }
            _ => {}
        }
    }

    cmp::max(max_fixed_len + 1, max_prefix_len + max_suffix_len)
}

/// An inclusive interval, used for precise integer exhaustiveness checking.
/// `IntRange`s always store a contiguous range. This means that values are
/// encoded such that `0` encodes the minimum value for the integer,
/// regardless of the signedness.
/// For example, the pattern `-128...127i8` is encoded as `0..=255`.
/// This makes comparisons and arithmetic on interval endpoints much more
/// straightforward. See `signed_bias` for details.
struct IntRange<'tcx> {
    pub range: RangeInclusive<u128>,
    pub ty: Ty<'tcx>,
}

impl<'tcx> IntRange<'tcx> {
    fn from_ctor(tcx: TyCtxt<'_, 'tcx, 'tcx>,
                 ctor: &Constructor<'tcx>)
                 -> Option<IntRange<'tcx>> {
        match ctor {
            ConstantRange(lo, hi, end) => {
                assert_eq!(lo.ty, hi.ty);
                let ty = lo.ty;
                let env_ty = ty::ParamEnv::empty().and(ty);
                if let Some(lo) = lo.assert_bits(tcx, env_ty) {
                    if let Some(hi) = hi.assert_bits(tcx, env_ty) {
                        // Perform a shift if the underlying types are signed,
                        // which makes the interval arithmetic simpler.
                        let bias = IntRange::signed_bias(tcx, ty);
                        let (lo, hi) = (lo ^ bias, hi ^ bias);
                        // Make sure the interval is well-formed.
                        return if lo > hi || lo == hi && *end == RangeEnd::Excluded {
                            None
                        } else {
                            let offset = (*end == RangeEnd::Excluded) as u128;
                            Some(IntRange { range: lo..=(hi - offset), ty })
                        };
                    }
                }
                None
            }
            ConstantValue(val) => {
                let ty = val.ty;
                if let Some(val) = val.assert_bits(tcx, ty::ParamEnv::empty().and(ty)) {
                    let bias = IntRange::signed_bias(tcx, ty);
                    let val = val ^ bias;
                    Some(IntRange { range: val..=val, ty })
                } else {
                    None
                }
            }
            Single | Variant(_) | Slice(_) => {
                None
            }
        }
    }

    // The return value of `signed_bias` should be
    // XORed with an endpoint to encode/decode it.
    fn signed_bias(tcx: TyCtxt<'_, 'tcx, 'tcx>, ty: Ty<'tcx>) -> u128 {
        match ty.sty {
            ty::TyInt(ity) => {
                let bits = Integer::from_attr(tcx, SignedInt(ity)).size().bits() as u128;
                1u128 << (bits - 1)
            }
            _ => 0
        }
    }

    /// Given an `IntRange` corresponding to a pattern in a `match` and a collection of
    /// ranges corresponding to the domain of values of a type (say, an integer), return
    /// a new collection of ranges corresponding to the original ranges minus the ranges
    /// covered by the `IntRange`.
    fn subtract_from(self,
                     tcx: TyCtxt<'_, 'tcx, 'tcx>,
                     ranges: Vec<Constructor<'tcx>>)
                     -> Vec<Constructor<'tcx>> {
        let ranges = ranges.into_iter().filter_map(|r| {
            IntRange::from_ctor(tcx, &r).map(|i| i.range)
        });
        // Convert a `RangeInclusive` to a `ConstantValue` or inclusive `ConstantRange`.
        let bias = IntRange::signed_bias(tcx, self.ty);
        let ty = ty::ParamEnv::empty().and(self.ty);
        let range_to_constant = |r: RangeInclusive<u128>| {
            let (lo, hi) = r.into_inner();
            if lo == hi {
                ConstantValue(ty::Const::from_bits(tcx, lo ^ bias, ty))
            } else {
                ConstantRange(ty::Const::from_bits(tcx, lo ^ bias, ty),
                              ty::Const::from_bits(tcx, hi ^ bias, ty),
                              RangeEnd::Included)
            }
        };
        let mut remaining_ranges = vec![];
        let (lo, hi) = self.range.into_inner();
        for subrange in ranges {
            let (subrange_lo, subrange_hi) = subrange.into_inner();
            if lo > subrange_hi || subrange_lo > hi  {
                // The pattern doesn't intersect with the subrange at all,
                // so the subrange remains untouched.
                remaining_ranges.push(range_to_constant(subrange_lo..=subrange_hi));
            } else {
                if lo > subrange_lo {
                    // The pattern intersects an upper section of the
                    // subrange, so a lower section will remain.
                    remaining_ranges.push(range_to_constant(subrange_lo..=(lo - 1)));
                }
                if hi < subrange_hi {
                    // The pattern intersects a lower section of the
                    // subrange, so an upper section will remain.
                    remaining_ranges.push(range_to_constant((hi + 1)..=subrange_hi));
                }
            }
        }
        remaining_ranges
    }
}

/// Algorithm from http://moscova.inria.fr/~maranget/papers/warn/index.html
/// The algorithm from the paper has been modified to correctly handle empty
/// types. The changes are:
///   (0) We don't exit early if the pattern matrix has zero rows. We just
///       continue to recurse over columns.
///   (1) all_constructors will only return constructors that are statically
///       possible. eg. it will only return Ok for Result<T, !>
///
/// This finds whether a (row) vector `v` of patterns is 'useful' in relation
/// to a set of such vectors `m` - this is defined as there being a set of
/// inputs that will match `v` but not any of the sets in `m`.
///
/// All the patterns at each column of the `matrix ++ v` matrix must
/// have the same type, except that wildcard (PatternKind::Wild) patterns
/// with type TyErr are also allowed, even if the "type of the column"
/// is not TyErr. That is used to represent private fields, as using their
/// real type would assert that they are inhabited.
///
/// This is used both for reachability checking (if a pattern isn't useful in
/// relation to preceding patterns, it is not reachable) and exhaustiveness
/// checking (if a wildcard pattern is useful in relation to a matrix, the
/// matrix isn't exhaustive).
pub fn is_useful<'p, 'a: 'p, 'tcx: 'a>(cx: &mut MatchCheckCtxt<'a, 'tcx>,
                                       matrix: &Matrix<'p, 'tcx>,
                                       v: &[&'p Pattern<'tcx>],
                                       witness: WitnessPreference)
                                       -> Usefulness<'tcx> {
    let &Matrix(ref rows) = matrix;
    debug!("is_useful({:#?}, {:#?})", matrix, v);

    // The base case. We are pattern-matching on () and the return value is
    // based on whether our matrix has a row or not.
    // NOTE: This could potentially be optimized by checking rows.is_empty()
    // first and then, if v is non-empty, the return value is based on whether
    // the type of the tuple we're checking is inhabited or not.
    if v.is_empty() {
        return if rows.is_empty() {
            match witness {
                ConstructWitness => UsefulWithWitness(vec![Witness(vec![])]),
                LeaveOutWitness => Useful,
            }
        } else {
            NotUseful
        }
    };

    assert!(rows.iter().all(|r| r.len() == v.len()));

    let pcx = PatternContext {
        // TyErr is used to represent the type of wildcard patterns matching
        // against inaccessible (private) fields of structs, so that we won't
        // be able to observe whether the types of the struct's fields are
        // inhabited.
        //
        // If the field is truly inaccessible, then all the patterns
        // matching against it must be wildcard patterns, so its type
        // does not matter.
        //
        // However, if we are matching against non-wildcard patterns, we
        // need to know the real type of the field so we can specialize
        // against it. This primarily occurs through constants - they
        // can include contents for fields that are inaccessible at the
        // location of the match. In that case, the field's type is
        // inhabited - by the constant - so we can just use it.
        //
        // FIXME: this might lead to "unstable" behavior with macro hygiene
        // introducing uninhabited patterns for inaccessible fields. We
        // need to figure out how to model that.
        ty: rows.iter().map(|r| r[0].ty).find(|ty| !ty.references_error())
            .unwrap_or(v[0].ty),
        max_slice_length: max_slice_length(cx, rows.iter().map(|r| r[0]).chain(Some(v[0])))
    };

    debug!("is_useful_expand_first_col: pcx={:#?}, expanding {:#?}", pcx, v[0]);

    if let Some(constructors) = pat_constructors(cx, v[0], pcx) {
        debug!("is_useful - expanding constructors: {:#?}", constructors);
        constructors.into_iter().map(|c|
            is_useful_specialized(cx, matrix, v, c.clone(), pcx.ty, witness)
        ).find(|result| result.is_useful()).unwrap_or(NotUseful)
    } else {
        debug!("is_useful - expanding wildcard");

        let used_ctors: Vec<Constructor> = rows.iter().flat_map(|row| {
            pat_constructors(cx, row[0], pcx).unwrap_or(vec![])
        }).collect();
        debug!("used_ctors = {:#?}", used_ctors);
        // `all_ctors` are all the constructors for the given type, which
        // should all be represented (or caught with the wild pattern `_`).
        let all_ctors = all_constructors(cx, pcx);
        debug!("all_ctors = {:#?}", all_ctors);

        // The only constructor patterns for which it is valid to
        // treat the values as constructors are ranges (see
        // `all_constructors` for details).
        let exhaustive_integer_patterns = cx.tcx.features().exhaustive_integer_patterns;
        let consider_value_constructors = exhaustive_integer_patterns
            && all_ctors.iter().all(|ctor| match ctor {
                ConstantRange(..) => true,
                _ => false,
            });

        // `missing_ctors` are those that should have appeared
        // as patterns in the `match` expression, but did not.
        let mut missing_ctors = vec![];
        for req_ctor in &all_ctors {
            let mut refined_ctors = vec![req_ctor.clone()];
            for used_ctor in &used_ctors {
                if used_ctor == req_ctor {
                    // If a constructor appears in a `match` arm, we can
                    // eliminate it straight away.
                    refined_ctors = vec![]
                } else if exhaustive_integer_patterns {
                    if let Some(interval) = IntRange::from_ctor(cx.tcx, used_ctor) {
                        // Refine the required constructors for the type by subtracting
                        // the range defined by the current constructor pattern.
                        refined_ctors = interval.subtract_from(cx.tcx, refined_ctors);
                    }
                }

                // If the constructor patterns that have been considered so far
                // already cover the entire range of values, then we the
                // constructor is not missing, and we can move on to the next one.
                if refined_ctors.is_empty() {
                    break;
                }
            }
            // If a constructor has not been matched, then it is missing.
            // We add `refined_ctors` instead of `req_ctor`, because then we can
            // provide more detailed error information about precisely which
            // ranges have been omitted.
            missing_ctors.extend(refined_ctors);
        }

        // `missing_ctors` is the set of constructors from the same type as the
        // first column of `matrix` that are matched only by wildcard patterns
        // from the first column.
        //
        // Therefore, if there is some pattern that is unmatched by `matrix`,
        // it will still be unmatched if the first constructor is replaced by
        // any of the constructors in `missing_ctors`
        //
        // However, if our scrutinee is *privately* an empty enum, we
        // must treat it as though it had an "unknown" constructor (in
        // that case, all other patterns obviously can't be variants)
        // to avoid exposing its emptyness. See the `match_privately_empty`
        // test for details.
        //
        // FIXME: currently the only way I know of something can
        // be a privately-empty enum is when the exhaustive_patterns
        // feature flag is not present, so this is only
        // needed for that case.

        let is_privately_empty =
            all_ctors.is_empty() && !cx.is_uninhabited(pcx.ty);
        let is_declared_nonexhaustive =
            cx.is_non_exhaustive_enum(pcx.ty) && !cx.is_local(pcx.ty);
        debug!("missing_ctors={:#?} is_privately_empty={:#?} is_declared_nonexhaustive={:#?}",
               missing_ctors, is_privately_empty, is_declared_nonexhaustive);

        // For privately empty and non-exhaustive enums, we work as if there were an "extra"
        // `_` constructor for the type, so we can never match over all constructors.
        let is_non_exhaustive = is_privately_empty || is_declared_nonexhaustive;

        if missing_ctors.is_empty() && !is_non_exhaustive {
            if consider_value_constructors {
                // If we've successfully matched every value
                // of the type, then we're done.
                NotUseful
            } else {
                all_ctors.into_iter().map(|c| {
                    is_useful_specialized(cx, matrix, v, c.clone(), pcx.ty, witness)
                }).find(|result| result.is_useful()).unwrap_or(NotUseful)
            }
        } else {
            let matrix = rows.iter().filter_map(|r| {
                if r[0].is_wildcard() {
                    Some(r[1..].to_vec())
                } else {
                    None
                }
            }).collect();
            match is_useful(cx, &matrix, &v[1..], witness) {
                UsefulWithWitness(pats) => {
                    let cx = &*cx;
                    // In this case, there's at least one "free"
                    // constructor that is only matched against by
                    // wildcard patterns.
                    //
                    // There are 2 ways we can report a witness here.
                    // Commonly, we can report all the "free"
                    // constructors as witnesses, e.g. if we have:
                    //
                    // ```
                    //     enum Direction { N, S, E, W }
                    //     let Direction::N = ...;
                    // ```
                    //
                    // we can report 3 witnesses: `S`, `E`, and `W`.
                    //
                    // However, there are 2 cases where we don't want
                    // to do this and instead report a single `_` witness:
                    //
                    // 1) If the user is matching against a non-exhaustive
                    // enum, there is no point in enumerating all possible
                    // variants, because the user can't actually match
                    // against them himself, e.g. in an example like:
                    // ```
                    //     let err: io::ErrorKind = ...;
                    //     match err {
                    //         io::ErrorKind::NotFound => {},
                    //     }
                    // ```
                    // we don't want to show every possible IO error,
                    // but instead have `_` as the witness (this is
                    // actually *required* if the user specified *all*
                    // IO errors, but is probably what we want in every
                    // case).
                    //
                    // 2) If the user didn't actually specify a constructor
                    // in this arm, e.g. in
                    // ```
                    //     let x: (Direction, Direction, bool) = ...;
                    //     let (_, _, false) = x;
                    // ```
                    // we don't want to show all 16 possible witnesses
                    // `(<direction-1>, <direction-2>, true)` - we are
                    // satisfied with `(_, _, true)`. In this case,
                    // `used_ctors` is empty.
                    let new_witnesses = if is_non_exhaustive || used_ctors.is_empty() {
                        // All constructors are unused. Add wild patterns
                        // rather than each individual constructor.
                        pats.into_iter().map(|mut witness| {
                            witness.0.push(Pattern {
                                ty: pcx.ty,
                                span: DUMMY_SP,
                                kind: box PatternKind::Wild,
                            });
                            witness
                        }).collect()
                    } else {
                        pats.into_iter().flat_map(|witness| {
                            missing_ctors.iter().map(move |ctor| {
                                // Extends the witness with a "wild" version of this
                                // constructor, that matches everything that can be built with
                                // it. For example, if `ctor` is a `Constructor::Variant` for
                                // `Option::Some`, this pushes the witness for `Some(_)`.
                                witness.clone().push_wild_constructor(cx, ctor, pcx.ty)
                            })
                        }).collect()
                    };
                    UsefulWithWitness(new_witnesses)
                }
                result => result
            }
        }
    }
}

fn is_useful_specialized<'p, 'a:'p, 'tcx: 'a>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    &Matrix(ref m): &Matrix<'p, 'tcx>,
    v: &[&'p Pattern<'tcx>],
    ctor: Constructor<'tcx>,
    lty: Ty<'tcx>,
    witness: WitnessPreference) -> Usefulness<'tcx>
{
    debug!("is_useful_specialized({:#?}, {:#?}, {:?})", v, ctor, lty);
    let sub_pat_tys = constructor_sub_pattern_tys(cx, &ctor, lty);
    let wild_patterns_owned: Vec<_> = sub_pat_tys.iter().map(|ty| {
        Pattern {
            ty,
            span: DUMMY_SP,
            kind: box PatternKind::Wild,
        }
    }).collect();
    let wild_patterns: Vec<_> = wild_patterns_owned.iter().collect();
    let matrix = Matrix(m.iter().flat_map(|r| {
        specialize(cx, &r, &ctor, &wild_patterns)
    }).collect());
    match specialize(cx, v, &ctor, &wild_patterns) {
        Some(v) => match is_useful(cx, &matrix, &v, witness) {
            UsefulWithWitness(witnesses) => UsefulWithWitness(
                witnesses.into_iter()
                    .map(|witness| witness.apply_constructor(cx, &ctor, lty))
                    .collect()
            ),
            result => result
        },
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
/// Returns None in case of a catch-all, which can't be specialized.
fn pat_constructors<'tcx>(cx: &mut MatchCheckCtxt,
                          pat: &Pattern<'tcx>,
                          pcx: PatternContext)
                          -> Option<Vec<Constructor<'tcx>>>
{
    match *pat.kind {
        PatternKind::Binding { .. } | PatternKind::Wild =>
            None,
        PatternKind::Leaf { .. } | PatternKind::Deref { .. } =>
            Some(vec![Single]),
        PatternKind::Variant { adt_def, variant_index, .. } =>
            Some(vec![Variant(adt_def.variants[variant_index].did)]),
        PatternKind::Constant { value } =>
            Some(vec![ConstantValue(value)]),
        PatternKind::Range { lo, hi, end } =>
            Some(vec![ConstantRange(lo, hi, end)]),
        PatternKind::Array { .. } => match pcx.ty.sty {
            ty::TyArray(_, length) => Some(vec![
                Slice(length.unwrap_usize(cx.tcx))
            ]),
            _ => span_bug!(pat.span, "bad ty {:?} for array pattern", pcx.ty)
        },
        PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
            let pat_len = prefix.len() as u64 + suffix.len() as u64;
            if slice.is_some() {
                Some((pat_len..pcx.max_slice_length+1).map(Slice).collect())
            } else {
                Some(vec![Slice(pat_len)])
            }
        }
    }
}

/// This computes the arity of a constructor. The arity of a constructor
/// is how many subpattern patterns of that constructor should be expanded to.
///
/// For instance, a tuple pattern (_, 42, Some([])) has the arity of 3.
/// A struct pattern's arity is the number of fields it contains, etc.
fn constructor_arity(_cx: &MatchCheckCtxt, ctor: &Constructor, ty: Ty) -> u64 {
    debug!("constructor_arity({:#?}, {:?})", ctor, ty);
    match ty.sty {
        ty::TyTuple(ref fs) => fs.len() as u64,
        ty::TySlice(..) | ty::TyArray(..) => match *ctor {
            Slice(length) => length,
            ConstantValue(_) => 0,
            _ => bug!("bad slice pattern {:?} {:?}", ctor, ty)
        },
        ty::TyRef(..) => 1,
        ty::TyAdt(adt, _) => {
            adt.variants[ctor.variant_index_for_adt(adt)].fields.len() as u64
        }
        _ => 0
    }
}

/// This computes the types of the sub patterns that a constructor should be
/// expanded to.
///
/// For instance, a tuple pattern (43u32, 'a') has sub pattern types [u32, char].
fn constructor_sub_pattern_tys<'a, 'tcx: 'a>(cx: &MatchCheckCtxt<'a, 'tcx>,
                                             ctor: &Constructor,
                                             ty: Ty<'tcx>) -> Vec<Ty<'tcx>>
{
    debug!("constructor_sub_pattern_tys({:#?}, {:?})", ctor, ty);
    match ty.sty {
        ty::TyTuple(ref fs) => fs.into_iter().map(|t| *t).collect(),
        ty::TySlice(ty) | ty::TyArray(ty, _) => match *ctor {
            Slice(length) => (0..length).map(|_| ty).collect(),
            ConstantValue(_) => vec![],
            _ => bug!("bad slice pattern {:?} {:?}", ctor, ty)
        },
        ty::TyRef(_, rty, _) => vec![rty],
        ty::TyAdt(adt, substs) => {
            if adt.is_box() {
                // Use T as the sub pattern type of Box<T>.
                vec![substs.type_at(0)]
            } else {
                adt.variants[ctor.variant_index_for_adt(adt)].fields.iter().map(|field| {
                    let is_visible = adt.is_enum()
                        || field.vis.is_accessible_from(cx.module, cx.tcx);
                    if is_visible {
                        field.ty(cx.tcx, substs)
                    } else {
                        // Treat all non-visible fields as TyErr. They
                        // can't appear in any other pattern from
                        // this match (because they are private),
                        // so their type does not matter - but
                        // we don't want to know they are
                        // uninhabited.
                        cx.tcx.types.err
                    }
                }).collect()
            }
        }
        _ => vec![],
    }
}

fn slice_pat_covered_by_constructor<'tcx>(
    tcx: TyCtxt<'_, 'tcx, '_>,
    _span: Span,
    ctor: &Constructor,
    prefix: &[Pattern<'tcx>],
    slice: &Option<Pattern<'tcx>>,
    suffix: &[Pattern<'tcx>]
) -> Result<bool, ErrorReported> {
    let data: &[u8] = match *ctor {
        ConstantValue(const_val) => {
            let val = match const_val.val {
                ConstValue::Unevaluated(..) |
                ConstValue::ByRef(..) => bug!("unexpected ConstValue: {:?}", const_val),
                ConstValue::Scalar(val) | ConstValue::ScalarPair(val, _) => val,
            };
            if let Ok(ptr) = val.to_ptr() {
                let is_array_ptr = const_val.ty
                    .builtin_deref(true)
                    .and_then(|t| t.ty.builtin_index())
                    .map_or(false, |t| t == tcx.types.u8);
                assert!(is_array_ptr);
                tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id).bytes.as_ref()
            } else {
                bug!("unexpected non-ptr ConstantValue")
            }
        }
        _ => bug!()
    };

    let pat_len = prefix.len() + suffix.len();
    if data.len() < pat_len || (slice.is_none() && data.len() > pat_len) {
        return Ok(false);
    }

    for (ch, pat) in
        data[..prefix.len()].iter().zip(prefix).chain(
            data[data.len()-suffix.len()..].iter().zip(suffix))
    {
        match pat.kind {
            box PatternKind::Constant { value } => {
                let b = value.unwrap_bits(tcx, ty::ParamEnv::empty().and(pat.ty));
                assert_eq!(b as u8 as u128, b);
                if b as u8 != *ch {
                    return Ok(false);
                }
            }
            _ => {}
        }
    }

    Ok(true)
}

fn constructor_covered_by_range<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ctor: &Constructor<'tcx>,
    from: &'tcx ty::Const<'tcx>, to: &'tcx ty::Const<'tcx>,
    end: RangeEnd,
    ty: Ty<'tcx>,
) -> Result<bool, ErrorReported> {
    trace!("constructor_covered_by_range {:#?}, {:#?}, {:#?}, {}", ctor, from, to, ty);
    let cmp_from = |c_from| compare_const_vals(tcx, c_from, from, ty::ParamEnv::empty().and(ty))
        .map(|res| res != Ordering::Less);
    let cmp_to = |c_to| compare_const_vals(tcx, c_to, to, ty::ParamEnv::empty().and(ty));
    macro_rules! some_or_ok {
        ($e:expr) => {
            match $e {
                Some(to) => to,
                None => return Ok(false), // not char or int
            }
        };
    }
    match *ctor {
        ConstantValue(value) => {
            let to = some_or_ok!(cmp_to(value));
            let end = (to == Ordering::Less) ||
                      (end == RangeEnd::Included && to == Ordering::Equal);
            Ok(some_or_ok!(cmp_from(value)) && end)
        },
        ConstantRange(from, to, RangeEnd::Included) => {
            let to = some_or_ok!(cmp_to(to));
            let end = (to == Ordering::Less) ||
                      (end == RangeEnd::Included && to == Ordering::Equal);
            Ok(some_or_ok!(cmp_from(from)) && end)
        },
        ConstantRange(from, to, RangeEnd::Excluded) => {
            let to = some_or_ok!(cmp_to(to));
            let end = (to == Ordering::Less) ||
                      (end == RangeEnd::Excluded && to == Ordering::Equal);
            Ok(some_or_ok!(cmp_from(from)) && end)
        }
        Single => Ok(true),
        _ => bug!(),
    }
}

fn patterns_for_variant<'p, 'a: 'p, 'tcx: 'a>(
    subpatterns: &'p [FieldPattern<'tcx>],
    wild_patterns: &[&'p Pattern<'tcx>])
    -> Vec<&'p Pattern<'tcx>>
{
    let mut result = wild_patterns.to_owned();

    for subpat in subpatterns {
        result[subpat.field.index()] = &subpat.pattern;
    }

    debug!("patterns_for_variant({:#?}, {:#?}) = {:#?}", subpatterns, wild_patterns, result);
    result
}

/// This is the main specialization step. It expands the first pattern in the given row
/// into `arity` patterns based on the constructor. For most patterns, the step is trivial,
/// for instance tuple patterns are flattened and box patterns expand into their inner pattern.
///
/// OTOH, slice patterns with a subslice pattern (..tail) can be expanded into multiple
/// different patterns.
/// Structure patterns with a partial wild pattern (Foo { a: 42, .. }) have their missing
/// fields filled with wild patterns.
fn specialize<'p, 'a: 'p, 'tcx: 'a>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    r: &[&'p Pattern<'tcx>],
    constructor: &Constructor<'tcx>,
    wild_patterns: &[&'p Pattern<'tcx>])
    -> Option<Vec<&'p Pattern<'tcx>>>
{
    let pat = &r[0];

    let head: Option<Vec<&Pattern>> = match *pat.kind {
        PatternKind::Binding { .. } | PatternKind::Wild => {
            Some(wild_patterns.to_owned())
        },

        PatternKind::Variant { adt_def, variant_index, ref subpatterns, .. } => {
            let ref variant = adt_def.variants[variant_index];
            if *constructor == Variant(variant.did) {
                Some(patterns_for_variant(subpatterns, wild_patterns))
            } else {
                None
            }
        }

        PatternKind::Leaf { ref subpatterns } => {
            Some(patterns_for_variant(subpatterns, wild_patterns))
        }

        PatternKind::Deref { ref subpattern } => {
            Some(vec![subpattern])
        }

        PatternKind::Constant { value } => {
            match *constructor {
                Slice(..) => {
                    if let Some(ptr) = value.to_ptr() {
                        let is_array_ptr = value.ty
                            .builtin_deref(true)
                            .and_then(|t| t.ty.builtin_index())
                            .map_or(false, |t| t == cx.tcx.types.u8);
                        assert!(is_array_ptr);
                        let data_len = cx.tcx
                            .alloc_map
                            .lock()
                            .unwrap_memory(ptr.alloc_id)
                            .bytes
                            .len();
                        if wild_patterns.len() == data_len {
                            Some(cx.lower_byte_str_pattern(pat))
                        } else {
                            None
                        }
                    } else {
                        span_bug!(pat.span,
                        "unexpected const-val {:?} with ctor {:?}", value, constructor)
                    }
                },
                _ => {
                    match constructor_covered_by_range(
                        cx.tcx,
                        constructor, value, value, RangeEnd::Included,
                        value.ty,
                            ) {
                        Ok(true) => Some(vec![]),
                        Ok(false) => None,
                        Err(ErrorReported) => None,
                    }
                }
            }
        }

        PatternKind::Range { lo, hi, ref end } => {
            match constructor_covered_by_range(
                cx.tcx,
                constructor, lo, hi, end.clone(), lo.ty,
            ) {
                Ok(true) => Some(vec![]),
                Ok(false) => None,
                Err(ErrorReported) => None,
            }
        }

        PatternKind::Array { ref prefix, ref slice, ref suffix } |
        PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
            match *constructor {
                Slice(..) => {
                    let pat_len = prefix.len() + suffix.len();
                    if let Some(slice_count) = wild_patterns.len().checked_sub(pat_len) {
                        if slice_count == 0 || slice.is_some() {
                            Some(
                                prefix.iter().chain(
                                wild_patterns.iter().map(|p| *p)
                                                    .skip(prefix.len())
                                                    .take(slice_count)
                                                    .chain(
                                suffix.iter()
                            )).collect())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                ConstantValue(..) => {
                    match slice_pat_covered_by_constructor(
                        cx.tcx, pat.span, constructor, prefix, slice, suffix
                            ) {
                        Ok(true) => Some(vec![]),
                        Ok(false) => None,
                        Err(ErrorReported) => None
                    }
                }
                _ => span_bug!(pat.span,
                    "unexpected ctor {:?} for slice pat", constructor)
            }
        }
    };
    debug!("specialize({:#?}, {:#?}) = {:#?}", r[0], wild_patterns, head);

    head.map(|mut head| {
        head.extend_from_slice(&r[1 ..]);
        head
    })
}
