/// This file includes the logic for exhaustiveness and usefulness checking for
/// pattern-matching. Specifically, given a list of patterns for a type, we can
/// tell whether:
/// (a) the patterns cover every possible constructor for the type [exhaustiveness]
/// (b) each pattern is necessary [usefulness]
///
/// The algorithm implemented here is a modified version of the one described in:
/// http://moscova.inria.fr/~maranget/papers/warn/index.html
/// However, to save future implementors from reading the original paper, we
/// summarise the algorithm here to hopefully save time and be a little clearer
/// (without being so rigorous).
///
/// The core of the algorithm revolves about a "usefulness" check. In particular, we
/// are trying to compute a predicate `U(P, p_{m + 1})` where `P` is a list of patterns
/// of length `m` for a compound (product) type with `n` components (we refer to this as
/// a matrix). `U(P, p_{m + 1})` represents whether, given an existing list of patterns
/// `p_1 ..= p_m`, adding a new pattern will be "useful" (that is, cover previously-
/// uncovered values of the type).
///
/// If we have this predicate, then we can easily compute both exhaustiveness of an
/// entire set of patterns and the individual usefulness of each one.
/// (a) the set of patterns is exhaustive iff `U(P, _)` is false (i.e., adding a wildcard
/// match doesn't increase the number of values we're matching)
/// (b) a pattern `p_i` is not useful if `U(P[0..=(i-1), p_i)` is false (i.e., adding a
/// pattern to those that have come before it doesn't increase the number of values
/// we're matching).
///
/// For example, say we have the following:
/// ```
///     // x: (Option<bool>, Result<()>)
///     match x {
///         (Some(true), _) => {}
///         (None, Err(())) => {}
///         (None, Err(_)) => {}
///     }
/// ```
/// Here, the matrix `P` is 3 x 2 (rows x columns).
/// [
///     [Some(true), _],
///     [None, Err(())],
///     [None, Err(_)],
/// ]
/// We can tell it's not exhaustive, because `U(P, _)` is true (we're not covering
/// `[Some(false), _]`, for instance). In addition, row 3 is not useful, because
/// all the values it covers are already covered by row 2.
///
/// To compute `U`, we must have two other concepts.
///     1. `S(c, P)` is a "specialized matrix", where `c` is a constructor (like `Some` or
///        `None`). You can think of it as filtering `P` to just the rows whose *first* pattern
///        can cover `c` (and expanding OR-patterns into distinct patterns), and then expanding
///        the constructor into all of its components.
///        The specialization of a row vector is computed by `specialize`.
///
///        It is computed as follows. For each row `p_i` of P, we have four cases:
///             1.1. `p_(i,1) = c(r_1, .., r_a)`. Then `S(c, P)` has a corresponding row:
///                     r_1, .., r_a, p_(i,2), .., p_(i,n)
///             1.2. `p_(i,1) = c'(r_1, .., r_a')` where `c ≠ c'`. Then `S(c, P)` has no
///                  corresponding row.
///             1.3. `p_(i,1) = _`. Then `S(c, P)` has a corresponding row:
///                     _, .., _, p_(i,2), .., p_(i,n)
///             1.4. `p_(i,1) = r_1 | r_2`. Then `S(c, P)` has corresponding rows inlined from:
///                     S(c, (r_1, p_(i,2), .., p_(i,n)))
///                     S(c, (r_2, p_(i,2), .., p_(i,n)))
///
///     2. `D(P)` is a "default matrix". This is used when we know there are missing
///        constructor cases, but there might be existing wildcard patterns, so to check the
///        usefulness of the matrix, we have to check all its *other* components.
///        The default matrix is computed inline in `is_useful`.
///
///         It is computed as follows. For each row `p_i` of P, we have three cases:
///             1.1. `p_(i,1) = c(r_1, .., r_a)`. Then `D(P)` has no corresponding row.
///             1.2. `p_(i,1) = _`. Then `D(P)` has a corresponding row:
///                     p_(i,2), .., p_(i,n)
///             1.3. `p_(i,1) = r_1 | r_2`. Then `D(P)` has corresponding rows inlined from:
///                     D((r_1, p_(i,2), .., p_(i,n)))
///                     D((r_2, p_(i,2), .., p_(i,n)))
///
///     Note that the OR-patterns are not always used directly in Rust, but are used to derive
///     the exhaustive integer matching rules, so they're written here for posterity.
///
/// The algorithm for computing `U`
/// -------------------------------
/// The algorithm is inductive (on the number of columns: i.e., components of tuple patterns).
/// That means we're going to check the components from left-to-right, so the algorithm
/// operates principally on the first component of the matrix and new pattern `p_{m + 1}`.
/// This algorithm is realised in the `is_useful` function.
///
/// Base case. (`n = 0`, i.e., an empty tuple pattern)
///     - If `P` already contains an empty pattern (i.e., if the number of patterns `m > 0`),
///       then `U(P, p_{m + 1})` is false.
///     - Otherwise, `P` must be empty, so `U(P, p_{m + 1})` is true.
///
/// Inductive step. (`n > 0`, i.e., whether there's at least one column
///                  [which may then be expanded into further columns later])
///     We're going to match on the new pattern, `p_{m + 1}`.
///         - If `p_{m + 1} == c(r_1, .., r_a)`, then we have a constructor pattern.
///           Thus, the usefulness of `p_{m + 1}` can be reduced to whether it is useful when
///           we ignore all the patterns in `P` that involve other constructors. This is where
///           `S(c, P)` comes in:
///           `U(P, p_{m + 1}) := U(S(c, P), S(c, p_{m + 1}))`
///           This special case is handled in `is_useful_specialized`.
///         - If `p_{m + 1} == _`, then we have two more cases:
///             + All the constructors of the first component of the type exist within
///               all the rows (after having expanded OR-patterns). In this case:
///               `U(P, p_{m + 1}) := ∨(k ϵ constructors) U(S(k, P), S(k, p_{m + 1}))`
///               I.e., the pattern `p_{m + 1}` is only useful when all the constructors are
///               present *if* its later components are useful for the respective constructors
///               covered by `p_{m + 1}` (usually a single constructor, but all in the case of `_`).
///             + Some constructors are not present in the existing rows (after having expanded
///               OR-patterns). However, there might be wildcard patterns (`_`) present. Thus, we
///               are only really concerned with the other patterns leading with wildcards. This is
///               where `D` comes in:
///               `U(P, p_{m + 1}) := U(D(P), p_({m + 1},2), ..,  p_({m + 1},n))`
///         - If `p_{m + 1} == r_1 | r_2`, then the usefulness depends on each separately:
///           `U(P, p_{m + 1}) := U(P, (r_1, p_({m + 1},2), .., p_({m + 1},n)))
///                            || U(P, (r_2, p_({m + 1},2), .., p_({m + 1},n)))`
///
/// Modifications to the algorithm
/// ------------------------------
/// The algorithm in the paper doesn't cover some of the special cases that arise in Rust, for
/// example uninhabited types and variable-length slice patterns. These are drawn attention to
/// throughout the code below. I'll make a quick note here about how exhaustive integer matching
/// is accounted for, though.
///
/// Exhaustive integer matching
/// ---------------------------
/// An integer type can be thought of as a (huge) sum type: 1 | 2 | 3 | ...
/// So to support exhaustive integer matching, we can make use of the logic in the paper for
/// OR-patterns. However, we obviously can't just treat ranges x..=y as individual sums, because
/// they are likely gigantic. So we instead treat ranges as constructors of the integers. This means
/// that we have a constructor *of* constructors (the integers themselves). We then need to work
/// through all the inductive step rules above, deriving how the ranges would be treated as
/// OR-patterns, and making sure that they're treated in the same way even when they're ranges.
/// There are really only four special cases here:
/// - When we match on a constructor that's actually a range, we have to treat it as if we would
///   an OR-pattern.
///     + It turns out that we can simply extend the case for single-value patterns in
///      `specialize` to either be *equal* to a value constructor, or *contained within* a range
///      constructor.
///     + When the pattern itself is a range, you just want to tell whether any of the values in
///       the pattern range coincide with values in the constructor range, which is precisely
///       intersection.
///   Since when encountering a range pattern for a value constructor, we also use inclusion, it
///   means that whenever the constructor is a value/range and the pattern is also a value/range,
///   we can simply use intersection to test usefulness.
/// - When we're testing for usefulness of a pattern and the pattern's first component is a
///   wildcard.
///     + If all the constructors appear in the matrix, we have a slight complication. By default,
///       the behaviour (i.e., a disjunction over specialised matrices for each constructor) is
///       invalid, because we want a disjunction over every *integer* in each range, not just a
///       disjunction over every range. This is a bit more tricky to deal with: essentially we need
///       to form equivalence classes of subranges of the constructor range for which the behaviour
///       of the matrix `P` and new pattern `p_{m + 1}` are the same. This is described in more
///       detail in `split_grouped_constructors`.
///     + If some constructors are missing from the matrix, it turns out we don't need to do
///       anything special (because we know none of the integers are actually wildcards: i.e., we
///       can't span wildcards using ranges).

use self::Constructor::*;
use self::Usefulness::*;
use self::WitnessPreference::*;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;

use super::{FieldPattern, Pattern, PatternKind, PatternRange};
use super::{PatternFoldable, PatternFolder, compare_const_vals};

use rustc::hir::def_id::DefId;
use rustc::hir::RangeEnd;
use rustc::ty::{self, Ty, TyCtxt, TypeFoldable, Const};
use rustc::ty::layout::{Integer, IntegerExt, VariantIdx, Size};

use rustc::mir::Field;
use rustc::mir::interpret::{ConstValue, Scalar, truncate, AllocId, Pointer};
use rustc::util::common::ErrorReported;

use syntax::attr::{SignedInt, UnsignedInt};
use syntax_pos::{Span, DUMMY_SP};

use arena::TypedArena;

use smallvec::{SmallVec, smallvec};
use std::cmp::{self, Ordering, min, max};
use std::fmt;
use std::iter::{FromIterator, IntoIterator};
use std::ops::RangeInclusive;
use std::u128;
use std::convert::TryInto;

pub fn expand_pattern<'a, 'tcx>(cx: &MatchCheckCtxt<'a, 'tcx>, pat: Pattern<'tcx>)
                                -> &'a Pattern<'tcx>
{
    cx.pattern_arena.alloc(LiteralExpander { tcx: cx.tcx }.fold_pattern(&pat))
}

struct LiteralExpander<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl LiteralExpander<'tcx> {
    /// Derefs `val` and potentially unsizes the value if `crty` is an array and `rty` a slice.
    ///
    /// `crty` and `rty` can differ because you can use array constants in the presence of slice
    /// patterns. So the pattern may end up being a slice, but the constant is an array. We convert
    /// the array to a slice in that case.
    fn fold_const_value_deref(
        &mut self,
        val: ConstValue<'tcx>,
        // the pattern's pointee type
        rty: Ty<'tcx>,
        // the constant's pointee type
        crty: Ty<'tcx>,
    ) -> ConstValue<'tcx> {
        debug!("fold_const_value_deref {:?} {:?} {:?}", val, rty, crty);
        match (val, &crty.sty, &rty.sty) {
            // the easy case, deref a reference
            (ConstValue::Scalar(Scalar::Ptr(p)), x, y) if x == y => {
                let alloc = self.tcx.alloc_map.lock().unwrap_memory(p.alloc_id);
                ConstValue::ByRef {
                    offset: p.offset,
                    // FIXME(oli-obk): this should be the type's layout
                    align: alloc.align,
                    alloc,
                }
            },
            // unsize array to slice if pattern is array but match value or other patterns are slice
            (ConstValue::Scalar(Scalar::Ptr(p)), ty::Array(t, n), ty::Slice(u)) => {
                assert_eq!(t, u);
                ConstValue::Slice {
                    data: self.tcx.alloc_map.lock().unwrap_memory(p.alloc_id),
                    start: p.offset.bytes().try_into().unwrap(),
                    end: n.unwrap_usize(self.tcx).try_into().unwrap(),
                }
            },
            // fat pointers stay the same
            | (ConstValue::Slice { .. }, _, _)
            | (_, ty::Slice(_), ty::Slice(_))
            | (_, ty::Str, ty::Str)
            => val,
            // FIXME(oli-obk): this is reachable for `const FOO: &&&u32 = &&&42;` being used
            _ => bug!("cannot deref {:#?}, {} -> {}", val, crty, rty),
        }
    }
}

impl PatternFolder<'tcx> for LiteralExpander<'tcx> {
    fn fold_pattern(&mut self, pat: &Pattern<'tcx>) -> Pattern<'tcx> {
        debug!("fold_pattern {:?} {:?} {:?}", pat, pat.ty.sty, pat.kind);
        match (&pat.ty.sty, &*pat.kind) {
            (
                &ty::Ref(_, rty, _),
                &PatternKind::Constant { value: Const {
                    val,
                    ty: ty::TyS { sty: ty::Ref(_, crty, _), .. },
                } },
            ) => {
                Pattern {
                    ty: pat.ty,
                    span: pat.span,
                    kind: box PatternKind::Deref {
                        subpattern: Pattern {
                            ty: rty,
                            span: pat.span,
                            kind: box PatternKind::Constant { value: self.tcx.mk_const(Const {
                                val: self.fold_const_value_deref(*val, rty, crty),
                                ty: rty,
                            }) },
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

/// A 2D matrix. Nx1 matrices are very common, which is why `SmallVec[_; 2]`
/// works well for each row.
pub struct Matrix<'p, 'tcx>(Vec<SmallVec<[&'p Pattern<'tcx>; 2]>>);

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    pub fn empty() -> Self {
        Matrix(vec![])
    }

    pub fn push(&mut self, row: SmallVec<[&'p Pattern<'tcx>; 2]>) {
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
impl<'p, 'tcx> fmt::Debug for Matrix<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl<'p, 'tcx> FromIterator<SmallVec<[&'p Pattern<'tcx>; 2]>> for Matrix<'p, 'tcx> {
    fn from_iter<T>(iter: T) -> Self
        where T: IntoIterator<Item=SmallVec<[&'p Pattern<'tcx>; 2]>>
    {
        Matrix(iter.into_iter().collect())
    }
}

pub struct MatchCheckCtxt<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,
    /// The module in which the match occurs. This is necessary for
    /// checking inhabited-ness of types because whether a type is (visibly)
    /// inhabited can depend on whether it was defined in the current module or
    /// not. E.g., `struct Foo { _private: ! }` cannot be seen to be empty
    /// outside it's module and should not be matchable with an empty match
    /// statement.
    pub module: DefId,
    param_env: ty::ParamEnv<'tcx>,
    pub pattern_arena: &'a TypedArena<Pattern<'tcx>>,
    pub byte_array_map: FxHashMap<*const Pattern<'tcx>, Vec<&'a Pattern<'tcx>>>,
}

impl<'a, 'tcx> MatchCheckCtxt<'a, 'tcx> {
    pub fn create_and_enter<F, R>(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        module: DefId,
        f: F,
    ) -> R
    where
        F: for<'b> FnOnce(MatchCheckCtxt<'b, 'tcx>) -> R,
    {
        let pattern_arena = TypedArena::default();

        f(MatchCheckCtxt {
            tcx,
            param_env,
            module,
            pattern_arena: &pattern_arena,
            byte_array_map: FxHashMap::default(),
        })
    }

    fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        if self.tcx.features().exhaustive_patterns {
            self.tcx.is_ty_uninhabited_from(self.module, ty)
        } else {
            false
        }
    }

    fn is_non_exhaustive_variant<'p>(&self, pattern: &'p Pattern<'tcx>) -> bool {
        match *pattern.kind {
            PatternKind::Variant { adt_def, variant_index, .. } => {
                let ref variant = adt_def.variants[variant_index];
                variant.is_field_list_non_exhaustive()
            }
            _ => false,
        }
    }

    fn is_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::Adt(adt_def, ..) => adt_def.is_variant_list_non_exhaustive(),
            _ => false,
        }
    }

    fn is_local(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::Adt(adt_def, ..) => adt_def.did.is_local(),
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Constructor<'tcx> {
    /// The constructor of all patterns that don't vary by constructor,
    /// e.g., struct patterns and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(DefId),
    /// Literal values.
    ConstantValue(&'tcx ty::Const<'tcx>),
    /// Ranges of literal values (`2..=5` and `2..5`).
    ConstantRange(u128, u128, Ty<'tcx>, RangeEnd),
    /// Array patterns of length n.
    Slice(u64),
}

impl<'tcx> Constructor<'tcx> {
    fn variant_index_for_adt<'a>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        adt: &'tcx ty::AdtDef,
    ) -> VariantIdx {
        match self {
            &Variant(id) => adt.variant_index_with_id(id),
            &Single => {
                assert!(!adt.is_enum());
                VariantIdx::new(0)
            }
            &ConstantValue(c) => crate::const_eval::const_variant_index(cx.tcx, cx.param_env, c),
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
    /// to reconstruct a complete witness, e.g., a pattern P' that covers a subset
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
                ty::Adt(..) |
                ty::Tuple(..) => {
                    let pats = pats.enumerate().map(|(i, p)| {
                        FieldPattern {
                            field: Field::new(i),
                            pattern: p
                        }
                    }).collect();

                    if let ty::Adt(adt, substs) = ty.sty {
                        if adt.is_enum() {
                            PatternKind::Variant {
                                adt_def: adt,
                                substs,
                                variant_index: ctor.variant_index_for_adt(cx, adt),
                                subpatterns: pats
                            }
                        } else {
                            PatternKind::Leaf { subpatterns: pats }
                        }
                    } else {
                        PatternKind::Leaf { subpatterns: pats }
                    }
                }

                ty::Ref(..) => {
                    PatternKind::Deref { subpattern: pats.nth(0).unwrap() }
                }

                ty::Slice(_) | ty::Array(..) => {
                    PatternKind::Slice {
                        prefix: pats.collect(),
                        slice: None,
                        suffix: vec![]
                    }
                }

                _ => {
                    match *ctor {
                        ConstantValue(value) => PatternKind::Constant { value },
                        ConstantRange(lo, hi, ty, end) => PatternKind::Range(PatternRange {
                            lo: ty::Const::from_bits(cx.tcx, lo, ty::ParamEnv::empty().and(ty)),
                            hi: ty::Const::from_bits(cx.tcx, hi, ty::ParamEnv::empty().and(ty)),
                            ty,
                            end,
                        }),
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
/// We make sure to omit constructors that are statically impossible. E.g., for
/// `Option<!>`, we do not include `Some(_)` in the returned list of constructors.
fn all_constructors<'a, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    pcx: PatternContext<'tcx>,
) -> Vec<Constructor<'tcx>> {
    debug!("all_constructors({:?})", pcx.ty);
    let ctors = match pcx.ty.sty {
        ty::Bool => {
            [true, false].iter().map(|&b| {
                ConstantValue(ty::Const::from_bool(cx.tcx, b))
            }).collect()
        }
        ty::Array(ref sub_ty, len) if len.assert_usize(cx.tcx).is_some() => {
            let len = len.unwrap_usize(cx.tcx);
            if len != 0 && cx.is_uninhabited(sub_ty) {
                vec![]
            } else {
                vec![Slice(len)]
            }
        }
        // Treat arrays of a constant but unknown length like slices.
        ty::Array(ref sub_ty, _) |
        ty::Slice(ref sub_ty) => {
            if cx.is_uninhabited(sub_ty) {
                vec![Slice(0)]
            } else {
                (0..pcx.max_slice_length+1).map(|length| Slice(length)).collect()
            }
        }
        ty::Adt(def, substs) if def.is_enum() => {
            def.variants.iter()
                .filter(|v| {
                    !cx.tcx.features().exhaustive_patterns ||
                    !v.uninhabited_from(cx.tcx, substs, def.adt_kind()).contains(cx.tcx, cx.module)
                })
                .map(|v| Variant(v.def_id))
                .collect()
        }
        ty::Char => {
            vec![
                // The valid Unicode Scalar Value ranges.
                ConstantRange('\u{0000}' as u128,
                              '\u{D7FF}' as u128,
                              cx.tcx.types.char,
                              RangeEnd::Included
                ),
                ConstantRange('\u{E000}' as u128,
                              '\u{10FFFF}' as u128,
                              cx.tcx.types.char,
                              RangeEnd::Included
                ),
            ]
        }
        ty::Int(ity) => {
            let bits = Integer::from_attr(&cx.tcx, SignedInt(ity)).size().bits() as u128;
            let min = 1u128 << (bits - 1);
            let max = min - 1;
            vec![ConstantRange(min, max, pcx.ty, RangeEnd::Included)]
        }
        ty::Uint(uty) => {
            let size = Integer::from_attr(&cx.tcx, UnsignedInt(uty)).size();
            let max = truncate(u128::max_value(), size);
            vec![ConstantRange(0, max, pcx.ty, RangeEnd::Included)]
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

fn max_slice_length<'p, 'a, 'tcx, I>(cx: &mut MatchCheckCtxt<'a, 'tcx>, patterns: I) -> u64
where
    I: Iterator<Item = &'p Pattern<'tcx>>,
    'tcx: 'p,
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
                // extract the length of an array/slice from a constant
                match (value.val, &value.ty.sty) {
                    (_, ty::Array(_, n)) => max_fixed_len = cmp::max(
                        max_fixed_len,
                        n.unwrap_usize(cx.tcx),
                    ),
                    (ConstValue::Slice{ start, end, .. }, ty::Slice(_)) => max_fixed_len = cmp::max(
                        max_fixed_len,
                        (end - start) as u64,
                    ),
                    _ => {},
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
/// For example, the pattern `-128..=127i8` is encoded as `0..=255`.
/// This makes comparisons and arithmetic on interval endpoints much more
/// straightforward. See `signed_bias` for details.
///
/// `IntRange` is never used to encode an empty range or a "range" that wraps
/// around the (offset) space: i.e., `range.lo <= range.hi`.
#[derive(Clone)]
struct IntRange<'tcx> {
    pub range: RangeInclusive<u128>,
    pub ty: Ty<'tcx>,
}

impl<'tcx> IntRange<'tcx> {
    fn from_ctor(tcx: TyCtxt<'tcx>, ctor: &Constructor<'tcx>) -> Option<IntRange<'tcx>> {
        // Floating-point ranges are permitted and we don't want
        // to consider them when constructing integer ranges.
        fn is_integral(ty: Ty<'_>) -> bool {
            match ty.sty {
                ty::Char | ty::Int(_) | ty::Uint(_) => true,
                _ => false,
            }
        }

        match ctor {
            ConstantRange(lo, hi, ty, end) if is_integral(ty) => {
                // Perform a shift if the underlying types are signed,
                // which makes the interval arithmetic simpler.
                let bias = IntRange::signed_bias(tcx, ty);
                let (lo, hi) = (lo ^ bias, hi ^ bias);
                // Make sure the interval is well-formed.
                if lo > hi || lo == hi && *end == RangeEnd::Excluded {
                    None
                } else {
                    let offset = (*end == RangeEnd::Excluded) as u128;
                    Some(IntRange { range: lo..=(hi - offset), ty })
                }
            }
            ConstantValue(val) if is_integral(val.ty) => {
                let ty = val.ty;
                if let Some(val) = val.assert_bits(tcx, ty::ParamEnv::empty().and(ty)) {
                    let bias = IntRange::signed_bias(tcx, ty);
                    let val = val ^ bias;
                    Some(IntRange { range: val..=val, ty })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn from_pat(tcx: TyCtxt<'tcx>, mut pat: &Pattern<'tcx>) -> Option<IntRange<'tcx>> {
        let range = loop {
            match pat.kind {
                box PatternKind::Constant { value } => break ConstantValue(value),
                box PatternKind::Range(PatternRange { lo, hi, ty, end }) => break ConstantRange(
                    lo.to_bits(tcx, ty::ParamEnv::empty().and(ty)).unwrap(),
                    hi.to_bits(tcx, ty::ParamEnv::empty().and(ty)).unwrap(),
                    ty,
                    end,
                ),
                box PatternKind::AscribeUserType { ref subpattern, .. } => {
                    pat = subpattern;
                },
                _ => return None,
            }
        };
        Self::from_ctor(tcx, &range)
    }

    // The return value of `signed_bias` should be XORed with an endpoint to encode/decode it.
    fn signed_bias(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> u128 {
        match ty.sty {
            ty::Int(ity) => {
                let bits = Integer::from_attr(&tcx, SignedInt(ity)).size().bits() as u128;
                1u128 << (bits - 1)
            }
            _ => 0
        }
    }

    /// Converts a `RangeInclusive` to a `ConstantValue` or inclusive `ConstantRange`.
    fn range_to_ctor(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        r: RangeInclusive<u128>,
    ) -> Constructor<'tcx> {
        let bias = IntRange::signed_bias(tcx, ty);
        let (lo, hi) = r.into_inner();
        if lo == hi {
            let ty = ty::ParamEnv::empty().and(ty);
            ConstantValue(ty::Const::from_bits(tcx, lo ^ bias, ty))
        } else {
            ConstantRange(lo ^ bias, hi ^ bias, ty, RangeEnd::Included)
        }
    }

    /// Returns a collection of ranges that spans the values covered by `ranges`, subtracted
    /// by the values covered by `self`: i.e., `ranges \ self` (in set notation).
    fn subtract_from(
        self,
        tcx: TyCtxt<'tcx>,
        ranges: Vec<Constructor<'tcx>>,
    ) -> Vec<Constructor<'tcx>> {
        let ranges = ranges.into_iter().filter_map(|r| {
            IntRange::from_ctor(tcx, &r).map(|i| i.range)
        });
        let mut remaining_ranges = vec![];
        let ty = self.ty;
        let (lo, hi) = self.range.into_inner();
        for subrange in ranges {
            let (subrange_lo, subrange_hi) = subrange.into_inner();
            if lo > subrange_hi || subrange_lo > hi  {
                // The pattern doesn't intersect with the subrange at all,
                // so the subrange remains untouched.
                remaining_ranges.push(Self::range_to_ctor(tcx, ty, subrange_lo..=subrange_hi));
            } else {
                if lo > subrange_lo {
                    // The pattern intersects an upper section of the
                    // subrange, so a lower section will remain.
                    remaining_ranges.push(Self::range_to_ctor(tcx, ty, subrange_lo..=(lo - 1)));
                }
                if hi < subrange_hi {
                    // The pattern intersects a lower section of the
                    // subrange, so an upper section will remain.
                    remaining_ranges.push(Self::range_to_ctor(tcx, ty, (hi + 1)..=subrange_hi));
                }
            }
        }
        remaining_ranges
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let ty = self.ty;
        let (lo, hi) = (*self.range.start(), *self.range.end());
        let (other_lo, other_hi) = (*other.range.start(), *other.range.end());
        if lo <= other_hi && other_lo <= hi {
            Some(IntRange { range: max(lo, other_lo)..=min(hi, other_hi), ty })
        } else {
            None
        }
    }
}

// A request for missing constructor data in terms of either:
// - whether or not there any missing constructors; or
// - the actual set of missing constructors.
#[derive(PartialEq)]
enum MissingCtorsInfo {
    Emptiness,
    Ctors,
}

// Used by `compute_missing_ctors`.
#[derive(Debug, PartialEq)]
enum MissingCtors<'tcx> {
    Empty,
    NonEmpty,

    // Note that the Vec can be empty.
    Ctors(Vec<Constructor<'tcx>>),
}

// When `info` is `MissingCtorsInfo::Ctors`, compute a set of constructors
// equivalent to `all_ctors \ used_ctors`. When `info` is
// `MissingCtorsInfo::Emptiness`, just determines if that set is empty or not.
// (The split logic gives a performance win, because we always need to know if
// the set is empty, but we rarely need the full set, and it can be expensive
// to compute the full set.)
fn compute_missing_ctors<'tcx>(
    info: MissingCtorsInfo,
    tcx: TyCtxt<'tcx>,
    all_ctors: &Vec<Constructor<'tcx>>,
    used_ctors: &Vec<Constructor<'tcx>>,
) -> MissingCtors<'tcx> {
    let mut missing_ctors = vec![];

    for req_ctor in all_ctors {
        let mut refined_ctors = vec![req_ctor.clone()];
        for used_ctor in used_ctors {
            if used_ctor == req_ctor {
                // If a constructor appears in a `match` arm, we can
                // eliminate it straight away.
                refined_ctors = vec![]
            } else if let Some(interval) = IntRange::from_ctor(tcx, used_ctor) {
                // Refine the required constructors for the type by subtracting
                // the range defined by the current constructor pattern.
                refined_ctors = interval.subtract_from(tcx, refined_ctors);
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
        if info == MissingCtorsInfo::Emptiness {
            if !refined_ctors.is_empty() {
                // The set is non-empty; return early.
                return MissingCtors::NonEmpty;
            }
        } else {
            missing_ctors.extend(refined_ctors);
        }
    }

    if info == MissingCtorsInfo::Emptiness {
        // If we reached here, the set is empty.
        MissingCtors::Empty
    } else {
        MissingCtors::Ctors(missing_ctors)
    }
}

/// Algorithm from http://moscova.inria.fr/~maranget/papers/warn/index.html.
/// The algorithm from the paper has been modified to correctly handle empty
/// types. The changes are:
///   (0) We don't exit early if the pattern matrix has zero rows. We just
///       continue to recurse over columns.
///   (1) all_constructors will only return constructors that are statically
///       possible. E.g., it will only return `Ok` for `Result<T, !>`.
///
/// This finds whether a (row) vector `v` of patterns is 'useful' in relation
/// to a set of such vectors `m` - this is defined as there being a set of
/// inputs that will match `v` but not any of the sets in `m`.
///
/// All the patterns at each column of the `matrix ++ v` matrix must
/// have the same type, except that wildcard (PatternKind::Wild) patterns
/// with type `TyErr` are also allowed, even if the "type of the column"
/// is not `TyErr`. That is used to represent private fields, as using their
/// real type would assert that they are inhabited.
///
/// This is used both for reachability checking (if a pattern isn't useful in
/// relation to preceding patterns, it is not reachable) and exhaustiveness
/// checking (if a wildcard pattern is useful in relation to a matrix, the
/// matrix isn't exhaustive).
pub fn is_useful<'p, 'a, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &[&Pattern<'tcx>],
    witness: WitnessPreference,
) -> Usefulness<'tcx> {
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
        ty: rows.iter().map(|r| r[0].ty).find(|ty| !ty.references_error()).unwrap_or(v[0].ty),
        max_slice_length: max_slice_length(cx, rows.iter().map(|r| r[0]).chain(Some(v[0])))
    };

    debug!("is_useful_expand_first_col: pcx={:#?}, expanding {:#?}", pcx, v[0]);

    if let Some(constructors) = pat_constructors(cx, v[0], pcx) {
        let is_declared_nonexhaustive = cx.is_non_exhaustive_variant(v[0]) && !cx.is_local(pcx.ty);
        debug!("is_useful - expanding constructors: {:#?}, is_declared_nonexhaustive: {:?}",
               constructors, is_declared_nonexhaustive);

        if is_declared_nonexhaustive {
            Useful
        } else {
            split_grouped_constructors(cx.tcx, constructors, matrix, pcx.ty).into_iter().map(|c|
                is_useful_specialized(cx, matrix, v, c, pcx.ty, witness)
            ).find(|result| result.is_useful()).unwrap_or(NotUseful)
        }
    } else {
        debug!("is_useful - expanding wildcard");

        let used_ctors: Vec<Constructor<'_>> = rows.iter().flat_map(|row| {
            pat_constructors(cx, row[0], pcx).unwrap_or(vec![])
        }).collect();
        debug!("used_ctors = {:#?}", used_ctors);
        // `all_ctors` are all the constructors for the given type, which
        // should all be represented (or caught with the wild pattern `_`).
        let all_ctors = all_constructors(cx, pcx);
        debug!("all_ctors = {:#?}", all_ctors);

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

        // Missing constructors are those that are not matched by any
        // non-wildcard patterns in the current column. We always determine if
        // the set is empty, but we only fully construct them on-demand,
        // because they're rarely used and can be big.
        let cheap_missing_ctors =
            compute_missing_ctors(MissingCtorsInfo::Emptiness, cx.tcx, &all_ctors, &used_ctors);

        let is_privately_empty = all_ctors.is_empty() && !cx.is_uninhabited(pcx.ty);
        let is_declared_nonexhaustive = cx.is_non_exhaustive_enum(pcx.ty) && !cx.is_local(pcx.ty);
        debug!("cheap_missing_ctors={:#?} is_privately_empty={:#?} is_declared_nonexhaustive={:#?}",
               cheap_missing_ctors, is_privately_empty, is_declared_nonexhaustive);

        // For privately empty and non-exhaustive enums, we work as if there were an "extra"
        // `_` constructor for the type, so we can never match over all constructors.
        let is_non_exhaustive = is_privately_empty || is_declared_nonexhaustive ||
            (pcx.ty.is_pointer_sized() && !cx.tcx.features().precise_pointer_size_matching);

        if cheap_missing_ctors == MissingCtors::Empty && !is_non_exhaustive {
            split_grouped_constructors(cx.tcx, all_ctors, matrix, pcx.ty).into_iter().map(|c| {
                is_useful_specialized(cx, matrix, v, c, pcx.ty, witness)
            }).find(|result| result.is_useful()).unwrap_or(NotUseful)
        } else {
            let matrix = rows.iter().filter_map(|r| {
                if r[0].is_wildcard() {
                    Some(SmallVec::from_slice(&r[1..]))
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
                    // constructors as witnesses, e.g., if we have:
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
                    // against them himself, e.g., in an example like:
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
                    // in this arm, e.g., in
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
                        let expensive_missing_ctors =
                            compute_missing_ctors(MissingCtorsInfo::Ctors, cx.tcx, &all_ctors,
                                                  &used_ctors);
                        if let MissingCtors::Ctors(missing_ctors) = expensive_missing_ctors {
                            pats.into_iter().flat_map(|witness| {
                                missing_ctors.iter().map(move |ctor| {
                                    // Extends the witness with a "wild" version of this
                                    // constructor, that matches everything that can be built with
                                    // it. For example, if `ctor` is a `Constructor::Variant` for
                                    // `Option::Some`, this pushes the witness for `Some(_)`.
                                    witness.clone().push_wild_constructor(cx, ctor, pcx.ty)
                                })
                            }).collect()
                        } else {
                            bug!("cheap missing ctors")
                        }
                    };
                    UsefulWithWitness(new_witnesses)
                }
                result => result
            }
        }
    }
}

/// A shorthand for the `U(S(c, P), S(c, q))` operation from the paper. I.e., `is_useful` applied
/// to the specialised version of both the pattern matrix `P` and the new pattern `q`.
fn is_useful_specialized<'p, 'a, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    &Matrix(ref m): &Matrix<'p, 'tcx>,
    v: &[&Pattern<'tcx>],
    ctor: Constructor<'tcx>,
    lty: Ty<'tcx>,
    witness: WitnessPreference,
) -> Usefulness<'tcx> {
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
        }
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
/// Returns `None` in case of a catch-all, which can't be specialized.
fn pat_constructors<'tcx>(cx: &mut MatchCheckCtxt<'_, 'tcx>,
                          pat: &Pattern<'tcx>,
                          pcx: PatternContext<'tcx>)
                          -> Option<Vec<Constructor<'tcx>>>
{
    match *pat.kind {
        PatternKind::AscribeUserType { ref subpattern, .. } =>
            pat_constructors(cx, subpattern, pcx),
        PatternKind::Binding { .. } | PatternKind::Wild => None,
        PatternKind::Leaf { .. } | PatternKind::Deref { .. } => Some(vec![Single]),
        PatternKind::Variant { adt_def, variant_index, .. } => {
            Some(vec![Variant(adt_def.variants[variant_index].def_id)])
        }
        PatternKind::Constant { value } => Some(vec![ConstantValue(value)]),
        PatternKind::Range(PatternRange { lo, hi, ty, end }) =>
            Some(vec![ConstantRange(
                lo.to_bits(cx.tcx, ty::ParamEnv::empty().and(ty)).unwrap(),
                hi.to_bits(cx.tcx, ty::ParamEnv::empty().and(ty)).unwrap(),
                ty,
                end,
            )]),
        PatternKind::Array { .. } => match pcx.ty.sty {
            ty::Array(_, length) => Some(vec![
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
/// For instance, a tuple pattern `(_, 42, Some([]))` has the arity of 3.
/// A struct pattern's arity is the number of fields it contains, etc.
fn constructor_arity(cx: &MatchCheckCtxt<'a, 'tcx>, ctor: &Constructor<'tcx>, ty: Ty<'tcx>) -> u64 {
    debug!("constructor_arity({:#?}, {:?})", ctor, ty);
    match ty.sty {
        ty::Tuple(ref fs) => fs.len() as u64,
        ty::Slice(..) | ty::Array(..) => match *ctor {
            Slice(length) => length,
            ConstantValue(_) => 0,
            _ => bug!("bad slice pattern {:?} {:?}", ctor, ty)
        }
        ty::Ref(..) => 1,
        ty::Adt(adt, _) => {
            adt.variants[ctor.variant_index_for_adt(cx, adt)].fields.len() as u64
        }
        _ => 0
    }
}

/// This computes the types of the sub patterns that a constructor should be
/// expanded to.
///
/// For instance, a tuple pattern (43u32, 'a') has sub pattern types [u32, char].
fn constructor_sub_pattern_tys<'a, 'tcx>(
    cx: &MatchCheckCtxt<'a, 'tcx>,
    ctor: &Constructor<'tcx>,
    ty: Ty<'tcx>,
) -> Vec<Ty<'tcx>> {
    debug!("constructor_sub_pattern_tys({:#?}, {:?})", ctor, ty);
    match ty.sty {
        ty::Tuple(ref fs) => fs.into_iter().map(|t| t.expect_ty()).collect(),
        ty::Slice(ty) | ty::Array(ty, _) => match *ctor {
            Slice(length) => (0..length).map(|_| ty).collect(),
            ConstantValue(_) => vec![],
            _ => bug!("bad slice pattern {:?} {:?}", ctor, ty)
        }
        ty::Ref(_, rty, _) => vec![rty],
        ty::Adt(adt, substs) => {
            if adt.is_box() {
                // Use T as the sub pattern type of Box<T>.
                vec![substs.type_at(0)]
            } else {
                adt.variants[ctor.variant_index_for_adt(cx, adt)].fields.iter().map(|field| {
                    let is_visible = adt.is_enum()
                        || field.vis.is_accessible_from(cx.module, cx.tcx);
                    if is_visible {
                        let ty = field.ty(cx.tcx, substs);
                        match ty.sty {
                            // If the field type returned is an array of an unknown
                            // size return an TyErr.
                            ty::Array(_, len) if len.assert_usize(cx.tcx).is_none() =>
                                cx.tcx.types.err,
                            _ => ty,
                        }
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

// checks whether a constant is equal to a user-written slice pattern. Only supports byte slices,
// meaning all other types will compare unequal and thus equal patterns often do not cause the
// second pattern to lint about unreachable match arms.
fn slice_pat_covered_by_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    _span: Span,
    const_val: &'tcx ty::Const<'tcx>,
    prefix: &[Pattern<'tcx>],
    slice: &Option<Pattern<'tcx>>,
    suffix: &[Pattern<'tcx>],
) -> Result<bool, ErrorReported> {
    let data: &[u8] = match (const_val.val, &const_val.ty.sty) {
        (ConstValue::ByRef { offset, alloc, .. }, ty::Array(t, n)) => {
            assert_eq!(*t, tcx.types.u8);
            let n = n.assert_usize(tcx).unwrap();
            let ptr = Pointer::new(AllocId(0), offset);
            alloc.get_bytes(&tcx, ptr, Size::from_bytes(n)).unwrap()
        },
        (ConstValue::Slice { data, start, end }, ty::Slice(t)) => {
            assert_eq!(*t, tcx.types.u8);
            let ptr = Pointer::new(AllocId(0), Size::from_bytes(start as u64));
            data.get_bytes(&tcx, ptr, Size::from_bytes((end - start) as u64)).unwrap()
        },
        // FIXME(oli-obk): create a way to extract fat pointers from ByRef
        (_, ty::Slice(_)) => return Ok(false),
        _ => bug!(
            "slice_pat_covered_by_const: {:#?}, {:#?}, {:#?}, {:#?}",
            const_val, prefix, slice, suffix,
        ),
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

// Whether to evaluate a constructor using exhaustive integer matching. This is true if the
// constructor is a range or constant with an integer type.
fn should_treat_range_exhaustively(tcx: TyCtxt<'tcx>, ctor: &Constructor<'tcx>) -> bool {
    let ty = match ctor {
        ConstantValue(value) => value.ty,
        ConstantRange(_, _, ty, _) => ty,
        _ => return false,
    };
    if let ty::Char | ty::Int(_) | ty::Uint(_) = ty.sty {
        !ty.is_pointer_sized() || tcx.features().precise_pointer_size_matching
    } else {
        false
    }
}

/// For exhaustive integer matching, some constructors are grouped within other constructors
/// (namely integer typed values are grouped within ranges). However, when specialising these
/// constructors, we want to be specialising for the underlying constructors (the integers), not
/// the groups (the ranges). Thus we need to split the groups up. Splitting them up naïvely would
/// mean creating a separate constructor for every single value in the range, which is clearly
/// impractical. However, observe that for some ranges of integers, the specialisation will be
/// identical across all values in that range (i.e., there are equivalence classes of ranges of
/// constructors based on their `is_useful_specialized` outcome). These classes are grouped by
/// the patterns that apply to them (in the matrix `P`). We can split the range whenever the
/// patterns that apply to that range (specifically: the patterns that *intersect* with that range)
/// change.
/// Our solution, therefore, is to split the range constructor into subranges at every single point
/// the group of intersecting patterns changes (using the method described below).
/// And voilà! We're testing precisely those ranges that we need to, without any exhaustive matching
/// on actual integers. The nice thing about this is that the number of subranges is linear in the
/// number of rows in the matrix (i.e., the number of cases in the `match` statement), so we don't
/// need to be worried about matching over gargantuan ranges.
///
/// Essentially, given the first column of a matrix representing ranges, looking like the following:
///
/// |------|  |----------| |-------|    ||
///    |-------| |-------|            |----| ||
///       |---------|
///
/// We split the ranges up into equivalence classes so the ranges are no longer overlapping:
///
/// |--|--|||-||||--||---|||-------|  |-|||| ||
///
/// The logic for determining how to split the ranges is fairly straightforward: we calculate
/// boundaries for each interval range, sort them, then create constructors for each new interval
/// between every pair of boundary points. (This essentially sums up to performing the intuitive
/// merging operation depicted above.)
fn split_grouped_constructors<'p, 'tcx>(
    tcx: TyCtxt<'tcx>,
    ctors: Vec<Constructor<'tcx>>,
    &Matrix(ref m): &Matrix<'p, 'tcx>,
    ty: Ty<'tcx>,
) -> Vec<Constructor<'tcx>> {
    let mut split_ctors = Vec::with_capacity(ctors.len());

    for ctor in ctors.into_iter() {
        match ctor {
            // For now, only ranges may denote groups of "subconstructors", so we only need to
            // special-case constant ranges.
            ConstantRange(..) if should_treat_range_exhaustively(tcx, &ctor) => {
                // We only care about finding all the subranges within the range of the constructor
                // range. Anything else is irrelevant, because it is guaranteed to result in
                // `NotUseful`, which is the default case anyway, and can be ignored.
                let ctor_range = IntRange::from_ctor(tcx, &ctor).unwrap();

                /// Represents a border between 2 integers. Because the intervals spanning borders
                /// must be able to cover every integer, we need to be able to represent
                /// 2^128 + 1 such borders.
                #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
                enum Border {
                    JustBefore(u128),
                    AfterMax,
                }

                // A function for extracting the borders of an integer interval.
                fn range_borders(r: IntRange<'_>) -> impl Iterator<Item = Border> {
                    let (lo, hi) = r.range.into_inner();
                    let from = Border::JustBefore(lo);
                    let to = match hi.checked_add(1) {
                        Some(m) => Border::JustBefore(m),
                        None => Border::AfterMax,
                    };
                    vec![from, to].into_iter()
                }

                // `borders` is the set of borders between equivalence classes: each equivalence
                // class lies between 2 borders.
                let row_borders = m.iter()
                    .flat_map(|row| IntRange::from_pat(tcx, row[0]))
                    .flat_map(|range| ctor_range.intersection(&range))
                    .flat_map(|range| range_borders(range));
                let ctor_borders = range_borders(ctor_range.clone());
                let mut borders: Vec<_> = row_borders.chain(ctor_borders).collect();
                borders.sort_unstable();

                // We're going to iterate through every pair of borders, making sure that each
                // represents an interval of nonnegative length, and convert each such interval
                // into a constructor.
                for IntRange { range, .. } in borders.windows(2).filter_map(|window| {
                    match (window[0], window[1]) {
                        (Border::JustBefore(n), Border::JustBefore(m)) => {
                            if n < m {
                                Some(IntRange { range: n..=(m - 1), ty })
                            } else {
                                None
                            }
                        }
                        (Border::JustBefore(n), Border::AfterMax) => {
                            Some(IntRange { range: n..=u128::MAX, ty })
                        }
                        (Border::AfterMax, _) => None,
                    }
                }) {
                    split_ctors.push(IntRange::range_to_ctor(tcx, ty, range));
                }
            }
            // Any other constructor can be used unchanged.
            _ => split_ctors.push(ctor),
        }
    }

    split_ctors
}

/// Checks whether there exists any shared value in either `ctor` or `pat` by intersecting them.
fn constructor_intersects_pattern<'p, 'tcx>(
    tcx: TyCtxt<'tcx>,
    ctor: &Constructor<'tcx>,
    pat: &'p Pattern<'tcx>,
) -> Option<SmallVec<[&'p Pattern<'tcx>; 2]>> {
    if should_treat_range_exhaustively(tcx, ctor) {
        match (IntRange::from_ctor(tcx, ctor), IntRange::from_pat(tcx, pat)) {
            (Some(ctor), Some(pat)) => {
                ctor.intersection(&pat).map(|_| {
                    let (pat_lo, pat_hi) = pat.range.into_inner();
                    let (ctor_lo, ctor_hi) = ctor.range.into_inner();
                    assert!(pat_lo <= ctor_lo && ctor_hi <= pat_hi);
                    smallvec![]
                })
            }
            _ => None,
        }
    } else {
        // Fallback for non-ranges and ranges that involve floating-point numbers, which are not
        // conveniently handled by `IntRange`. For these cases, the constructor may not be a range
        // so intersection actually devolves into being covered by the pattern.
        match constructor_covered_by_range(tcx, ctor, pat) {
            Ok(true) => Some(smallvec![]),
            Ok(false) | Err(ErrorReported) => None,
        }
    }
}

fn constructor_covered_by_range<'tcx>(
    tcx: TyCtxt<'tcx>,
    ctor: &Constructor<'tcx>,
    pat: &Pattern<'tcx>,
) -> Result<bool, ErrorReported> {
    let (from, to, end, ty) = match pat.kind {
        box PatternKind::Constant { value } => (value, value, RangeEnd::Included, value.ty),
        box PatternKind::Range(PatternRange { lo, hi, end, ty }) => (lo, hi, end, ty),
        _ => bug!("`constructor_covered_by_range` called with {:?}", pat),
    };
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
        ConstantRange(from, to, ty, RangeEnd::Included) => {
            let to = some_or_ok!(cmp_to(ty::Const::from_bits(
                tcx,
                to,
                ty::ParamEnv::empty().and(ty),
            )));
            let end = (to == Ordering::Less) ||
                      (end == RangeEnd::Included && to == Ordering::Equal);
            Ok(some_or_ok!(cmp_from(ty::Const::from_bits(
                tcx,
                from,
                ty::ParamEnv::empty().and(ty),
            ))) && end)
        },
        ConstantRange(from, to, ty, RangeEnd::Excluded) => {
            let to = some_or_ok!(cmp_to(ty::Const::from_bits(
                tcx,
                to,
                ty::ParamEnv::empty().and(ty)
            )));
            let end = (to == Ordering::Less) ||
                      (end == RangeEnd::Excluded && to == Ordering::Equal);
            Ok(some_or_ok!(cmp_from(ty::Const::from_bits(
                tcx,
                from,
                ty::ParamEnv::empty().and(ty)))
            ) && end)
        }
        Single => Ok(true),
        _ => bug!(),
    }
}

fn patterns_for_variant<'p, 'tcx>(
    subpatterns: &'p [FieldPattern<'tcx>],
    wild_patterns: &[&'p Pattern<'tcx>])
    -> SmallVec<[&'p Pattern<'tcx>; 2]>
{
    let mut result = SmallVec::from_slice(wild_patterns);

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
fn specialize<'p, 'a: 'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    r: &[&'p Pattern<'tcx>],
    constructor: &Constructor<'tcx>,
    wild_patterns: &[&'p Pattern<'tcx>],
) -> Option<SmallVec<[&'p Pattern<'tcx>; 2]>> {
    let pat = &r[0];

    let head = match *pat.kind {
        PatternKind::AscribeUserType { ref subpattern, .. } => {
            specialize(cx, ::std::slice::from_ref(&subpattern), constructor, wild_patterns)
        }

        PatternKind::Binding { .. } | PatternKind::Wild => {
            Some(SmallVec::from_slice(wild_patterns))
        }

        PatternKind::Variant { adt_def, variant_index, ref subpatterns, .. } => {
            let ref variant = adt_def.variants[variant_index];
            Some(Variant(variant.def_id))
                .filter(|variant_constructor| variant_constructor == constructor)
                .map(|_| patterns_for_variant(subpatterns, wild_patterns))
        }

        PatternKind::Leaf { ref subpatterns } => {
            Some(patterns_for_variant(subpatterns, wild_patterns))
        }

        PatternKind::Deref { ref subpattern } => {
            Some(smallvec![subpattern])
        }

        PatternKind::Constant { value } => {
            match *constructor {
                Slice(..) => {
                    // we extract an `Option` for the pointer because slices of zero elements don't
                    // necessarily point to memory, they are usually just integers. The only time
                    // they should be pointing to memory is when they are subslices of nonzero
                    // slices
                    let (alloc, offset, n, ty) = match value.ty.sty {
                        ty::Array(t, n) => {
                            match value.val {
                                ConstValue::ByRef { offset, alloc, .. } => (
                                    alloc,
                                    offset,
                                    n.unwrap_usize(cx.tcx),
                                    t,
                                ),
                                _ => span_bug!(
                                    pat.span,
                                    "array pattern is {:?}", value,
                                ),
                            }
                        },
                        ty::Slice(t) => {
                            match value.val {
                                ConstValue::Slice { data, start, end } => (
                                    data,
                                    Size::from_bytes(start as u64),
                                    (end - start) as u64,
                                    t,
                                ),
                                ConstValue::ByRef { .. } => {
                                    // FIXME(oli-obk): implement `deref` for `ConstValue`
                                    return None;
                                },
                                _ => span_bug!(
                                    pat.span,
                                    "slice pattern constant must be scalar pair but is {:?}",
                                    value,
                                ),
                            }
                        },
                        _ => span_bug!(
                            pat.span,
                            "unexpected const-val {:?} with ctor {:?}",
                            value,
                            constructor,
                        ),
                    };
                    if wild_patterns.len() as u64 == n {
                        // convert a constant slice/array pattern to a list of patterns.
                        let layout = cx.tcx.layout_of(cx.param_env.and(ty)).ok()?;
                        let ptr = Pointer::new(AllocId(0), offset);
                        (0..n).map(|i| {
                            let ptr = ptr.offset(layout.size * i, &cx.tcx).ok()?;
                            let scalar = alloc.read_scalar(
                                &cx.tcx, ptr, layout.size,
                            ).ok()?;
                            let scalar = scalar.not_undef().ok()?;
                            let value = ty::Const::from_scalar(cx.tcx, scalar, ty);
                            let pattern = Pattern {
                                ty,
                                span: pat.span,
                                kind: box PatternKind::Constant { value },
                            };
                            Some(&*cx.pattern_arena.alloc(pattern))
                        }).collect()
                    } else {
                        None
                    }
                }
                _ => {
                    // If the constructor is a:
                    //      Single value: add a row if the constructor equals the pattern.
                    //      Range: add a row if the constructor contains the pattern.
                    constructor_intersects_pattern(cx.tcx, constructor, pat)
                }
            }
        }

        PatternKind::Range { .. } => {
            // If the constructor is a:
            //      Single value: add a row if the pattern contains the constructor.
            //      Range: add a row if the constructor intersects the pattern.
            constructor_intersects_pattern(cx.tcx, constructor, pat)
        }

        PatternKind::Array { ref prefix, ref slice, ref suffix } |
        PatternKind::Slice { ref prefix, ref slice, ref suffix } => {
            match *constructor {
                Slice(..) => {
                    let pat_len = prefix.len() + suffix.len();
                    if let Some(slice_count) = wild_patterns.len().checked_sub(pat_len) {
                        if slice_count == 0 || slice.is_some() {
                            Some(prefix.iter().chain(
                                    wild_patterns.iter().map(|p| *p)
                                                 .skip(prefix.len())
                                                 .take(slice_count)
                                                 .chain(suffix.iter())
                            ).collect())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                ConstantValue(cv) => {
                    match slice_pat_covered_by_const(cx.tcx, pat.span, cv, prefix, slice, suffix) {
                        Ok(true) => Some(smallvec![]),
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
