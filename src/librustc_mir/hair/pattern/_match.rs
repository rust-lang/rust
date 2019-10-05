/// Note: most tests relevant to this file can be found (at the time of writing)
/// in src/tests/ui/pattern/usefulness. Also look out for rfc2008 (feature
/// non_exhaustive) tests.
///
/// # Introduction
///
/// This file includes the logic for exhaustiveness and usefulness checking for
/// pattern-matching. Specifically, given a list of patterns for a type, we can
/// tell whether:
/// (a) the patterns cover every possible constructor for the type [exhaustiveness]
/// (b) each pattern is necessary [usefulness]
///
/// The algorithm implemented here is based on the one described in:
/// http://moscova.inria.fr/~maranget/papers/warn/index.html
/// However, various modifications have been made to it so we keep it only as reference
/// and will describe the extended algorithm here (without being so rigorous).
///
/// The core of the algorithm revolves about a "usefulness" check. In particular, we
/// are trying to compute a predicate `U(P, q)` where `P` is a list of patterns.
/// `U(P, q)` represents whether, given an existing list of patterns
/// `P_1 ..= P_m`, adding a new pattern `q` will be "useful" (that is, cover previously-
/// uncovered values of the type).
///
/// If we have this predicate, then we can easily compute both exhaustiveness of an
/// entire set of patterns and the individual usefulness of each one.
/// (a) the set of patterns is exhaustive iff `U(P, _)` is false (i.e., adding a wildcard
/// match doesn't increase the number of values we're matching)
/// (b) a pattern `P_i` is not useful (i.e. unreachable) if `U(P[0..=(i-1), P_i)` is
/// false (i.e., adding a pattern to those that have come before it doesn't match any value
/// that wasn't matched previously).
///
///
/// # Pattern-stacks and matrices
///
/// The basic datastructure that we will make use of in the algorithm is a list of patterns that
/// the paper calls "pattern-vector" and that we call "pattern-stack". The idea is that we
/// start with a single pattern of interest,
/// and repeatedly unpack the top constructor to reveal its arguments. We keep the yet-untreated
/// arguments in the tail of the stack.
///
/// For example, say we start with the pattern `Foo(Bar(1, 2), Some(true), false)`. The
/// pattern-stack might then evolve as follows:
///   [Foo(Bar(1, 2), Some(_), false)] // Initially we have a single pattern in the stack
///   [Bar(1, 2), Some(_), false] // After unpacking the `Foo` constructor
///   [1, 2, Some(_), false] // After unpacking the `Bar` constructor
///   [2, Some(_), false] // After unpacking the `1` constructor
///   // etc.
///
/// We call the operation of popping the constructor on top of the stack "specialization", and we
/// write it `S(c, p)`, where `p` is a pattern-stack and `c` a specific constructor (like `Some`
/// or `None`). This operation returns zero or more altered pattern-stacks, as follows.
/// We look at the pattern `p_1` on top of the stack, and we have four cases:
///      1. `p_1 = c(r_1, .., r_a)`, i.e. the top of the stack has constructor `c`. We push
///         onto the stack the arguments of this constructor, and return the result:
///              r_1, .., r_a, p_2, .., p_n
///      2. `p_1 = c'(r_1, .., r_a')` where `c ≠ c'`. We discard the current stack and return
///         nothing.
///      3. `p_1 = _`. We push onto the stack as many wildcards as the constructor `c`
///         has arguments (its arity), and return the resulting stack:
///              _, .., _, p_2, .., p_n
///      4. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting stack:
///              S(c, (r_1, p_2, .., p_n))
///              S(c, (r_2, p_2, .., p_n))
///
/// Note that when the required constructor does not match the constructor on top of the stack, we
/// return nothing. Thus specialization filters pattern-stacks by the constructor on top of them.
///
/// We call a list of pattern-stacks a "matrix", because in the run of the algorithm they will
/// keep a rectangular shape. `S` operation extends straightforwardly to matrices by
/// working row-by-row using flat_map.
///
///
/// # Abstract algorithm
///
/// The algorithm itself is a function `U`, that takes as arguments a matrix `M` and a new pattern
/// `p`, both with the same number `n` of columns.
/// The algorithm is inductive (on the number of columns: i.e., components of pattern-stacks).
/// The algorithm is realised in the `is_useful` function.
///
/// Base case. (`n = 0`, i.e., an empty tuple pattern)
///     - If `M` already contains an empty pattern (i.e., if the number of patterns `m > 0`),
///       then `U(M, p)` is false.
///     - Otherwise, `M` must be empty, so `U(M, p)` is true.
///
/// Inductive step. (`n > 0`)
///     We look at `p_1`, the head of the pattern-stack `p`.
///
///     We first generate the list of constructors that are covered by a pattern `pat`. We name
///     this operation `pat_constructors`.
///         - If `pat == c(r_1, .., r_a)`, i.e. we have a constructor pattern. Then we just
///         return `c`:
///             `pat_constructors(pat) = [c]`
///
///         - If `pat == _`, then we return the list of all possible constructors for the
///         relevant type:
///             `pat_constructors(pat) = all_constructors(pat.ty)`
///
///         - If `pat == r_1 | r_2`, then we return the constructors for either branch of the
///         OR-pattern:
///             `pat_constructors(pat) = pat_constructors(r_1) + pat_constructors(r_2)`
///
///     Then for each constructor `c` in `pat_constructors(p_1)`, we want to check whether a value
///     that starts with this constructor may show that `p` is useful, i.e. may match `p` but not
///     be matched by the matrix above.
///     For that, we only care about those rows of `M` whose first component covers the
///     constructor `c`; and for those rows that do, we want to unpack the arguments to `c` to check
///     further that `p` matches additional values.
///     This is where specialization comes in: this check amounts to computing `U(S(c, M), S(c,
///     p))`. More details can be found in the paper.
///
///     Thus we get: `U(M, p) := ∃(c ϵ pat_constructors(p_1)) U(S(c, M), S(c, p))`
///
///     Note that for c ϵ pat_constructors(p_1), `S(c, P)` always returns exactly one element, so
///     the formula above makes sense.
///
/// This algorithm however has a lot of practical issues. Most importantly, it may not terminate
/// in the presence of recursive types, since we always unpack all constructors as much
/// as possible. And it would be stupidly slow anyways for types with a lot of constructors,
/// like `u64` of `&[bool]`. We therefore present a modified version after the example.
///
///
/// # Example run of the algorithm
///
/// Assume we have the following match. We want to know whether it is exhaustive, i.e. whether
/// an additional `_` pattern would be useful (would be reachable).
/// ```
///     match x {
///         Some(true) => {}
///         None => {}
///     }
/// ```
///
/// We start with the following `M` and `p`:
/// M = [ [Some(true)],
///       [None] ]
/// p =   [_]
/// `pat_constructors(p)` returns `[None, Some]`
///
/// We specialize on the `None` constructor first:
/// S(None, M) = [ [] ]
/// S(None, p) =   []
/// We hit the base case n = 0: since bool is inhabited, `U(S(None, M), S(None, p)) = false`.
///
/// We specialize on the `Some` constructor second:
/// S(Some, M) = [ [true] ]
/// S(Some, p) =   [_]
/// Let M' := S(Some, M) and p' := S(Some, p).
///
/// `pat_constructors(p')` returns `[true, false]`
/// S(true, M') = [ [] ]
/// S(true, p') =   []
/// So `U(S(true, M'), S(true, p')) = false`
///
/// S(false, M') = []
/// S(false, p') = []
/// So `U(S(false, M'), S(false, p')) = true`
///
/// Therefore `U(M, p) = true`, indeed by following the steps taken we can recover that
/// the pattern `Some(false)` was not covered by the initial match.
///
///
/// # Concrete algorithm
///
/// To make the algorithm tractable, we introduce the notion of meta-constructors. A
/// meta-constructor stands for a particular group of constructors. The typical example
/// is the wildcard `_`, which stands for all the constructors of a given type.
///
/// In practice, the meta-constructors we make use of in this file are the following:
///     - any normal constructor is also a metaconstructor with exactly one member;
///     - the wildcard `_`, that captures all constructors of a given type;
///     - the constant range `x..y` that captures a range of values for types that support
///     it, like integers;
///     - the variable-length slice `[x, y, .., z]`, that captures all slice constructors
///     from a given length onwards;
///     - the "missing constructors" metaconstructor, that captures a provided arbitrary group
///     of constructors.
///
/// We first redefine `pat_constructors` to potentially return a metaconstructor when relevant
/// for a pattern.
///
/// We then add a step to the algorithm: a function `split_metaconstructor(mc, M)` that returns
/// a list of metaconstructors, with the following properties:
///     - the set of base constructors covered by the output must be the same as covered by `mc`;
///     - for each metaconstructor `k` in the output, all the `c ϵ k` behave the same relative
///     to `M`. More precisely, we want that for any two `c1` and `c2` in `k`,
///     `U(S(c1, M), S(c1, p))` iff `U(S(c2, M), S(c2, p))`;
///     - if the first column of `M` is only wildcards, then the function returns at most
///     `[mc]` on its own;
///     - if the relevant type is uninhabited, the function returns nothing.
/// Any function that has those properties ensures correctness of the algorithm. We will of course
/// try to pick a function that also ensures good performance.
/// The idea is that we still need to try different constructors, but we try to keep them grouped
/// together when possible to avoid doing redundant work.
///
/// Here is roughly how splitting works for us:
///     - for wildcards, there are two cases:
///         - if all the possible constructors of the relevant type exist in the first column
///         of `M`, then we return the list of all those constructors, like we did before;
///         - if however some constructors are missing, then it turns out that considering
///         those missing constructors is enough. We return a "missing constructors" meta-
///         contructor that carries the missing constructors in question.
///     (Note the similarity with the algorithm from the paper. It is not a coincidence)
///     - for ranges, we split the range into a disjoint set of subranges, see the code for details;
///     - for slices, we split the slice into a number of fixed-length slices and one longer
///     variable-length slice, again see code;
///
/// Thus we get the new inductive step (i.e. when `n > 0`):
///     `U(M, p) :=
///         ∃(mc ϵ pat_constructors(p_1))
///         ∃(mc' ϵ split_metaconstructor(mc, M))
///         U(S(c, M), S(c, p)) for some c ϵ mc'`
/// Note: in the case of an uninhabited type, there won't be any `mc'` so this just returns false.
///
/// Note that the termination of the algorithm now depends on the behaviour of the splitting
/// phase. However, from the third property of the splitting function,
/// we can see that the depth of splitting of the algorithm is bounded by some
/// function of the depths of the patterns fed to it initially. So we're confident that
/// it terminates.
///
/// This algorithm is equivalent to the one presented in the paper if we only consider
/// wildcards. Thus this mostly extends the original algorithm to ranges and variable-length
/// slices, while removing the special-casing of the wildcard pattern. We also additionally
/// support uninhabited types.
use self::Constructor::*;
use self::Usefulness::*;
use self::WitnessPreference::*;

use rustc_index::vec::Idx;

use super::{compare_const_vals, PatternFoldable, PatternFolder};
use super::{FieldPat, Pat, PatKind, PatRange};

use rustc::hir::def_id::DefId;
use rustc::hir::RangeEnd;
use rustc::ty::layout::{Integer, IntegerExt, Size, VariantIdx};
use rustc::ty::{self, Const, Ty, TyCtxt, TypeFoldable};

use rustc::mir::interpret::{truncate, AllocId, ConstValue, Pointer, Scalar};
use rustc::mir::Field;
use rustc::util::common::ErrorReported;

use syntax::attr::{SignedInt, UnsignedInt};
use syntax_pos::{Span, DUMMY_SP};

use arena::TypedArena;

use smallvec::{smallvec, SmallVec};
use std::cmp::{self, max, min, Ordering};
use std::convert::TryInto;
use std::fmt;
use std::iter::{self, FromIterator, IntoIterator};
use std::ops::RangeInclusive;
use std::u128;

pub fn expand_pattern<'a, 'tcx>(cx: &MatchCheckCtxt<'a, 'tcx>, pat: Pat<'tcx>) -> &'a Pat<'tcx> {
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
        match (val, &crty.kind, &rty.kind) {
            // the easy case, deref a reference
            (ConstValue::Scalar(Scalar::Ptr(p)), x, y) if x == y => {
                let alloc = self.tcx.alloc_map.lock().unwrap_memory(p.alloc_id);
                ConstValue::ByRef { alloc, offset: p.offset }
            }
            // unsize array to slice if pattern is array but match value or other patterns are slice
            (ConstValue::Scalar(Scalar::Ptr(p)), ty::Array(t, n), ty::Slice(u)) => {
                assert_eq!(t, u);
                ConstValue::Slice {
                    data: self.tcx.alloc_map.lock().unwrap_memory(p.alloc_id),
                    start: p.offset.bytes().try_into().unwrap(),
                    end: n.eval_usize(self.tcx, ty::ParamEnv::empty()).try_into().unwrap(),
                }
            }
            // fat pointers stay the same
            (ConstValue::Slice { .. }, _, _)
            | (_, ty::Slice(_), ty::Slice(_))
            | (_, ty::Str, ty::Str) => val,
            // FIXME(oli-obk): this is reachable for `const FOO: &&&u32 = &&&42;` being used
            _ => bug!("cannot deref {:#?}, {} -> {}", val, crty, rty),
        }
    }
}

impl PatternFolder<'tcx> for LiteralExpander<'tcx> {
    fn fold_pattern(&mut self, pat: &Pat<'tcx>) -> Pat<'tcx> {
        debug!("fold_pattern {:?} {:?} {:?}", pat, pat.ty.kind, pat.kind);
        match (&pat.ty.kind, &*pat.kind) {
            (
                &ty::Ref(_, rty, _),
                &PatKind::Constant {
                    value: Const { val, ty: ty::TyS { kind: ty::Ref(_, crty, _), .. } },
                },
            ) => Pat {
                ty: pat.ty,
                span: pat.span,
                kind: box PatKind::Deref {
                    subpattern: Pat {
                        ty: rty,
                        span: pat.span,
                        kind: box PatKind::Constant {
                            value: self.tcx.mk_const(Const {
                                val: self.fold_const_value_deref(*val, rty, crty),
                                ty: rty,
                            }),
                        },
                    },
                },
            },
            (_, &PatKind::Binding { subpattern: Some(ref s), .. }) => s.fold_with(self),
            _ => pat.super_fold_with(self),
        }
    }
}

/// A row of a matrix. Rows of len 1 are very common, which is why `SmallVec[_; 2]`
/// works well.
#[derive(Debug, Clone)]
pub struct PatStack<'p, 'tcx>(SmallVec<[&'p Pat<'tcx>; 2]>);

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    pub fn from_pattern(pat: &'p Pat<'tcx>) -> Self {
        PatStack(smallvec![pat])
    }
    fn empty() -> Self {
        PatStack(smallvec![])
    }
    fn from_vec(vec: SmallVec<[&'p Pat<'tcx>; 2]>) -> Self {
        PatStack(vec)
    }
    fn from_slice(s: &[&'p Pat<'tcx>]) -> Self {
        PatStack(SmallVec::from_slice(s))
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn head<'a, 'p2>(&'a self) -> &'p2 Pat<'tcx>
    where
        'p: 'p2,
    {
        self.0[0]
    }
    fn iter(&self) -> impl Iterator<Item = &Pat<'tcx>> {
        self.0.iter().map(|p| *p)
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize<'a, 'p2>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        constructor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &[&'p2 Pat<'tcx>],
    ) -> SmallVec<[PatStack<'p2, 'tcx>; 1]>
    where
        'a: 'p2,
        'p: 'p2,
    {
        let new_heads = specialize_one_pattern(cx, self.head(), constructor, ctor_wild_subpatterns);
        let result = new_heads
            .into_iter()
            .map(|mut new_head| {
                new_head.0.extend_from_slice(&self.0[1..]);
                new_head
            })
            .collect();
        debug!("specialize({:#?}, {:#?}) = {:#?}", self, constructor, result);
        result
    }
}

impl<'p, 'tcx> Default for PatStack<'p, 'tcx> {
    fn default() -> Self {
        PatStack(smallvec![])
    }
}

impl<'p, 'tcx> FromIterator<&'p Pat<'tcx>> for PatStack<'p, 'tcx> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'p Pat<'tcx>>,
    {
        PatStack(iter.into_iter().collect())
    }
}

/// A 2D matrix.
pub struct Matrix<'p, 'tcx>(Vec<PatStack<'p, 'tcx>>);

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    pub fn empty() -> Self {
        Matrix(vec![])
    }

    pub fn push(&mut self, row: PatStack<'p, 'tcx>) {
        self.0.push(row)
    }

    /// Iterate over the first component of each row
    // Can't return impl Iterator because of hidden lifetime capture.
    fn heads<'a, 'p2>(
        &'a self,
    ) -> iter::Map<
        std::slice::Iter<'a, PatStack<'p, 'tcx>>,
        impl FnMut(&'a PatStack<'p, 'tcx>) -> &'p2 Pat<'tcx>,
    >
    where
        'p: 'p2,
        'a: 'p2,
    {
        self.0.iter().map(|r| r.head())
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize<'a, 'p2>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        constructor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &[&'p2 Pat<'tcx>],
    ) -> Matrix<'p2, 'tcx>
    where
        'a: 'p2,
        'p: 'p2,
    {
        Matrix(
            self.0
                .iter()
                .flat_map(|r| r.specialize(cx, constructor, ctor_wild_subpatterns))
                .collect(),
        )
    }
}

/// Pretty-printer for matrices of patterns, example:
/// +++++++++++++++++++++++++++++
/// + _     + []                +
/// +++++++++++++++++++++++++++++
/// + true  + [First]           +
/// +++++++++++++++++++++++++++++
/// + true  + [Second(true)]    +
/// +++++++++++++++++++++++++++++
/// + false + [_]               +
/// +++++++++++++++++++++++++++++
/// + _     + [_, _, tail @ ..] +
/// +++++++++++++++++++++++++++++
impl<'p, 'tcx> fmt::Debug for Matrix<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n")?;

        let &Matrix(ref m) = self;
        let pretty_printed_matrix: Vec<Vec<String>> =
            m.iter().map(|row| row.iter().map(|pat| format!("{:?}", pat)).collect()).collect();

        let column_count = m.iter().map(|row| row.len()).max().unwrap_or(0);
        assert!(m.iter().all(|row| row.len() == column_count));
        let column_widths: Vec<usize> = (0..column_count)
            .map(|col| pretty_printed_matrix.iter().map(|row| row[col].len()).max().unwrap_or(0))
            .collect();

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

impl<'p, 'tcx> FromIterator<PatStack<'p, 'tcx>> for Matrix<'p, 'tcx> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = PatStack<'p, 'tcx>>,
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
    pub pattern_arena: &'a TypedArena<Pat<'tcx>>,
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

        f(MatchCheckCtxt { tcx, param_env, module, pattern_arena: &pattern_arena })
    }

    fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        if self.tcx.features().exhaustive_patterns {
            self.tcx.is_ty_uninhabited_from(self.module, ty)
        } else {
            false
        }
    }

    fn is_non_exhaustive_variant<'p>(&self, pattern: &'p Pat<'tcx>) -> bool {
        match *pattern.kind {
            PatKind::Variant { adt_def, variant_index, .. } => {
                let ref variant = adt_def.variants[variant_index];
                variant.is_field_list_non_exhaustive()
            }
            _ => false,
        }
    }

    fn is_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
        match ty.kind {
            ty::Adt(adt_def, ..) => adt_def.is_variant_list_non_exhaustive(),
            _ => false,
        }
    }

    fn is_local(&self, ty: Ty<'tcx>) -> bool {
        match ty.kind {
            ty::Adt(adt_def, ..) => adt_def.did.is_local(),
            _ => false,
        }
    }
}

/// Constructors and metaconstructors.
#[derive(Clone, Debug, PartialEq)]
enum Constructor<'tcx> {
    // Base constructors
    /// The constructor of all patterns that don't vary by constructor,
    /// e.g., struct patterns and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(DefId),
    /// Literal values.
    ConstantValue(&'tcx ty::Const<'tcx>),
    /// Array patterns of length n.
    FixedLenSlice(u64),

    // Meta-constructors
    /// Ranges of literal values (`2..=5` and `2..5`).
    ConstantRange(u128, u128, Ty<'tcx>, RangeEnd),
    /// Slice patterns. Captures any array constructor of length >= i+j.
    VarLenSlice(u64, u64),
    /// Wildcard metaconstructor.
    Wildcard,
    /// List of constructors that were _not_ present in the first column
    /// of the matrix when encountering a wildcard. The contained list must
    /// be nonempty.
    /// This is only used in the output of splitting the wildcard metaconstructor.
    MissingConstructors(MissingConstructors<'tcx>),
}

impl<'tcx> Constructor<'tcx> {
    fn is_slice(&self) -> bool {
        match self {
            FixedLenSlice { .. } => true,
            _ => false,
        }
    }

    fn is_wildcard(&self) -> bool {
        match self {
            Wildcard => true,
            MissingConstructors(_) => bug!(
                "Not sure if MissingConstructors should be a wildcard. Shouldn't happen anyways."
            ),
            _ => false,
        }
    }

    fn variant_index_for_adt<'a>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        adt: &'tcx ty::AdtDef,
    ) -> VariantIdx {
        match self {
            Variant(id) => adt.variant_index_with_id(*id),
            Single => {
                assert!(!adt.is_enum());
                VariantIdx::new(0)
            }
            ConstantValue(c) => crate::const_eval::const_variant_index(cx.tcx, cx.param_env, c),
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    /// Split a constructor into equivalence classes of constructors that behave the same
    /// for the given matrix. See description of the algorithm for details.
    /// Note: We can rely on this returning an empty list if the type is uninhabited and
    /// we're not in the privately_empty case.
    fn split_meta_constructor(
        self,
        cx: &MatchCheckCtxt<'_, 'tcx>,
        pcx: PatCtxt<'tcx>,
        head_ctors: &Vec<Constructor<'tcx>>,
    ) -> SmallVec<[Constructor<'tcx>; 1]> {
        debug!("split_meta_constructor {:?}", self);
        assert!(!head_ctors.iter().any(|c| c.is_wildcard()));

        match &self {
            // Any base constructor can be used unchanged.
            Single | Variant(_) | ConstantValue(_) | FixedLenSlice(_) => smallvec![self],
            ConstantRange(..) if should_treat_range_exhaustively(cx.tcx, &self) => {
                // For exhaustive integer matching, some constructors are grouped within other
                // constructors (namely integer typed values are grouped within ranges). However,
                // when specialising these constructors, we want to be specialising for the
                // underlying constructors (the integers), not the groups (the ranges). Thus we
                // need to split the groups up. Splitting them up naïvely would mean creating a
                // separate constructor for every single value in the range, which is clearly
                // impractical. However, observe that for some ranges of integers, the
                // specialisation will be identical across all values in that range (i.e., there
                // are equivalence classes of ranges of constructors based on their
                // `is_useful_specialized` outcome). These classes are grouped by the patterns that
                // apply to them (in the matrix `P`). We can split the range whenever the patterns
                // that apply to that range (specifically: the patterns that *intersect* with that
                // range) change.
                // Our solution, therefore, is to split the range constructor into subranges at
                // every single point the group of intersecting patterns changes (using the method
                // described below). And voilà! We're testing precisely those ranges that we need
                // to, without any exhaustive matching on actual integers. The nice thing about
                // this is that the number of subranges is linear in the number of rows in the
                // matrix (i.e., the number of cases in the `match` statement), so we don't need to
                // be worried about matching over gargantuan ranges.
                //
                // Essentially, given the first column of a matrix representing ranges, looking
                // like the following:
                //
                // |------|  |----------| |-------|    ||
                //    |-------| |-------|            |----| ||
                //       |---------|
                //
                // We split the ranges up into equivalence classes so the ranges are no longer
                // overlapping:
                //
                // |--|--|||-||||--||---|||-------|  |-|||| ||
                //
                // The logic for determining how to split the ranges is fairly straightforward: we
                // calculate boundaries for each interval range, sort them, then create
                // constructors for each new interval between every pair of boundary points. (This
                // essentially sums up to performing the intuitive merging operation depicted
                // above.)

                // We only care about finding all the subranges within the range of the constructor
                // range. Anything else is irrelevant, because it is guaranteed to result in
                // `NotUseful`, which is the default case anyway, and can be ignored.
                let ctor_range = IntRange::from_ctor(cx.tcx, cx.param_env, &self).unwrap();

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
                let row_borders = head_ctors
                    .iter()
                    .flat_map(|ctor| IntRange::from_ctor(cx.tcx, cx.param_env, ctor))
                    .flat_map(|range| ctor_range.intersection(&range))
                    .flat_map(|range| range_borders(range));
                let ctor_borders = range_borders(ctor_range.clone());
                let mut borders: Vec<_> = row_borders.chain(ctor_borders).collect();
                borders.sort_unstable();

                // We're going to iterate through every adjacent pair of borders, making sure that
                // each represents an interval of nonnegative length, and convert each such
                // interval into a constructor.
                borders
                    .windows(2)
                    .filter_map(|window| match (window[0], window[1]) {
                        (Border::JustBefore(n), Border::JustBefore(m)) => {
                            if n < m {
                                Some(n..=(m - 1))
                            } else {
                                None
                            }
                        }
                        (Border::JustBefore(n), Border::AfterMax) => Some(n..=u128::MAX),
                        (Border::AfterMax, _) => None,
                    })
                    .map(|range| IntRange::range_to_ctor(cx.tcx, pcx.ty, range))
                    .collect()
            }
            ConstantRange(..) => smallvec![self],
            VarLenSlice(prefix, suffix) => {
                (prefix + suffix..pcx.max_slice_length + 1).map(FixedLenSlice).collect()
            }
            Wildcard => {
                let is_declared_nonexhaustive =
                    !cx.is_local(pcx.ty) && cx.is_non_exhaustive_enum(pcx.ty);

                // `all_ctors` are all the constructors for the given type, which
                // should all be represented (or caught with the wild pattern `_`).
                let all_ctors = all_constructors(cx, pcx);

                let is_privately_empty = all_ctors.is_empty() && !cx.is_uninhabited(pcx.ty);

                // For privately empty and non-exhaustive enums, we work as if there were an "extra"
                // `_` constructor for the type, so we can never match over all constructors.
                let is_non_exhaustive = is_privately_empty
                    || is_declared_nonexhaustive
                    || (pcx.ty.is_ptr_sized_integral()
                        && !cx.tcx.features().precise_pointer_size_matching);

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
                // non-wildcard patterns in the current column.
                let missing_ctors = MissingConstructors::new(
                    pcx,
                    cx.tcx,
                    cx.param_env,
                    all_ctors,
                    head_ctors.clone(),
                );
                debug!(
                    "missing_ctors.is_empty()={:#?} is_non_exhaustive={:#?}",
                    missing_ctors.is_empty(),
                    is_non_exhaustive,
                );

                // If there are some missing constructors, we only need to specialize relative
                // to them and we can ignore the other ones. Otherwise, we have to try all
                // existing constructors one-by-one.
                if is_non_exhaustive {
                    // We pretend the type has an additional `_` constructor, that counts as a
                    // missing constructor. So we return that constructor.
                    smallvec![Wildcard]
                } else if !missing_ctors.is_empty() {
                    if head_ctors.is_empty() {
                        // If head_ctors is empty, then all constructors of the type behave the same
                        // so we can keep the Wildcard metaconstructor.
                        smallvec![Wildcard]
                    } else {
                        // Otherwise, we have a set of missing constructors that is neither empty
                        // not equal to all_constructors. Since all missing constructors will
                        // behave the same (i.e. will be matched only by wildcards), we return a
                        // metaconstructor that contains all of them at once.
                        smallvec![MissingConstructors(missing_ctors)]
                    }
                } else {
                    // Here we know there are no missing constructors, so we have to try all
                    // existing constructors one-by-one.
                    let (all_ctors, _) = missing_ctors.into_inner();
                    // Recursively split newly generated list of constructors. This list must not
                    // contain any wildcards so we don't recurse infinitely.
                    all_ctors
                        .into_iter()
                        .flat_map(|ctor| ctor.split_meta_constructor(cx, pcx, head_ctors))
                        .collect()
                }
            }
            MissingConstructors(_) => bug!("shouldn't try to split constructor {:?}", self),
        }
    }

    /// Returns a collection of constructors that spans the constructors covered by `self`,
    /// subtracted by the constructors covered by `head_ctors`: i.e., `self \ head_ctors` (in set
    /// notation).
    fn subtract_meta_constructor(
        self,
        _pcx: PatCtxt<'tcx>,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        used_ctors: &Vec<Constructor<'tcx>>,
    ) -> SmallVec<[Constructor<'tcx>; 1]> {
        debug!("subtract_meta_constructor {:?}", self);
        assert!(!used_ctors.iter().any(|c| c.is_wildcard()));

        match self {
            // Those constructors can't match a non-wildcard metaconstructor, so we're fine
            // just comparing for equality.
            Single | Variant(_) => {
                if used_ctors.iter().any(|c| c == &self) {
                    smallvec![]
                } else {
                    smallvec![self]
                }
            }
            FixedLenSlice(self_len) => {
                let overlaps = |c: &Constructor<'_>| match c {
                    FixedLenSlice(other_len) => *other_len == self_len,
                    VarLenSlice(prefix, suffix) => prefix + suffix <= self_len,
                    _ => false,
                };
                if used_ctors.iter().any(overlaps) { smallvec![] } else { smallvec![self] }
            }
            VarLenSlice(self_prefix, self_suffix) => {
                let self_len = self_prefix + self_suffix;
                // Initially we cover all slice lengths starting from self_len.

                // If there is a VarLenSlice(n) in used_ctors, then we have to discard
                // all lengths >= n. So we pick the smallest one.
                let max_len: Option<_> = used_ctors
                    .iter()
                    .filter_map(|c: &Constructor<'tcx>| match c {
                        VarLenSlice(prefix, suffix) => Some(prefix + suffix),
                        _ => None,
                    })
                    .min();

                // If max_len <= self_len there are no lengths remaining.
                if let Some(max_len) = max_len {
                    if max_len <= self_len {
                        return smallvec![];
                    }
                }

                // The remaining range of lengths is now either `self_len..`
                // or `self_len..max_len`. We then remove from that range all the
                // individual FixedLenSlice lengths in used_ctors. For that,
                // we extract all those lengths that are in our remaining range and
                // sort them. Every such length becomes a boundary between ranges
                // of lengths that will remain.
                #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
                enum Length {
                    Start,
                    Boundary(u64),
                }
                use Length::*;

                let mut lengths: Vec<_> = used_ctors
                    .iter()
                    .filter_map(|c: &Constructor<'tcx>| match c {
                        FixedLenSlice(other_len) => Some(*other_len),
                        _ => None,
                    })
                    .filter(|l| *l >= self_len)
                    .filter(|l| match max_len {
                        Some(max_len) => *l < max_len,
                        None => true,
                    })
                    .chain(max_len) // Add max_len as the final boundary
                    .map(Boundary)
                    .chain(Some(Start)) // Add a special starting boundary
                    .collect();
                lengths.sort_unstable();
                lengths.dedup();

                let mut remaining_ctors: SmallVec<_> = lengths
                    .windows(2)
                    .flat_map(|window| match (window[0], window[1]) {
                        (Boundary(n), Boundary(m)) => (n + 1..m),
                        (Start, Boundary(m)) => (self_len..m),
                        _ => bug!(),
                    })
                    .map(FixedLenSlice)
                    .collect();

                // If there was a max_len, then we're done. Otherwise, we
                // still need to include all lengths starting from the longest
                // one til infinity, using VarLenSlice.
                if max_len.is_none() {
                    let final_length = match lengths.last().unwrap() {
                        Start => self_len,
                        Boundary(n) => n + 1,
                    };
                    remaining_ctors.push(VarLenSlice(final_length, 0));
                }

                remaining_ctors
            }
            ConstantRange(..) | ConstantValue(..) => {
                let mut remaining_ctors = smallvec![self];

                // For each used ctor, subtract from the current set of constructors.
                for used_ctor in used_ctors {
                    remaining_ctors = remaining_ctors
                        .into_iter()
                        .filter(|ctor| ctor != used_ctor)
                        .flat_map(|ctor| -> SmallVec<[Constructor<'tcx>; 2]> {
                            if let Some(interval) = IntRange::from_ctor(tcx, param_env, used_ctor) {
                                interval.subtract_from(tcx, param_env, ctor)
                            } else {
                                smallvec![ctor]
                            }
                        })
                        .collect();

                    // If the constructors that have been considered so far already cover
                    // the entire range of `self`, no need to look at more constructors.
                    if remaining_ctors.is_empty() {
                        break;
                    }
                }

                remaining_ctors
            }
            Wildcard | MissingConstructors(_) => {
                bug!("shouldn't try to subtract constructor {:?}", self)
            }
        }
    }

    /// This returns one wildcard pattern for each argument to this constructor.
    fn wildcard_subpatterns<'a>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        ty: Ty<'tcx>,
    ) -> impl Iterator<Item = Pat<'tcx>> + DoubleEndedIterator {
        debug!("wildcard_subpatterns({:#?}, {:?})", self, ty);
        let subpattern_types = match *self {
            Single | Variant(_) => match ty.kind {
                ty::Tuple(ref fs) => fs.into_iter().map(|t| t.expect_ty()).collect(),
                ty::Ref(_, rty, _) => vec![rty],
                ty::Adt(adt, substs) => {
                    if adt.is_box() {
                        // Use T as the sub pattern type of Box<T>.
                        vec![substs.type_at(0)]
                    } else {
                        adt.variants[self.variant_index_for_adt(cx, adt)]
                            .fields
                            .iter()
                            .map(|field| {
                                let is_visible = adt.is_enum()
                                    || field.vis.is_accessible_from(cx.module, cx.tcx);
                                if is_visible {
                                    let ty = field.ty(cx.tcx, substs);
                                    match ty.kind {
                                        // If the field type returned is an array of an unknown
                                        // size return an TyErr.
                                        ty::Array(_, len)
                                            if len
                                                .try_eval_usize(cx.tcx, cx.param_env)
                                                .is_none() =>
                                        {
                                            cx.tcx.types.err
                                        }
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
                            })
                            .collect()
                    }
                }
                ty::Slice(ty) | ty::Array(ty, _) => bug!("bad slice pattern {:?} {:?}", self, ty),
                _ => vec![],
            },
            FixedLenSlice(length) => match ty.kind {
                ty::Slice(ty) | ty::Array(ty, _) => (0..length).map(|_| ty).collect(),
                _ => bug!("bad slice pattern {:?} {:?}", self, ty),
            },
            VarLenSlice(prefix, suffix) => match ty.kind {
                ty::Slice(ty) | ty::Array(ty, _) => (0..prefix + suffix).map(|_| ty).collect(),
                _ => bug!("bad slice pattern {:?} {:?}", self, ty),
            },
            ConstantValue(_) | MissingConstructors(_) | ConstantRange(..) | Wildcard => vec![],
        };

        subpattern_types.into_iter().map(|ty| Pat { ty, span: DUMMY_SP, kind: box PatKind::Wild })
    }

    /// This computes the arity of a constructor. The arity of a constructor
    /// is how many subpattern patterns of that constructor should be expanded to.
    ///
    /// For instance, a tuple pattern `(_, 42, Some([]))` has the arity of 3.
    /// A struct pattern's arity is the number of fields it contains, etc.
    fn arity<'a>(&self, cx: &MatchCheckCtxt<'a, 'tcx>, ty: Ty<'tcx>) -> u64 {
        debug!("Constructor::arity({:#?}, {:?})", self, ty);
        match *self {
            Single | Variant(_) => match ty.kind {
                ty::Tuple(ref fs) => fs.len() as u64,
                ty::Ref(..) => 1,
                ty::Adt(adt, _) => {
                    adt.variants[self.variant_index_for_adt(cx, adt)].fields.len() as u64
                }
                ty::Slice(..) | ty::Array(..) => bug!("bad slice pattern {:?} {:?}", self, ty),
                _ => 0,
            },
            FixedLenSlice(length) => length,
            VarLenSlice(prefix, suffix) => prefix + suffix,
            ConstantValue(_) | ConstantRange(..) | Wildcard | MissingConstructors(_) => 0,
        }
    }

    /// Apply a constructor to a list of patterns, yielding a new pattern. `pats`
    /// must have as many elements as this constructor's arity.
    ///
    /// Examples:
    /// self: Single
    /// ty: tuple of 3 elements
    /// pats: [10, 20, _]           => (10, 20, _)
    ///
    /// self: Option::Some
    /// ty: Option<bool>
    /// pats: [false]  => Some(false)
    fn apply<'a>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        pcx: PatCtxt<'tcx>,
        ty: Ty<'tcx>,
        pats: impl IntoIterator<Item = Pat<'tcx>>,
    ) -> SmallVec<[Pat<'tcx>; 1]> {
        let mut pats = pats.into_iter();
        let pat = match self {
            Single | Variant(_) => match ty.kind {
                ty::Adt(..) | ty::Tuple(..) => {
                    let pats = pats
                        .enumerate()
                        .map(|(i, p)| FieldPat { field: Field::new(i), pattern: p })
                        .collect();

                    if let ty::Adt(adt, substs) = ty.kind {
                        if adt.is_enum() {
                            PatKind::Variant {
                                adt_def: adt,
                                substs,
                                variant_index: self.variant_index_for_adt(cx, adt),
                                subpatterns: pats,
                            }
                        } else {
                            PatKind::Leaf { subpatterns: pats }
                        }
                    } else {
                        PatKind::Leaf { subpatterns: pats }
                    }
                }
                ty::Ref(..) => PatKind::Deref { subpattern: pats.nth(0).unwrap() },
                _ => PatKind::Wild,
            },
            FixedLenSlice(_) => {
                PatKind::Slice { prefix: pats.collect(), slice: None, suffix: vec![] }
            }
            VarLenSlice(_, _) => match ty.kind {
                ty::Slice(ty) | ty::Array(ty, _) => {
                    let prefix = pats.collect();
                    if cx.tcx.features().slice_patterns {
                        let wild = Pat { ty, span: DUMMY_SP, kind: Box::new(PatKind::Wild) };
                        PatKind::Slice { prefix, slice: Some(wild), suffix: vec![] }
                    } else {
                        // We don't want to output a variable-length slice pattern if the
                        // slice_patterns feature is not enabled.
                        // The constructor covers infinitely many slice lengths, but for diagnostic
                        // purposes it is correct to return only some examples of non-covered
                        // patterns. So we just return the smallest length pattern here.
                        PatKind::Slice { prefix, slice: None, suffix: vec![] }
                    }
                }
                _ => bug!("bad slice pattern {:?} {:?}", self, ty),
            },
            ConstantValue(value) => PatKind::Constant { value },
            ConstantRange(lo, hi, ty, end) => PatKind::Range(PatRange {
                lo: ty::Const::from_bits(cx.tcx, *lo, ty::ParamEnv::empty().and(ty)),
                hi: ty::Const::from_bits(cx.tcx, *hi, ty::ParamEnv::empty().and(ty)),
                end: *end,
            }),
            Wildcard => PatKind::Wild,
            MissingConstructors(missing_ctors) => {
                // Construct for each missing constructor a "wildcard" version of this
                // constructor, that matches everything that can be built with
                // it. For example, if `ctor` is a `Constructor::Variant` for
                // `Option::Some`, we get the pattern `Some(_)`.
                return missing_ctors
                    .iter()
                    .flat_map(|ctor| ctor.apply_wildcards(cx, pcx, ty))
                    .collect();
            }
        };

        smallvec![Pat { ty, span: DUMMY_SP, kind: Box::new(pat) }]
    }

    /// Like `apply`, but where all the subpatterns are wildcards `_`.
    fn apply_wildcards<'a>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        pcx: PatCtxt<'tcx>,
        ty: Ty<'tcx>,
    ) -> SmallVec<[Pat<'tcx>; 1]> {
        let pats = self.wildcard_subpatterns(cx, ty).rev();
        self.apply(cx, pcx, ty, pats)
    }
}

#[derive(Clone, Debug)]
pub enum Usefulness<'tcx> {
    Useful,
    UsefulWithWitness(Vec<Witness<'tcx>>),
    NotUseful,
}

impl<'tcx> Usefulness<'tcx> {
    fn new_useful(preference: WitnessPreference) -> Self {
        match preference {
            ConstructWitness => UsefulWithWitness(vec![Witness(vec![])]),
            LeaveOutWitness => Useful,
        }
    }

    fn is_useful(&self) -> bool {
        match *self {
            NotUseful => false,
            _ => true,
        }
    }

    fn apply_constructor(
        self,
        cx: &MatchCheckCtxt<'_, 'tcx>,
        pcx: PatCtxt<'tcx>,
        ctor: &Constructor<'tcx>,
        lty: Ty<'tcx>,
    ) -> Self {
        match self {
            UsefulWithWitness(witnesses) => UsefulWithWitness(
                witnesses
                    .into_iter()
                    .flat_map(|witness| witness.apply_constructor(cx, pcx, &ctor, lty))
                    .collect(),
            ),
            x => x,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness,
}

#[derive(Copy, Clone, Debug)]
struct PatCtxt<'tcx> {
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
pub struct Witness<'tcx>(Vec<Pat<'tcx>>);

impl<'tcx> Witness<'tcx> {
    pub fn single_pattern(self) -> Pat<'tcx> {
        assert_eq!(self.0.len(), 1);
        self.0.into_iter().next().unwrap()
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
        cx: &MatchCheckCtxt<'a, 'tcx>,
        pcx: PatCtxt<'tcx>,
        ctor: &Constructor<'tcx>,
        ty: Ty<'tcx>,
    ) -> SmallVec<[Self; 1]> {
        let arity = ctor.arity(cx, ty);
        let applied_pats = {
            let len = self.0.len() as u64;
            let pats = self.0.drain((len - arity) as usize..).rev();
            ctor.apply(cx, pcx, ty, pats)
        };

        applied_pats
            .into_iter()
            .map(|pat| {
                let mut w = self.clone();
                w.0.push(pat);
                w
            })
            .collect()
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
    cx: &MatchCheckCtxt<'a, 'tcx>,
    pcx: PatCtxt<'tcx>,
) -> Vec<Constructor<'tcx>> {
    debug!("all_constructors({:?})", pcx.ty);
    let ctors = match pcx.ty.kind {
        ty::Bool => {
            [true, false].iter().map(|&b| ConstantValue(ty::Const::from_bool(cx.tcx, b))).collect()
        }
        ty::Array(ref sub_ty, len) if len.try_eval_usize(cx.tcx, cx.param_env).is_some() => {
            let len = len.eval_usize(cx.tcx, cx.param_env);
            if len != 0 && cx.is_uninhabited(sub_ty) { vec![] } else { vec![FixedLenSlice(len)] }
        }
        // Treat arrays of a constant but unknown length like slices.
        ty::Array(ref sub_ty, _) | ty::Slice(ref sub_ty) => {
            if cx.is_uninhabited(sub_ty) {
                vec![FixedLenSlice(0)]
            } else {
                vec![VarLenSlice(0, 0)]
            }
        }
        ty::Adt(def, substs) if def.is_enum() => def
            .variants
            .iter()
            .filter(|v| {
                !cx.tcx.features().exhaustive_patterns
                    || !v
                        .uninhabited_from(cx.tcx, substs, def.adt_kind())
                        .contains(cx.tcx, cx.module)
            })
            .map(|v| Variant(v.def_id))
            .collect(),
        ty::Char => {
            vec![
                // The valid Unicode Scalar Value ranges.
                ConstantRange(
                    '\u{0000}' as u128,
                    '\u{D7FF}' as u128,
                    cx.tcx.types.char,
                    RangeEnd::Included,
                ),
                ConstantRange(
                    '\u{E000}' as u128,
                    '\u{10FFFF}' as u128,
                    cx.tcx.types.char,
                    RangeEnd::Included,
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

fn max_slice_length<'p, 'a, 'tcx, I>(cx: &MatchCheckCtxt<'a, 'tcx>, patterns: I) -> u64
where
    I: Iterator<Item = &'p Pat<'tcx>>,
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
            PatKind::Constant { value } => {
                // extract the length of an array/slice from a constant
                match (value.val, &value.ty.kind) {
                    (_, ty::Array(_, n)) => {
                        max_fixed_len = cmp::max(max_fixed_len, n.eval_usize(cx.tcx, cx.param_env))
                    }
                    (ConstValue::Slice { start, end, .. }, ty::Slice(_)) => {
                        max_fixed_len = cmp::max(max_fixed_len, (end - start) as u64)
                    }
                    _ => {}
                }
            }
            PatKind::Slice { ref prefix, slice: None, ref suffix } => {
                let fixed_len = prefix.len() as u64 + suffix.len() as u64;
                max_fixed_len = cmp::max(max_fixed_len, fixed_len);
            }
            PatKind::Slice { ref prefix, slice: Some(_), ref suffix } => {
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
    #[inline]
    fn is_integral(ty: Ty<'_>) -> bool {
        match ty.kind {
            ty::Char | ty::Int(_) | ty::Uint(_) => true,
            _ => false,
        }
    }

    #[inline]
    fn integral_size_and_signed_bias(tcx: TyCtxt<'tcx>, ty: Ty<'_>) -> Option<(Size, u128)> {
        match ty.kind {
            ty::Char => Some((Size::from_bytes(4), 0)),
            ty::Int(ity) => {
                let size = Integer::from_attr(&tcx, SignedInt(ity)).size();
                Some((size, 1u128 << (size.bits() as u128 - 1)))
            }
            ty::Uint(uty) => Some((Integer::from_attr(&tcx, UnsignedInt(uty)).size(), 0)),
            _ => None,
        }
    }

    #[inline]
    fn from_const(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: &Const<'tcx>,
    ) -> Option<IntRange<'tcx>> {
        if let Some((target_size, bias)) = Self::integral_size_and_signed_bias(tcx, value.ty) {
            let ty = value.ty;
            let val = if let ConstValue::Scalar(Scalar::Raw { data, size }) = value.val {
                // For this specific pattern we can skip a lot of effort and go
                // straight to the result, after doing a bit of checking. (We
                // could remove this branch and just use the next branch, which
                // is more general but much slower.)
                Scalar::<()>::check_raw(data, size, target_size);
                data
            } else if let Some(val) = value.try_eval_bits(tcx, param_env, ty) {
                // This is a more general form of the previous branch.
                val
            } else {
                return None;
            };
            let val = val ^ bias;
            Some(IntRange { range: val..=val, ty })
        } else {
            None
        }
    }

    #[inline]
    fn from_range(
        tcx: TyCtxt<'tcx>,
        lo: u128,
        hi: u128,
        ty: Ty<'tcx>,
        end: &RangeEnd,
    ) -> Option<IntRange<'tcx>> {
        if Self::is_integral(ty) {
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
        } else {
            None
        }
    }

    fn from_ctor(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ctor: &Constructor<'tcx>,
    ) -> Option<IntRange<'tcx>> {
        // Floating-point ranges are permitted and we don't want
        // to consider them when constructing integer ranges.
        match ctor {
            ConstantRange(lo, hi, ty, end) => Self::from_range(tcx, *lo, *hi, ty, end),
            ConstantValue(val) => Self::from_const(tcx, param_env, val),
            _ => None,
        }
    }

    // The return value of `signed_bias` should be XORed with an endpoint to encode/decode it.
    fn signed_bias(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> u128 {
        match ty.kind {
            ty::Int(ity) => {
                let bits = Integer::from_attr(&tcx, SignedInt(ity)).size().bits() as u128;
                1u128 << (bits - 1)
            }
            _ => 0,
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

    /// Returns a collection of ranges that spans the values covered by `ctor`, subtracted
    /// by the values covered by `self`: i.e., `ctor \ self` (in set notation).
    fn subtract_from(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ctor: Constructor<'tcx>,
    ) -> SmallVec<[Constructor<'tcx>; 2]> {
        let range = match IntRange::from_ctor(tcx, param_env, &ctor) {
            None => return smallvec![],
            Some(int_range) => int_range.range,
        };

        let ty = self.ty;
        let (lo, hi) = (*self.range.start(), *self.range.end());
        let (range_lo, range_hi) = range.into_inner();
        let mut remaining_ranges = smallvec![];
        if lo > range_hi || range_lo > hi {
            // The pattern doesn't intersect with the range at all,
            // so the range remains untouched.
            remaining_ranges.push(Self::range_to_ctor(tcx, ty, range_lo..=range_hi));
        } else {
            if lo > range_lo {
                // The pattern intersects an upper section of the
                // range, so a lower section will remain.
                remaining_ranges.push(Self::range_to_ctor(tcx, ty, range_lo..=(lo - 1)));
            }
            if hi < range_hi {
                // The pattern intersects a lower section of the
                // range, so an upper section will remain.
                remaining_ranges.push(Self::range_to_ctor(tcx, ty, (hi + 1)..=range_hi));
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

// A struct to compute a set of constructors equivalent to `all_ctors \ used_ctors`.
#[derive(Clone)]
struct MissingConstructors<'tcx> {
    pcx: PatCtxt<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    all_ctors: Vec<Constructor<'tcx>>,
    used_ctors: Vec<Constructor<'tcx>>,
}

type MissingConstructorsIter<'a, 'tcx, F> = std::iter::FlatMap<
    std::slice::Iter<'a, Constructor<'tcx>>,
    SmallVec<[Constructor<'tcx>; 1]>,
    F,
>;

impl<'tcx> MissingConstructors<'tcx> {
    fn new(
        pcx: PatCtxt<'tcx>,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        all_ctors: Vec<Constructor<'tcx>>,
        used_ctors: Vec<Constructor<'tcx>>,
    ) -> Self {
        MissingConstructors { pcx, tcx, param_env, all_ctors, used_ctors }
    }

    fn into_inner(self) -> (Vec<Constructor<'tcx>>, Vec<Constructor<'tcx>>) {
        (self.all_ctors, self.used_ctors)
    }

    fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }

    /// Iterate over all_ctors \ used_ctors
    // Can't use impl Iterator because of lifetime shenanigans
    fn iter<'a>(
        &'a self,
    ) -> MissingConstructorsIter<
        'a,
        'tcx,
        impl FnMut(&'a Constructor<'tcx>) -> SmallVec<[Constructor<'tcx>; 1]>,
    > {
        self.all_ctors.iter().flat_map(move |req_ctor| {
            req_ctor.clone().subtract_meta_constructor(
                self.pcx,
                self.tcx,
                self.param_env,
                &self.used_ctors,
            )
        })
    }
}

impl<'tcx> fmt::Debug for MissingConstructors<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ctors: Vec<_> = self.iter().collect();
        f.debug_tuple("MissingConstructors").field(&ctors).finish()
    }
}

/// The implementation panics because this should not happen
impl<'tcx> PartialEq<Self> for MissingConstructors<'tcx> {
    fn eq(&self, _other: &Self) -> bool {
        bug!("Tried to compare MissingConstructors for equality")
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
/// have the same type, except that wildcard (PatKind::Wild) patterns
/// with type `TyErr` are also allowed, even if the "type of the column"
/// is not `TyErr`. That is used to represent private fields, as using their
/// real type would assert that they are inhabited.
///
/// This is used both for reachability checking (if a pattern isn't useful in
/// relation to preceding patterns, it is not reachable) and exhaustiveness
/// checking (if a wildcard pattern is useful in relation to a matrix, the
/// matrix isn't exhaustive).
pub fn is_useful<'p, 'a, 'tcx>(
    cx: &MatchCheckCtxt<'a, 'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &PatStack<'_, 'tcx>,
    witness_preference: WitnessPreference,
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
            Usefulness::new_useful(witness_preference)
        } else {
            NotUseful
        };
    };

    assert!(rows.iter().all(|r| r.len() == v.len()));

    let pcx = PatCtxt {
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
        ty: matrix.heads().map(|p| p.ty).find(|ty| !ty.references_error()).unwrap_or(v.head().ty),
        max_slice_length: max_slice_length(cx, matrix.heads().chain(Some(v.head()))),
    };

    debug!("is_useful_expand_first_col: pcx={:#?}, expanding {:#?}", pcx, v.head());

    let v_constructors = pat_constructors(cx.tcx, cx.param_env, v.head(), pcx);

    if cx.is_non_exhaustive_variant(v.head())
        && !cx.is_local(pcx.ty)
        && !v_constructors.iter().any(|ctor| ctor.is_wildcard())
    {
        debug!("is_useful - shortcut because declared non-exhaustive");
        // FIXME(#65157)
        return Useful;
    }

    let matrix_head_ctors: Vec<Constructor<'_>> = matrix
        .heads()
        .flat_map(|p| pat_constructors(cx.tcx, cx.param_env, p, pcx))
        .filter(|ctor| !ctor.is_wildcard())
        .collect();
    debug!("matrix_head_ctors = {:#?}", matrix_head_ctors);

    v_constructors
        .into_iter()
        .flat_map(|ctor| ctor.split_meta_constructor(cx, pcx, &matrix_head_ctors))
        .map(|c| is_useful_specialized(cx, pcx, matrix, v, c, pcx.ty, witness_preference))
        .find(|result| result.is_useful())
        .unwrap_or(NotUseful)
}

/// A shorthand for the `U(S(c, M), S(c, q))` operation. I.e., `is_useful` applied
/// to the specialised version of both the pattern matrix `M` and the new pattern `q`.
fn is_useful_specialized<'p, 'a, 'tcx>(
    cx: &MatchCheckCtxt<'a, 'tcx>,
    pcx: PatCtxt<'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &PatStack<'_, 'tcx>,
    ctor: Constructor<'tcx>,
    lty: Ty<'tcx>,
    witness_preference: WitnessPreference,
) -> Usefulness<'tcx> {
    debug!("is_useful_specialized({:#?}, {:#?}, {:?})", v, ctor, lty);

    let ctor_wild_subpatterns_owned: Vec<_> = ctor.wildcard_subpatterns(cx, lty).collect();
    let ctor_wild_subpatterns: Vec<_> = ctor_wild_subpatterns_owned.iter().collect();
    let matrix = matrix.specialize(cx, &ctor, &ctor_wild_subpatterns);
    let ret = v
        .specialize(cx, &ctor, &ctor_wild_subpatterns)
        .into_iter()
        .map(|v| is_useful(cx, &matrix, &v, witness_preference))
        .map(|u| u.apply_constructor(cx, pcx, &ctor, lty))
        .find(|result| result.is_useful())
        .unwrap_or(NotUseful);
    ret
}

/// Determines the constructors that the given pattern can be specialized to.
///
/// In most cases, there's only one constructor that a specific pattern
/// represents, such as a specific enum variant or a specific literal value.
/// Slice patterns, however, can match slices of different lengths. For instance,
/// `[a, b, tail @ ..]` can match a slice of length 2, 3, 4 and so on.
///
/// Returns `None` in case of a catch-all, which can't be specialized.
fn pat_constructors<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pat: &Pat<'tcx>,
    pcx: PatCtxt<'tcx>,
) -> SmallVec<[Constructor<'tcx>; 1]> {
    match *pat.kind {
        PatKind::AscribeUserType { ref subpattern, .. } => {
            pat_constructors(tcx, param_env, subpattern, pcx)
        }
        PatKind::Binding { .. } | PatKind::Wild => smallvec![Wildcard],
        PatKind::Leaf { .. } | PatKind::Deref { .. } => smallvec![Single],
        PatKind::Variant { adt_def, variant_index, .. } => {
            smallvec![Variant(adt_def.variants[variant_index].def_id)]
        }
        PatKind::Constant { value } => smallvec![ConstantValue(value)],
        PatKind::Range(PatRange { lo, hi, end }) => smallvec![ConstantRange(
            lo.eval_bits(tcx, param_env, lo.ty),
            hi.eval_bits(tcx, param_env, hi.ty),
            lo.ty,
            end,
        )],
        PatKind::Array { .. } => match pcx.ty.kind {
            ty::Array(_, length) => smallvec![FixedLenSlice(length.eval_usize(tcx, param_env))],
            _ => span_bug!(pat.span, "bad ty {:?} for array pattern", pcx.ty),
        },
        PatKind::Slice { ref prefix, ref slice, ref suffix } => {
            let prefix = prefix.len() as u64;
            let suffix = suffix.len() as u64;
            if slice.is_some() {
                smallvec![VarLenSlice(prefix, suffix)]
            } else {
                smallvec![FixedLenSlice(prefix + suffix)]
            }
        }
        PatKind::Or { .. } => {
            bug!("support for or-patterns has not been fully implemented yet.");
        }
    }
}

// Checks whether a constant is equal to a user-written slice pattern. Only supports byte slices,
// meaning all other types will compare unequal and thus equal patterns often do not cause the
// second pattern to lint about unreachable match arms.
fn slice_pat_covered_by_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    _span: Span,
    const_val: &'tcx ty::Const<'tcx>,
    prefix: &[Pat<'tcx>],
    slice: &Option<Pat<'tcx>>,
    suffix: &[Pat<'tcx>],
    param_env: ty::ParamEnv<'tcx>,
) -> Result<bool, ErrorReported> {
    let data: &[u8] = match (const_val.val, &const_val.ty.kind) {
        (ConstValue::ByRef { offset, alloc, .. }, ty::Array(t, n)) => {
            assert_eq!(*t, tcx.types.u8);
            let n = n.eval_usize(tcx, param_env);
            let ptr = Pointer::new(AllocId(0), offset);
            alloc.get_bytes(&tcx, ptr, Size::from_bytes(n)).unwrap()
        }
        (ConstValue::Slice { data, start, end }, ty::Slice(t)) => {
            assert_eq!(*t, tcx.types.u8);
            let ptr = Pointer::new(AllocId(0), Size::from_bytes(start as u64));
            data.get_bytes(&tcx, ptr, Size::from_bytes((end - start) as u64)).unwrap()
        }
        // FIXME(oli-obk): create a way to extract fat pointers from ByRef
        (_, ty::Slice(_)) => return Ok(false),
        _ => bug!(
            "slice_pat_covered_by_const: {:#?}, {:#?}, {:#?}, {:#?}",
            const_val,
            prefix,
            slice,
            suffix,
        ),
    };

    let pat_len = prefix.len() + suffix.len();
    if data.len() < pat_len || (slice.is_none() && data.len() > pat_len) {
        return Ok(false);
    }

    for (ch, pat) in data[..prefix.len()]
        .iter()
        .zip(prefix)
        .chain(data[data.len() - suffix.len()..].iter().zip(suffix))
    {
        match pat.kind {
            box PatKind::Constant { value } => {
                let b = value.eval_bits(tcx, param_env, pat.ty);
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
    if let ty::Char | ty::Int(_) | ty::Uint(_) = ty.kind {
        !ty.is_ptr_sized_integral() || tcx.features().precise_pointer_size_matching
    } else {
        false
    }
}

/// Checks whether there exists any shared value in either `ctor` or `pat` by intersecting them.
fn constructor_intersects_pattern<'p, 'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ctor: &Constructor<'tcx>,
    pat: &'p Pat<'tcx>,
) -> Option<PatStack<'p, 'tcx>> {
    if should_treat_range_exhaustively(tcx, ctor) {
        let range = match *pat.kind {
            PatKind::Constant { value } => ConstantValue(value),
            PatKind::Range(PatRange { lo, hi, end }) => ConstantRange(
                lo.eval_bits(tcx, param_env, lo.ty),
                hi.eval_bits(tcx, param_env, hi.ty),
                lo.ty,
                end,
            ),
            _ => bug!("`constructor_intersects_pattern` called with {:?}", pat),
        };

        let pat = IntRange::from_ctor(tcx, param_env, &range)?;
        let ctor = IntRange::from_ctor(tcx, param_env, ctor)?;
        ctor.intersection(&pat)?;

        let (pat_lo, pat_hi) = pat.range.into_inner();
        let (ctor_lo, ctor_hi) = ctor.range.into_inner();
        assert!(pat_lo <= ctor_lo && ctor_hi <= pat_hi);
        Some(PatStack::default())
    } else {
        // Fallback for non-ranges and ranges that involve floating-point numbers, which are not
        // conveniently handled by `IntRange`. For these cases, the constructor may not be a range
        // so intersection actually devolves into being covered by the pattern.
        let (from, to, end, ty) = match *pat.kind {
            PatKind::Constant { value } => (value, value, RangeEnd::Included, value.ty),
            PatKind::Range(PatRange { lo, hi, end }) => (lo, hi, end, lo.ty),
            _ => bug!("`constructor_intersects_pattern` called with {:?}", pat),
        };
        trace!("constructor_intersects_pattern {:#?}, {:#?}, {:#?}, {}", ctor, from, to, ty);
        let cmp_from = |c_from| compare_const_vals(tcx, c_from, from, param_env, ty);
        let cmp_to = |c_to| compare_const_vals(tcx, c_to, to, param_env, ty);
        let result = match *ctor {
            ConstantValue(value) => {
                let to = cmp_to(value)?;
                let from = cmp_from(value)?;
                let end =
                    (to == Ordering::Less) || (end == RangeEnd::Included && to == Ordering::Equal);
                (from != Ordering::Less) && end
            }
            ConstantRange(from, to, ty, range_end) => {
                let to = cmp_to(ty::Const::from_bits(tcx, to, ty::ParamEnv::empty().and(ty)))?;
                let from =
                    cmp_from(ty::Const::from_bits(tcx, from, ty::ParamEnv::empty().and(ty)))?;
                let end = (to == Ordering::Less) || (end == range_end && to == Ordering::Equal);
                (from != Ordering::Less) && end
            }
            Single => true,
            _ => bug!(),
        };

        if result { Some(PatStack::default()) } else { None }
    }
}

fn patterns_for_variant<'p, 'tcx>(
    subpatterns: &'p [FieldPat<'tcx>],
    ctor_wild_subpatterns: &[&'p Pat<'tcx>],
) -> PatStack<'p, 'tcx> {
    let mut result = SmallVec::from_slice(ctor_wild_subpatterns);

    for subpat in subpatterns {
        result[subpat.field.index()] = &subpat.pattern;
    }

    debug!(
        "patterns_for_variant({:#?}, {:#?}) = {:#?}",
        subpatterns, ctor_wild_subpatterns, result
    );
    PatStack::from_vec(result)
}

/// This is the main specialization step. It expands the pattern
/// into `arity` patterns based on the constructor. For most patterns, the step is trivial,
/// for instance tuple patterns are flattened and box patterns expand into their inner pattern.
/// Returns `None` if the pattern does not have the given constructor.
///
/// OTOH, slice patterns with a subslice pattern (tail @ ..) can be expanded into multiple
/// different patterns.
/// Structure patterns with a partial wild pattern (Foo { a: 42, .. }) have their missing
/// fields filled with wild patterns.
fn specialize_one_pattern<'p, 'a: 'p, 'p2: 'p, 'tcx>(
    cx: &MatchCheckCtxt<'a, 'tcx>,
    mut pat: &'p2 Pat<'tcx>,
    constructor: &Constructor<'tcx>,
    ctor_wild_subpatterns: &[&'p Pat<'tcx>],
) -> SmallVec<[PatStack<'p, 'tcx>; 1]> {
    while let PatKind::AscribeUserType { ref subpattern, .. } = *pat.kind {
        pat = subpattern;
    }

    if let MissingConstructors(_) = constructor {
        // By construction of MissingConstructors, we know that all non-wildcard constructors
        // should be discarded.
        return match *pat.kind {
            PatKind::Binding { .. } | PatKind::Wild => smallvec![PatStack::empty()],
            _ => smallvec![],
        };
    } else if let Wildcard = constructor {
        // If we get here, either there were only wildcards in the first component of the
        // matrix, or we are in a special non_exhaustive case where we pretend the type has
        // an extra `_` constructor to prevent exhaustive matching. In both cases, all
        // non-wildcard constructors should be discarded.
        return match *pat.kind {
            PatKind::Binding { .. } | PatKind::Wild => smallvec![PatStack::empty()],
            _ => smallvec![],
        };
    }

    match *pat.kind {
        PatKind::AscribeUserType { .. } => unreachable!(), // Handled above

        PatKind::Binding { .. } | PatKind::Wild => {
            smallvec![PatStack::from_slice(ctor_wild_subpatterns)]
        }

        PatKind::Variant { adt_def, variant_index, ref subpatterns, .. } => {
            let ref variant = adt_def.variants[variant_index];
            if Variant(variant.def_id) == *constructor {
                smallvec![patterns_for_variant(subpatterns, ctor_wild_subpatterns)]
            } else {
                smallvec![]
            }
        }

        PatKind::Leaf { ref subpatterns } => {
            smallvec![patterns_for_variant(subpatterns, ctor_wild_subpatterns)]
        }

        PatKind::Deref { ref subpattern } => smallvec![PatStack::from_pattern(subpattern)],

        PatKind::Constant { value } if constructor.is_slice() => {
            // We extract an `Option` for the pointer because slices of zero
            // elements don't necessarily point to memory, they are usually
            // just integers. The only time they should be pointing to memory
            // is when they are subslices of nonzero slices.
            let (alloc, offset, n, ty) = match value.ty.kind {
                ty::Array(t, n) => match value.val {
                    ConstValue::ByRef { offset, alloc, .. } => {
                        (alloc, offset, n.eval_usize(cx.tcx, cx.param_env), t)
                    }
                    _ => span_bug!(pat.span, "array pattern is {:?}", value,),
                },
                ty::Slice(t) => {
                    match value.val {
                        ConstValue::Slice { data, start, end } => {
                            (data, Size::from_bytes(start as u64), (end - start) as u64, t)
                        }
                        ConstValue::ByRef { .. } => {
                            // FIXME(oli-obk): implement `deref` for `ConstValue`
                            return smallvec![];
                        }
                        _ => span_bug!(
                            pat.span,
                            "slice pattern constant must be scalar pair but is {:?}",
                            value,
                        ),
                    }
                }
                _ => span_bug!(
                    pat.span,
                    "unexpected const-val {:?} with ctor {:?}",
                    value,
                    constructor,
                ),
            };
            if ctor_wild_subpatterns.len() as u64 == n {
                // convert a constant slice/array pattern to a list of patterns.
                let layout = if let Ok(layout) = cx.tcx.layout_of(cx.param_env.and(ty)) {
                    layout
                } else {
                    return smallvec![];
                };
                let ptr = Pointer::new(AllocId(0), offset);
                let stack: Option<PatStack<'_, '_>> = (0..n)
                    .map(|i| {
                        let ptr = ptr.offset(layout.size * i, &cx.tcx).ok()?;
                        let scalar = alloc.read_scalar(&cx.tcx, ptr, layout.size).ok()?;
                        let scalar = scalar.not_undef().ok()?;
                        let value = ty::Const::from_scalar(cx.tcx, scalar, ty);
                        let pattern =
                            Pat { ty, span: pat.span, kind: box PatKind::Constant { value } };
                        Some(&*cx.pattern_arena.alloc(pattern))
                    })
                    .collect();
                stack.into_iter().collect()
            } else {
                smallvec![]
            }
        }

        PatKind::Constant { .. } | PatKind::Range { .. } => {
            // If the constructor is a:
            // - Single value: add a row if the pattern contains the constructor.
            // - Range: add a row if the constructor intersects the pattern.
            constructor_intersects_pattern(cx.tcx, cx.param_env, constructor, pat)
                .into_iter()
                .collect()
        }

        PatKind::Array { ref prefix, ref slice, ref suffix }
        | PatKind::Slice { ref prefix, ref slice, ref suffix } => match *constructor {
            FixedLenSlice(..) => {
                let pat_len = prefix.len() + suffix.len();
                if let Some(slice_count) = ctor_wild_subpatterns.len().checked_sub(pat_len) {
                    if slice_count == 0 || slice.is_some() {
                        smallvec![
                            prefix
                                .iter()
                                .chain(
                                    ctor_wild_subpatterns
                                        .iter()
                                        .map(|p| *p)
                                        .skip(prefix.len())
                                        .take(slice_count)
                                        .chain(suffix.iter()),
                                )
                                .collect(),
                        ]
                    } else {
                        smallvec![]
                    }
                } else {
                    smallvec![]
                }
            }
            ConstantValue(cv) => {
                match slice_pat_covered_by_const(
                    cx.tcx,
                    pat.span,
                    cv,
                    prefix,
                    slice,
                    suffix,
                    cx.param_env,
                ) {
                    Ok(true) => smallvec![PatStack::default()],
                    Ok(false) | Err(ErrorReported) => smallvec![],
                }
            }
            _ => span_bug!(pat.span, "unexpected ctor {:?} for slice pat", constructor),
        },

        PatKind::Or { .. } => {
            bug!("support for or-patterns has not been fully implemented yet.");
        }
    }
}
