//! Note: most of the tests relevant to this file can be found (at the time of writing) in
//! src/tests/ui/pattern/usefulness.
//!
//! This file includes the logic for exhaustiveness and usefulness checking for
//! pattern-matching. Specifically, given a list of patterns for a type, we can
//! tell whether:
//! (a) the patterns cover every possible constructor for the type (exhaustiveness)
//! (b) each pattern is necessary (usefulness)
//!
//! The algorithm implemented here is a modified version of the one described in:
//! http://moscova.inria.fr/~maranget/papers/warn/index.html
//! However, to save future implementors from reading the original paper, we
//! summarise the algorithm here to hopefully save time and be a little clearer
//! (without being so rigorous).
//!
//! # Premise
//!
//! The core of the algorithm revolves about a "usefulness" check. In particular, we
//! are trying to compute a predicate `U(P, p)` where `P` is a list of patterns (we refer to this as
//! a matrix). `U(P, p)` represents whether, given an existing list of patterns
//! `P_1 ..= P_m`, adding a new pattern `p` will be "useful" (that is, cover previously-
//! uncovered values of the type).
//!
//! If we have this predicate, then we can easily compute both exhaustiveness of an
//! entire set of patterns and the individual usefulness of each one.
//! (a) the set of patterns is exhaustive iff `U(P, _)` is false (i.e., adding a wildcard
//! match doesn't increase the number of values we're matching)
//! (b) a pattern `P_i` is not useful if `U(P[0..=(i-1), P_i)` is false (i.e., adding a
//! pattern to those that have come before it doesn't increase the number of values
//! we're matching).
//!
//! # Core concept
//!
//! The idea that powers everything that is done in this file is the following: a value is made
//! from a constructor applied to some fields. Examples of constructors are `Some`, `None`, `(,)`
//! (the 2-tuple constructor), `Foo {..}` (the constructor for a struct `Foo`), and `2` (the
//! constructor for the number `2`). Fields are just a (possibly empty) list of values.
//!
//! Some of the constructors listed above might feel weird: `None` and `2` don't take any
//! arguments. This is part of what makes constructors so general: we will consider plain values
//! like numbers and string literals to be constructors that take no arguments, also called "0-ary
//! constructors"; they are the simplest case of constructors. This allows us to see any value as
//! made up from a tree of constructors, each having a given number of children. For example:
//! `(None, Ok(0))` is made from 4 different constructors.
//!
//! This idea can be extended to patterns: a pattern captures a set of possible values, and we can
//! describe this set using constructors. For example, `Err(_)` captures all values of the type
//! `Result<T, E>` that start with the `Err` constructor (for some choice of `T` and `E`). The
//! wildcard `_` captures all values of the given type starting with any of the constructors for
//! that type.
//!
//! We use this to compute whether different patterns might capture a same value. Do the patterns
//! `Ok("foo")` and `Err(_)` capture a common value? The answer is no, because the first pattern
//! captures only values starting with the `Ok` constructor and the second only values starting
//! with the `Err` constructor. Do the patterns `Some(42)` and `Some(1..10)` intersect? They might,
//! since they both capture values starting with `Some`. To be certain, we need to dig under the
//! `Some` constructor and continue asking the question. This is the main idea behind the
//! exhaustiveness algorithm: by looking at patterns constructor-by-constructor, we can efficiently
//! figure out if some new pattern might capture a value that hadn't been captured by previous
//! patterns.
//!
//! Constructors are represented by the `Constructor` enum, and its fields by the `Fields` enum.
//! Most of the complexity of this file resides in transforming between patterns and
//! (`Constructor`, `Fields`) pairs, handling all the special cases correctly.
//!
//! Caveat: this constructors/fields distinction doesn't quite cover every Rust value. For example
//! a value of type `Rc<u64>` doesn't fit this idea very well, nor do various other things.
//! However, this idea covers most of the cases that are relevant to exhaustiveness checking.
//!
//!
//! # Algorithm
//!
//! Recall that `U(P, p)` represents whether, given an existing list of patterns (aka matrix) `P`,
//! adding a new pattern `p` will cover previously-uncovered values of the type.
//! During the course of the algorithm, the rows of the matrix won't just be individual patterns,
//! but rather partially-deconstructed patterns in the form of a list of fields. The paper
//! calls those pattern-vectors, and we will call them pattern-stacks. The same holds for the
//! new pattern `p`.
//!
//! For example, say we have the following:
//! ```
//!     // x: (Option<bool>, Result<()>)
//!     match x {
//!         (Some(true), _) => {}
//!         (None, Err(())) => {}
//!         (None, Err(_)) => {}
//!     }
//! ```
//! Here, the matrix `P` starts as:
//! [
//!     [(Some(true), _)],
//!     [(None, Err(()))],
//!     [(None, Err(_))],
//! ]
//! We can tell it's not exhaustive, because `U(P, _)` is true (we're not covering
//! `[(Some(false), _)]`, for instance). In addition, row 3 is not useful, because
//! all the values it covers are already covered by row 2.
//!
//! A list of patterns can be thought of as a stack, because we are mainly interested in the top of
//! the stack at any given point, and we can pop or apply constructors to get new pattern-stacks.
//! To match the paper, the top of the stack is at the beginning / on the left.
//!
//! There are two important operations on pattern-stacks necessary to understand the algorithm:
//!
//! 1. We can pop a given constructor off the top of a stack. This operation is called
//!    `specialize`, and is denoted `S(c, p)` where `c` is a constructor (like `Some` or
//!    `None`) and `p` a pattern-stack.
//!    If the pattern on top of the stack can cover `c`, this removes the constructor and
//!    pushes its arguments onto the stack. It also expands OR-patterns into distinct patterns.
//!    Otherwise the pattern-stack is discarded.
//!    This essentially filters those pattern-stacks whose top covers the constructor `c` and
//!    discards the others.
//!
//!    For example, the first pattern above initially gives a stack `[(Some(true), _)]`. If we
//!    pop the tuple constructor, we are left with `[Some(true), _]`, and if we then pop the
//!    `Some` constructor we get `[true, _]`. If we had popped `None` instead, we would get
//!    nothing back.
//!
//!    This returns zero or more new pattern-stacks, as follows. We look at the pattern `p_1`
//!    on top of the stack, and we have four cases:
//!         1.1. `p_1 = c(r_1, .., r_a)`, i.e. the top of the stack has constructor `c`. We
//!              push onto the stack the arguments of this constructor, and return the result:
//!                 r_1, .., r_a, p_2, .., p_n
//!         1.2. `p_1 = c'(r_1, .., r_a')` where `c ≠ c'`. We discard the current stack and
//!              return nothing.
//!         1.3. `p_1 = _`. We push onto the stack as many wildcards as the constructor `c` has
//!              arguments (its arity), and return the resulting stack:
//!                 _, .., _, p_2, .., p_n
//!         1.4. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting
//!              stack:
//!                 S(c, (r_1, p_2, .., p_n))
//!                 S(c, (r_2, p_2, .., p_n))
//!
//! 2. We can pop a wildcard off the top of the stack. This is called `D(p)`, where `p` is
//!    a pattern-stack.
//!    This is used when we know there are missing constructor cases, but there might be
//!    existing wildcard patterns, so to check the usefulness of the matrix, we have to check
//!    all its *other* components.
//!
//!    It is computed as follows. We look at the pattern `p_1` on top of the stack,
//!    and we have three cases:
//!         1.1. `p_1 = c(r_1, .., r_a)`. We discard the current stack and return nothing.
//!         1.2. `p_1 = _`. We return the rest of the stack:
//!                 p_2, .., p_n
//!         1.3. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting
//!           stack.
//!                 D((r_1, p_2, .., p_n))
//!                 D((r_2, p_2, .., p_n))
//!
//! Note that the OR-patterns are not always used directly in Rust, but are used to derive the
//! exhaustive integer matching rules, so they're written here for posterity.
//!
//! Both those operations extend straightforwardly to a list or pattern-stacks, i.e. a matrix, by
//! working row-by-row. Popping a constructor ends up keeping only the matrix rows that start with
//! the given constructor, and popping a wildcard keeps those rows that start with a wildcard.
//!
//!
//! The algorithm for computing `U`
//! -------------------------------
//! The algorithm is inductive (on the number of columns: i.e., components of tuple patterns).
//! That means we're going to check the components from left-to-right, so the algorithm
//! operates principally on the first component of the matrix and new pattern-stack `p`.
//! This algorithm is realised in the `is_useful` function.
//!
//! Base case. (`n = 0`, i.e., an empty tuple pattern)
//!     - If `P` already contains an empty pattern (i.e., if the number of patterns `m > 0`),
//!       then `U(P, p)` is false.
//!     - Otherwise, `P` must be empty, so `U(P, p)` is true.
//!
//! Inductive step. (`n > 0`, i.e., whether there's at least one column
//!                  [which may then be expanded into further columns later])
//! We're going to match on the top of the new pattern-stack, `p_1`.
//!     - If `p_1 == c(r_1, .., r_a)`, i.e. we have a constructor pattern.
//! Then, the usefulness of `p_1` can be reduced to whether it is useful when
//! we ignore all the patterns in the first column of `P` that involve other constructors.
//! This is where `S(c, P)` comes in:
//! `U(P, p) := U(S(c, P), S(c, p))`
//! This special case is handled in `is_useful_specialized`.
//!
//! For example, if `P` is:
//! [
//! [Some(true), _],
//! [None, 0],
//! ]
//! and `p` is [Some(false), 0], then we don't care about row 2 since we know `p` only
//! matches values that row 2 doesn't. For row 1 however, we need to dig into the
//! arguments of `Some` to know whether some new value is covered. So we compute
//! `U([[true, _]], [false, 0])`.
//!
//!   - If `p_1 == _`, then we look at the list of constructors that appear in the first
//! component of the rows of `P`:
//!   + If there are some constructors that aren't present, then we might think that the
//! wildcard `_` is useful, since it covers those constructors that weren't covered
//! before.
//! That's almost correct, but only works if there were no wildcards in those first
//! components. So we need to check that `p` is useful with respect to the rows that
//! start with a wildcard, if there are any. This is where `D` comes in:
//! `U(P, p) := U(D(P), D(p))`
//!
//! For example, if `P` is:
//! [
//!     [_, true, _],
//!     [None, false, 1],
//! ]
//! and `p` is [_, false, _], the `Some` constructor doesn't appear in `P`. So if we
//! only had row 2, we'd know that `p` is useful. However row 1 starts with a
//! wildcard, so we need to check whether `U([[true, _]], [false, 1])`.
//!
//!   + Otherwise, all possible constructors (for the relevant type) are present. In this
//! case we must check whether the wildcard pattern covers any unmatched value. For
//! that, we can think of the `_` pattern as a big OR-pattern that covers all
//! possible constructors. For `Option`, that would mean `_ = None | Some(_)` for
//! example. The wildcard pattern is useful in this case if it is useful when
//! specialized to one of the possible constructors. So we compute:
//! `U(P, p) := ∃(k ϵ constructors) U(S(k, P), S(k, p))`
//!
//! For example, if `P` is:
//! [
//!     [Some(true), _],
//!     [None, false],
//! ]
//! and `p` is [_, false], both `None` and `Some` constructors appear in the first
//! components of `P`. We will therefore try popping both constructors in turn: we
//! compute `U([[true, _]], [_, false])` for the `Some` constructor, and `U([[false]],
//! [false])` for the `None` constructor. The first case returns true, so we know that
//! `p` is useful for `P`. Indeed, it matches `[Some(false), _]` that wasn't matched
//! before.
//!
//!   - If `p_1 == r_1 | r_2`, then the usefulness depends on each `r_i` separately:
//! `U(P, p) := U(P, (r_1, p_2, .., p_n))
//!  || U(P, (r_2, p_2, .., p_n))`
//!
//! Modifications to the algorithm
//! ------------------------------
//! The algorithm in the paper doesn't cover some of the special cases that arise in Rust, for
//! example uninhabited types and variable-length slice patterns. These are drawn attention to
//! throughout the code below. I'll make a quick note here about how exhaustive integer matching is
//! accounted for, though.
//!
//! Exhaustive integer matching
//! ---------------------------
//! An integer type can be thought of as a (huge) sum type: 1 | 2 | 3 | ...
//! So to support exhaustive integer matching, we can make use of the logic in the paper for
//! OR-patterns. However, we obviously can't just treat ranges x..=y as individual sums, because
//! they are likely gigantic. So we instead treat ranges as constructors of the integers. This means
//! that we have a constructor *of* constructors (the integers themselves). We then need to work
//! through all the inductive step rules above, deriving how the ranges would be treated as
//! OR-patterns, and making sure that they're treated in the same way even when they're ranges.
//! There are really only four special cases here:
//! - When we match on a constructor that's actually a range, we have to treat it as if we would
//!   an OR-pattern.
//!     + It turns out that we can simply extend the case for single-value patterns in
//!      `specialize` to either be *equal* to a value constructor, or *contained within* a range
//!      constructor.
//!     + When the pattern itself is a range, you just want to tell whether any of the values in
//!       the pattern range coincide with values in the constructor range, which is precisely
//!       intersection.
//!   Since when encountering a range pattern for a value constructor, we also use inclusion, it
//!   means that whenever the constructor is a value/range and the pattern is also a value/range,
//!   we can simply use intersection to test usefulness.
//! - When we're testing for usefulness of a pattern and the pattern's first component is a
//!   wildcard.
//!     + If all the constructors appear in the matrix, we have a slight complication. By default,
//!       the behaviour (i.e., a disjunction over specialised matrices for each constructor) is
//!       invalid, because we want a disjunction over every *integer* in each range, not just a
//!       disjunction over every range. This is a bit more tricky to deal with: essentially we need
//!       to form equivalence classes of subranges of the constructor range for which the behaviour
//!       of the matrix `P` and new pattern `p` are the same. This is described in more
//!       detail in `split_grouped_constructors`.
//!     + If some constructors are missing from the matrix, it turns out we don't need to do
//!       anything special (because we know none of the integers are actually wildcards: i.e., we
//!       can't span wildcards using ranges).
use self::Constructor::*;
use self::SliceKind::*;
use self::Usefulness::*;
use self::WitnessPreference::*;

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_index::vec::Idx;

use super::{compare_const_vals, PatternFoldable, PatternFolder};
use super::{FieldPat, Pat, PatKind, PatRange};

use rustc_arena::TypedArena;
use rustc_attr::{SignedInt, UnsignedInt};
use rustc_errors::ErrorReported;
use rustc_hir::def_id::DefId;
use rustc_hir::{HirId, RangeEnd};
use rustc_middle::mir::interpret::{truncate, AllocId, ConstValue, Pointer, Scalar};
use rustc_middle::mir::Field;
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_session::lint;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{Integer, Size, VariantIdx};

use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;
use std::cmp::{self, max, min, Ordering};
use std::convert::TryInto;
use std::fmt;
use std::iter::{FromIterator, IntoIterator};
use std::ops::RangeInclusive;

crate fn expand_pattern<'a, 'tcx>(cx: &MatchCheckCtxt<'a, 'tcx>, pat: Pat<'tcx>) -> Pat<'tcx> {
    LiteralExpander { tcx: cx.tcx, param_env: cx.param_env }.fold_pattern(&pat)
}

struct LiteralExpander<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> LiteralExpander<'tcx> {
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
        match (val, &crty.kind(), &rty.kind()) {
            // the easy case, deref a reference
            (ConstValue::Scalar(p), x, y) if x == y => {
                match p {
                    Scalar::Ptr(p) => {
                        let alloc = self.tcx.global_alloc(p.alloc_id).unwrap_memory();
                        ConstValue::ByRef { alloc, offset: p.offset }
                    }
                    Scalar::Raw { .. } => {
                        let layout = self.tcx.layout_of(self.param_env.and(rty)).unwrap();
                        if layout.is_zst() {
                            // Deref of a reference to a ZST is a nop.
                            ConstValue::Scalar(Scalar::zst())
                        } else {
                            // FIXME(oli-obk): this is reachable for `const FOO: &&&u32 = &&&42;`
                            bug!("cannot deref {:#?}, {} -> {}", val, crty, rty);
                        }
                    }
                }
            }
            // unsize array to slice if pattern is array but match value or other patterns are slice
            (ConstValue::Scalar(Scalar::Ptr(p)), ty::Array(t, n), ty::Slice(u)) => {
                assert_eq!(t, u);
                ConstValue::Slice {
                    data: self.tcx.global_alloc(p.alloc_id).unwrap_memory(),
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

impl<'tcx> PatternFolder<'tcx> for LiteralExpander<'tcx> {
    fn fold_pattern(&mut self, pat: &Pat<'tcx>) -> Pat<'tcx> {
        debug!("fold_pattern {:?} {:?} {:?}", pat, pat.ty.kind(), pat.kind);
        match (pat.ty.kind(), &*pat.kind) {
            (&ty::Ref(_, rty, _), &PatKind::Constant { value: Const { val, ty: const_ty } })
                if const_ty.is_ref() =>
            {
                let crty =
                    if let ty::Ref(_, crty, _) = const_ty.kind() { crty } else { unreachable!() };
                if let ty::ConstKind::Value(val) = val {
                    Pat {
                        ty: pat.ty,
                        span: pat.span,
                        kind: box PatKind::Deref {
                            subpattern: Pat {
                                ty: rty,
                                span: pat.span,
                                kind: box PatKind::Constant {
                                    value: Const::from_value(
                                        self.tcx,
                                        self.fold_const_value_deref(*val, rty, crty),
                                        rty,
                                    ),
                                },
                            },
                        },
                    }
                } else {
                    bug!("cannot deref {:#?}, {} -> {}", val, crty, rty)
                }
            }

            (_, &PatKind::Binding { subpattern: Some(ref s), .. }) => s.fold_with(self),
            (_, &PatKind::AscribeUserType { subpattern: ref s, .. }) => s.fold_with(self),
            _ => pat.super_fold_with(self),
        }
    }
}

impl<'tcx> Pat<'tcx> {
    pub(super) fn is_wildcard(&self) -> bool {
        match *self.kind {
            PatKind::Binding { subpattern: None, .. } | PatKind::Wild => true,
            _ => false,
        }
    }
}

/// A row of a matrix. Rows of len 1 are very common, which is why `SmallVec[_; 2]`
/// works well.
#[derive(Debug, Clone)]
crate struct PatStack<'p, 'tcx>(SmallVec<[&'p Pat<'tcx>; 2]>);

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    crate fn from_pattern(pat: &'p Pat<'tcx>) -> Self {
        PatStack(smallvec![pat])
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

    fn head(&self) -> &'p Pat<'tcx> {
        self.0[0]
    }

    fn to_tail(&self) -> Self {
        PatStack::from_slice(&self.0[1..])
    }

    fn iter(&self) -> impl Iterator<Item = &Pat<'tcx>> {
        self.0.iter().copied()
    }

    // If the first pattern is an or-pattern, expand this pattern. Otherwise, return `None`.
    fn expand_or_pat(&self) -> Option<Vec<Self>> {
        if self.is_empty() {
            None
        } else if let PatKind::Or { pats } = &*self.head().kind {
            Some(
                pats.iter()
                    .map(|pat| {
                        let mut new_patstack = PatStack::from_pattern(pat);
                        new_patstack.0.extend_from_slice(&self.0[1..]);
                        new_patstack
                    })
                    .collect(),
            )
        } else {
            None
        }
    }

    /// This computes `D(self)`. See top of the file for explanations.
    fn specialize_wildcard(&self) -> Option<Self> {
        if self.head().is_wildcard() { Some(self.to_tail()) } else { None }
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        cx: &mut MatchCheckCtxt<'p, 'tcx>,
        constructor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Option<PatStack<'p, 'tcx>> {
        let new_fields =
            specialize_one_pattern(cx, self.head(), constructor, ctor_wild_subpatterns)?;
        Some(new_fields.push_on_patstack(&self.0[1..]))
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
#[derive(Clone)]
crate struct Matrix<'p, 'tcx>(Vec<PatStack<'p, 'tcx>>);

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    crate fn empty() -> Self {
        Matrix(vec![])
    }

    /// Pushes a new row to the matrix. If the row starts with an or-pattern, this expands it.
    crate fn push(&mut self, row: PatStack<'p, 'tcx>) {
        if let Some(rows) = row.expand_or_pat() {
            for row in rows {
                // We recursively expand the or-patterns of the new rows.
                // This is necessary as we might have `0 | (1 | 2)` or e.g., `x @ 0 | x @ (1 | 2)`.
                self.push(row)
            }
        } else {
            self.0.push(row);
        }
    }

    /// Iterate over the first component of each row
    fn heads<'a>(&'a self) -> impl Iterator<Item = &'a Pat<'tcx>> + Captures<'p> {
        self.0.iter().map(|r| r.head())
    }

    /// This computes `D(self)`. See top of the file for explanations.
    fn specialize_wildcard(&self) -> Self {
        self.0.iter().filter_map(|r| r.specialize_wildcard()).collect()
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        cx: &mut MatchCheckCtxt<'p, 'tcx>,
        constructor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Matrix<'p, 'tcx> {
        self.0
            .iter()
            .filter_map(|r| r.specialize_constructor(cx, constructor, ctor_wild_subpatterns))
            .collect()
    }
}

/// Pretty-printer for matrices of patterns, example:
///
/// ```text
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
        let mut matrix = Matrix::empty();
        for x in iter {
            // Using `push` ensures we correctly expand or-patterns.
            matrix.push(x);
        }
        matrix
    }
}

crate struct MatchCheckCtxt<'a, 'tcx> {
    crate tcx: TyCtxt<'tcx>,
    /// The module in which the match occurs. This is necessary for
    /// checking inhabited-ness of types because whether a type is (visibly)
    /// inhabited can depend on whether it was defined in the current module or
    /// not. E.g., `struct Foo { _private: ! }` cannot be seen to be empty
    /// outside it's module and should not be matchable with an empty match
    /// statement.
    crate module: DefId,
    crate param_env: ty::ParamEnv<'tcx>,
    crate pattern_arena: &'a TypedArena<Pat<'tcx>>,
}

impl<'a, 'tcx> MatchCheckCtxt<'a, 'tcx> {
    fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        if self.tcx.features().exhaustive_patterns {
            self.tcx.is_ty_uninhabited_from(self.module, ty, self.param_env)
        } else {
            false
        }
    }

    /// Returns whether the given type is an enum from another crate declared `#[non_exhaustive]`.
    crate fn is_foreign_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
        match ty.kind() {
            ty::Adt(def, ..) => {
                def.is_enum() && def.is_variant_list_non_exhaustive() && !def.did.is_local()
            }
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SliceKind {
    /// Patterns of length `n` (`[x, y]`).
    FixedLen(u64),
    /// Patterns using the `..` notation (`[x, .., y]`).
    /// Captures any array constructor of `length >= i + j`.
    /// In the case where `array_len` is `Some(_)`,
    /// this indicates that we only care about the first `i` and the last `j` values of the array,
    /// and everything in between is a wildcard `_`.
    VarLen(u64, u64),
}

impl SliceKind {
    fn arity(self) -> u64 {
        match self {
            FixedLen(length) => length,
            VarLen(prefix, suffix) => prefix + suffix,
        }
    }

    /// Whether this pattern includes patterns of length `other_len`.
    fn covers_length(self, other_len: u64) -> bool {
        match self {
            FixedLen(len) => len == other_len,
            VarLen(prefix, suffix) => prefix + suffix <= other_len,
        }
    }

    /// Returns a collection of slices that spans the values covered by `self`, subtracted by the
    /// values covered by `other`: i.e., `self \ other` (in set notation).
    fn subtract(self, other: Self) -> SmallVec<[Self; 1]> {
        // Remember, `VarLen(i, j)` covers the union of `FixedLen` from `i + j` to infinity.
        // Naming: we remove the "neg" constructors from the "pos" ones.
        match self {
            FixedLen(pos_len) => {
                if other.covers_length(pos_len) {
                    smallvec![]
                } else {
                    smallvec![self]
                }
            }
            VarLen(pos_prefix, pos_suffix) => {
                let pos_len = pos_prefix + pos_suffix;
                match other {
                    FixedLen(neg_len) => {
                        if neg_len < pos_len {
                            smallvec![self]
                        } else {
                            (pos_len..neg_len)
                                .map(FixedLen)
                                // We know that `neg_len + 1 >= pos_len >= pos_suffix`.
                                .chain(Some(VarLen(neg_len + 1 - pos_suffix, pos_suffix)))
                                .collect()
                        }
                    }
                    VarLen(neg_prefix, neg_suffix) => {
                        let neg_len = neg_prefix + neg_suffix;
                        if neg_len <= pos_len {
                            smallvec![]
                        } else {
                            (pos_len..neg_len).map(FixedLen).collect()
                        }
                    }
                }
            }
        }
    }
}

/// A constructor for array and slice patterns.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Slice {
    /// `None` if the matched value is a slice, `Some(n)` if it is an array of size `n`.
    array_len: Option<u64>,
    /// The kind of pattern it is: fixed-length `[x, y]` or variable length `[x, .., y]`.
    kind: SliceKind,
}

impl Slice {
    /// Returns what patterns this constructor covers: either fixed-length patterns or
    /// variable-length patterns.
    fn pattern_kind(self) -> SliceKind {
        match self {
            Slice { array_len: Some(len), kind: VarLen(prefix, suffix) }
                if prefix + suffix == len =>
            {
                FixedLen(len)
            }
            _ => self.kind,
        }
    }

    /// Returns what values this constructor covers: either values of only one given length, or
    /// values of length above a given length.
    /// This is different from `pattern_kind()` because in some cases the pattern only takes into
    /// account a subset of the entries of the array, but still only captures values of a given
    /// length.
    fn value_kind(self) -> SliceKind {
        match self {
            Slice { array_len: Some(len), kind: VarLen(_, _) } => FixedLen(len),
            _ => self.kind,
        }
    }

    fn arity(self) -> u64 {
        self.pattern_kind().arity()
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// the constructor. See also `Fields`.
///
/// `pat_constructor` retrieves the constructor corresponding to a pattern.
/// `specialize_one_pattern` returns the list of fields corresponding to a pattern, given a
/// constructor. `Constructor::apply` reconstructs the pattern from a pair of `Constructor` and
/// `Fields`.
#[derive(Clone, Debug, PartialEq)]
enum Constructor<'tcx> {
    /// The constructor for patterns that have a single constructor, like tuples, struct patterns
    /// and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(DefId),
    /// Literal values.
    ConstantValue(&'tcx ty::Const<'tcx>),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange<'tcx>),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    FloatRange(&'tcx ty::Const<'tcx>, &'tcx ty::Const<'tcx>, RangeEnd),
    /// Array and slice patterns.
    Slice(Slice),
    /// Fake extra constructor for enums that aren't allowed to be matched exhaustively.
    NonExhaustive,
}

impl<'tcx> Constructor<'tcx> {
    fn is_slice(&self) -> bool {
        match self {
            Slice(_) => true,
            _ => false,
        }
    }

    fn variant_index_for_adt<'a>(
        &self,
        cx: &MatchCheckCtxt<'a, 'tcx>,
        adt: &'tcx ty::AdtDef,
    ) -> VariantIdx {
        match *self {
            Variant(id) => adt.variant_index_with_id(id),
            Single => {
                assert!(!adt.is_enum());
                VariantIdx::new(0)
            }
            ConstantValue(c) => cx
                .tcx
                .destructure_const(cx.param_env.and(c))
                .variant
                .expect("destructed const of adt without variant id"),
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    // Returns the set of constructors covered by `self` but not by
    // anything in `other_ctors`.
    fn subtract_ctors(&self, other_ctors: &Vec<Constructor<'tcx>>) -> Vec<Constructor<'tcx>> {
        if other_ctors.is_empty() {
            return vec![self.clone()];
        }

        match self {
            // Those constructors can only match themselves.
            Single | Variant(_) | ConstantValue(..) | FloatRange(..) => {
                if other_ctors.iter().any(|c| c == self) { vec![] } else { vec![self.clone()] }
            }
            &Slice(slice) => {
                let mut other_slices = other_ctors
                    .iter()
                    .filter_map(|c: &Constructor<'_>| match c {
                        Slice(slice) => Some(*slice),
                        // FIXME(oli-obk): implement `deref` for `ConstValue`
                        ConstantValue(..) => None,
                        _ => bug!("bad slice pattern constructor {:?}", c),
                    })
                    .map(Slice::value_kind);

                match slice.value_kind() {
                    FixedLen(self_len) => {
                        if other_slices.any(|other_slice| other_slice.covers_length(self_len)) {
                            vec![]
                        } else {
                            vec![Slice(slice)]
                        }
                    }
                    kind @ VarLen(..) => {
                        let mut remaining_slices = vec![kind];

                        // For each used slice, subtract from the current set of slices.
                        for other_slice in other_slices {
                            remaining_slices = remaining_slices
                                .into_iter()
                                .flat_map(|remaining_slice| remaining_slice.subtract(other_slice))
                                .collect();

                            // If the constructors that have been considered so far already cover
                            // the entire range of `self`, no need to look at more constructors.
                            if remaining_slices.is_empty() {
                                break;
                            }
                        }

                        remaining_slices
                            .into_iter()
                            .map(|kind| Slice { array_len: slice.array_len, kind })
                            .map(Slice)
                            .collect()
                    }
                }
            }
            IntRange(self_range) => {
                let mut remaining_ranges = vec![self_range.clone()];
                for other_ctor in other_ctors {
                    if let IntRange(other_range) = other_ctor {
                        if other_range == self_range {
                            // If the `self` range appears directly in a `match` arm, we can
                            // eliminate it straight away.
                            remaining_ranges = vec![];
                        } else {
                            // Otherwise explicitly compute the remaining ranges.
                            remaining_ranges = other_range.subtract_from(remaining_ranges);
                        }

                        // If the ranges that have been considered so far already cover the entire
                        // range of values, we can return early.
                        if remaining_ranges.is_empty() {
                            break;
                        }
                    }
                }

                // Convert the ranges back into constructors.
                remaining_ranges.into_iter().map(IntRange).collect()
            }
            // This constructor is never covered by anything else
            NonExhaustive => vec![NonExhaustive],
        }
    }

    /// Apply a constructor to a list of patterns, yielding a new pattern. `pats`
    /// must have as many elements as this constructor's arity.
    ///
    /// This is roughly the inverse of `specialize_one_pattern`.
    ///
    /// Examples:
    /// `self`: `Constructor::Single`
    /// `ty`: `(u32, u32, u32)`
    /// `pats`: `[10, 20, _]`
    /// returns `(10, 20, _)`
    ///
    /// `self`: `Constructor::Variant(Option::Some)`
    /// `ty`: `Option<bool>`
    /// `pats`: `[false]`
    /// returns `Some(false)`
    fn apply<'p>(
        &self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        ty: Ty<'tcx>,
        fields: Fields<'p, 'tcx>,
    ) -> Pat<'tcx> {
        let mut subpatterns = fields.all_patterns();

        let pat = match self {
            Single | Variant(_) => match ty.kind() {
                ty::Adt(..) | ty::Tuple(..) => {
                    let subpatterns = subpatterns
                        .enumerate()
                        .map(|(i, p)| FieldPat { field: Field::new(i), pattern: p })
                        .collect();

                    if let ty::Adt(adt, substs) = ty.kind() {
                        if adt.is_enum() {
                            PatKind::Variant {
                                adt_def: adt,
                                substs,
                                variant_index: self.variant_index_for_adt(cx, adt),
                                subpatterns,
                            }
                        } else {
                            PatKind::Leaf { subpatterns }
                        }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                ty::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                ty::Slice(_) | ty::Array(..) => bug!("bad slice pattern {:?} {:?}", self, ty),
                _ => PatKind::Wild,
            },
            Slice(slice) => match slice.pattern_kind() {
                FixedLen(_) => {
                    PatKind::Slice { prefix: subpatterns.collect(), slice: None, suffix: vec![] }
                }
                VarLen(prefix, _) => {
                    let mut prefix: Vec<_> = subpatterns.by_ref().take(prefix as usize).collect();
                    if slice.array_len.is_some() {
                        // Improves diagnostics a bit: if the type is a known-size array, instead
                        // of reporting `[x, _, .., _, y]`, we prefer to report `[x, .., y]`.
                        // This is incorrect if the size is not known, since `[_, ..]` captures
                        // arrays of lengths `>= 1` whereas `[..]` captures any length.
                        while !prefix.is_empty() && prefix.last().unwrap().is_wildcard() {
                            prefix.pop();
                        }
                    }
                    let suffix: Vec<_> = if slice.array_len.is_some() {
                        // Same as above.
                        subpatterns.skip_while(Pat::is_wildcard).collect()
                    } else {
                        subpatterns.collect()
                    };
                    let wild = Pat::wildcard_from_ty(ty);
                    PatKind::Slice { prefix, slice: Some(wild), suffix }
                }
            },
            &ConstantValue(value) => PatKind::Constant { value },
            &FloatRange(lo, hi, end) => PatKind::Range(PatRange { lo, hi, end }),
            IntRange(range) => return range.to_pat(cx.tcx),
            NonExhaustive => PatKind::Wild,
        };

        Pat { ty, span: DUMMY_SP, kind: Box::new(pat) }
    }

    /// Like `apply`, but where all the subpatterns are wildcards `_`.
    fn apply_wildcards<'a>(&self, cx: &MatchCheckCtxt<'a, 'tcx>, ty: Ty<'tcx>) -> Pat<'tcx> {
        self.apply(cx, ty, Fields::wildcards(cx, self, ty))
    }
}

/// Some fields need to be explicitly hidden away in certain cases; see the comment above the
/// `Fields` struct. This struct represents such a potentially-hidden field. When a field is hidden
/// we still keep its type around.
#[derive(Debug, Copy, Clone)]
enum FilteredField<'p, 'tcx> {
    Kept(&'p Pat<'tcx>),
    Hidden(Ty<'tcx>),
}

impl<'p, 'tcx> FilteredField<'p, 'tcx> {
    fn kept(self) -> Option<&'p Pat<'tcx>> {
        match self {
            FilteredField::Kept(p) => Some(p),
            FilteredField::Hidden(_) => None,
        }
    }

    fn to_pattern(self) -> Pat<'tcx> {
        match self {
            FilteredField::Kept(p) => p.clone(),
            FilteredField::Hidden(ty) => Pat::wildcard_from_ty(ty),
        }
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// those fields, generalized to allow patterns in each field. See also `Constructor`.
///
/// If a private or `non_exhaustive` field is uninhabited, the code mustn't observe that it is
/// uninhabited. For that, we filter these fields out of the matrix. This is subtle because we
/// still need to have those fields back when going to/from a `Pat`. Most of this is handled
/// automatically in `Fields`, but when constructing or deconstructing `Fields` you need to be
/// careful. As a rule, when going to/from the matrix, use the filtered field list; when going
/// to/from `Pat`, use the full field list.
/// This filtering is uncommon in practice, because uninhabited fields are rarely used, so we avoid
/// it when possible to preserve performance.
#[derive(Debug, Clone)]
enum Fields<'p, 'tcx> {
    /// Lists of patterns that don't contain any filtered fields.
    /// `Slice` and `Vec` behave the same; the difference is only to avoid allocating and
    /// triple-dereferences when possible. Frankly this is premature optimization, I (Nadrieril)
    /// have not measured if it really made a difference.
    Slice(&'p [Pat<'tcx>]),
    Vec(SmallVec<[&'p Pat<'tcx>; 2]>),
    /// Patterns where some of the fields need to be hidden. `kept_count` caches the number of
    /// non-hidden fields.
    Filtered {
        fields: SmallVec<[FilteredField<'p, 'tcx>; 2]>,
        kept_count: usize,
    },
}

impl<'p, 'tcx> Fields<'p, 'tcx> {
    fn empty() -> Self {
        Fields::Slice(&[])
    }

    /// Construct a new `Fields` from the given pattern. Must not be used if the pattern is a field
    /// of a struct/tuple/variant.
    fn from_single_pattern(pat: &'p Pat<'tcx>) -> Self {
        Fields::Slice(std::slice::from_ref(pat))
    }

    /// Construct a new `Fields` from the given patterns. You must be sure those patterns can't
    /// contain fields that need to be filtered out. When in doubt, prefer `replace_fields`.
    fn from_slice_unfiltered(pats: &'p [Pat<'tcx>]) -> Self {
        Fields::Slice(pats)
    }

    /// Convenience; internal use.
    fn wildcards_from_tys(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        tys: impl IntoIterator<Item = Ty<'tcx>>,
    ) -> Self {
        let wilds = tys.into_iter().map(Pat::wildcard_from_ty);
        let pats = cx.pattern_arena.alloc_from_iter(wilds);
        Fields::Slice(pats)
    }

    /// Creates a new list of wildcard fields for a given constructor.
    fn wildcards(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        constructor: &Constructor<'tcx>,
        ty: Ty<'tcx>,
    ) -> Self {
        let wildcard_from_ty = |ty| &*cx.pattern_arena.alloc(Pat::wildcard_from_ty(ty));

        let ret = match constructor {
            Single | Variant(_) => match ty.kind() {
                ty::Tuple(ref fs) => {
                    Fields::wildcards_from_tys(cx, fs.into_iter().map(|ty| ty.expect_ty()))
                }
                ty::Ref(_, rty, _) => Fields::from_single_pattern(wildcard_from_ty(rty)),
                ty::Adt(adt, substs) => {
                    if adt.is_box() {
                        // Use T as the sub pattern type of Box<T>.
                        Fields::from_single_pattern(wildcard_from_ty(substs.type_at(0)))
                    } else {
                        let variant = &adt.variants[constructor.variant_index_for_adt(cx, adt)];
                        // Whether we must not match the fields of this variant exhaustively.
                        let is_non_exhaustive =
                            variant.is_field_list_non_exhaustive() && !adt.did.is_local();
                        let field_tys = variant.fields.iter().map(|field| field.ty(cx.tcx, substs));
                        // In the following cases, we don't need to filter out any fields. This is
                        // the vast majority of real cases, since uninhabited fields are uncommon.
                        let has_no_hidden_fields = (adt.is_enum() && !is_non_exhaustive)
                            || !field_tys.clone().any(|ty| cx.is_uninhabited(ty));

                        if has_no_hidden_fields {
                            Fields::wildcards_from_tys(cx, field_tys)
                        } else {
                            let mut kept_count = 0;
                            let fields = variant
                                .fields
                                .iter()
                                .map(|field| {
                                    let ty = field.ty(cx.tcx, substs);
                                    let is_visible = adt.is_enum()
                                        || field.vis.is_accessible_from(cx.module, cx.tcx);
                                    let is_uninhabited = cx.is_uninhabited(ty);

                                    // In the cases of either a `#[non_exhaustive]` field list
                                    // or a non-public field, we hide uninhabited fields in
                                    // order not to reveal the uninhabitedness of the whole
                                    // variant.
                                    if is_uninhabited && (!is_visible || is_non_exhaustive) {
                                        FilteredField::Hidden(ty)
                                    } else {
                                        kept_count += 1;
                                        FilteredField::Kept(wildcard_from_ty(ty))
                                    }
                                })
                                .collect();
                            Fields::Filtered { fields, kept_count }
                        }
                    }
                }
                _ => Fields::empty(),
            },
            Slice(slice) => match *ty.kind() {
                ty::Slice(ty) | ty::Array(ty, _) => {
                    let arity = slice.arity();
                    Fields::wildcards_from_tys(cx, (0..arity).map(|_| ty))
                }
                _ => bug!("bad slice pattern {:?} {:?}", constructor, ty),
            },
            ConstantValue(..) | FloatRange(..) | IntRange(..) | NonExhaustive => Fields::empty(),
        };
        debug!("Fields::wildcards({:?}, {:?}) = {:#?}", constructor, ty, ret);
        ret
    }

    /// Returns the number of patterns from the viewpoint of match-checking, i.e. excluding hidden
    /// fields. This is what we want in most cases in this file, the only exception being
    /// conversion to/from `Pat`.
    fn len(&self) -> usize {
        match self {
            Fields::Slice(pats) => pats.len(),
            Fields::Vec(pats) => pats.len(),
            Fields::Filtered { kept_count, .. } => *kept_count,
        }
    }

    /// Returns the complete list of patterns, including hidden fields.
    fn all_patterns(self) -> impl Iterator<Item = Pat<'tcx>> {
        let pats: SmallVec<[_; 2]> = match self {
            Fields::Slice(pats) => pats.iter().cloned().collect(),
            Fields::Vec(pats) => pats.into_iter().cloned().collect(),
            Fields::Filtered { fields, .. } => {
                // We don't skip any fields here.
                fields.into_iter().map(|p| p.to_pattern()).collect()
            }
        };
        pats.into_iter()
    }

    /// Overrides some of the fields with the provided patterns. Exactly like
    /// `replace_fields_indexed`, except that it takes `FieldPat`s as input.
    fn replace_with_fieldpats(
        &self,
        new_pats: impl IntoIterator<Item = &'p FieldPat<'tcx>>,
    ) -> Self {
        self.replace_fields_indexed(
            new_pats.into_iter().map(|pat| (pat.field.index(), &pat.pattern)),
        )
    }

    /// Overrides some of the fields with the provided patterns. This is used when a pattern
    /// defines some fields but not all, for example `Foo { field1: Some(_), .. }`: here we start with a
    /// `Fields` that is just one wildcard per field of the `Foo` struct, and override the entry
    /// corresponding to `field1` with the pattern `Some(_)`. This is also used for slice patterns
    /// for the same reason.
    fn replace_fields_indexed(
        &self,
        new_pats: impl IntoIterator<Item = (usize, &'p Pat<'tcx>)>,
    ) -> Self {
        let mut fields = self.clone();
        if let Fields::Slice(pats) = fields {
            fields = Fields::Vec(pats.iter().collect());
        }

        match &mut fields {
            Fields::Vec(pats) => {
                for (i, pat) in new_pats {
                    pats[i] = pat
                }
            }
            Fields::Filtered { fields, .. } => {
                for (i, pat) in new_pats {
                    if let FilteredField::Kept(p) = &mut fields[i] {
                        *p = pat
                    }
                }
            }
            Fields::Slice(_) => unreachable!(),
        }
        fields
    }

    /// Replaces contained fields with the given filtered list of patterns, e.g. taken from the
    /// matrix. There must be `len()` patterns in `pats`.
    fn replace_fields(
        &self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        pats: impl IntoIterator<Item = Pat<'tcx>>,
    ) -> Self {
        let pats: &[_] = cx.pattern_arena.alloc_from_iter(pats);

        match self {
            Fields::Filtered { fields, kept_count } => {
                let mut pats = pats.iter();
                let mut fields = fields.clone();
                for f in &mut fields {
                    if let FilteredField::Kept(p) = f {
                        // We take one input pattern for each `Kept` field, in order.
                        *p = pats.next().unwrap();
                    }
                }
                Fields::Filtered { fields, kept_count: *kept_count }
            }
            _ => Fields::Slice(pats),
        }
    }

    fn push_on_patstack(self, stack: &[&'p Pat<'tcx>]) -> PatStack<'p, 'tcx> {
        let pats: SmallVec<_> = match self {
            Fields::Slice(pats) => pats.iter().chain(stack.iter().copied()).collect(),
            Fields::Vec(mut pats) => {
                pats.extend_from_slice(stack);
                pats
            }
            Fields::Filtered { fields, .. } => {
                // We skip hidden fields here
                fields.into_iter().filter_map(|p| p.kept()).chain(stack.iter().copied()).collect()
            }
        };
        PatStack::from_vec(pats)
    }
}

#[derive(Clone, Debug)]
crate enum Usefulness<'tcx> {
    /// Carries a list of unreachable subpatterns. Used only in the presence of or-patterns.
    Useful(Vec<Span>),
    /// Carries a list of witnesses of non-exhaustiveness.
    UsefulWithWitness(Vec<Witness<'tcx>>),
    NotUseful,
}

impl<'tcx> Usefulness<'tcx> {
    fn new_useful(preference: WitnessPreference) -> Self {
        match preference {
            ConstructWitness => UsefulWithWitness(vec![Witness(vec![])]),
            LeaveOutWitness => Useful(vec![]),
        }
    }

    fn is_useful(&self) -> bool {
        match *self {
            NotUseful => false,
            _ => true,
        }
    }

    fn apply_constructor<'p>(
        self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ty: Ty<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Self {
        match self {
            UsefulWithWitness(witnesses) => UsefulWithWitness(
                witnesses
                    .into_iter()
                    .map(|witness| witness.apply_constructor(cx, &ctor, ty, ctor_wild_subpatterns))
                    .collect(),
            ),
            x => x,
        }
    }

    fn apply_wildcard(self, ty: Ty<'tcx>) -> Self {
        match self {
            UsefulWithWitness(witnesses) => {
                let wild = Pat::wildcard_from_ty(ty);
                UsefulWithWitness(
                    witnesses
                        .into_iter()
                        .map(|mut witness| {
                            witness.0.push(wild.clone());
                            witness
                        })
                        .collect(),
                )
            }
            x => x,
        }
    }

    fn apply_missing_ctors(
        self,
        cx: &MatchCheckCtxt<'_, 'tcx>,
        ty: Ty<'tcx>,
        missing_ctors: &MissingConstructors<'tcx>,
    ) -> Self {
        match self {
            UsefulWithWitness(witnesses) => {
                let new_patterns: Vec<_> =
                    missing_ctors.iter().map(|ctor| ctor.apply_wildcards(cx, ty)).collect();
                // Add the new patterns to each witness
                UsefulWithWitness(
                    witnesses
                        .into_iter()
                        .flat_map(|witness| {
                            new_patterns.iter().map(move |pat| {
                                let mut witness = witness.clone();
                                witness.0.push(pat.clone());
                                witness
                            })
                        })
                        .collect(),
                )
            }
            x => x,
        }
    }
}

#[derive(Copy, Clone, Debug)]
crate enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness,
}

#[derive(Copy, Clone, Debug)]
struct PatCtxt<'tcx> {
    ty: Ty<'tcx>,
    span: Span,
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
crate struct Witness<'tcx>(Vec<Pat<'tcx>>);

impl<'tcx> Witness<'tcx> {
    crate fn single_pattern(self) -> Pat<'tcx> {
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
    fn apply_constructor<'p>(
        mut self,
        cx: &MatchCheckCtxt<'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ty: Ty<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Self {
        let pat = {
            let len = self.0.len();
            let arity = ctor_wild_subpatterns.len();
            let pats = self.0.drain((len - arity)..).rev();
            let fields = ctor_wild_subpatterns.replace_fields(cx, pats);
            ctor.apply(cx, ty, fields)
        };

        self.0.push(pat);

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
/// Invariant: this returns an empty `Vec` if and only if the type is uninhabited (as determined by
/// `cx.is_uninhabited()`).
fn all_constructors<'a, 'tcx>(
    cx: &mut MatchCheckCtxt<'a, 'tcx>,
    pcx: PatCtxt<'tcx>,
) -> Vec<Constructor<'tcx>> {
    debug!("all_constructors({:?})", pcx.ty);
    let make_range = |start, end| {
        IntRange(
            // `unwrap()` is ok because we know the type is an integer.
            IntRange::from_range(cx.tcx, start, end, pcx.ty, &RangeEnd::Included, pcx.span)
                .unwrap(),
        )
    };
    match *pcx.ty.kind() {
        ty::Bool => {
            [true, false].iter().map(|&b| ConstantValue(ty::Const::from_bool(cx.tcx, b))).collect()
        }
        ty::Array(ref sub_ty, len) if len.try_eval_usize(cx.tcx, cx.param_env).is_some() => {
            let len = len.eval_usize(cx.tcx, cx.param_env);
            if len != 0 && cx.is_uninhabited(sub_ty) {
                vec![]
            } else {
                vec![Slice(Slice { array_len: Some(len), kind: VarLen(0, 0) })]
            }
        }
        // Treat arrays of a constant but unknown length like slices.
        ty::Array(ref sub_ty, _) | ty::Slice(ref sub_ty) => {
            let kind = if cx.is_uninhabited(sub_ty) { FixedLen(0) } else { VarLen(0, 0) };
            vec![Slice(Slice { array_len: None, kind })]
        }
        ty::Adt(def, substs) if def.is_enum() => {
            let ctors: Vec<_> = if cx.tcx.features().exhaustive_patterns {
                // If `exhaustive_patterns` is enabled, we exclude variants known to be
                // uninhabited.
                def.variants
                    .iter()
                    .filter(|v| {
                        !v.uninhabited_from(cx.tcx, substs, def.adt_kind(), cx.param_env)
                            .contains(cx.tcx, cx.module)
                    })
                    .map(|v| Variant(v.def_id))
                    .collect()
            } else {
                def.variants.iter().map(|v| Variant(v.def_id)).collect()
            };

            // If the enum is declared as `#[non_exhaustive]`, we treat it as if it had an
            // additional "unknown" constructor.
            // There is no point in enumerating all possible variants, because the user can't
            // actually match against them all themselves. So we always return only the fictitious
            // constructor.
            // E.g., in an example like:
            // ```
            //     let err: io::ErrorKind = ...;
            //     match err {
            //         io::ErrorKind::NotFound => {},
            //     }
            // ```
            // we don't want to show every possible IO error, but instead have only `_` as the
            // witness.
            let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(pcx.ty);

            // If `exhaustive_patterns` is disabled and our scrutinee is an empty enum, we treat it
            // as though it had an "unknown" constructor to avoid exposing its emptyness. Note that
            // an empty match will still be considered exhaustive because that case is handled
            // separately in `check_match`.
            let is_secretly_empty =
                def.variants.is_empty() && !cx.tcx.features().exhaustive_patterns;

            if is_secretly_empty || is_declared_nonexhaustive { vec![NonExhaustive] } else { ctors }
        }
        ty::Char => {
            vec![
                // The valid Unicode Scalar Value ranges.
                make_range('\u{0000}' as u128, '\u{D7FF}' as u128),
                make_range('\u{E000}' as u128, '\u{10FFFF}' as u128),
            ]
        }
        ty::Int(_) | ty::Uint(_)
            if pcx.ty.is_ptr_sized_integral()
                && !cx.tcx.features().precise_pointer_size_matching =>
        {
            // `usize`/`isize` are not allowed to be matched exhaustively unless the
            // `precise_pointer_size_matching` feature is enabled. So we treat those types like
            // `#[non_exhaustive]` enums by returning a special unmatcheable constructor.
            vec![NonExhaustive]
        }
        ty::Int(ity) => {
            let bits = Integer::from_attr(&cx.tcx, SignedInt(ity)).size().bits() as u128;
            let min = 1u128 << (bits - 1);
            let max = min - 1;
            vec![make_range(min, max)]
        }
        ty::Uint(uty) => {
            let size = Integer::from_attr(&cx.tcx, UnsignedInt(uty)).size();
            let max = truncate(u128::MAX, size);
            vec![make_range(0, max)]
        }
        _ => {
            if cx.is_uninhabited(pcx.ty) {
                vec![]
            } else {
                vec![Single]
            }
        }
    }
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
#[derive(Clone, Debug)]
struct IntRange<'tcx> {
    range: RangeInclusive<u128>,
    ty: Ty<'tcx>,
    span: Span,
}

impl<'tcx> IntRange<'tcx> {
    #[inline]
    fn is_integral(ty: Ty<'_>) -> bool {
        match ty.kind() {
            ty::Char | ty::Int(_) | ty::Uint(_) => true,
            _ => false,
        }
    }

    fn is_singleton(&self) -> bool {
        self.range.start() == self.range.end()
    }

    fn boundaries(&self) -> (u128, u128) {
        (*self.range.start(), *self.range.end())
    }

    /// Don't treat `usize`/`isize` exhaustively unless the `precise_pointer_size_matching` feature
    /// is enabled.
    fn treat_exhaustively(&self, tcx: TyCtxt<'tcx>) -> bool {
        !self.ty.is_ptr_sized_integral() || tcx.features().precise_pointer_size_matching
    }

    #[inline]
    fn integral_size_and_signed_bias(tcx: TyCtxt<'tcx>, ty: Ty<'_>) -> Option<(Size, u128)> {
        match *ty.kind() {
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
        span: Span,
    ) -> Option<IntRange<'tcx>> {
        if let Some((target_size, bias)) = Self::integral_size_and_signed_bias(tcx, value.ty) {
            let ty = value.ty;
            let val = (|| {
                if let ty::ConstKind::Value(ConstValue::Scalar(scalar)) = value.val {
                    // For this specific pattern we can skip a lot of effort and go
                    // straight to the result, after doing a bit of checking. (We
                    // could remove this branch and just fall through, which
                    // is more general but much slower.)
                    if let Ok(bits) = scalar.to_bits_or_ptr(target_size, &tcx) {
                        return Some(bits);
                    }
                }
                // This is a more general form of the previous case.
                value.try_eval_bits(tcx, param_env, ty)
            })()?;
            let val = val ^ bias;
            Some(IntRange { range: val..=val, ty, span })
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
        span: Span,
    ) -> Option<IntRange<'tcx>> {
        if Self::is_integral(ty) {
            // Perform a shift if the underlying types are signed,
            // which makes the interval arithmetic simpler.
            let bias = IntRange::signed_bias(tcx, ty);
            let (lo, hi) = (lo ^ bias, hi ^ bias);
            let offset = (*end == RangeEnd::Excluded) as u128;
            if lo > hi || (lo == hi && *end == RangeEnd::Excluded) {
                // This should have been caught earlier by E0030.
                bug!("malformed range pattern: {}..={}", lo, (hi - offset));
            }
            Some(IntRange { range: lo..=(hi - offset), ty, span })
        } else {
            None
        }
    }

    fn from_pat(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        pat: &Pat<'tcx>,
    ) -> Option<IntRange<'tcx>> {
        match pat_constructor(tcx, param_env, pat)? {
            IntRange(range) => Some(range),
            _ => None,
        }
    }

    // The return value of `signed_bias` should be XORed with an endpoint to encode/decode it.
    fn signed_bias(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> u128 {
        match *ty.kind() {
            ty::Int(ity) => {
                let bits = Integer::from_attr(&tcx, SignedInt(ity)).size().bits() as u128;
                1u128 << (bits - 1)
            }
            _ => 0,
        }
    }

    /// Returns a collection of ranges that spans the values covered by `ranges`, subtracted
    /// by the values covered by `self`: i.e., `ranges \ self` (in set notation).
    fn subtract_from(&self, ranges: Vec<IntRange<'tcx>>) -> Vec<IntRange<'tcx>> {
        let mut remaining_ranges = vec![];
        let ty = self.ty;
        let span = self.span;
        let (lo, hi) = self.boundaries();
        for subrange in ranges {
            let (subrange_lo, subrange_hi) = subrange.range.into_inner();
            if lo > subrange_hi || subrange_lo > hi {
                // The pattern doesn't intersect with the subrange at all,
                // so the subrange remains untouched.
                remaining_ranges.push(IntRange { range: subrange_lo..=subrange_hi, ty, span });
            } else {
                if lo > subrange_lo {
                    // The pattern intersects an upper section of the
                    // subrange, so a lower section will remain.
                    remaining_ranges.push(IntRange { range: subrange_lo..=(lo - 1), ty, span });
                }
                if hi < subrange_hi {
                    // The pattern intersects a lower section of the
                    // subrange, so an upper section will remain.
                    remaining_ranges.push(IntRange { range: (hi + 1)..=subrange_hi, ty, span });
                }
            }
        }
        remaining_ranges
    }

    fn is_subrange(&self, other: &Self) -> bool {
        other.range.start() <= self.range.start() && self.range.end() <= other.range.end()
    }

    fn intersection(&self, tcx: TyCtxt<'tcx>, other: &Self) -> Option<Self> {
        let ty = self.ty;
        let (lo, hi) = self.boundaries();
        let (other_lo, other_hi) = other.boundaries();
        if self.treat_exhaustively(tcx) {
            if lo <= other_hi && other_lo <= hi {
                let span = other.span;
                Some(IntRange { range: max(lo, other_lo)..=min(hi, other_hi), ty, span })
            } else {
                None
            }
        } else {
            // If the range should not be treated exhaustively, fallback to checking for inclusion.
            if self.is_subrange(other) { Some(self.clone()) } else { None }
        }
    }

    fn suspicious_intersection(&self, other: &Self) -> bool {
        // `false` in the following cases:
        // 1     ----      // 1  ----------   // 1 ----        // 1       ----
        // 2  ----------   // 2     ----      // 2       ----  // 2 ----
        //
        // The following are currently `false`, but could be `true` in the future (#64007):
        // 1 ---------       // 1     ---------
        // 2     ----------  // 2 ----------
        //
        // `true` in the following cases:
        // 1 -------          // 1       -------
        // 2       --------   // 2 -------
        let (lo, hi) = self.boundaries();
        let (other_lo, other_hi) = other.boundaries();
        lo == other_hi || hi == other_lo
    }

    fn to_pat(&self, tcx: TyCtxt<'tcx>) -> Pat<'tcx> {
        let (lo, hi) = self.boundaries();

        let bias = IntRange::signed_bias(tcx, self.ty);
        let (lo, hi) = (lo ^ bias, hi ^ bias);

        let ty = ty::ParamEnv::empty().and(self.ty);
        let lo_const = ty::Const::from_bits(tcx, lo, ty);
        let hi_const = ty::Const::from_bits(tcx, hi, ty);

        let kind = if lo == hi {
            PatKind::Constant { value: lo_const }
        } else {
            PatKind::Range(PatRange { lo: lo_const, hi: hi_const, end: RangeEnd::Included })
        };

        // This is a brand new pattern, so we don't reuse `self.span`.
        Pat { ty: self.ty, span: DUMMY_SP, kind: Box::new(kind) }
    }
}

/// Ignore spans when comparing, they don't carry semantic information as they are only for lints.
impl<'tcx> std::cmp::PartialEq for IntRange<'tcx> {
    fn eq(&self, other: &Self) -> bool {
        self.range == other.range && self.ty == other.ty
    }
}

// A struct to compute a set of constructors equivalent to `all_ctors \ used_ctors`.
struct MissingConstructors<'tcx> {
    all_ctors: Vec<Constructor<'tcx>>,
    used_ctors: Vec<Constructor<'tcx>>,
}

impl<'tcx> MissingConstructors<'tcx> {
    fn new(all_ctors: Vec<Constructor<'tcx>>, used_ctors: Vec<Constructor<'tcx>>) -> Self {
        MissingConstructors { all_ctors, used_ctors }
    }

    fn into_inner(self) -> (Vec<Constructor<'tcx>>, Vec<Constructor<'tcx>>) {
        (self.all_ctors, self.used_ctors)
    }

    fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
    /// Whether this contains all the constructors for the given type or only a
    /// subset.
    fn all_ctors_are_missing(&self) -> bool {
        self.used_ctors.is_empty()
    }

    /// Iterate over all_ctors \ used_ctors
    fn iter<'a>(&'a self) -> impl Iterator<Item = Constructor<'tcx>> + Captures<'a> {
        self.all_ctors.iter().flat_map(move |req_ctor| req_ctor.subtract_ctors(&self.used_ctors))
    }
}

impl<'tcx> fmt::Debug for MissingConstructors<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ctors: Vec<_> = self.iter().collect();
        write!(f, "{:?}", ctors)
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
/// All the patterns at each column of the `matrix ++ v` matrix must have the same type.
///
/// This is used both for reachability checking (if a pattern isn't useful in
/// relation to preceding patterns, it is not reachable) and exhaustiveness
/// checking (if a wildcard pattern is useful in relation to a matrix, the
/// matrix isn't exhaustive).
///
/// `is_under_guard` is used to inform if the pattern has a guard. If it
/// has one it must not be inserted into the matrix. This shouldn't be
/// relied on for soundness.
crate fn is_useful<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &PatStack<'p, 'tcx>,
    witness_preference: WitnessPreference,
    hir_id: HirId,
    is_under_guard: bool,
    is_top_level: bool,
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

    // If the first pattern is an or-pattern, expand it.
    if let Some(vs) = v.expand_or_pat() {
        // We need to push the already-seen patterns into the matrix in order to detect redundant
        // branches like `Some(_) | Some(0)`. We also keep track of the unreachable subpatterns.
        let mut matrix = matrix.clone();
        // `Vec` of all the unreachable branches of the current or-pattern.
        let mut unreachable_branches = Vec::new();
        // Subpatterns that are unreachable from all branches. E.g. in the following case, the last
        // `true` is unreachable only from one branch, so it is overall reachable.
        // ```
        // match (true, true) {
        //     (true, true) => {}
        //     (false | true, false | true) => {}
        // }
        // ```
        let mut unreachable_subpats = FxHashSet::default();
        // Whether any branch at all is useful.
        let mut any_is_useful = false;

        for v in vs {
            let res = is_useful(cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false);
            match res {
                Useful(pats) => {
                    if !any_is_useful {
                        any_is_useful = true;
                        // Initialize with the first set of unreachable subpatterns encountered.
                        unreachable_subpats = pats.into_iter().collect();
                    } else {
                        // Keep the patterns unreachable from both this and previous branches.
                        unreachable_subpats =
                            pats.into_iter().filter(|p| unreachable_subpats.contains(p)).collect();
                    }
                }
                NotUseful => unreachable_branches.push(v.head().span),
                UsefulWithWitness(_) => {
                    bug!("Encountered or-pat in `v` during exhaustiveness checking")
                }
            }
            // If pattern has a guard don't add it to the matrix
            if !is_under_guard {
                matrix.push(v);
            }
        }
        if any_is_useful {
            // Collect all the unreachable patterns.
            unreachable_branches.extend(unreachable_subpats);
            return Useful(unreachable_branches);
        } else {
            return NotUseful;
        }
    }

    // FIXME(Nadrieril): Hack to work around type normalization issues (see #72476).
    let ty = matrix.heads().next().map(|r| r.ty).unwrap_or(v.head().ty);
    let pcx = PatCtxt { ty, span: v.head().span };

    debug!("is_useful_expand_first_col: pcx={:#?}, expanding {:#?}", pcx, v.head());

    let ret = if let Some(constructor) = pat_constructor(cx.tcx, cx.param_env, v.head()) {
        debug!("is_useful - expanding constructor: {:#?}", constructor);
        split_grouped_constructors(
            cx.tcx,
            cx.param_env,
            pcx,
            vec![constructor],
            matrix,
            pcx.span,
            Some(hir_id),
        )
        .into_iter()
        .map(|c| {
            is_useful_specialized(
                cx,
                matrix,
                v,
                c,
                pcx.ty,
                witness_preference,
                hir_id,
                is_under_guard,
            )
        })
        .find(|result| result.is_useful())
        .unwrap_or(NotUseful)
    } else {
        debug!("is_useful - expanding wildcard");

        let used_ctors: Vec<Constructor<'_>> =
            matrix.heads().filter_map(|p| pat_constructor(cx.tcx, cx.param_env, p)).collect();
        debug!("is_useful_used_ctors = {:#?}", used_ctors);
        // `all_ctors` are all the constructors for the given type, which
        // should all be represented (or caught with the wild pattern `_`).
        let all_ctors = all_constructors(cx, pcx);
        debug!("is_useful_all_ctors = {:#?}", all_ctors);

        // `missing_ctors` is the set of constructors from the same type as the
        // first column of `matrix` that are matched only by wildcard patterns
        // from the first column.
        //
        // Therefore, if there is some pattern that is unmatched by `matrix`,
        // it will still be unmatched if the first constructor is replaced by
        // any of the constructors in `missing_ctors`

        // Missing constructors are those that are not matched by any non-wildcard patterns in the
        // current column. We only fully construct them on-demand, because they're rarely used and
        // can be big.
        let missing_ctors = MissingConstructors::new(all_ctors, used_ctors);

        debug!("is_useful_missing_ctors.empty()={:#?}", missing_ctors.is_empty(),);

        if missing_ctors.is_empty() {
            let (all_ctors, _) = missing_ctors.into_inner();
            split_grouped_constructors(cx.tcx, cx.param_env, pcx, all_ctors, matrix, DUMMY_SP, None)
                .into_iter()
                .map(|c| {
                    is_useful_specialized(
                        cx,
                        matrix,
                        v,
                        c,
                        pcx.ty,
                        witness_preference,
                        hir_id,
                        is_under_guard,
                    )
                })
                .find(|result| result.is_useful())
                .unwrap_or(NotUseful)
        } else {
            let matrix = matrix.specialize_wildcard();
            let v = v.to_tail();
            let usefulness =
                is_useful(cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false);

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
            // However, there is a case where we don't want
            // to do this and instead report a single `_` witness:
            // if the user didn't actually specify a constructor
            // in this arm, e.g., in
            // ```
            //     let x: (Direction, Direction, bool) = ...;
            //     let (_, _, false) = x;
            // ```
            // we don't want to show all 16 possible witnesses
            // `(<direction-1>, <direction-2>, true)` - we are
            // satisfied with `(_, _, true)`. In this case,
            // `used_ctors` is empty.
            // The exception is: if we are at the top-level, for example in an empty match, we
            // sometimes prefer reporting the list of constructors instead of just `_`.
            let report_ctors_rather_than_wildcard = is_top_level && !IntRange::is_integral(pcx.ty);
            if missing_ctors.all_ctors_are_missing() && !report_ctors_rather_than_wildcard {
                // All constructors are unused. Add a wild pattern
                // rather than each individual constructor.
                usefulness.apply_wildcard(pcx.ty)
            } else {
                // Construct for each missing constructor a "wild" version of this
                // constructor, that matches everything that can be built with
                // it. For example, if `ctor` is a `Constructor::Variant` for
                // `Option::Some`, we get the pattern `Some(_)`.
                usefulness.apply_missing_ctors(cx, pcx.ty, &missing_ctors)
            }
        }
    };
    debug!("is_useful::returns({:#?}, {:#?}) = {:?}", matrix, v, ret);
    ret
}

/// A shorthand for the `U(S(c, P), S(c, q))` operation from the paper. I.e., `is_useful` applied
/// to the specialised version of both the pattern matrix `P` and the new pattern `q`.
fn is_useful_specialized<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &PatStack<'p, 'tcx>,
    ctor: Constructor<'tcx>,
    ty: Ty<'tcx>,
    witness_preference: WitnessPreference,
    hir_id: HirId,
    is_under_guard: bool,
) -> Usefulness<'tcx> {
    debug!("is_useful_specialized({:#?}, {:#?}, {:?})", v, ctor, ty);

    // We cache the result of `Fields::wildcards` because it is used a lot.
    let ctor_wild_subpatterns = Fields::wildcards(cx, &ctor, ty);
    let matrix = matrix.specialize_constructor(cx, &ctor, &ctor_wild_subpatterns);
    v.specialize_constructor(cx, &ctor, &ctor_wild_subpatterns)
        .map(|v| is_useful(cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false))
        .map(|u| u.apply_constructor(cx, &ctor, ty, &ctor_wild_subpatterns))
        .unwrap_or(NotUseful)
}

/// Determines the constructor that the given pattern can be specialized to.
/// Returns `None` in case of a catch-all, which can't be specialized.
fn pat_constructor<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pat: &Pat<'tcx>,
) -> Option<Constructor<'tcx>> {
    match *pat.kind {
        PatKind::AscribeUserType { .. } => bug!(), // Handled by `expand_pattern`
        PatKind::Binding { .. } | PatKind::Wild => None,
        PatKind::Leaf { .. } | PatKind::Deref { .. } => Some(Single),
        PatKind::Variant { adt_def, variant_index, .. } => {
            Some(Variant(adt_def.variants[variant_index].def_id))
        }
        PatKind::Constant { value } => {
            if let Some(int_range) = IntRange::from_const(tcx, param_env, value, pat.span) {
                Some(IntRange(int_range))
            } else {
                match (value.val, &value.ty.kind()) {
                    (_, ty::Array(_, n)) => {
                        let len = n.eval_usize(tcx, param_env);
                        Some(Slice(Slice { array_len: Some(len), kind: FixedLen(len) }))
                    }
                    (ty::ConstKind::Value(ConstValue::Slice { start, end, .. }), ty::Slice(_)) => {
                        let len = (end - start) as u64;
                        Some(Slice(Slice { array_len: None, kind: FixedLen(len) }))
                    }
                    // FIXME(oli-obk): implement `deref` for `ConstValue`
                    // (ty::ConstKind::Value(ConstValue::ByRef { .. }), ty::Slice(_)) => { ... }
                    _ => Some(ConstantValue(value)),
                }
            }
        }
        PatKind::Range(PatRange { lo, hi, end }) => {
            let ty = lo.ty;
            if let Some(int_range) = IntRange::from_range(
                tcx,
                lo.eval_bits(tcx, param_env, lo.ty),
                hi.eval_bits(tcx, param_env, hi.ty),
                ty,
                &end,
                pat.span,
            ) {
                Some(IntRange(int_range))
            } else {
                Some(FloatRange(lo, hi, end))
            }
        }
        PatKind::Array { ref prefix, ref slice, ref suffix }
        | PatKind::Slice { ref prefix, ref slice, ref suffix } => {
            let array_len = match pat.ty.kind() {
                ty::Array(_, length) => Some(length.eval_usize(tcx, param_env)),
                ty::Slice(_) => None,
                _ => span_bug!(pat.span, "bad ty {:?} for slice pattern", pat.ty),
            };
            let prefix = prefix.len() as u64;
            let suffix = suffix.len() as u64;
            let kind =
                if slice.is_some() { VarLen(prefix, suffix) } else { FixedLen(prefix + suffix) };
            Some(Slice(Slice { array_len, kind }))
        }
        PatKind::Or { .. } => bug!("Or-pattern should have been expanded earlier on."),
    }
}

// checks whether a constant is equal to a user-written slice pattern. Only supports byte slices,
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
    let const_val_val = if let ty::ConstKind::Value(val) = const_val.val {
        val
    } else {
        bug!(
            "slice_pat_covered_by_const: {:#?}, {:#?}, {:#?}, {:#?}",
            const_val,
            prefix,
            slice,
            suffix,
        )
    };

    let data: &[u8] = match (const_val_val, &const_val.ty.kind()) {
        (ConstValue::ByRef { offset, alloc, .. }, ty::Array(t, n)) => {
            assert_eq!(*t, tcx.types.u8);
            let n = n.eval_usize(tcx, param_env);
            let ptr = Pointer::new(AllocId(0), offset);
            alloc.get_bytes(&tcx, ptr, Size::from_bytes(n)).unwrap()
        }
        (ConstValue::Slice { data, start, end }, ty::Slice(t)) => {
            assert_eq!(*t, tcx.types.u8);
            let ptr = Pointer::new(AllocId(0), Size::from_bytes(start));
            data.get_bytes(&tcx, ptr, Size::from_bytes(end - start)).unwrap()
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
        if let box PatKind::Constant { value } = pat.kind {
            let b = value.eval_bits(tcx, param_env, pat.ty);
            assert_eq!(b as u8 as u128, b);
            if b as u8 != *ch {
                return Ok(false);
            }
        }
    }

    Ok(true)
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
///
/// `hir_id` is `None` when we're evaluating the wildcard pattern, do not lint for overlapping in
/// ranges that case.
///
/// This also splits variable-length slices into fixed-length slices.
fn split_grouped_constructors<'p, 'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    pcx: PatCtxt<'tcx>,
    ctors: Vec<Constructor<'tcx>>,
    matrix: &Matrix<'p, 'tcx>,
    span: Span,
    hir_id: Option<HirId>,
) -> Vec<Constructor<'tcx>> {
    let ty = pcx.ty;
    let mut split_ctors = Vec::with_capacity(ctors.len());
    debug!("split_grouped_constructors({:#?}, {:#?})", matrix, ctors);

    for ctor in ctors.into_iter() {
        match ctor {
            IntRange(ctor_range) if ctor_range.treat_exhaustively(tcx) => {
                // Fast-track if the range is trivial. In particular, don't do the overlapping
                // ranges check.
                if ctor_range.is_singleton() {
                    split_ctors.push(IntRange(ctor_range));
                    continue;
                }

                /// Represents a border between 2 integers. Because the intervals spanning borders
                /// must be able to cover every integer, we need to be able to represent
                /// 2^128 + 1 such borders.
                #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
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

                // Collect the span and range of all the intersecting ranges to lint on likely
                // incorrect range patterns. (#63987)
                let mut overlaps = vec![];
                // `borders` is the set of borders between equivalence classes: each equivalence
                // class lies between 2 borders.
                let row_borders = matrix
                    .0
                    .iter()
                    .flat_map(|row| {
                        IntRange::from_pat(tcx, param_env, row.head()).map(|r| (r, row.len()))
                    })
                    .flat_map(|(range, row_len)| {
                        let intersection = ctor_range.intersection(tcx, &range);
                        let should_lint = ctor_range.suspicious_intersection(&range);
                        if let (Some(range), 1, true) = (&intersection, row_len, should_lint) {
                            // FIXME: for now, only check for overlapping ranges on simple range
                            // patterns. Otherwise with the current logic the following is detected
                            // as overlapping:
                            //   match (10u8, true) {
                            //    (0 ..= 125, false) => {}
                            //    (126 ..= 255, false) => {}
                            //    (0 ..= 255, true) => {}
                            //  }
                            overlaps.push(range.clone());
                        }
                        intersection
                    })
                    .flat_map(range_borders);
                let ctor_borders = range_borders(ctor_range.clone());
                let mut borders: Vec<_> = row_borders.chain(ctor_borders).collect();
                borders.sort_unstable();

                lint_overlapping_patterns(tcx, hir_id, ctor_range, ty, overlaps);

                // We're going to iterate through every adjacent pair of borders, making sure that
                // each represents an interval of nonnegative length, and convert each such
                // interval into a constructor.
                split_ctors.extend(
                    borders
                        .windows(2)
                        .filter_map(|window| match (window[0], window[1]) {
                            (Border::JustBefore(n), Border::JustBefore(m)) => {
                                if n < m {
                                    Some(IntRange { range: n..=(m - 1), ty, span })
                                } else {
                                    None
                                }
                            }
                            (Border::JustBefore(n), Border::AfterMax) => {
                                Some(IntRange { range: n..=u128::MAX, ty, span })
                            }
                            (Border::AfterMax, _) => None,
                        })
                        .map(IntRange),
                );
            }
            Slice(Slice { array_len, kind: VarLen(self_prefix, self_suffix) }) => {
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
                // we can pick `L = max(max(FIXED_LEN)+1, max(PREFIX_LEN) + max(SUFFIX_LEN))`
                //
                // for example, with the above pair of patterns, all elements
                // but the first and last can be added/removed, so any
                // witness of length ≥2 (say, `[false, false, true]`) can be
                // turned to a witness from any other length ≥2.

                let mut max_prefix_len = self_prefix;
                let mut max_suffix_len = self_suffix;
                let mut max_fixed_len = 0;

                let head_ctors =
                    matrix.heads().filter_map(|pat| pat_constructor(tcx, param_env, pat));
                for ctor in head_ctors {
                    if let Slice(slice) = ctor {
                        match slice.pattern_kind() {
                            FixedLen(len) => {
                                max_fixed_len = cmp::max(max_fixed_len, len);
                            }
                            VarLen(prefix, suffix) => {
                                max_prefix_len = cmp::max(max_prefix_len, prefix);
                                max_suffix_len = cmp::max(max_suffix_len, suffix);
                            }
                        }
                    }
                }

                // For diagnostics, we keep the prefix and suffix lengths separate, so in the case
                // where `max_fixed_len + 1` is the largest, we adapt `max_prefix_len` accordingly,
                // so that `L = max_prefix_len + max_suffix_len`.
                if max_fixed_len + 1 >= max_prefix_len + max_suffix_len {
                    // The subtraction can't overflow thanks to the above check.
                    // The new `max_prefix_len` is also guaranteed to be larger than its previous
                    // value.
                    max_prefix_len = max_fixed_len + 1 - max_suffix_len;
                }

                match array_len {
                    Some(len) => {
                        let kind = if max_prefix_len + max_suffix_len < len {
                            VarLen(max_prefix_len, max_suffix_len)
                        } else {
                            FixedLen(len)
                        };
                        split_ctors.push(Slice(Slice { array_len, kind }));
                    }
                    None => {
                        // `ctor` originally covered the range `(self_prefix +
                        // self_suffix..infinity)`. We now split it into two: lengths smaller than
                        // `max_prefix_len + max_suffix_len` are treated independently as
                        // fixed-lengths slices, and lengths above are captured by a final VarLen
                        // constructor.
                        split_ctors.extend(
                            (self_prefix + self_suffix..max_prefix_len + max_suffix_len)
                                .map(|len| Slice(Slice { array_len, kind: FixedLen(len) })),
                        );
                        split_ctors.push(Slice(Slice {
                            array_len,
                            kind: VarLen(max_prefix_len, max_suffix_len),
                        }));
                    }
                }
            }
            // Any other constructor can be used unchanged.
            _ => split_ctors.push(ctor),
        }
    }

    debug!("split_grouped_constructors(..)={:#?}", split_ctors);
    split_ctors
}

fn lint_overlapping_patterns<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_id: Option<HirId>,
    ctor_range: IntRange<'tcx>,
    ty: Ty<'tcx>,
    overlaps: Vec<IntRange<'tcx>>,
) {
    if let (true, Some(hir_id)) = (!overlaps.is_empty(), hir_id) {
        tcx.struct_span_lint_hir(
            lint::builtin::OVERLAPPING_PATTERNS,
            hir_id,
            ctor_range.span,
            |lint| {
                let mut err = lint.build("multiple patterns covering the same range");
                err.span_label(ctor_range.span, "overlapping patterns");
                for int_range in overlaps {
                    // Use the real type for user display of the ranges:
                    err.span_label(
                        int_range.span,
                        &format!(
                            "this range overlaps on `{}`",
                            IntRange { range: int_range.range, ty, span: DUMMY_SP }.to_pat(tcx),
                        ),
                    );
                }
                err.emit();
            },
        );
    }
}

fn constructor_covered_by_range<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    ctor: &Constructor<'tcx>,
    pat: &Pat<'tcx>,
) -> Option<()> {
    if let Single = ctor {
        return Some(());
    }

    let (pat_from, pat_to, pat_end, ty) = match *pat.kind {
        PatKind::Constant { value } => (value, value, RangeEnd::Included, value.ty),
        PatKind::Range(PatRange { lo, hi, end }) => (lo, hi, end, lo.ty),
        _ => bug!("`constructor_covered_by_range` called with {:?}", pat),
    };
    let (ctor_from, ctor_to, ctor_end) = match *ctor {
        ConstantValue(value) => (value, value, RangeEnd::Included),
        FloatRange(from, to, ctor_end) => (from, to, ctor_end),
        _ => bug!("`constructor_covered_by_range` called with {:?}", ctor),
    };
    trace!("constructor_covered_by_range {:#?}, {:#?}, {:#?}, {}", ctor, pat_from, pat_to, ty);

    let to = compare_const_vals(tcx, ctor_to, pat_to, param_env, ty)?;
    let from = compare_const_vals(tcx, ctor_from, pat_from, param_env, ty)?;
    let intersects = (from == Ordering::Greater || from == Ordering::Equal)
        && (to == Ordering::Less || (pat_end == ctor_end && to == Ordering::Equal));
    if intersects { Some(()) } else { None }
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
///
/// This is roughly the inverse of `Constructor::apply`.
fn specialize_one_pattern<'p, 'tcx>(
    cx: &mut MatchCheckCtxt<'p, 'tcx>,
    pat: &'p Pat<'tcx>,
    constructor: &Constructor<'tcx>,
    ctor_wild_subpatterns: &Fields<'p, 'tcx>,
) -> Option<Fields<'p, 'tcx>> {
    if let NonExhaustive = constructor {
        // Only a wildcard pattern can match the special extra constructor
        if !pat.is_wildcard() {
            return None;
        }
        return Some(Fields::empty());
    }

    let result = match *pat.kind {
        PatKind::AscribeUserType { .. } => bug!(), // Handled by `expand_pattern`

        PatKind::Binding { .. } | PatKind::Wild => Some(ctor_wild_subpatterns.clone()),

        PatKind::Variant { adt_def, variant_index, ref subpatterns, .. } => {
            let variant = &adt_def.variants[variant_index];
            if constructor != &Variant(variant.def_id) {
                return None;
            }
            Some(ctor_wild_subpatterns.replace_with_fieldpats(subpatterns))
        }

        PatKind::Leaf { ref subpatterns } => {
            Some(ctor_wild_subpatterns.replace_with_fieldpats(subpatterns))
        }

        PatKind::Deref { ref subpattern } => Some(Fields::from_single_pattern(subpattern)),

        PatKind::Constant { value } if constructor.is_slice() => {
            // We extract an `Option` for the pointer because slices of zero
            // elements don't necessarily point to memory, they are usually
            // just integers. The only time they should be pointing to memory
            // is when they are subslices of nonzero slices.
            let (alloc, offset, n, ty) = match value.ty.kind() {
                ty::Array(t, n) => {
                    let n = n.eval_usize(cx.tcx, cx.param_env);
                    // Shortcut for `n == 0` where no matter what `alloc` and `offset` we produce,
                    // the result would be exactly what we early return here.
                    if n == 0 {
                        if ctor_wild_subpatterns.len() as u64 != n {
                            return None;
                        }
                        return Some(Fields::empty());
                    }
                    match value.val {
                        ty::ConstKind::Value(ConstValue::ByRef { offset, alloc, .. }) => {
                            (Cow::Borrowed(alloc), offset, n, t)
                        }
                        _ => span_bug!(pat.span, "array pattern is {:?}", value,),
                    }
                }
                ty::Slice(t) => {
                    match value.val {
                        ty::ConstKind::Value(ConstValue::Slice { data, start, end }) => {
                            let offset = Size::from_bytes(start);
                            let n = (end - start) as u64;
                            (Cow::Borrowed(data), offset, n, t)
                        }
                        ty::ConstKind::Value(ConstValue::ByRef { .. }) => {
                            // FIXME(oli-obk): implement `deref` for `ConstValue`
                            return None;
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
            if ctor_wild_subpatterns.len() as u64 != n {
                return None;
            }

            // Convert a constant slice/array pattern to a list of patterns.
            let layout = cx.tcx.layout_of(cx.param_env.and(ty)).ok()?;
            let ptr = Pointer::new(AllocId(0), offset);
            let pats = cx.pattern_arena.alloc_from_iter((0..n).filter_map(|i| {
                let ptr = ptr.offset(layout.size * i, &cx.tcx).ok()?;
                let scalar = alloc.read_scalar(&cx.tcx, ptr, layout.size).ok()?;
                let scalar = scalar.check_init().ok()?;
                let value = ty::Const::from_scalar(cx.tcx, scalar, ty);
                let pattern = Pat { ty, span: pat.span, kind: box PatKind::Constant { value } };
                Some(pattern)
            }));
            // Ensure none of the dereferences failed.
            if pats.len() as u64 != n {
                return None;
            }
            Some(Fields::from_slice_unfiltered(pats))
        }

        PatKind::Constant { .. } | PatKind::Range { .. } => {
            // If the constructor is a:
            // - Single value: add a row if the pattern contains the constructor.
            // - Range: add a row if the constructor intersects the pattern.
            if let IntRange(ctor) = constructor {
                let pat = IntRange::from_pat(cx.tcx, cx.param_env, pat)?;
                ctor.intersection(cx.tcx, &pat)?;
                // Constructor splitting should ensure that all intersections we encounter
                // are actually inclusions.
                assert!(ctor.is_subrange(&pat));
            } else {
                // Fallback for non-ranges and ranges that involve
                // floating-point numbers, which are not conveniently handled
                // by `IntRange`. For these cases, the constructor may not be a
                // range so intersection actually devolves into being covered
                // by the pattern.
                constructor_covered_by_range(cx.tcx, cx.param_env, constructor, pat)?;
            }
            Some(Fields::empty())
        }

        PatKind::Array { ref prefix, ref slice, ref suffix }
        | PatKind::Slice { ref prefix, ref slice, ref suffix } => match *constructor {
            Slice(_) => {
                // Number of subpatterns for this pattern
                let pat_len = prefix.len() + suffix.len();
                // Number of subpatterns for this constructor
                let arity = ctor_wild_subpatterns.len();

                if (slice.is_none() && arity != pat_len) || pat_len > arity {
                    return None;
                }

                // Replace the prefix and the suffix with the given patterns, leaving wildcards in
                // the middle if there was a subslice pattern `..`.
                let prefix = prefix.iter().enumerate();
                let suffix = suffix.iter().enumerate().map(|(i, p)| (arity - suffix.len() + i, p));
                Some(ctor_wild_subpatterns.replace_fields_indexed(prefix.chain(suffix)))
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
                    Ok(true) => Some(Fields::empty()),
                    Ok(false) => None,
                    Err(ErrorReported) => None,
                }
            }
            _ => span_bug!(pat.span, "unexpected ctor {:?} for slice pat", constructor),
        },

        PatKind::Or { .. } => bug!("Or-pattern should have been expanded earlier on."),
    };
    debug!(
        "specialize({:#?}, {:#?}, {:#?}) = {:#?}",
        pat, constructor, ctor_wild_subpatterns, result
    );

    result
}
