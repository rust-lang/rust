//! Note: tests specific to this file can be found in:
//!
//!   - `ui/pattern/usefulness`
//!   - `ui/or-patterns`
//!   - `ui/consts/const_in_pattern`
//!   - `ui/rfc-2008-non-exhaustive`
//!   - `ui/half-open-range-patterns`
//!   - probably many others
//!
//! I (Nadrieril) prefer to put new tests in `ui/pattern/usefulness` unless there's a specific
//! reason not to, for example if they depend on a particular feature like `or_patterns`.
//!
//! -----
//!
//! This file includes the logic for exhaustiveness and usefulness checking for
//! pattern-matching. Specifically, given a list of patterns for a type, we can
//! tell whether:
//! (a) the patterns cover every possible constructor for the type (exhaustiveness)
//! (b) each pattern is necessary (usefulness)
//!
//! The algorithm implemented here is a modified version of the one described in
//! [this paper](http://moscova.inria.fr/~maranget/papers/warn/index.html).
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
//!
//! ```
//! // x: (Option<bool>, Result<()>)
//! match x {
//!     (Some(true), _) => {}
//!     (None, Err(())) => {}
//!     (None, Err(_)) => {}
//! }
//! ```
//!
//! Here, the matrix `P` starts as:
//!
//! ```
//! [
//!     [(Some(true), _)],
//!     [(None, Err(()))],
//!     [(None, Err(_))],
//! ]
//! ```
//!
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
//!
//!      1.1. `p_1 = c(r_1, .., r_a)`, i.e. the top of the stack has constructor `c`. We
//!           push onto the stack the arguments of this constructor, and return the result:
//!              `r_1, .., r_a, p_2, .., p_n`
//!
//!      1.2. `p_1 = c'(r_1, .., r_a')` where `c ≠ c'`. We discard the current stack and
//!           return nothing.
//!
//!         1.3. `p_1 = _`. We push onto the stack as many wildcards as the constructor `c` has
//!              arguments (its arity), and return the resulting stack:
//!                 `_, .., _, p_2, .., p_n`
//!
//!         1.4. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting
//!              stack:
//!                 - `S(c, (r_1, p_2, .., p_n))`
//!                 - `S(c, (r_2, p_2, .., p_n))`
//!
//! 2. We can pop a wildcard off the top of the stack. This is called `S(_, p)`, where `p` is
//!    a pattern-stack. Note: the paper calls this `D(p)`.
//!    This is used when we know there are missing constructor cases, but there might be
//!    existing wildcard patterns, so to check the usefulness of the matrix, we have to check
//!    all its *other* components.
//!
//!    It is computed as follows. We look at the pattern `p_1` on top of the stack,
//!    and we have three cases:
//!         2.1. `p_1 = c(r_1, .., r_a)`. We discard the current stack and return nothing.
//!         2.2. `p_1 = _`. We return the rest of the stack:
//!                 p_2, .., p_n
//!         2.3. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting
//!           stack.
//!                 - `S(_, (r_1, p_2, .., p_n))`
//!                 - `S(_, (r_2, p_2, .., p_n))`
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
//!
//! For example, if `P` is:
//!
//! ```
//! [
//!     [Some(true), _],
//!     [None, 0],
//! ]
//! ```
//!
//! and `p` is `[Some(false), 0]`, then we don't care about row 2 since we know `p` only
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
//! start with a wildcard, if there are any. This is where `S(_, x)` comes in:
//! `U(P, p) := U(S(_, P), S(_, p))`
//!
//! For example, if `P` is:
//!
//! ```
//! [
//!     [_, true, _],
//!     [None, false, 1],
//! ]
//! ```
//!
//! and `p` is `[_, false, _]`, the `Some` constructor doesn't appear in `P`. So if we
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
//!
//! ```
//! [
//!     [Some(true), _],
//!     [None, false],
//! ]
//! ```
//!
//! and `p` is `[_, false]`, both `None` and `Some` constructors appear in the first
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
//!       detail in `Constructor::split`.
//!     + If some constructors are missing from the matrix, it turns out we don't need to do
//!       anything special (because we know none of the integers are actually wildcards: i.e., we
//!       can't span wildcards using ranges).

use self::Usefulness::*;
use self::WitnessPreference::*;

use super::deconstruct_pat::{Constructor, Fields, MissingConstructors};
use super::{Pat, PatKind};
use super::{PatternFoldable, PatternFolder};

use rustc_data_structures::captures::Captures;
use rustc_data_structures::sync::OnceCell;

use rustc_arena::TypedArena;
use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::iter::{FromIterator, IntoIterator};

crate struct MatchCheckCtxt<'a, 'tcx> {
    crate tcx: TyCtxt<'tcx>,
    /// The module in which the match occurs. This is necessary for
    /// checking inhabited-ness of types because whether a type is (visibly)
    /// inhabited can depend on whether it was defined in the current module or
    /// not. E.g., `struct Foo { _private: ! }` cannot be seen to be empty
    /// outside its module and should not be matchable with an empty match statement.
    crate module: DefId,
    crate param_env: ty::ParamEnv<'tcx>,
    crate pattern_arena: &'a TypedArena<Pat<'tcx>>,
}

impl<'a, 'tcx> MatchCheckCtxt<'a, 'tcx> {
    pub(super) fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        if self.tcx.features().exhaustive_patterns {
            self.tcx.is_ty_uninhabited_from(self.module, ty, self.param_env)
        } else {
            false
        }
    }

    /// Returns whether the given type is an enum from another crate declared `#[non_exhaustive]`.
    pub(super) fn is_foreign_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
        match ty.kind() {
            ty::Adt(def, ..) => {
                def.is_enum() && def.is_variant_list_non_exhaustive() && !def.did.is_local()
            }
            _ => false,
        }
    }
}

#[derive(Copy, Clone)]
pub(super) struct PatCtxt<'a, 'p, 'tcx> {
    pub(super) cx: &'a MatchCheckCtxt<'p, 'tcx>,
    /// Current state of the matrix.
    pub(super) matrix: &'a Matrix<'p, 'tcx>,
    /// Type of the current column under investigation.
    pub(super) ty: Ty<'tcx>,
    /// Span of the current pattern under investigation.
    pub(super) span: Span,
    /// Whether the current pattern is the whole pattern as found in a match arm, or if it's a
    /// subpattern.
    pub(super) is_top_level: bool,
}

crate fn expand_pattern<'tcx>(pat: Pat<'tcx>) -> Pat<'tcx> {
    LiteralExpander.fold_pattern(&pat)
}

struct LiteralExpander;

impl<'tcx> PatternFolder<'tcx> for LiteralExpander {
    fn fold_pattern(&mut self, pat: &Pat<'tcx>) -> Pat<'tcx> {
        debug!("fold_pattern {:?} {:?} {:?}", pat, pat.ty.kind(), pat.kind);
        match (pat.ty.kind(), pat.kind.as_ref()) {
            (_, PatKind::Binding { subpattern: Some(s), .. }) => s.fold_with(self),
            (_, PatKind::AscribeUserType { subpattern: s, .. }) => s.fold_with(self),
            (ty::Ref(_, t, _), PatKind::Constant { .. }) if t.is_str() => {
                // Treat string literal patterns as deref patterns to a `str` constant, i.e.
                // `&CONST`. This expands them like other const patterns. This could have been done
                // in `const_to_pat`, but that causes issues with the rest of the matching code.
                let mut new_pat = pat.super_fold_with(self);
                // Make a fake const pattern of type `str` (instead of `&str`). That the carried
                // constant value still knows it is of type `&str`.
                new_pat.ty = t;
                Pat {
                    kind: Box::new(PatKind::Deref { subpattern: new_pat }),
                    span: pat.span,
                    ty: pat.ty,
                }
            }
            _ => pat.super_fold_with(self),
        }
    }
}

impl<'tcx> Pat<'tcx> {
    pub(super) fn is_wildcard(&self) -> bool {
        matches!(*self.kind, PatKind::Binding { subpattern: None, .. } | PatKind::Wild)
    }
}

/// A row of a matrix. Rows of len 1 are very common, which is why `SmallVec[_; 2]`
/// works well.
#[derive(Debug, Clone)]
struct PatStack<'p, 'tcx> {
    pats: SmallVec<[&'p Pat<'tcx>; 2]>,
    /// Cache for the constructor of the head
    head_ctor: OnceCell<Constructor<'tcx>>,
}

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    fn from_pattern(pat: &'p Pat<'tcx>) -> Self {
        Self::from_vec(smallvec![pat])
    }

    fn from_vec(vec: SmallVec<[&'p Pat<'tcx>; 2]>) -> Self {
        PatStack { pats: vec, head_ctor: OnceCell::new() }
    }

    fn is_empty(&self) -> bool {
        self.pats.is_empty()
    }

    fn len(&self) -> usize {
        self.pats.len()
    }

    fn head(&self) -> &'p Pat<'tcx> {
        self.pats[0]
    }

    fn head_ctor<'a>(&'a self, cx: &MatchCheckCtxt<'p, 'tcx>) -> &'a Constructor<'tcx> {
        self.head_ctor.get_or_init(|| Constructor::from_pat(cx, self.head()))
    }

    fn iter(&self) -> impl Iterator<Item = &Pat<'tcx>> {
        self.pats.iter().copied()
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
                        new_patstack.pats.extend_from_slice(&self.pats[1..]);
                        new_patstack
                    })
                    .collect(),
            )
        } else {
            None
        }
    }

    /// This computes `S(self.head_ctor(), self)`. See top of the file for explanations.
    ///
    /// Structure patterns with a partial wild pattern (Foo { a: 42, .. }) have their missing
    /// fields filled with wild patterns.
    ///
    /// This is roughly the inverse of `Constructor::apply`.
    fn pop_head_constructor(&self, ctor_wild_subpatterns: &Fields<'p, 'tcx>) -> PatStack<'p, 'tcx> {
        // We pop the head pattern and push the new fields extracted from the arguments of
        // `self.head()`.
        let mut new_fields =
            ctor_wild_subpatterns.replace_with_pattern_arguments(self.head()).filtered_patterns();
        new_fields.extend_from_slice(&self.pats[1..]);
        PatStack::from_vec(new_fields)
    }
}

impl<'p, 'tcx> Default for PatStack<'p, 'tcx> {
    fn default() -> Self {
        Self::from_vec(smallvec![])
    }
}

impl<'p, 'tcx> PartialEq for PatStack<'p, 'tcx> {
    fn eq(&self, other: &Self) -> bool {
        self.pats == other.pats
    }
}

impl<'p, 'tcx> FromIterator<&'p Pat<'tcx>> for PatStack<'p, 'tcx> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'p Pat<'tcx>>,
    {
        Self::from_vec(iter.into_iter().collect())
    }
}

/// A 2D matrix.
#[derive(Clone, PartialEq)]
pub(super) struct Matrix<'p, 'tcx> {
    patterns: Vec<PatStack<'p, 'tcx>>,
}

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    fn empty() -> Self {
        Matrix { patterns: vec![] }
    }

    /// Number of columns of this matrix. `None` is the matrix is empty.
    pub(super) fn column_count(&self) -> Option<usize> {
        self.patterns.get(0).map(|r| r.len())
    }

    /// Pushes a new row to the matrix. If the row starts with an or-pattern, this expands it.
    fn push(&mut self, row: PatStack<'p, 'tcx>) {
        if let Some(rows) = row.expand_or_pat() {
            for row in rows {
                // We recursively expand the or-patterns of the new rows.
                // This is necessary as we might have `0 | (1 | 2)` or e.g., `x @ 0 | x @ (1 | 2)`.
                self.push(row)
            }
        } else {
            self.patterns.push(row);
        }
    }

    /// Iterate over the first component of each row
    fn heads<'a>(&'a self) -> impl Iterator<Item = &'a Pat<'tcx>> + Captures<'p> {
        self.patterns.iter().map(|r| r.head())
    }

    /// Iterate over the first constructor of each row.
    pub(super) fn head_ctors<'a>(
        &'a self,
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'p> {
        self.patterns.iter().map(move |r| r.head_ctor(cx))
    }

    /// Iterate over the first constructor and the corresponding span of each row.
    pub(super) fn head_ctors_and_spans<'a>(
        &'a self,
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
    ) -> impl Iterator<Item = (&'a Constructor<'tcx>, Span)> + Captures<'p> {
        self.patterns.iter().map(move |r| (r.head_ctor(cx), r.head().span))
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Matrix<'p, 'tcx> {
        self.patterns
            .iter()
            .filter(|r| ctor.is_covered_by(pcx, r.head_ctor(pcx.cx)))
            .map(|r| r.pop_head_constructor(ctor_wild_subpatterns))
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
/// ```
impl<'p, 'tcx> fmt::Debug for Matrix<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n")?;

        let Matrix { patterns: m, .. } = self;
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

/// Represents a set of `Span`s closed under the containment relation. That is, if a `Span` is
/// contained in the set then all `Span`s contained in it are also implicitly contained in the set.
/// In particular this means that when intersecting two sets, taking the intersection of some span
/// and one of its subspans returns the subspan, whereas a simple `HashSet` would have returned an
/// empty intersection.
/// It is assumed that two spans don't overlap without one being contained in the other; in other
/// words, that the inclusion structure forms a tree and not a DAG.
/// Intersection is not very efficient. It compares everything pairwise. If needed it could be made
/// faster by sorting the `Span`s and merging cleverly.
#[derive(Debug, Clone, Default)]
pub(crate) struct SpanSet {
    /// The minimal set of `Span`s required to represent the whole set. If A and B are `Span`s in
    /// the `SpanSet`, and A is a descendant of B, then only B will be in `root_spans`.
    /// Invariant: the spans are disjoint.
    root_spans: Vec<Span>,
}

impl SpanSet {
    /// Creates an empty set.
    fn new() -> Self {
        Self::default()
    }

    /// Tests whether the set is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.root_spans.is_empty()
    }

    /// Iterate over the disjoint list of spans at the roots of this set.
    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item = Span> + Captures<'a> {
        self.root_spans.iter().copied()
    }

    /// Tests whether the set contains a given Span.
    fn contains(&self, span: Span) -> bool {
        self.iter().any(|root_span| root_span.contains(span))
    }

    /// Add a span to the set if we know the span has no intersection in this set.
    fn push_nonintersecting(&mut self, new_span: Span) {
        self.root_spans.push(new_span);
    }

    fn intersection_mut(&mut self, other: &Self) {
        if self.is_empty() || other.is_empty() {
            *self = Self::new();
            return;
        }
        // Those that were in `self` but not contained in `other`
        let mut leftover = SpanSet::new();
        // We keep the elements in `self` that are also in `other`.
        self.root_spans.retain(|span| {
            let retain = other.contains(*span);
            if !retain {
                leftover.root_spans.push(*span);
            }
            retain
        });
        // We keep the elements in `other` that are also in the original `self`. You might think
        // this is not needed because `self` already contains the intersection. But those aren't
        // just sets of things. If `self = [a]`, `other = [b]` and `a` contains `b`, then `b`
        // belongs in the intersection but we didn't catch it in the filtering above. We look at
        // `leftover` instead of the full original `self` to avoid duplicates.
        for span in other.iter() {
            if leftover.contains(span) {
                self.root_spans.push(span);
            }
        }
    }
}

#[derive(Clone, Debug)]
crate enum Usefulness<'tcx> {
    /// Pontentially carries a set of sub-branches that have been found to be unreachable. Used
    /// only in the presence of or-patterns, otherwise it stays empty.
    Useful(SpanSet),
    /// Carries a list of witnesses of non-exhaustiveness.
    UsefulWithWitness(Vec<Witness<'tcx>>),
    NotUseful,
}

impl<'tcx> Usefulness<'tcx> {
    fn new_useful(preference: WitnessPreference) -> Self {
        match preference {
            ConstructWitness => UsefulWithWitness(vec![Witness(vec![])]),
            LeaveOutWitness => Useful(Default::default()),
        }
    }

    /// When trying several branches and each returns a `Usefulness`, we need to combine the
    /// results together.
    fn merge(usefulnesses: impl Iterator<Item = Self>) -> Self {
        // If we have detected some unreachable sub-branches, we only want to keep them when they
        // were unreachable in _all_ branches. Eg. in the following, the last `true` is unreachable
        // in the second branch of the first or-pattern, but not otherwise. Therefore we don't want
        // to lint that it is unreachable.
        // ```
        // match (true, true) {
        //     (true, true) => {}
        //     (false | true, false | true) => {}
        // }
        // ```
        // Here however we _do_ want to lint that the last `false` is unreachable. So we don't want
        // to intersect the spans that come directly from the or-pattern, since each branch of the
        // or-pattern brings a new disjoint pattern.
        // ```
        // match None {
        //     Some(false) => {}
        //     None | Some(true | false) => {}
        // }
        // ```

        // Is `None` when no branch was useful. Will often be `Some(Spanset::new())` because the
        // sets are only non-empty in the presence of or-patterns.
        let mut unreachables: Option<SpanSet> = None;
        // Witnesses of usefulness, if any.
        let mut witnesses = Vec::new();

        for u in usefulnesses {
            match u {
                Useful(spans) if spans.is_empty() => {
                    // Once we reach the empty set, more intersections won't change the result.
                    return Useful(SpanSet::new());
                }
                Useful(spans) => {
                    if let Some(unreachables) = &mut unreachables {
                        if !unreachables.is_empty() {
                            unreachables.intersection_mut(&spans);
                        }
                        if unreachables.is_empty() {
                            return Useful(SpanSet::new());
                        }
                    } else {
                        unreachables = Some(spans);
                    }
                }
                NotUseful => {}
                UsefulWithWitness(wits) => {
                    witnesses.extend(wits);
                }
            }
        }

        if !witnesses.is_empty() {
            UsefulWithWitness(witnesses)
        } else if let Some(unreachables) = unreachables {
            Useful(unreachables)
        } else {
            NotUseful
        }
    }

    /// After calculating the usefulness for a branch of an or-pattern, call this to make this
    /// usefulness mergeable with those from the other branches.
    fn unsplit_or_pat(self, this_span: Span, or_pat_spans: &[Span]) -> Self {
        match self {
            Useful(mut spans) => {
                // We register the spans of the other branches of this or-pattern as being
                // unreachable from this one. This ensures that intersecting together the sets of
                // spans returns what we want.
                // Until we optimize `SpanSet` however, intersecting this entails a number of
                // comparisons quadratic in the number of branches.
                for &span in or_pat_spans {
                    if span != this_span {
                        spans.push_nonintersecting(span);
                    }
                }
                Useful(spans)
            }
            x => x,
        }
    }

    /// After calculating usefulness after a specialization, call this to recontruct a usefulness
    /// that makes sense for the matrix pre-specialization. This new usefulness can then be merged
    /// with the results of specializing with the other constructors.
    fn apply_constructor<'p>(
        self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Self {
        match self {
            UsefulWithWitness(witnesses) => {
                let new_witnesses = if ctor.is_wildcard() {
                    let missing_ctors = MissingConstructors::new(pcx);
                    let new_patterns = missing_ctors.report_patterns(pcx);
                    witnesses
                        .into_iter()
                        .flat_map(|witness| {
                            new_patterns.iter().map(move |pat| {
                                let mut witness = witness.clone();
                                witness.0.push(pat.clone());
                                witness
                            })
                        })
                        .collect()
                } else {
                    witnesses
                        .into_iter()
                        .map(|witness| witness.apply_constructor(pcx, &ctor, ctor_wild_subpatterns))
                        .collect()
                };
                UsefulWithWitness(new_witnesses)
            }
            x => x,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum WitnessPreference {
    ConstructWitness,
    LeaveOutWitness,
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
///
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
    /// Asserts that the witness contains a single pattern, and returns it.
    fn single_pattern(self) -> Pat<'tcx> {
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
        pcx: PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Self {
        let pat = {
            let len = self.0.len();
            let arity = ctor_wild_subpatterns.len();
            let pats = self.0.drain((len - arity)..).rev();
            ctor_wild_subpatterns.replace_fields(pcx.cx, pats).apply(pcx, ctor)
        };

        self.0.push(pat);

        self
    }
}

/// Algorithm from <http://moscova.inria.fr/~maranget/papers/warn/index.html>.
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
fn is_useful<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &PatStack<'p, 'tcx>,
    witness_preference: WitnessPreference,
    hir_id: HirId,
    is_under_guard: bool,
    is_top_level: bool,
) -> Usefulness<'tcx> {
    let Matrix { patterns: rows, .. } = matrix;
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

    // FIXME(Nadrieril): Hack to work around type normalization issues (see #72476).
    let ty = matrix.heads().next().map(|r| r.ty).unwrap_or(v.head().ty);
    let pcx = PatCtxt { cx, matrix, ty, span: v.head().span, is_top_level };

    debug!("is_useful_expand_first_col: ty={:#?}, expanding {:#?}", pcx.ty, v.head());

    // If the first pattern is an or-pattern, expand it.
    let ret = if let Some(vs) = v.expand_or_pat() {
        let subspans: Vec<_> = vs.iter().map(|v| v.head().span).collect();
        // We expand the or pattern, trying each of its branches in turn and keeping careful track
        // of possible unreachable sub-branches.
        let mut matrix = matrix.clone();
        let usefulnesses = vs.into_iter().map(|v| {
            let v_span = v.head().span;
            let usefulness =
                is_useful(cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false);
            // If pattern has a guard don't add it to the matrix.
            if !is_under_guard {
                // We push the already-seen patterns into the matrix in order to detect redundant
                // branches like `Some(_) | Some(0)`.
                matrix.push(v);
            }
            usefulness.unsplit_or_pat(v_span, &subspans)
        });
        Usefulness::merge(usefulnesses)
    } else {
        // We split the head constructor of `v`.
        let ctors = v.head_ctor(cx).split(pcx, Some(hir_id));
        // For each constructor, we compute whether there's a value that starts with it that would
        // witness the usefulness of `v`.
        let usefulnesses = ctors.into_iter().map(|ctor| {
            // We cache the result of `Fields::wildcards` because it is used a lot.
            let ctor_wild_subpatterns = Fields::wildcards(pcx, &ctor);
            let matrix = pcx.matrix.specialize_constructor(pcx, &ctor, &ctor_wild_subpatterns);
            let v = v.pop_head_constructor(&ctor_wild_subpatterns);
            let usefulness =
                is_useful(pcx.cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false);
            usefulness.apply_constructor(pcx, &ctor, &ctor_wild_subpatterns)
        });
        Usefulness::merge(usefulnesses)
    };
    debug!("is_useful::returns({:#?}, {:#?}) = {:?}", matrix, v, ret);
    ret
}

/// The arm of a match expression.
#[derive(Clone, Copy)]
crate struct MatchArm<'p, 'tcx> {
    /// The pattern must have been lowered through `MatchVisitor::lower_pattern`.
    crate pat: &'p super::Pat<'tcx>,
    crate hir_id: HirId,
    crate has_guard: bool,
}

/// The output of checking a match for exhaustiveness and arm reachability.
crate struct UsefulnessReport<'p, 'tcx> {
    /// For each arm of the input, whether that arm is reachable after the arms above it.
    crate arm_usefulness: Vec<(MatchArm<'p, 'tcx>, Usefulness<'tcx>)>,
    /// If the match is exhaustive, this is empty. If not, this contains witnesses for the lack of
    /// exhaustiveness.
    crate non_exhaustiveness_witnesses: Vec<super::Pat<'tcx>>,
}

/// The entrypoint for the usefulness algorithm. Computes whether a match is exhaustive and which
/// of its arms are reachable.
///
/// Note: the input patterns must have been lowered through `MatchVisitor::lower_pattern`.
crate fn compute_match_usefulness<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    arms: &[MatchArm<'p, 'tcx>],
    scrut_hir_id: HirId,
    scrut_ty: Ty<'tcx>,
) -> UsefulnessReport<'p, 'tcx> {
    let mut matrix = Matrix::empty();
    let arm_usefulness: Vec<_> = arms
        .iter()
        .copied()
        .map(|arm| {
            let v = PatStack::from_pattern(arm.pat);
            let usefulness =
                is_useful(cx, &matrix, &v, LeaveOutWitness, arm.hir_id, arm.has_guard, true);
            if !arm.has_guard {
                matrix.push(v);
            }
            (arm, usefulness)
        })
        .collect();

    let wild_pattern = cx.pattern_arena.alloc(super::Pat::wildcard_from_ty(scrut_ty));
    let v = PatStack::from_pattern(wild_pattern);
    let usefulness = is_useful(cx, &matrix, &v, ConstructWitness, scrut_hir_id, false, true);
    let non_exhaustiveness_witnesses = match usefulness {
        NotUseful => vec![], // Wildcard pattern isn't useful, so the match is exhaustive.
        UsefulWithWitness(pats) => {
            if pats.is_empty() {
                bug!("Exhaustiveness check returned no witnesses")
            } else {
                pats.into_iter().map(|w| w.single_pattern()).collect()
            }
        }
        Useful(_) => bug!(),
    };
    UsefulnessReport { arm_usefulness, non_exhaustiveness_witnesses }
}
