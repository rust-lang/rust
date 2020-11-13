//! Note: tests specific to this file can be found in:
//!     - ui/pattern/usefulness
//!     - ui/or-patterns
//!     - ui/consts/const_in_pattern
//!     - ui/rfc-2008-non-exhaustive
//!     - probably many others
//! I (Nadrieril) prefer to put new tests in `ui/pattern/usefulness` unless there's a specific
//! reason not to, for example if they depend on a particular feature like or_patterns.
//!
//! This file includes the logic for exhaustiveness and usefulness checking for
//! pattern-matching. Specifically, given a list of patterns for a type, we can
//! tell whether:
//! (a) the patterns cover every possible constructor for the type (exhaustiveness)
//! (b) each pattern is necessary (usefulness)
//!
//! The algorithm implemented here is a modified version of the one described in:
//! <http://moscova.inria.fr/~maranget/papers/warn/index.html>
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
//!                 S(_, (r_1, p_2, .., p_n))
//!                 S(_, (r_2, p_2, .., p_n))
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
//!
//! ```
//! [
//!     [Some(true), _],
//!     [None, false],
//! ]
//! ```
//!
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
//!       detail in `Constructor::split`.
//!     + If some constructors are missing from the matrix, it turns out we don't need to do
//!       anything special (because we know none of the integers are actually wildcards: i.e., we
//!       can't span wildcards using ranges).
use self::Constructor::*;
use self::SliceKind::*;
use self::Usefulness::*;
use self::WitnessPreference::*;

use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::OnceCell;
use rustc_index::vec::Idx;

use super::{compare_const_vals, PatternFoldable, PatternFolder};
use super::{FieldPat, Pat, PatKind, PatRange};

use rustc_arena::TypedArena;
use rustc_attr::{SignedInt, UnsignedInt};
use rustc_hir::def_id::DefId;
use rustc_hir::{HirId, RangeEnd};
use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::mir::Field;
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_session::lint;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{Integer, Size, VariantIdx};

use smallvec::{smallvec, SmallVec};
use std::cmp::{self, max, min, Ordering};
use std::fmt;
use std::iter::{FromIterator, IntoIterator};
use std::ops::RangeInclusive;

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
        self.head_ctor.get_or_init(|| pat_constructor(cx, self.head()))
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
        let new_fields = ctor_wild_subpatterns.replace_with_pattern_arguments(self.head());
        new_fields.push_on_patstack(&self.pats[1..])
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
struct Matrix<'p, 'tcx> {
    patterns: Vec<PatStack<'p, 'tcx>>,
}

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    fn empty() -> Self {
        Matrix { patterns: vec![] }
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

    /// Iterate over the first constructor of each row
    fn head_ctors<'a>(
        &'a self,
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'a> + Captures<'p> {
        self.patterns.iter().map(move |r| r.head_ctor(cx))
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
    fn is_foreign_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
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
    fn new(array_len: Option<u64>, kind: SliceKind) -> Self {
        let kind = match (array_len, kind) {
            // If the middle `..` is empty, we effectively have a fixed-length pattern.
            (Some(len), VarLen(prefix, suffix)) if prefix + suffix >= len => FixedLen(len),
            _ => kind,
        };
        Slice { array_len, kind }
    }

    fn arity(self) -> u64 {
        self.kind.arity()
    }

    /// The exhaustiveness-checking paper does not include any details on
    /// checking variable-length slice patterns. However, they may be
    /// matched by an infinite collection of fixed-length array patterns.
    ///
    /// Checking the infinite set directly would take an infinite amount
    /// of time. However, it turns out that for each finite set of
    /// patterns `P`, all sufficiently large array lengths are equivalent:
    ///
    /// Each slice `s` with a "sufficiently-large" length `l ≥ L` that applies
    /// to exactly the subset `Pₜ` of `P` can be transformed to a slice
    /// `sₘ` for each sufficiently-large length `m` that applies to exactly
    /// the same subset of `P`.
    ///
    /// Because of that, each witness for reachability-checking of one
    /// of the sufficiently-large lengths can be transformed to an
    /// equally-valid witness of any other length, so we only have
    /// to check slices of the "minimal sufficiently-large length"
    /// and less.
    ///
    /// Note that the fact that there is a *single* `sₘ` for each `m`
    /// not depending on the specific pattern in `P` is important: if
    /// you look at the pair of patterns
    ///     `[true, ..]`
    ///     `[.., false]`
    /// Then any slice of length ≥1 that matches one of these two
    /// patterns can be trivially turned to a slice of any
    /// other length ≥1 that matches them and vice-versa,
    /// but the slice of length 2 `[false, true]` that matches neither
    /// of these patterns can't be turned to a slice from length 1 that
    /// matches neither of these patterns, so we have to consider
    /// slices from length 2 there.
    ///
    /// Now, to see that that length exists and find it, observe that slice
    /// patterns are either "fixed-length" patterns (`[_, _, _]`) or
    /// "variable-length" patterns (`[_, .., _]`).
    ///
    /// For fixed-length patterns, all slices with lengths *longer* than
    /// the pattern's length have the same outcome (of not matching), so
    /// as long as `L` is greater than the pattern's length we can pick
    /// any `sₘ` from that length and get the same result.
    ///
    /// For variable-length patterns, the situation is more complicated,
    /// because as seen above the precise value of `sₘ` matters.
    ///
    /// However, for each variable-length pattern `p` with a prefix of length
    /// `plₚ` and suffix of length `slₚ`, only the first `plₚ` and the last
    /// `slₚ` elements are examined.
    ///
    /// Therefore, as long as `L` is positive (to avoid concerns about empty
    /// types), all elements after the maximum prefix length and before
    /// the maximum suffix length are not examined by any variable-length
    /// pattern, and therefore can be added/removed without affecting
    /// them - creating equivalent patterns from any sufficiently-large
    /// length.
    ///
    /// Of course, if fixed-length patterns exist, we must be sure
    /// that our length is large enough to miss them all, so
    /// we can pick `L = max(max(FIXED_LEN)+1, max(PREFIX_LEN) + max(SUFFIX_LEN))`
    ///
    /// for example, with the above pair of patterns, all elements
    /// but the first and last can be added/removed, so any
    /// witness of length ≥2 (say, `[false, false, true]`) can be
    /// turned to a witness from any other length ≥2.
    fn split<'p, 'tcx>(self, pcx: PatCtxt<'_, 'p, 'tcx>) -> SmallVec<[Constructor<'tcx>; 1]> {
        let (self_prefix, self_suffix) = match self.kind {
            VarLen(self_prefix, self_suffix) => (self_prefix, self_suffix),
            _ => return smallvec![Slice(self)],
        };

        let head_ctors = pcx.matrix.head_ctors(pcx.cx).filter(|c| !c.is_wildcard());

        let mut max_prefix_len = self_prefix;
        let mut max_suffix_len = self_suffix;
        let mut max_fixed_len = 0;

        for ctor in head_ctors {
            if let Slice(slice) = ctor {
                match slice.kind {
                    FixedLen(len) => {
                        max_fixed_len = cmp::max(max_fixed_len, len);
                    }
                    VarLen(prefix, suffix) => {
                        max_prefix_len = cmp::max(max_prefix_len, prefix);
                        max_suffix_len = cmp::max(max_suffix_len, suffix);
                    }
                }
            } else {
                bug!("unexpected ctor for slice type: {:?}", ctor);
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

        let final_slice = VarLen(max_prefix_len, max_suffix_len);
        let final_slice = Slice::new(self.array_len, final_slice);
        match self.array_len {
            Some(_) => smallvec![Slice(final_slice)],
            None => {
                // `self` originally covered the range `(self.arity()..infinity)`. We split that
                // range into two: lengths smaller than `final_slice.arity()` are treated
                // independently as fixed-lengths slices, and lengths above are captured by
                // `final_slice`.
                let smaller_lengths = (self.arity()..final_slice.arity()).map(FixedLen);
                smaller_lengths
                    .map(|kind| Slice::new(self.array_len, kind))
                    .chain(Some(final_slice))
                    .map(Slice)
                    .collect()
            }
        }
    }

    /// See `Constructor::is_covered_by`
    fn is_covered_by(self, other: Self) -> bool {
        other.kind.covers_length(self.arity())
    }
}

/// A value can be decomposed into a constructor applied to some fields. This struct represents
/// the constructor. See also `Fields`.
///
/// `pat_constructor` retrieves the constructor corresponding to a pattern.
/// `specialize_constructor` returns the list of fields corresponding to a pattern, given a
/// constructor. `Constructor::apply` reconstructs the pattern from a pair of `Constructor` and
/// `Fields`.
#[derive(Clone, Debug, PartialEq)]
enum Constructor<'tcx> {
    /// The constructor for patterns that have a single constructor, like tuples, struct patterns
    /// and fixed-length arrays.
    Single,
    /// Enum variants.
    Variant(DefId),
    /// Ranges of integer literal values (`2`, `2..=5` or `2..5`).
    IntRange(IntRange<'tcx>),
    /// Ranges of floating-point literal values (`2.0..=5.2`).
    FloatRange(&'tcx ty::Const<'tcx>, &'tcx ty::Const<'tcx>, RangeEnd),
    /// String literals. Strings are not quite the same as `&[u8]` so we treat them separately.
    Str(&'tcx ty::Const<'tcx>),
    /// Array and slice patterns.
    Slice(Slice),
    /// Constants that must not be matched structurally. They are treated as black
    /// boxes for the purposes of exhaustiveness: we must not inspect them, and they
    /// don't count towards making a match exhaustive.
    Opaque,
    /// Fake extra constructor for enums that aren't allowed to be matched exhaustively. Also used
    /// for those types for which we cannot list constructors explicitly, like `f64` and `str`.
    NonExhaustive,
    /// Wildcard pattern.
    Wildcard,
}

impl<'tcx> Constructor<'tcx> {
    fn is_wildcard(&self) -> bool {
        matches!(self, Wildcard)
    }

    fn as_int_range(&self) -> Option<&IntRange<'tcx>> {
        match self {
            IntRange(range) => Some(range),
            _ => None,
        }
    }

    fn as_slice(&self) -> Option<Slice> {
        match self {
            Slice(slice) => Some(*slice),
            _ => None,
        }
    }

    fn variant_index_for_adt(&self, adt: &'tcx ty::AdtDef) -> VariantIdx {
        match *self {
            Variant(id) => adt.variant_index_with_id(id),
            Single => {
                assert!(!adt.is_enum());
                VariantIdx::new(0)
            }
            _ => bug!("bad constructor {:?} for adt {:?}", self, adt),
        }
    }

    /// Some constructors (namely `Wildcard`, `IntRange` and `Slice`) actually stand for a set of actual
    /// constructors (like variants, integers or fixed-sized slices). When specializing for these
    /// constructors, we want to be specialising for the actual underlying constructors.
    /// Naively, we would simply return the list of constructors they correspond to. We instead are
    /// more clever: if there are constructors that we know will behave the same wrt the current
    /// matrix, we keep them grouped. For example, all slices of a sufficiently large length
    /// will either be all useful or all non-useful with a given matrix.
    ///
    /// See the branches for details on how the splitting is done.
    ///
    /// This function may discard some irrelevant constructors if this preserves behavior and
    /// diagnostics. Eg. for the `_` case, we ignore the constructors already present in the
    /// matrix, unless all of them are.
    ///
    /// `hir_id` is `None` when we're evaluating the wildcard pattern. In that case we do not want
    /// to lint for overlapping ranges.
    fn split<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>, hir_id: Option<HirId>) -> SmallVec<[Self; 1]> {
        debug!("Constructor::split({:#?}, {:#?})", self, pcx.matrix);

        match self {
            Wildcard => Constructor::split_wildcard(pcx),
            // Fast-track if the range is trivial. In particular, we don't do the overlapping
            // ranges check.
            IntRange(ctor_range)
                if ctor_range.treat_exhaustively(pcx.cx.tcx) && !ctor_range.is_singleton() =>
            {
                ctor_range.split(pcx, hir_id)
            }
            Slice(slice @ Slice { kind: VarLen(..), .. }) => slice.split(pcx),
            // Any other constructor can be used unchanged.
            _ => smallvec![self.clone()],
        }
    }

    /// For wildcards, there are two groups of constructors: there are the constructors actually
    /// present in the matrix (`head_ctors`), and the constructors not present (`missing_ctors`).
    /// Two constructors that are not in the matrix will either both be caught (by a wildcard), or
    /// both not be caught. Therefore we can keep the missing constructors grouped together.
    fn split_wildcard<'p>(pcx: PatCtxt<'_, 'p, 'tcx>) -> SmallVec<[Self; 1]> {
        // Missing constructors are those that are not matched by any non-wildcard patterns in the
        // current column. We only fully construct them on-demand, because they're rarely used and
        // can be big.
        let missing_ctors = MissingConstructors::new(pcx);
        if missing_ctors.is_empty(pcx) {
            // All the constructors are present in the matrix, so we just go through them all.
            // We must also split them first.
            missing_ctors.all_ctors
        } else {
            // Some constructors are missing, thus we can specialize with the wildcard constructor,
            // which will stand for those constructors that are missing, and behaves like any of
            // them.
            smallvec![Wildcard]
        }
    }

    /// Returns whether `self` is covered by `other`, i.e. whether `self` is a subset of `other`.
    /// For the simple cases, this is simply checking for equality. For the "grouped" constructors,
    /// this checks for inclusion.
    fn is_covered_by<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>, other: &Self) -> bool {
        // This must be kept in sync with `is_covered_by_any`.
        match (self, other) {
            // Wildcards cover anything
            (_, Wildcard) => true,
            // Wildcards are only covered by wildcards
            (Wildcard, _) => false,

            (Single, Single) => true,
            (Variant(self_id), Variant(other_id)) => self_id == other_id,

            (IntRange(self_range), IntRange(other_range)) => {
                self_range.is_covered_by(pcx, other_range)
            }
            (
                FloatRange(self_from, self_to, self_end),
                FloatRange(other_from, other_to, other_end),
            ) => {
                match (
                    compare_const_vals(pcx.cx.tcx, self_to, other_to, pcx.cx.param_env, pcx.ty),
                    compare_const_vals(pcx.cx.tcx, self_from, other_from, pcx.cx.param_env, pcx.ty),
                ) {
                    (Some(to), Some(from)) => {
                        (from == Ordering::Greater || from == Ordering::Equal)
                            && (to == Ordering::Less
                                || (other_end == self_end && to == Ordering::Equal))
                    }
                    _ => false,
                }
            }
            (Str(self_val), Str(other_val)) => {
                // FIXME: there's probably a more direct way of comparing for equality
                match compare_const_vals(pcx.cx.tcx, self_val, other_val, pcx.cx.param_env, pcx.ty)
                {
                    Some(comparison) => comparison == Ordering::Equal,
                    None => false,
                }
            }
            (Slice(self_slice), Slice(other_slice)) => self_slice.is_covered_by(*other_slice),

            // We are trying to inspect an opaque constant. Thus we skip the row.
            (Opaque, _) | (_, Opaque) => false,
            // Only a wildcard pattern can match the special extra constructor.
            (NonExhaustive, _) => false,

            _ => span_bug!(
                pcx.span,
                "trying to compare incompatible constructors {:?} and {:?}",
                self,
                other
            ),
        }
    }

    /// Faster version of `is_covered_by` when applied to many constructors. `used_ctors` is
    /// assumed to be built from `matrix.head_ctors()` with wildcards filtered out, and `self` is
    /// assumed to have been split from a wildcard.
    fn is_covered_by_any<'p>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        used_ctors: &[Constructor<'tcx>],
    ) -> bool {
        if used_ctors.is_empty() {
            return false;
        }

        // This must be kept in sync with `is_covered_by`.
        match self {
            // If `self` is `Single`, `used_ctors` cannot contain anything else than `Single`s.
            Single => !used_ctors.is_empty(),
            Variant(_) => used_ctors.iter().any(|c| c == self),
            IntRange(range) => used_ctors
                .iter()
                .filter_map(|c| c.as_int_range())
                .any(|other| range.is_covered_by(pcx, other)),
            Slice(slice) => used_ctors
                .iter()
                .filter_map(|c| c.as_slice())
                .any(|other| slice.is_covered_by(other)),
            // This constructor is never covered by anything else
            NonExhaustive => false,
            Str(..) | FloatRange(..) | Opaque | Wildcard => {
                bug!("found unexpected ctor in all_ctors: {:?}", self)
            }
        }
    }

    /// Apply a constructor to a list of patterns, yielding a new pattern. `pats`
    /// must have as many elements as this constructor's arity.
    ///
    /// This is roughly the inverse of `specialize_constructor`.
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
    fn apply<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>, fields: Fields<'p, 'tcx>) -> Pat<'tcx> {
        let mut subpatterns = fields.all_patterns();

        let pat = match self {
            Single | Variant(_) => match pcx.ty.kind() {
                ty::Adt(..) | ty::Tuple(..) => {
                    let subpatterns = subpatterns
                        .enumerate()
                        .map(|(i, p)| FieldPat { field: Field::new(i), pattern: p })
                        .collect();

                    if let ty::Adt(adt, substs) = pcx.ty.kind() {
                        if adt.is_enum() {
                            PatKind::Variant {
                                adt_def: adt,
                                substs,
                                variant_index: self.variant_index_for_adt(adt),
                                subpatterns,
                            }
                        } else {
                            PatKind::Leaf { subpatterns }
                        }
                    } else {
                        PatKind::Leaf { subpatterns }
                    }
                }
                // Note: given the expansion of `&str` patterns done in `expand_pattern`, we should
                // be careful to reconstruct the correct constant pattern here. However a string
                // literal pattern will never be reported as a non-exhaustiveness witness, so we
                // can ignore this issue.
                ty::Ref(..) => PatKind::Deref { subpattern: subpatterns.next().unwrap() },
                ty::Slice(_) | ty::Array(..) => bug!("bad slice pattern {:?} {:?}", self, pcx.ty),
                _ => PatKind::Wild,
            },
            Slice(slice) => match slice.kind {
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
                    let wild = Pat::wildcard_from_ty(pcx.ty);
                    PatKind::Slice { prefix, slice: Some(wild), suffix }
                }
            },
            &Str(value) => PatKind::Constant { value },
            &FloatRange(lo, hi, end) => PatKind::Range(PatRange { lo, hi, end }),
            IntRange(range) => return range.to_pat(pcx.cx.tcx),
            NonExhaustive => PatKind::Wild,
            Opaque => bug!("we should not try to apply an opaque constructor"),
            Wildcard => bug!(
                "trying to apply a wildcard constructor; this should have been done in `apply_constructors`"
            ),
        };

        Pat { ty: pcx.ty, span: DUMMY_SP, kind: Box::new(pat) }
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
    fn wildcards(pcx: PatCtxt<'_, 'p, 'tcx>, constructor: &Constructor<'tcx>) -> Self {
        let ty = pcx.ty;
        let cx = pcx.cx;
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
                        let variant = &adt.variants[constructor.variant_index_for_adt(adt)];
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
                _ => bug!("Unexpected type for `Single` constructor: {:?}", ty),
            },
            Slice(slice) => match *ty.kind() {
                ty::Slice(ty) | ty::Array(ty, _) => {
                    let arity = slice.arity();
                    Fields::wildcards_from_tys(cx, (0..arity).map(|_| ty))
                }
                _ => bug!("bad slice pattern {:?} {:?}", constructor, ty),
            },
            Str(..) | FloatRange(..) | IntRange(..) | NonExhaustive | Opaque | Wildcard => {
                Fields::empty()
            }
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

    /// Replaces contained fields with the arguments of the given pattern. Only use on a pattern
    /// that is compatible with the constructor used to build `self`.
    /// This is meant to be used on the result of `Fields::wildcards()`. The idea is that
    /// `wildcards` constructs a list of fields where all entries are wildcards, and the pattern
    /// provided to this function fills some of the fields with non-wildcards.
    /// In the following example `Fields::wildcards` would return `[_, _, _, _]`. If we call
    /// `replace_with_pattern_arguments` on it with the pattern, the result will be `[Some(0), _,
    /// _, _]`.
    /// ```rust
    /// let x: [Option<u8>; 4] = foo();
    /// match x {
    ///     [Some(0), ..] => {}
    /// }
    /// ```
    /// This is guaranteed to preserve the number of patterns in `self`.
    fn replace_with_pattern_arguments(&self, pat: &'p Pat<'tcx>) -> Self {
        match pat.kind.as_ref() {
            PatKind::Deref { subpattern } => {
                assert_eq!(self.len(), 1);
                Fields::from_single_pattern(subpattern)
            }
            PatKind::Leaf { subpatterns } | PatKind::Variant { subpatterns, .. } => {
                self.replace_with_fieldpats(subpatterns)
            }
            PatKind::Array { prefix, suffix, .. } | PatKind::Slice { prefix, suffix, .. } => {
                // Number of subpatterns for the constructor
                let ctor_arity = self.len();

                // Replace the prefix and the suffix with the given patterns, leaving wildcards in
                // the middle if there was a subslice pattern `..`.
                let prefix = prefix.iter().enumerate();
                let suffix =
                    suffix.iter().enumerate().map(|(i, p)| (ctor_arity - suffix.len() + i, p));
                self.replace_fields_indexed(prefix.chain(suffix))
            }
            _ => self.clone(),
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
    /// Carries, for each column in the matrix, a set of sub-branches that have been found to be
    /// unreachable. Used only in the presence of or-patterns, otherwise it stays empty.
    Useful(Vec<FxHashSet<Span>>),
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
        !matches!(*self, NotUseful)
    }

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
            Useful(mut unreachables) => {
                if !unreachables.is_empty() {
                    // When we apply a constructor, there are `arity` columns of the matrix that
                    // corresponded to its arguments. All the unreachables found in these columns
                    // will, after `apply`, come from the first column. So we take the union of all
                    // the corresponding sets and put them in the first column.
                    // Note that `arity` may be 0, in which case we just push a new empty set.
                    let len = unreachables.len();
                    let arity = ctor_wild_subpatterns.len();
                    let mut unioned = FxHashSet::default();
                    for set in unreachables.drain((len - arity)..) {
                        unioned.extend(set)
                    }
                    unreachables.push(unioned);
                }
                Useful(unreachables)
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

#[derive(Copy, Clone)]
struct PatCtxt<'a, 'p, 'tcx> {
    cx: &'a MatchCheckCtxt<'p, 'tcx>,
    /// Current state of the matrix.
    matrix: &'a Matrix<'p, 'tcx>,
    /// Type of the current column under investigation.
    ty: Ty<'tcx>,
    /// Span of the current pattern under investigation.
    span: Span,
    /// Whether the current pattern is the whole pattern as found in a match arm, or if it's a
    /// subpattern.
    is_top_level: bool,
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
            let fields = ctor_wild_subpatterns.replace_fields(pcx.cx, pats);
            ctor.apply(pcx, fields)
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
fn all_constructors<'p, 'tcx>(pcx: PatCtxt<'_, 'p, 'tcx>) -> Vec<Constructor<'tcx>> {
    debug!("all_constructors({:?})", pcx.ty);
    let cx = pcx.cx;
    let make_range = |start, end| {
        IntRange(
            // `unwrap()` is ok because we know the type is an integer.
            IntRange::from_range(cx.tcx, start, end, pcx.ty, &RangeEnd::Included, pcx.span)
                .unwrap(),
        )
    };
    match pcx.ty.kind() {
        ty::Bool => vec![make_range(0, 1)],
        ty::Array(sub_ty, len) if len.try_eval_usize(cx.tcx, cx.param_env).is_some() => {
            let len = len.eval_usize(cx.tcx, cx.param_env);
            if len != 0 && cx.is_uninhabited(sub_ty) {
                vec![]
            } else {
                vec![Slice(Slice::new(Some(len), VarLen(0, 0)))]
            }
        }
        // Treat arrays of a constant but unknown length like slices.
        ty::Array(sub_ty, _) | ty::Slice(sub_ty) => {
            let kind = if cx.is_uninhabited(sub_ty) { FixedLen(0) } else { VarLen(0, 0) };
            vec![Slice(Slice::new(None, kind))]
        }
        ty::Adt(def, substs) if def.is_enum() => {
            // If the enum is declared as `#[non_exhaustive]`, we treat it as if it had an
            // additional "unknown" constructor.
            // There is no point in enumerating all possible variants, because the user can't
            // actually match against them all themselves. So we always return only the fictitious
            // constructor.
            // E.g., in an example like:
            //
            // ```
            //     let err: io::ErrorKind = ...;
            //     match err {
            //         io::ErrorKind::NotFound => {},
            //     }
            // ```
            //
            // we don't want to show every possible IO error, but instead have only `_` as the
            // witness.
            let is_declared_nonexhaustive = cx.is_foreign_non_exhaustive_enum(pcx.ty);

            // If `exhaustive_patterns` is disabled and our scrutinee is an empty enum, we treat it
            // as though it had an "unknown" constructor to avoid exposing its emptiness. The
            // exception is if the pattern is at the top level, because we want empty matches to be
            // considered exhaustive.
            let is_secretly_empty = def.variants.is_empty()
                && !cx.tcx.features().exhaustive_patterns
                && !pcx.is_top_level;

            if is_secretly_empty || is_declared_nonexhaustive {
                vec![NonExhaustive]
            } else if cx.tcx.features().exhaustive_patterns {
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
            }
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
        &ty::Int(ity) => {
            let bits = Integer::from_attr(&cx.tcx, SignedInt(ity)).size().bits() as u128;
            let min = 1u128 << (bits - 1);
            let max = min - 1;
            vec![make_range(min, max)]
        }
        &ty::Uint(uty) => {
            let size = Integer::from_attr(&cx.tcx, UnsignedInt(uty)).size();
            let max = size.truncate(u128::MAX);
            vec![make_range(0, max)]
        }
        // If `exhaustive_patterns` is disabled and our scrutinee is the never type, we cannot
        // expose its emptiness. The exception is if the pattern is at the top level, because we
        // want empty matches to be considered exhaustive.
        ty::Never if !cx.tcx.features().exhaustive_patterns && !pcx.is_top_level => {
            vec![NonExhaustive]
        }
        ty::Never => vec![],
        _ if cx.is_uninhabited(pcx.ty) => vec![],
        ty::Adt(..) | ty::Tuple(..) | ty::Ref(..) => vec![Single],
        // This type is one for which we cannot list constructors, like `str` or `f64`.
        _ => vec![NonExhaustive],
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
        matches!(ty.kind(), ty::Char | ty::Int(_) | ty::Uint(_) | ty::Bool)
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
            ty::Bool => Some((Size::from_bytes(1), 0)),
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

    /// For exhaustive integer matching, some constructors are grouped within other constructors
    /// (namely integer typed values are grouped within ranges). However, when specialising these
    /// constructors, we want to be specialising for the underlying constructors (the integers), not
    /// the groups (the ranges). Thus we need to split the groups up. Splitting them up naïvely would
    /// mean creating a separate constructor for every single value in the range, which is clearly
    /// impractical. However, observe that for some ranges of integers, the specialisation will be
    /// identical across all values in that range (i.e., there are equivalence classes of ranges of
    /// constructors based on their `U(S(c, P), S(c, p))` outcome). These classes are grouped by
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
    fn split<'p>(
        &self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        hir_id: Option<HirId>,
    ) -> SmallVec<[Constructor<'tcx>; 1]> {
        let ty = pcx.ty;

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
        let row_len = pcx.matrix.patterns.get(0).map(|r| r.len()).unwrap_or(0);
        // `borders` is the set of borders between equivalence classes: each equivalence
        // class lies between 2 borders.
        let row_borders = pcx
            .matrix
            .head_ctors(pcx.cx)
            .filter_map(|ctor| ctor.as_int_range())
            .filter_map(|range| {
                let intersection = self.intersection(pcx.cx.tcx, &range);
                let should_lint = self.suspicious_intersection(&range);
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
        let self_borders = range_borders(self.clone());
        let mut borders: Vec<_> = row_borders.chain(self_borders).collect();
        borders.sort_unstable();

        self.lint_overlapping_patterns(pcx.cx.tcx, hir_id, ty, overlaps);

        // We're going to iterate through every adjacent pair of borders, making sure that
        // each represents an interval of nonnegative length, and convert each such
        // interval into a constructor.
        borders
            .array_windows()
            .filter_map(|&pair| match pair {
                [Border::JustBefore(n), Border::JustBefore(m)] => {
                    if n < m {
                        Some(n..=(m - 1))
                    } else {
                        None
                    }
                }
                [Border::JustBefore(n), Border::AfterMax] => Some(n..=u128::MAX),
                [Border::AfterMax, _] => None,
            })
            .map(|range| IntRange { range, ty, span: pcx.span })
            .map(IntRange)
            .collect()
    }

    fn lint_overlapping_patterns(
        &self,
        tcx: TyCtxt<'tcx>,
        hir_id: Option<HirId>,
        ty: Ty<'tcx>,
        overlaps: Vec<IntRange<'tcx>>,
    ) {
        if let (true, Some(hir_id)) = (!overlaps.is_empty(), hir_id) {
            tcx.struct_span_lint_hir(
                lint::builtin::OVERLAPPING_PATTERNS,
                hir_id,
                self.span,
                |lint| {
                    let mut err = lint.build("multiple patterns covering the same range");
                    err.span_label(self.span, "overlapping patterns");
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

    /// See `Constructor::is_covered_by`
    fn is_covered_by<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>, other: &Self) -> bool {
        if self.intersection(pcx.cx.tcx, other).is_some() {
            // Constructor splitting should ensure that all intersections we encounter are actually
            // inclusions.
            assert!(self.is_subrange(other));
            true
        } else {
            false
        }
    }
}

/// Ignore spans when comparing, they don't carry semantic information as they are only for lints.
impl<'tcx> std::cmp::PartialEq for IntRange<'tcx> {
    fn eq(&self, other: &Self) -> bool {
        self.range == other.range && self.ty == other.ty
    }
}

// A struct to compute a set of constructors equivalent to `all_ctors \ used_ctors`.
#[derive(Debug)]
struct MissingConstructors<'tcx> {
    all_ctors: SmallVec<[Constructor<'tcx>; 1]>,
    used_ctors: Vec<Constructor<'tcx>>,
}

impl<'tcx> MissingConstructors<'tcx> {
    fn new<'p>(pcx: PatCtxt<'_, 'p, 'tcx>) -> Self {
        let used_ctors: Vec<Constructor<'_>> =
            pcx.matrix.head_ctors(pcx.cx).cloned().filter(|c| !c.is_wildcard()).collect();
        // Since `all_ctors` never contains wildcards, this won't recurse further.
        let all_ctors =
            all_constructors(pcx).into_iter().flat_map(|ctor| ctor.split(pcx, None)).collect();

        MissingConstructors { all_ctors, used_ctors }
    }

    fn is_empty<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>) -> bool {
        self.iter(pcx).next().is_none()
    }

    /// Iterate over all_ctors \ used_ctors
    fn iter<'a, 'p>(
        &'a self,
        pcx: PatCtxt<'a, 'p, 'tcx>,
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'p> {
        self.all_ctors.iter().filter(move |ctor| !ctor.is_covered_by_any(pcx, &self.used_ctors))
    }

    /// List the patterns corresponding to the missing constructors. In some cases, instead of
    /// listing all constructors of a given type, we prefer to simply report a wildcard.
    fn report_patterns<'p>(&self, pcx: PatCtxt<'_, 'p, 'tcx>) -> SmallVec<[Pat<'tcx>; 1]> {
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
        //
        // ```
        //     let x: (Direction, Direction, bool) = ...;
        //     let (_, _, false) = x;
        // ```
        //
        // we don't want to show all 16 possible witnesses
        // `(<direction-1>, <direction-2>, true)` - we are
        // satisfied with `(_, _, true)`. In this case,
        // `used_ctors` is empty.
        // The exception is: if we are at the top-level, for example in an empty match, we
        // sometimes prefer reporting the list of constructors instead of just `_`.
        let report_when_all_missing = pcx.is_top_level && !IntRange::is_integral(pcx.ty);
        if self.used_ctors.is_empty() && !report_when_all_missing {
            // All constructors are unused. Report only a wildcard
            // rather than each individual constructor.
            smallvec![Pat::wildcard_from_ty(pcx.ty)]
        } else {
            // Construct for each missing constructor a "wild" version of this
            // constructor, that matches everything that can be built with
            // it. For example, if `ctor` is a `Constructor::Variant` for
            // `Option::Some`, we get the pattern `Some(_)`.
            self.iter(pcx)
                .map(|missing_ctor| {
                    let fields = Fields::wildcards(pcx, &missing_ctor);
                    missing_ctor.apply(pcx, fields)
                })
                .collect()
        }
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

    // If the first pattern is an or-pattern, expand it.
    if let Some(vs) = v.expand_or_pat() {
        // We expand the or pattern, trying each of its branches in turn and keeping careful track
        // of possible unreachable sub-branches.
        //
        // If two branches have detected some unreachable sub-branches, we need to be careful. If
        // they were detected in columns that are not the current one, we want to keep only the
        // sub-branches that were unreachable in _all_ branches. Eg. in the following, the last
        // `true` is unreachable in the second branch of the first or-pattern, but not otherwise.
        // Therefore we don't want to lint that it is unreachable.
        //
        // ```
        // match (true, true) {
        //     (true, true) => {}
        //     (false | true, false | true) => {}
        // }
        // ```
        // If however the sub-branches come from the current column, they come from the inside of
        // the current or-pattern, and we want to keep them all. Eg. in the following, we _do_ want
        // to lint that the last `false` is unreachable.
        // ```
        // match None {
        //     Some(false) => {}
        //     None | Some(true | false) => {}
        // }
        // ```

        let mut matrix = matrix.clone();
        // We keep track of sub-branches separately depending on whether they come from this column
        // or from others.
        let mut unreachables_this_column: FxHashSet<Span> = FxHashSet::default();
        let mut unreachables_other_columns: Vec<FxHashSet<Span>> = Vec::default();
        // Whether at least one branch is reachable.
        let mut any_is_useful = false;

        for v in vs {
            let res = is_useful(cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false);
            match res {
                Useful(unreachables) => {
                    if let Some((this_column, other_columns)) = unreachables.split_last() {
                        // We keep the union of unreachables found in the first column.
                        unreachables_this_column.extend(this_column);
                        // We keep the intersection of unreachables found in other columns.
                        if unreachables_other_columns.is_empty() {
                            unreachables_other_columns = other_columns.to_vec();
                        } else {
                            unreachables_other_columns = unreachables_other_columns
                                .into_iter()
                                .zip(other_columns)
                                .map(|(x, y)| x.intersection(&y).copied().collect())
                                .collect();
                        }
                    }
                    any_is_useful = true;
                }
                NotUseful => {
                    unreachables_this_column.insert(v.head().span);
                }
                UsefulWithWitness(_) => bug!(
                    "encountered or-pat in the expansion of `_` during exhaustiveness checking"
                ),
            }

            // If pattern has a guard don't add it to the matrix.
            if !is_under_guard {
                // We push the already-seen patterns into the matrix in order to detect redundant
                // branches like `Some(_) | Some(0)`.
                matrix.push(v);
            }
        }

        return if any_is_useful {
            let mut unreachables = if unreachables_other_columns.is_empty() {
                let n_columns = v.len();
                (0..n_columns - 1).map(|_| FxHashSet::default()).collect()
            } else {
                unreachables_other_columns
            };
            unreachables.push(unreachables_this_column);
            Useful(unreachables)
        } else {
            NotUseful
        };
    }

    // FIXME(Nadrieril): Hack to work around type normalization issues (see #72476).
    let ty = matrix.heads().next().map(|r| r.ty).unwrap_or(v.head().ty);
    let pcx = PatCtxt { cx, matrix, ty, span: v.head().span, is_top_level };

    debug!("is_useful_expand_first_col: ty={:#?}, expanding {:#?}", pcx.ty, v.head());

    let ret = v
        .head_ctor(cx)
        .split(pcx, Some(hir_id))
        .into_iter()
        .map(|ctor| {
            // We cache the result of `Fields::wildcards` because it is used a lot.
            let ctor_wild_subpatterns = Fields::wildcards(pcx, &ctor);
            let matrix = pcx.matrix.specialize_constructor(pcx, &ctor, &ctor_wild_subpatterns);
            let v = v.pop_head_constructor(&ctor_wild_subpatterns);
            let usefulness =
                is_useful(pcx.cx, &matrix, &v, witness_preference, hir_id, is_under_guard, false);
            usefulness.apply_constructor(pcx, &ctor, &ctor_wild_subpatterns)
        })
        .find(|result| result.is_useful())
        .unwrap_or(NotUseful);
    debug!("is_useful::returns({:#?}, {:#?}) = {:?}", matrix, v, ret);
    ret
}

/// Determines the constructor that the given pattern can be specialized to.
/// Returns `None` in case of a catch-all, which can't be specialized.
fn pat_constructor<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    pat: &'p Pat<'tcx>,
) -> Constructor<'tcx> {
    match pat.kind.as_ref() {
        PatKind::AscribeUserType { .. } => bug!(), // Handled by `expand_pattern`
        PatKind::Binding { .. } | PatKind::Wild => Wildcard,
        PatKind::Leaf { .. } | PatKind::Deref { .. } => Single,
        &PatKind::Variant { adt_def, variant_index, .. } => {
            Variant(adt_def.variants[variant_index].def_id)
        }
        PatKind::Constant { value } => {
            if let Some(int_range) = IntRange::from_const(cx.tcx, cx.param_env, value, pat.span) {
                IntRange(int_range)
            } else {
                match pat.ty.kind() {
                    ty::Float(_) => FloatRange(value, value, RangeEnd::Included),
                    // In `expand_pattern`, we convert string literals to `&CONST` patterns with
                    // `CONST` a pattern of type `str`. In truth this contains a constant of type
                    // `&str`.
                    ty::Str => Str(value),
                    // All constants that can be structurally matched have already been expanded
                    // into the corresponding `Pat`s by `const_to_pat`. Constants that remain are
                    // opaque.
                    _ => Opaque,
                }
            }
        }
        &PatKind::Range(PatRange { lo, hi, end }) => {
            let ty = lo.ty;
            if let Some(int_range) = IntRange::from_range(
                cx.tcx,
                lo.eval_bits(cx.tcx, cx.param_env, lo.ty),
                hi.eval_bits(cx.tcx, cx.param_env, hi.ty),
                ty,
                &end,
                pat.span,
            ) {
                IntRange(int_range)
            } else {
                FloatRange(lo, hi, end)
            }
        }
        PatKind::Array { prefix, slice, suffix } | PatKind::Slice { prefix, slice, suffix } => {
            let array_len = match pat.ty.kind() {
                ty::Array(_, length) => Some(length.eval_usize(cx.tcx, cx.param_env)),
                ty::Slice(_) => None,
                _ => span_bug!(pat.span, "bad ty {:?} for slice pattern", pat.ty),
            };
            let prefix = prefix.len() as u64;
            let suffix = suffix.len() as u64;
            let kind =
                if slice.is_some() { VarLen(prefix, suffix) } else { FixedLen(prefix + suffix) };
            Slice(Slice::new(array_len, kind))
        }
        PatKind::Or { .. } => bug!("Or-pattern should have been expanded earlier on."),
    }
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
