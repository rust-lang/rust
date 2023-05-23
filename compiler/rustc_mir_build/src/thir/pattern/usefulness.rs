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
//! This file includes the logic for exhaustiveness and reachability checking for pattern-matching.
//! Specifically, given a list of patterns for a type, we can tell whether:
//! (a) each pattern is reachable (reachability)
//! (b) the patterns cover every possible value for the type (exhaustiveness)
//!
//! The algorithm implemented here is a modified version of the one described in [this
//! paper](http://moscova.inria.fr/~maranget/papers/warn/index.html). We have however generalized
//! it to accommodate the variety of patterns that Rust supports. We thus explain our version here,
//! without being as rigorous.
//!
//!
//! # Summary
//!
//! The core of the algorithm is the notion of "usefulness". A pattern `q` is said to be *useful*
//! relative to another pattern `p` of the same type if there is a value that is matched by `q` and
//! not matched by `p`. This generalizes to many `p`s: `q` is useful w.r.t. a list of patterns
//! `p_1 .. p_n` if there is a value that is matched by `q` and by none of the `p_i`. We write
//! `usefulness(p_1 .. p_n, q)` for a function that returns a list of such values. The aim of this
//! file is to compute it efficiently.
//!
//! This is enough to compute reachability: a pattern in a `match` expression is reachable iff it
//! is useful w.r.t. the patterns above it:
//! ```rust
//! # fn foo(x: Option<i32>) {
//! match x {
//!     Some(_) => {},
//!     None => {},    // reachable: `None` is matched by this but not the branch above
//!     Some(0) => {}, // unreachable: all the values this matches are already matched by
//!                    // `Some(_)` above
//! }
//! # }
//! ```
//!
//! This is also enough to compute exhaustiveness: a match is exhaustive iff the wildcard `_`
//! pattern is _not_ useful w.r.t. the patterns in the match. The values returned by `usefulness`
//! are used to tell the user which values are missing.
//! ```compile_fail,E0004
//! # fn foo(x: Option<i32>) {
//! match x {
//!     Some(0) => {},
//!     None => {},
//!     // not exhaustive: `_` is useful because it matches `Some(1)`
//! }
//! # }
//! ```
//!
//! The entrypoint of this file is the [`compute_match_usefulness`] function, which computes
//! reachability for each match branch and exhaustiveness for the whole match.
//!
//!
//! # Constructors and fields
//!
//! Note: we will often abbreviate "constructor" as "ctor".
//!
//! The idea that powers everything that is done in this file is the following: a (matchable)
//! value is made from a constructor applied to a number of subvalues. Examples of constructors are
//! `Some`, `None`, `(,)` (the 2-tuple constructor), `Foo {..}` (the constructor for a struct
//! `Foo`), and `2` (the constructor for the number `2`). This is natural when we think of
//! pattern-matching, and this is the basis for what follows.
//!
//! Some of the ctors listed above might feel weird: `None` and `2` don't take any arguments.
//! That's ok: those are ctors that take a list of 0 arguments; they are the simplest case of
//! ctors. We treat `2` as a ctor because `u64` and other number types behave exactly like a huge
//! `enum`, with one variant for each number. This allows us to see any matchable value as made up
//! from a tree of ctors, each having a set number of children. For example: `Foo { bar: None,
//! baz: Ok(0) }` is made from 4 different ctors, namely `Foo{..}`, `None`, `Ok` and `0`.
//!
//! This idea can be extended to patterns: they are also made from constructors applied to fields.
//! A pattern for a given type is allowed to use all the ctors for values of that type (which we
//! call "value constructors"), but there are also pattern-only ctors. The most important one is
//! the wildcard (`_`), and the others are integer ranges (`0..=10`), variable-length slices (`[x,
//! ..]`), and or-patterns (`Ok(0) | Err(_)`). Examples of valid patterns are `42`, `Some(_)`, `Foo
//! { bar: Some(0) | None, baz: _ }`. Note that a binder in a pattern (e.g. `Some(x)`) matches the
//! same values as a wildcard (e.g. `Some(_)`), so we treat both as wildcards.
//!
//! From this deconstruction we can compute whether a given value matches a given pattern; we
//! simply look at ctors one at a time. Given a pattern `p` and a value `v`, we want to compute
//! `matches!(v, p)`. It's mostly straightforward: we compare the head ctors and when they match
//! we compare their fields recursively. A few representative examples:
//!
//! - `matches!(v, _) := true`
//! - `matches!((v0,  v1), (p0,  p1)) := matches!(v0, p0) && matches!(v1, p1)`
//! - `matches!(Foo { bar: v0, baz: v1 }, Foo { bar: p0, baz: p1 }) := matches!(v0, p0) && matches!(v1, p1)`
//! - `matches!(Ok(v0), Ok(p0)) := matches!(v0, p0)`
//! - `matches!(Ok(v0), Err(p0)) := false` (incompatible variants)
//! - `matches!(v, 1..=100) := matches!(v, 1) || ... || matches!(v, 100)`
//! - `matches!([v0], [p0, .., p1]) := false` (incompatible lengths)
//! - `matches!([v0, v1, v2], [p0, .., p1]) := matches!(v0, p0) && matches!(v2, p1)`
//! - `matches!(v, p0 | p1) := matches!(v, p0) || matches!(v, p1)`
//!
//! Constructors, fields and relevant operations are defined in the [`super::deconstruct_pat`] module.
//!
//! Note: this constructors/fields distinction may not straightforwardly apply to every Rust type.
//! For example a value of type `Rc<u64>` can't be deconstructed that way, and `&str` has an
//! infinitude of constructors. There are also subtleties with visibility of fields and
//! uninhabitedness and various other things. The constructors idea can be extended to handle most
//! of these subtleties though; caveats are documented where relevant throughout the code.
//!
//! Whether constructors cover each other is computed by [`Constructor::is_covered_by`].
//!
//!
//! # Specialization
//!
//! Recall that we wish to compute `usefulness(p_1 .. p_n, q)`: given a list of patterns `p_1 ..
//! p_n` and a pattern `q`, all of the same type, we want to find a list of values (called
//! "witnesses") that are matched by `q` and by none of the `p_i`. We obviously don't just
//! enumerate all possible values. From the discussion above we see that we can proceed
//! ctor-by-ctor: for each value ctor of the given type, we ask "is there a value that starts with
//! this constructor and matches `q` and none of the `p_i`?". As we saw above, there's a lot we can
//! say from knowing only the first constructor of our candidate value.
//!
//! Let's take the following example:
//! ```compile_fail,E0004
//! # enum Enum { Variant1(()), Variant2(Option<bool>, u32)}
//! # fn foo(x: Enum) {
//! match x {
//!     Enum::Variant1(_) => {} // `p1`
//!     Enum::Variant2(None, 0) => {} // `p2`
//!     Enum::Variant2(Some(_), 0) => {} // `q`
//! }
//! # }
//! ```
//!
//! We can easily see that if our candidate value `v` starts with `Variant1` it will not match `q`.
//! If `v = Variant2(v0, v1)` however, whether or not it matches `p2` and `q` will depend on `v0`
//! and `v1`. In fact, such a `v` will be a witness of usefulness of `q` exactly when the tuple
//! `(v0, v1)` is a witness of usefulness of `q'` in the following reduced match:
//!
//! ```compile_fail,E0004
//! # fn foo(x: (Option<bool>, u32)) {
//! match x {
//!     (None, 0) => {} // `p2'`
//!     (Some(_), 0) => {} // `q'`
//! }
//! # }
//! ```
//!
//! This motivates a new step in computing usefulness, that we call _specialization_.
//! Specialization consist of filtering a list of patterns for those that match a constructor, and
//! then looking into the constructor's fields. This enables usefulness to be computed recursively.
//!
//! Instead of acting on a single pattern in each row, we will consider a list of patterns for each
//! row, and we call such a list a _pattern-stack_. The idea is that we will specialize the
//! leftmost pattern, which amounts to popping the constructor and pushing its fields, which feels
//! like a stack. We note a pattern-stack simply with `[p_1 ... p_n]`.
//! Here's a sequence of specializations of a list of pattern-stacks, to illustrate what's
//! happening:
//! ```ignore (illustrative)
//! [Enum::Variant1(_)]
//! [Enum::Variant2(None, 0)]
//! [Enum::Variant2(Some(_), 0)]
//! //==>> specialize with `Variant2`
//! [None, 0]
//! [Some(_), 0]
//! //==>> specialize with `Some`
//! [_, 0]
//! //==>> specialize with `true` (say the type was `bool`)
//! [0]
//! //==>> specialize with `0`
//! []
//! ```
//!
//! The function `specialize(c, p)` takes a value constructor `c` and a pattern `p`, and returns 0
//! or more pattern-stacks. If `c` does not match the head constructor of `p`, it returns nothing;
//! otherwise if returns the fields of the constructor. This only returns more than one
//! pattern-stack if `p` has a pattern-only constructor.
//!
//! - Specializing for the wrong constructor returns nothing
//!
//!   `specialize(None, Some(p0)) := []`
//!
//! - Specializing for the correct constructor returns a single row with the fields
//!
//!   `specialize(Variant1, Variant1(p0, p1, p2)) := [[p0, p1, p2]]`
//!
//!   `specialize(Foo{..}, Foo { bar: p0, baz: p1 }) := [[p0, p1]]`
//!
//! - For or-patterns, we specialize each branch and concatenate the results
//!
//!   `specialize(c, p0 | p1) := specialize(c, p0) ++ specialize(c, p1)`
//!
//! - We treat the other pattern constructors as if they were a large or-pattern of all the
//!   possibilities:
//!
//!   `specialize(c, _) := specialize(c, Variant1(_) | Variant2(_, _) | ...)`
//!
//!   `specialize(c, 1..=100) := specialize(c, 1 | ... | 100)`
//!
//!   `specialize(c, [p0, .., p1]) := specialize(c, [p0, p1] | [p0, _, p1] | [p0, _, _, p1] | ...)`
//!
//! - If `c` is a pattern-only constructor, `specialize` is defined on a case-by-case basis. See
//!   the discussion about constructor splitting in [`super::deconstruct_pat`].
//!
//!
//! We then extend this function to work with pattern-stacks as input, by acting on the first
//! column and keeping the other columns untouched.
//!
//! Specialization for the whole matrix is done in [`Matrix::specialize_constructor`]. Note that
//! or-patterns in the first column are expanded before being stored in the matrix. Specialization
//! for a single patstack is done from a combination of [`Constructor::is_covered_by`] and
//! [`PatStack::pop_head_constructor`]. The internals of how it's done mostly live in the
//! [`Fields`] struct.
//!
//!
//! # Computing usefulness
//!
//! We now have all we need to compute usefulness. The inputs to usefulness are a list of
//! pattern-stacks `p_1 ... p_n` (one per row), and a new pattern_stack `q`. The paper and this
//! file calls the list of patstacks a _matrix_. They must all have the same number of columns and
//! the patterns in a given column must all have the same type. `usefulness` returns a (possibly
//! empty) list of witnesses of usefulness. These witnesses will also be pattern-stacks.
//!
//! - base case: `n_columns == 0`.
//!     Since a pattern-stack functions like a tuple of patterns, an empty one functions like the
//!     unit type. Thus `q` is useful iff there are no rows above it, i.e. if `n == 0`.
//!
//! - inductive case: `n_columns > 0`.
//!     We need a way to list the constructors we want to try. We will be more clever in the next
//!     section but for now assume we list all value constructors for the type of the first column.
//!
//!     - for each such ctor `c`:
//!
//!         - for each `q'` returned by `specialize(c, q)`:
//!
//!             - we compute `usefulness(specialize(c, p_1) ... specialize(c, p_n), q')`
//!
//!         - for each witness found, we revert specialization by pushing the constructor `c` on top.
//!
//!     - We return the concatenation of all the witnesses found, if any.
//!
//! Example:
//! ```ignore (illustrative)
//! [Some(true)] // p_1
//! [None] // p_2
//! [Some(_)] // q
//! //==>> try `None`: `specialize(None, q)` returns nothing
//! //==>> try `Some`: `specialize(Some, q)` returns a single row
//! [true] // p_1'
//! [_] // q'
//! //==>> try `true`: `specialize(true, q')` returns a single row
//! [] // p_1''
//! [] // q''
//! //==>> base case; `n != 0` so `q''` is not useful.
//! //==>> go back up a step
//! [true] // p_1'
//! [_] // q'
//! //==>> try `false`: `specialize(false, q')` returns a single row
//! [] // q''
//! //==>> base case; `n == 0` so `q''` is useful. We return the single witness `[]`
//! witnesses:
//! []
//! //==>> undo the specialization with `false`
//! witnesses:
//! [false]
//! //==>> undo the specialization with `Some`
//! witnesses:
//! [Some(false)]
//! //==>> we have tried all the constructors. The output is the single witness `[Some(false)]`.
//! ```
//!
//! This computation is done in `is_useful`. In practice we don't care about the list of
//! witnesses when computing reachability; we only need to know whether any exist. We do keep the
//! witnesses when computing exhaustiveness to report them to the user.
//!
//!
//! # Making usefulness tractable: constructor splitting
//!
//! We're missing one last detail: which constructors do we list? Naively listing all value
//! constructors cannot work for types like `u64` or `&str`, so we need to be more clever. The
//! first obvious insight is that we only want to list constructors that are covered by the head
//! constructor of `q`. If it's a value constructor, we only try that one. If it's a pattern-only
//! constructor, we use the final clever idea for this algorithm: _constructor splitting_, where we
//! group together constructors that behave the same.
//!
//! The details are not necessary to understand this file, so we explain them in
//! [`super::deconstruct_pat`]. Splitting is done by the `Constructor::split` function.
//!
//! # Constants in patterns
//!
//! There are two kinds of constants in patterns:
//!
//! * literals (`1`, `true`, `"foo"`)
//! * named or inline consts (`FOO`, `const { 5 + 6 }`)
//!
//! The latter are converted into other patterns with literals at the leaves. For example
//! `const_to_pat(const { [1, 2, 3] })` becomes an `Array(vec![Const(1), Const(2), Const(3)])`
//! pattern. This gets problematic when comparing the constant via `==` would behave differently
//! from matching on the constant converted to a pattern. Situations like that can occur, when
//! the user implements `PartialEq` manually, and thus could make `==` behave arbitrarily different.
//! In order to honor the `==` implementation, constants of types that implement `PartialEq` manually
//! stay as a full constant and become an `Opaque` pattern. These `Opaque` patterns do not participate
//! in exhaustiveness, specialization or overlap checking.

use super::deconstruct_pat::{Constructor, ConstructorSet, DeconstructedPat, Fields};
use crate::errors::{NonExhaustiveOmittedPattern, Uncovered};

use rustc_arena::TypedArena;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint::builtin::NON_EXHAUSTIVE_OMITTED_PATTERNS;
use rustc_span::{Span, DUMMY_SP};

use smallvec::{smallvec, SmallVec};
use std::fmt;

pub(crate) struct MatchCheckCtxt<'p, 'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    /// The module in which the match occurs. This is necessary for
    /// checking inhabited-ness of types because whether a type is (visibly)
    /// inhabited can depend on whether it was defined in the current module or
    /// not. E.g., `struct Foo { _private: ! }` cannot be seen to be empty
    /// outside its module and should not be matchable with an empty match statement.
    pub(crate) module: DefId,
    pub(crate) param_env: ty::ParamEnv<'tcx>,
    pub(crate) pattern_arena: &'p TypedArena<DeconstructedPat<'p, 'tcx>>,
    /// Only produce `NON_EXHAUSTIVE_OMITTED_PATTERNS` lint on refutable patterns.
    pub(crate) refutable: bool,
}

impl<'a, 'tcx> MatchCheckCtxt<'a, 'tcx> {
    pub(super) fn is_uninhabited(&self, ty: Ty<'tcx>) -> bool {
        if self.tcx.features().exhaustive_patterns {
            !ty.is_inhabited_from(self.tcx, self.module, self.param_env)
        } else {
            false
        }
    }

    /// Returns whether the given type is an enum from another crate declared `#[non_exhaustive]`.
    pub(super) fn is_foreign_non_exhaustive_enum(&self, ty: Ty<'tcx>) -> bool {
        match ty.kind() {
            ty::Adt(def, ..) => {
                def.is_enum() && def.is_variant_list_non_exhaustive() && !def.did().is_local()
            }
            _ => false,
        }
    }
}

#[derive(Copy, Clone)]
pub(super) struct PatCtxt<'a, 'p, 'tcx> {
    pub(super) cx: &'a MatchCheckCtxt<'p, 'tcx>,
    /// Type of the current column under investigation.
    pub(super) ty: Ty<'tcx>,
}

impl<'a, 'p, 'tcx> fmt::Debug for PatCtxt<'a, 'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PatCtxt").field("ty", &self.ty).finish()
    }
}

/// A row of a matrix. Rows of len 1 are very common, which is why `SmallVec[_; 2]`
/// works well.
#[derive(Clone)]
struct PatStack<'p, 'tcx> {
    pats: SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]>,
    /// The row (in the matrix) of the `PatStack` from which this one is derived. When there is
    /// none, this is the id of the arm.
    parent_row: usize,
    is_under_guard: bool,
    arm_hir_id: HirId,
    is_useful: bool,
}

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    fn from_pattern(
        pat: &'p DeconstructedPat<'p, 'tcx>,
        parent_row: usize,
        is_under_guard: bool,
        arm_hir_id: HirId,
    ) -> Self {
        Self::from_vec(smallvec![pat], parent_row, is_under_guard, arm_hir_id)
    }

    fn from_vec(
        vec: SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]>,
        parent_row: usize,
        is_under_guard: bool,
        arm_hir_id: HirId,
    ) -> Self {
        PatStack { pats: vec, parent_row, is_under_guard, arm_hir_id, is_useful: false }
    }

    fn is_empty(&self) -> bool {
        self.pats.is_empty()
    }

    fn len(&self) -> usize {
        self.pats.len()
    }

    fn head(&self) -> &'p DeconstructedPat<'p, 'tcx> {
        self.pats[0]
    }

    fn iter(&self) -> impl Iterator<Item = &DeconstructedPat<'p, 'tcx>> {
        self.pats.iter().copied()
    }

    // Expand the first pattern into its subpatterns. Only useful if the pattern is an
    // or-pattern. Panics if `self` is empty.
    fn expand_or_pat<'a>(&'a self) -> impl Iterator<Item = PatStack<'p, 'tcx>> + Captures<'a> {
        self.head().iter_fields().map(move |pat| {
            let mut new_patstack =
                PatStack::from_pattern(pat, self.parent_row, self.is_under_guard, self.arm_hir_id);
            new_patstack.pats.extend_from_slice(&self.pats[1..]);
            new_patstack
        })
    }

    /// This computes `S(self.head().ctor(), self)`. See top of the file for explanations.
    ///
    /// Structure patterns with a partial wild pattern (Foo { a: 42, .. }) have their missing
    /// fields filled with wild patterns.
    ///
    /// This is roughly the inverse of `Constructor::apply`.
    fn pop_head_constructor(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        parent_row: usize,
    ) -> PatStack<'p, 'tcx> {
        // We pop the head pattern and push the new fields extracted from the arguments of
        // `self.head()`.
        let mut new_fields: SmallVec<[_; 2]> = self.head().specialize(pcx, ctor);
        new_fields.extend_from_slice(&self.pats[1..]);
        PatStack::from_vec(new_fields, parent_row, self.is_under_guard, self.arm_hir_id)
    }
}

/// Pretty-printing for matrix row.
impl<'p, 'tcx> fmt::Debug for PatStack<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for pat in self.iter() {
            write!(f, " {:?} +", pat)?;
        }
        Ok(())
    }
}

/// A 2D matrix.
#[derive(Clone)]
struct Matrix<'p, 'tcx> {
    /// The rows of this matrix.
    /// Invariant: `row.head()` is never an or-pattern.
    rows: Vec<PatStack<'p, 'tcx>>,
    /// Fake row that stores wildcard patterns that match the other rows. This is used only to track
    /// the type and number of columns of the matrix.
    wildcard_row: PatStack<'p, 'tcx>,
}

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    fn empty(wildcard_row: PatStack<'p, 'tcx>) -> Self {
        Matrix { rows: vec![], wildcard_row }
    }

    /// Pushes a new row to the matrix. If the row starts with an or-pattern, this recursively
    /// expands it.
    fn push(&mut self, row: PatStack<'p, 'tcx>) {
        if !row.is_empty() && row.head().is_or_pat() {
            for new_row in row.expand_or_pat() {
                self.push(new_row);
            }
        } else {
            self.rows.push(row);
        }
    }

    fn column_count(&self) -> usize {
        self.wildcard_row.len()
    }
    fn head_ty(&self) -> Ty<'tcx> {
        let mut ty = self.wildcard_row.head().ty();
        // Opaque types can't get destructured/split, but the patterns can
        // actually hint at hidden types, so we use the patterns' types instead.
        if let ty::Alias(ty::Opaque, ..) = ty.kind() {
            if let Some(row) = self.rows().next() {
                ty = row.head().ty();
            }
        }
        ty
    }

    fn rows<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a PatStack<'p, 'tcx>> + Clone + DoubleEndedIterator + ExactSizeIterator
    {
        self.rows.iter()
    }
    fn rows_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut PatStack<'p, 'tcx>> + DoubleEndedIterator + ExactSizeIterator
    {
        self.rows.iter_mut()
    }
    /// Iterate over the first component of each row
    fn heads<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Clone + Captures<'a> {
        self.rows().map(|r| r.head())
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
    ) -> Matrix<'p, 'tcx> {
        let mut matrix =
            Matrix::empty(self.wildcard_row.pop_head_constructor(pcx, ctor, usize::MAX));
        for (i, row) in self.rows().enumerate() {
            if ctor.is_covered_by(pcx, row.head().ctor()) {
                let new_row = row.pop_head_constructor(pcx, ctor, i);
                matrix.push(new_row);
            }
        }
        matrix
    }
}

/// Pretty-printer for matrices of patterns, example:
///
/// ```text
/// + _     + []                +
/// + true  + [First]           +
/// + true  + [Second(true)]    +
/// + false + [_]               +
/// + _     + [_, _, tail @ ..] +
/// ```
impl<'p, 'tcx> fmt::Debug for Matrix<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n")?;

        let Matrix { rows: m, .. } = self;
        let pretty_printed_matrix: Vec<Vec<String>> =
            m.iter().map(|row| row.iter().map(|pat| format!("{:?}", pat)).collect()).collect();

        let column_count = m.iter().map(|row| row.len()).next().unwrap_or(0);
        assert!(m.iter().all(|row| row.len() == column_count));
        let column_widths: Vec<usize> = (0..column_count)
            .map(|col| pretty_printed_matrix.iter().map(|row| row[col].len()).max().unwrap_or(0))
            .collect();

        for row in pretty_printed_matrix {
            write!(f, "+")?;
            for (column, pat_str) in row.into_iter().enumerate() {
                write!(f, " ")?;
                write!(f, "{:1$}", pat_str, column_widths[column])?;
                write!(f, " +")?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

/// A partially-constructed witness of non-exhaustiveness for error reporting, represented as a list
/// of patterns (in reverse order of construction). This is very similar to `PatStack`, with two
/// differences: the patterns are stored in the opposite order for historical reasons, and where the
/// API of `PatStack` is oriented towards deconstructing it by peeling layers off, this is oriented
/// towards reconstructing by rewrapping layers.
///
/// This should have the same length and types as the `PatStack`s and `Matrix` it is constructed
/// nearby of. At the end of the algorithm this will have length 1 and represent and actual pattern.
///
/// For example, if we are constructing a witness for the match against
///
/// ```compile_fail,E0004
/// struct Pair(Option<(u32, u32)>, bool);
/// # fn foo(p: Pair) {
/// match p {
///    Pair(None, _) => {}
///    Pair(_, false) => {}
/// }
/// # }
/// ```
///
/// We'll perform the following steps:
/// 1. Start with an empty witness
///     `WitnessStack(vec![])`
/// 2. Push a witness `true` against the `false`
///     `WitnessStack(vec![true])`
/// 3. Push a witness `Some(_)` against the `None`
///     `WitnessStack(vec![true, Some(_)])`
/// 4. Apply the `Pair` constructor to the witnesses
///     `WitnessStack(vec![Pair(Some(_), true)])`
///
/// The final `Pair(Some(_), true)` is then the resulting witness.
#[derive(Debug, Clone)]
struct WitnessStack<'p, 'tcx>(Vec<DeconstructedPat<'p, 'tcx>>);

impl<'p, 'tcx> WitnessStack<'p, 'tcx> {
    /// Asserts that the witness contains a single pattern, and returns it.
    fn single_pattern(self) -> DeconstructedPat<'p, 'tcx> {
        assert_eq!(self.0.len(), 1);
        self.0.into_iter().next().unwrap()
    }

    /// Reverses specialization. Given a witness obtained after specialization, this constructs a
    /// new witness valid for before specialization. Examples:
    ///
    /// ctor: tuple of 2 elements
    /// pats: [false, "foo", _, true]
    /// result: [(false, "foo"), _, true]
    ///
    /// ctor: Enum::Variant { a: (bool, &'static str), b: usize}
    /// pats: [(false, "foo"), _, true]
    /// result: [Enum::Variant { a: (false, "foo"), b: _ }, true]
    fn apply_constructor(&mut self, pcx: &PatCtxt<'_, 'p, 'tcx>, ctor: &Constructor<'tcx>) {
        let len = self.0.len();
        let arity = ctor.arity(pcx);
        let fields = {
            let pats = self.0.drain((len - arity)..).rev();
            Fields::from_iter(pcx.cx, pats)
        };
        let pat = DeconstructedPat::new(ctor.clone(), fields, pcx.ty, DUMMY_SP);

        self.0.push(pat);
    }

    /// Reverses specialization by the `Missing` constructor by pushing a whole new pattern.
    fn push_pattern(&mut self, pat: DeconstructedPat<'p, 'tcx>) {
        self.0.push(pat);
    }
}

/// Represents a set of partially-constructed witnesses of non-exhaustiveness for error reporting.
/// This has similar invariants as `Matrix` does.
/// Throughout the exhaustiveness phase of the algorithm, `is_useful` maintains the invariant that
/// the union of the `Matrix` and the `WitnessMatrix` together matches the type exhaustively. By the
/// end of the algorithm, this has a single column, which contains the patterns that are missing for
/// the match to be exhaustive.
#[derive(Debug, Clone)]
pub struct WitnessMatrix<'p, 'tcx>(Vec<WitnessStack<'p, 'tcx>>);

impl<'p, 'tcx> WitnessMatrix<'p, 'tcx> {
    /// New matrix with no rows.
    fn new_empty() -> Self {
        WitnessMatrix(vec![])
    }
    /// New matrix with one row and no columns.
    fn new_unit() -> Self {
        WitnessMatrix(vec![WitnessStack(vec![])])
    }

    /// Whether this has any rows.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Asserts that there is a single column and returns the patterns in it.
    fn single_column(self) -> Vec<DeconstructedPat<'p, 'tcx>> {
        self.0.into_iter().map(|w| w.single_pattern()).collect()
    }

    /// Reverses specialization by the `Missing` constructor. This constructs a "wild" version of
    /// `ctor`, that matches everything that can be built with it. For example, if `ctor` is a
    /// `Constructor::Variant` for `Option::Some`, we get the pattern `Some(_)`. We push repeated
    /// copies of that pattern to all rows.
    fn push_wild_ctor(&mut self, pcx: &PatCtxt<'_, 'p, 'tcx>, ctor: Constructor<'tcx>) {
        let pat = DeconstructedPat::wild_from_ctor(pcx, ctor);
        for witness in self.0.iter_mut() {
            witness.push_pattern(pat.clone())
        }
    }

    /// Reverses specialization by `ctor`.
    fn apply_constructor(
        &mut self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        missing_ctors: &[Constructor<'tcx>],
        ctor: &Constructor<'tcx>,
    ) {
        if self.is_empty() {
            return;
        }
        if !matches!(ctor, Constructor::Missing { .. }) {
            for witness in self.0.iter_mut() {
                witness.apply_constructor(pcx, ctor)
            }
        } else if missing_ctors.iter().any(|c| c.is_non_exhaustive()) {
            // Here we don't want the user to try to list all variants, we want them to add a
            // wildcard, so we only suggest that.
            self.push_wild_ctor(pcx, Constructor::NonExhaustive);
        } else {
            // We got the special `Missing` constructor, so each of the missing constructors gives a
            // new pattern that is not caught by the match. We list those patterns and push them
            // onto our current witnesses.
            let old_witnesses = std::mem::replace(self, Self::new_empty());
            for missing_ctor in missing_ctors {
                let mut witnesses_with_missing_ctor = old_witnesses.clone();
                witnesses_with_missing_ctor.push_wild_ctor(pcx, missing_ctor.clone());
                self.extend(witnesses_with_missing_ctor)
            }
        }
    }

    /// Merges the rows of two witness matrices. Their column types must match.
    fn extend(&mut self, other: Self) {
        self.0.extend(other.0)
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
#[instrument(level = "debug", skip(cx, is_top_level), ret)]
fn compute_usefulness<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    matrix: &mut Matrix<'p, 'tcx>,
    collect_witnesses: bool,
    lint_root: HirId,
    is_top_level: bool,
) -> WitnessMatrix<'p, 'tcx> {
    debug_assert!(matrix.rows().all(|r| r.len() == matrix.column_count()));

    // The base case. We are pattern-matching on () and the return value is based on whether our
    // matrix has a row or not.
    if matrix.column_count() == 0 {
        let mut useful = true;
        for row in matrix.rows_mut() {
            row.is_useful = useful;
            useful = useful && row.is_under_guard;
            if !useful {
                break;
            }
        }
        if useful && collect_witnesses {
            return WitnessMatrix::new_unit();
        } else {
            return WitnessMatrix::new_empty();
        }
    }

    let ty = matrix.head_ty();
    debug!("ty: {ty:?}");
    let pcx = &PatCtxt { cx, ty };

    let ctor_set = ConstructorSet::new(pcx);
    let (split_ctors, missing_ctors) =
        ctor_set.split(pcx, matrix.heads().map(|p| p.ctor()), is_top_level);

    // For each constructor, we compute whether there's a value that starts with it that would
    // witness the usefulness of `v`.
    let mut ret = WitnessMatrix::new_empty();
    let orig_column_count = matrix.column_count();
    for ctor in split_ctors {
        // If some ctors are missing we only report those. Could report all if that's useful for
        // some applications.
        let collect_witnesses = collect_witnesses
            && (missing_ctors.is_empty()
                || matches!(ctor, Constructor::Wildcard | Constructor::Missing));
        debug!("specialize({:?})", ctor);
        let mut spec_matrix = matrix.specialize_constructor(pcx, &ctor);
        let mut witnesses = ensure_sufficient_stack(|| {
            compute_usefulness(cx, &mut spec_matrix, collect_witnesses, lint_root, false)
        });
        if collect_witnesses {
            witnesses.apply_constructor(pcx, &missing_ctors, &ctor);
            ret.extend(witnesses);
        }

        // Lint on likely incorrect range patterns (#63987)
        if spec_matrix.rows().len() >= 2 {
            if let Constructor::IntRange(overlap_range) = &ctor {
                // If two ranges overlap on their boundaries, that boundary will be found as a singleton
                // range after splitting.
                // We limit to a single column for now, see `lint_overlapping_range_endpoints`.
                if overlap_range.is_singleton() && orig_column_count == 1 {
                    overlap_range.lint_overlapping_range_endpoints(
                        pcx,
                        spec_matrix.rows().map(|child_row| &matrix.rows[child_row.parent_row]).map(
                            |parent_row| {
                                (
                                    parent_row.head(),
                                    parent_row.is_under_guard,
                                    parent_row.arm_hir_id,
                                )
                            },
                        ),
                        orig_column_count,
                    );
                }
            }
        }

        // When all the conditions are met we have a match with a `non_exhaustive` enum
        // that has the potential to trigger the `non_exhaustive_omitted_patterns` lint.
        if cx.refutable
            && matches!(&ctor, Constructor::Missing)
            && matches!(&ctor_set, ConstructorSet::Variants { non_exhaustive: true, .. })
            && spec_matrix.rows().len() != 0
        {
            let patterns = missing_ctors
                .iter()
                // We want to list only real variants.
                .filter(|c| !(c.is_non_exhaustive() || c.is_wildcard()))
                .cloned()
                .map(|missing_ctor| DeconstructedPat::wild_from_ctor(pcx, missing_ctor))
                .collect::<Vec<_>>();

            if !patterns.is_empty() {
                let first_spec_row = spec_matrix.rows().next().unwrap();
                let first_wildcard_row = &matrix.rows[first_spec_row.parent_row];
                let wildcard_span = first_wildcard_row.head().span();
                // Report that a match of a `non_exhaustive` enum marked with `non_exhaustive_omitted_patterns`
                // is not exhaustive enough.
                // NB: The partner lint for structs lives in `compiler/rustc_hir_analysis/src/check/pat.rs`.
                cx.tcx.emit_spanned_lint(
                    NON_EXHAUSTIVE_OMITTED_PATTERNS,
                    first_wildcard_row.arm_hir_id,
                    wildcard_span,
                    NonExhaustiveOmittedPattern {
                        scrut_ty: pcx.ty,
                        uncovered: Uncovered::new(wildcard_span, pcx.cx, patterns),
                    },
                );
            }
        }

        for child_row in spec_matrix.rows() {
            let parent_row = &mut matrix.rows[child_row.parent_row];
            parent_row.is_useful = parent_row.is_useful || child_row.is_useful;
        }
    }

    for row in matrix.rows() {
        if row.is_useful {
            row.head().set_reachable();
        }
    }

    ret
}

/// The arm of a match expression.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatchArm<'p, 'tcx> {
    /// The pattern must have been lowered through `check_match::MatchVisitor::lower_pattern`.
    pub(crate) pat: &'p DeconstructedPat<'p, 'tcx>,
    pub(crate) hir_id: HirId,
    pub(crate) has_guard: bool,
}

/// Indicates whether or not a given arm is reachable.
#[derive(Clone, Debug)]
pub(crate) enum Reachability {
    /// The arm is reachable. This additionally carries a set of or-pattern branches that have been
    /// found to be unreachable despite the overall arm being reachable. Used only in the presence
    /// of or-patterns, otherwise it stays empty.
    Reachable(Vec<Span>),
    /// The arm is unreachable.
    Unreachable,
}

/// The output of checking a match for exhaustiveness and arm reachability.
pub(crate) struct UsefulnessReport<'p, 'tcx> {
    /// For each arm of the input, whether that arm is reachable after the arms above it.
    pub(crate) arm_usefulness: Vec<(MatchArm<'p, 'tcx>, Reachability)>,
    /// If the match is exhaustive, this is empty. If not, this contains witnesses for the lack of
    /// exhaustiveness.
    pub(crate) non_exhaustiveness_witnesses: Vec<DeconstructedPat<'p, 'tcx>>,
}

/// The entrypoint for the usefulness algorithm. Computes whether a match is exhaustive and which
/// of its arms are reachable.
///
/// Note: the input patterns must have been lowered through
/// `check_match::MatchVisitor::lower_pattern`.
#[instrument(skip(cx, arms), level = "debug")]
pub(crate) fn compute_match_usefulness<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    arms: &[MatchArm<'p, 'tcx>],
    lint_root: HirId,
    scrut_ty: Ty<'tcx>,
    scrut_span: Span,
) -> UsefulnessReport<'p, 'tcx> {
    let wild_pattern = cx.pattern_arena.alloc(DeconstructedPat::wildcard(scrut_ty, DUMMY_SP));
    let wildcard_row = PatStack::from_pattern(wild_pattern, usize::MAX, false, lint_root);
    let mut matrix = Matrix::empty(wildcard_row);
    for (row_id, arm) in arms.iter().enumerate() {
        let v = PatStack::from_pattern(arm.pat, row_id, arm.has_guard, arm.hir_id);
        matrix.push(v);
    }

    let non_exhaustiveness_witnesses = compute_usefulness(cx, &mut matrix, true, lint_root, true);
    let non_exhaustiveness_witnesses: Vec<_> = non_exhaustiveness_witnesses.single_column();
    let arm_usefulness: Vec<_> = arms
        .iter()
        .copied()
        .map(|arm| {
            debug!(?arm);
            let reachability = if arm.pat.is_reachable() {
                Reachability::Reachable(arm.pat.unreachable_spans())
            } else {
                Reachability::Unreachable
            };
            (arm, reachability)
        })
        .collect();

    UsefulnessReport { arm_usefulness, non_exhaustiveness_witnesses }
}
