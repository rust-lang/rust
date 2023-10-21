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
//! [`super::deconstruct_pat::Fields`] struct.
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
//! This computation is done in [`is_useful`]. In practice we don't care about the list of
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
//! [`super::deconstruct_pat`]. Splitting is done by the [`Constructor::split`] function.
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

use self::ArmType::*;
use self::Usefulness::*;
use super::deconstruct_pat::{Constructor, ConstructorSet, DeconstructedPat, WitnessPat};
use crate::errors::{NonExhaustiveOmittedPattern, Uncovered};

use rustc_data_structures::captures::Captures;

use rustc_arena::TypedArena;
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
    /// Span of the current pattern under investigation.
    pub(super) span: Span,
    /// Whether the current pattern is the whole pattern as found in a match arm, or if it's a
    /// subpattern.
    pub(super) is_top_level: bool,
}

impl<'a, 'p, 'tcx> fmt::Debug for PatCtxt<'a, 'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PatCtxt").field("ty", &self.ty).finish()
    }
}

/// A row of a matrix. Rows of len 1 are very common, which is why `SmallVec[_; 2]`
/// works well.
#[derive(Clone)]
pub(crate) struct PatStack<'p, 'tcx> {
    pub(crate) pats: SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]>,
}

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    fn from_pattern(pat: &'p DeconstructedPat<'p, 'tcx>) -> Self {
        Self::from_vec(smallvec![pat])
    }

    fn from_vec(vec: SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]>) -> Self {
        PatStack { pats: vec }
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

    // Recursively expand the first pattern into its subpatterns. Only useful if the pattern is an
    // or-pattern. Panics if `self` is empty.
    fn expand_or_pat<'a>(&'a self) -> impl Iterator<Item = PatStack<'p, 'tcx>> + Captures<'a> {
        self.head().iter_fields().map(move |pat| {
            let mut new_patstack = PatStack::from_pattern(pat);
            new_patstack.pats.extend_from_slice(&self.pats[1..]);
            new_patstack
        })
    }

    // Recursively expand all patterns into their subpatterns and push each `PatStack` to matrix.
    fn expand_and_extend<'a>(&'a self, matrix: &mut Matrix<'p, 'tcx>) {
        if !self.is_empty() && self.head().is_or_pat() {
            for pat in self.head().iter_fields() {
                let mut new_patstack = PatStack::from_pattern(pat);
                new_patstack.pats.extend_from_slice(&self.pats[1..]);
                if !new_patstack.is_empty() && new_patstack.head().is_or_pat() {
                    new_patstack.expand_and_extend(matrix);
                } else if !new_patstack.is_empty() {
                    matrix.push(new_patstack);
                }
            }
        }
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
    ) -> PatStack<'p, 'tcx> {
        // We pop the head pattern and push the new fields extracted from the arguments of
        // `self.head()`.
        let mut new_fields: SmallVec<[_; 2]> = self.head().specialize(pcx, ctor);
        new_fields.extend_from_slice(&self.pats[1..]);
        PatStack::from_vec(new_fields)
    }
}

/// Pretty-printing for matrix row.
impl<'p, 'tcx> fmt::Debug for PatStack<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for pat in self.iter() {
            write!(f, " {pat:?} +")?;
        }
        Ok(())
    }
}

/// A 2D matrix.
#[derive(Clone)]
pub(super) struct Matrix<'p, 'tcx> {
    pub patterns: Vec<PatStack<'p, 'tcx>>,
}

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    fn empty() -> Self {
        Matrix { patterns: vec![] }
    }

    /// Number of columns of this matrix. `None` is the matrix is empty.
    pub(super) fn column_count(&self) -> Option<usize> {
        self.patterns.get(0).map(|r| r.len())
    }

    /// Pushes a new row to the matrix. If the row starts with an or-pattern, this recursively
    /// expands it.
    fn push(&mut self, row: PatStack<'p, 'tcx>) {
        if !row.is_empty() && row.head().is_or_pat() {
            row.expand_and_extend(self);
        } else {
            self.patterns.push(row);
        }
    }

    /// Iterate over the first component of each row
    fn heads<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Clone + Captures<'a> {
        self.patterns.iter().map(|r| r.head())
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
    ) -> Matrix<'p, 'tcx> {
        let mut matrix = Matrix::empty();
        for row in &self.patterns {
            if ctor.is_covered_by(pcx, row.head().ctor()) {
                let new_row = row.pop_head_constructor(pcx, ctor);
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

        let Matrix { patterns: m, .. } = self;
        let pretty_printed_matrix: Vec<Vec<String>> =
            m.iter().map(|row| row.iter().map(|pat| format!("{pat:?}")).collect()).collect();

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

/// This carries the results of computing usefulness, as described at the top of the file. When
/// checking usefulness of a match branch, we use the `NoWitnesses` variant, which also keeps track
/// of potential unreachable sub-patterns (in the presence of or-patterns). When checking
/// exhaustiveness of a whole match, we use the `WithWitnesses` variant, which carries a list of
/// witnesses of non-exhaustiveness when there are any.
/// Which variant to use is dictated by `ArmType`.
#[derive(Debug, Clone)]
enum Usefulness<'tcx> {
    /// If we don't care about witnesses, simply remember if the pattern was useful.
    NoWitnesses { useful: bool },
    /// Carries a list of witnesses of non-exhaustiveness. If empty, indicates that the whole
    /// pattern is unreachable.
    WithWitnesses(Vec<WitnessStack<'tcx>>),
}

impl<'tcx> Usefulness<'tcx> {
    fn new_useful(preference: ArmType) -> Self {
        match preference {
            // A single (empty) witness of reachability.
            FakeExtraWildcard => WithWitnesses(vec![WitnessStack(vec![])]),
            RealArm => NoWitnesses { useful: true },
        }
    }

    fn new_not_useful(preference: ArmType) -> Self {
        match preference {
            FakeExtraWildcard => WithWitnesses(vec![]),
            RealArm => NoWitnesses { useful: false },
        }
    }

    fn is_useful(&self) -> bool {
        match self {
            Usefulness::NoWitnesses { useful } => *useful,
            Usefulness::WithWitnesses(witnesses) => !witnesses.is_empty(),
        }
    }

    /// Combine usefulnesses from two branches. This is an associative operation.
    fn extend(&mut self, other: Self) {
        match (&mut *self, other) {
            (WithWitnesses(_), WithWitnesses(o)) if o.is_empty() => {}
            (WithWitnesses(s), WithWitnesses(o)) if s.is_empty() => *self = WithWitnesses(o),
            (WithWitnesses(s), WithWitnesses(o)) => s.extend(o),
            (NoWitnesses { useful: s_useful }, NoWitnesses { useful: o_useful }) => {
                *s_useful = *s_useful || o_useful
            }
            _ => unreachable!(),
        }
    }

    /// After calculating usefulness after a specialization, call this to reconstruct a usefulness
    /// that makes sense for the matrix pre-specialization. This new usefulness can then be merged
    /// with the results of specializing with the other constructors.
    fn apply_constructor(
        self,
        pcx: &PatCtxt<'_, '_, 'tcx>,
        matrix: &Matrix<'_, 'tcx>, // used to compute missing ctors
        ctor: &Constructor<'tcx>,
    ) -> Self {
        match self {
            NoWitnesses { .. } => self,
            WithWitnesses(ref witnesses) if witnesses.is_empty() => self,
            WithWitnesses(witnesses) => {
                let new_witnesses = if let Constructor::Missing { .. } = ctor {
                    let mut missing = ConstructorSet::for_ty(pcx.cx, pcx.ty)
                        .compute_missing(pcx, matrix.heads().map(DeconstructedPat::ctor));
                    if missing.iter().any(|c| c.is_non_exhaustive()) {
                        // We only report `_` here; listing other constructors would be redundant.
                        missing = vec![Constructor::NonExhaustive];
                    }

                    // We got the special `Missing` constructor, so each of the missing constructors
                    // gives a new pattern that is not caught by the match.
                    // We construct for each missing constructor a version of this constructor with
                    // wildcards for fields, i.e. that matches everything that can be built with it.
                    // For example, if `ctor` is a `Constructor::Variant` for `Option::Some`, we get
                    // the pattern `Some(_)`.
                    let new_patterns: Vec<WitnessPat<'_>> = missing
                        .into_iter()
                        .map(|missing_ctor| WitnessPat::wild_from_ctor(pcx, missing_ctor.clone()))
                        .collect();

                    witnesses
                        .into_iter()
                        .flat_map(|witness| {
                            new_patterns.iter().map(move |pat| {
                                let mut stack = witness.clone();
                                stack.0.push(pat.clone());
                                stack
                            })
                        })
                        .collect()
                } else {
                    witnesses
                        .into_iter()
                        .map(|witness| witness.apply_constructor(pcx, &ctor))
                        .collect()
                };
                WithWitnesses(new_witnesses)
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ArmType {
    FakeExtraWildcard,
    RealArm,
}

/// A witness-tuple of non-exhaustiveness for error reporting, represented as a list of patterns (in
/// reverse order of construction) with wildcards inside to represent elements that can take any
/// inhabitant of the type as a value.
///
/// This mirrors `PatStack`: they function similarly, except `PatStack` contains user patterns we
/// are inspecting, and `WitnessStack` contains witnesses we are constructing.
/// FIXME(Nadrieril): use the same order of patterns for both
///
/// A `WitnessStack` should have the same types and length as the `PatStacks` we are inspecting
/// (except we store the patterns in reverse order). Because Rust `match` is always against a single
/// pattern, at the end the stack will have length 1. In the middle of the algorithm, it can contain
/// multiple patterns.
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
/// We'll perform the following steps (among others):
/// - Start with a matrix representing the match
///     `PatStack(vec![Pair(None, _)])`
///     `PatStack(vec![Pair(_, false)])`
/// - Specialize with `Pair`
///     `PatStack(vec![None, _])`
///     `PatStack(vec![_, false])`
/// - Specialize with `Some`
///     `PatStack(vec![_, false])`
/// - Specialize with `_`
///     `PatStack(vec![false])`
/// - Specialize with `true`
///     // no patstacks left
/// - This is a non-exhaustive match: we have the empty witness stack as a witness.
///     `WitnessStack(vec![])`
/// - Apply `true`
///     `WitnessStack(vec![true])`
/// - Apply `_`
///     `WitnessStack(vec![true, _])`
/// - Apply `Some`
///     `WitnessStack(vec![true, Some(_)])`
/// - Apply `Pair`
///     `WitnessStack(vec![Pair(Some(_), true)])`
///
/// The final `Pair(Some(_), true)` is then the resulting witness.
#[derive(Debug, Clone)]
pub(crate) struct WitnessStack<'tcx>(Vec<WitnessPat<'tcx>>);

impl<'tcx> WitnessStack<'tcx> {
    /// Asserts that the witness contains a single pattern, and returns it.
    fn single_pattern(self) -> WitnessPat<'tcx> {
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
    fn apply_constructor(mut self, pcx: &PatCtxt<'_, '_, 'tcx>, ctor: &Constructor<'tcx>) -> Self {
        let pat = {
            let len = self.0.len();
            let arity = ctor.arity(pcx);
            let fields = self.0.drain((len - arity)..).rev().collect();
            WitnessPat::new(ctor.clone(), fields, pcx.ty)
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
#[instrument(level = "debug", skip(cx, matrix, lint_root), ret)]
fn is_useful<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    matrix: &Matrix<'p, 'tcx>,
    v: &PatStack<'p, 'tcx>,
    witness_preference: ArmType,
    lint_root: HirId,
    is_under_guard: bool,
    is_top_level: bool,
) -> Usefulness<'tcx> {
    debug!(?matrix, ?v);
    let Matrix { patterns: rows, .. } = matrix;

    // The base case. We are pattern-matching on () and the return value is
    // based on whether our matrix has a row or not.
    // NOTE: This could potentially be optimized by checking rows.is_empty()
    // first and then, if v is non-empty, the return value is based on whether
    // the type of the tuple we're checking is inhabited or not.
    if v.is_empty() {
        let ret = if rows.is_empty() {
            Usefulness::new_useful(witness_preference)
        } else {
            Usefulness::new_not_useful(witness_preference)
        };
        debug!(?ret);
        return ret;
    }

    debug_assert!(rows.iter().all(|r| r.len() == v.len()));

    // If the first pattern is an or-pattern, expand it.
    let mut ret = Usefulness::new_not_useful(witness_preference);
    if v.head().is_or_pat() {
        debug!("expanding or-pattern");
        // We try each or-pattern branch in turn.
        let mut matrix = matrix.clone();
        for v in v.expand_or_pat() {
            debug!(?v);
            let usefulness = ensure_sufficient_stack(|| {
                is_useful(cx, &matrix, &v, witness_preference, lint_root, is_under_guard, false)
            });
            debug!(?usefulness);
            ret.extend(usefulness);
            // If pattern has a guard don't add it to the matrix.
            if !is_under_guard {
                // We push the already-seen patterns into the matrix in order to detect redundant
                // branches like `Some(_) | Some(0)`.
                matrix.push(v);
            }
        }
    } else {
        let mut ty = v.head().ty();

        // Opaque types can't get destructured/split, but the patterns can
        // actually hint at hidden types, so we use the patterns' types instead.
        if let ty::Alias(ty::Opaque, ..) = ty.kind() {
            if let Some(row) = rows.first() {
                ty = row.head().ty();
            }
        }
        debug!("v.head: {:?}, v.span: {:?}", v.head(), v.head().span());
        let pcx = &PatCtxt { cx, ty, span: v.head().span(), is_top_level };

        let v_ctor = v.head().ctor();
        debug!(?v_ctor);
        if let Constructor::IntRange(ctor_range) = &v_ctor {
            // Lint on likely incorrect range patterns (#63987)
            ctor_range.lint_overlapping_range_endpoints(
                pcx,
                matrix.heads(),
                matrix.column_count().unwrap_or(0),
                lint_root,
            )
        }
        // We split the head constructor of `v`.
        let split_ctors = v_ctor.split(pcx, matrix.heads().map(DeconstructedPat::ctor));
        // For each constructor, we compute whether there's a value that starts with it that would
        // witness the usefulness of `v`.
        let start_matrix = &matrix;
        for ctor in split_ctors {
            debug!("specialize({:?})", ctor);
            // We cache the result of `Fields::wildcards` because it is used a lot.
            let spec_matrix = start_matrix.specialize_constructor(pcx, &ctor);
            let v = v.pop_head_constructor(pcx, &ctor);
            let usefulness = ensure_sufficient_stack(|| {
                is_useful(
                    cx,
                    &spec_matrix,
                    &v,
                    witness_preference,
                    lint_root,
                    is_under_guard,
                    false,
                )
            });
            let usefulness = usefulness.apply_constructor(pcx, start_matrix, &ctor);
            ret.extend(usefulness);
        }
    }

    if ret.is_useful() {
        v.head().set_reachable();
    }

    ret
}

/// Traverse the patterns to collect any variants of a non_exhaustive enum that fail to be mentioned
/// in a given column. This traverses patterns column-by-column, where a column is the intuitive
/// notion of "subpatterns that inspect the same subvalue".
/// Despite similarities with `is_useful`, this traversal is different. Notably this is linear in the
/// depth of patterns, whereas `is_useful` is worst-case exponential (exhaustiveness is NP-complete).
fn collect_nonexhaustive_missing_variants<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    column: &[&DeconstructedPat<'p, 'tcx>],
) -> Vec<WitnessPat<'tcx>> {
    if column.is_empty() {
        return Vec::new();
    }
    let ty = column[0].ty();
    let pcx = &PatCtxt { cx, ty, span: DUMMY_SP, is_top_level: false };

    let set = ConstructorSet::for_ty(pcx.cx, pcx.ty).split(pcx, column.iter().map(|p| p.ctor()));
    if set.present.is_empty() {
        // We can't consistently handle the case where no constructors are present (since this would
        // require digging deep through any type in case there's a non_exhaustive enum somewhere),
        // so for consistency we refuse to handle the top-level case, where we could handle it.
        return vec![];
    }

    let mut witnesses = Vec::new();
    if cx.is_foreign_non_exhaustive_enum(ty) {
        witnesses.extend(
            set.missing
                .into_iter()
                // This will list missing visible variants.
                .filter(|c| !matches!(c, Constructor::Hidden | Constructor::NonExhaustive))
                .map(|missing_ctor| WitnessPat::wild_from_ctor(pcx, missing_ctor)),
        )
    }

    // Recurse into the fields.
    for ctor in set.present {
        let arity = ctor.arity(pcx);
        if arity == 0 {
            continue;
        }

        // We specialize the column by `ctor`. This gives us `arity`-many columns of patterns. These
        // columns may have different lengths in the presence of or-patterns (this is why we can't
        // reuse `Matrix`).
        let mut specialized_columns: Vec<Vec<_>> = (0..arity).map(|_| Vec::new()).collect();
        let relevant_patterns = column.iter().filter(|pat| ctor.is_covered_by(pcx, pat.ctor()));
        for pat in relevant_patterns {
            let specialized = pat.specialize(pcx, &ctor);
            for (subpat, sub_column) in specialized.iter().zip(&mut specialized_columns) {
                if subpat.is_or_pat() {
                    sub_column.extend(subpat.iter_fields())
                } else {
                    sub_column.push(subpat)
                }
            }
        }
        debug_assert!(
            !specialized_columns[0].is_empty(),
            "ctor {ctor:?} was listed as present but isn't"
        );

        let wild_pat = WitnessPat::wild_from_ctor(pcx, ctor);
        for (i, col_i) in specialized_columns.iter().enumerate() {
            // Compute witnesses for each column.
            let wits_for_col_i = collect_nonexhaustive_missing_variants(cx, col_i.as_slice());
            // For each witness, we build a new pattern in the shape of `ctor(_, _, wit, _, _)`,
            // adding enough wildcards to match `arity`.
            for wit in wits_for_col_i {
                let mut pat = wild_pat.clone();
                pat.fields[i] = wit;
                witnesses.push(pat);
            }
        }
    }
    witnesses
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
    pub(crate) non_exhaustiveness_witnesses: Vec<WitnessPat<'tcx>>,
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
    let mut matrix = Matrix::empty();
    let arm_usefulness: Vec<_> = arms
        .iter()
        .copied()
        .map(|arm| {
            debug!(?arm);
            let v = PatStack::from_pattern(arm.pat);
            is_useful(cx, &matrix, &v, RealArm, arm.hir_id, arm.has_guard, true);
            if !arm.has_guard {
                matrix.push(v);
            }
            let reachability = if arm.pat.is_reachable() {
                Reachability::Reachable(arm.pat.unreachable_spans())
            } else {
                Reachability::Unreachable
            };
            (arm, reachability)
        })
        .collect();

    let wild_pattern = cx.pattern_arena.alloc(DeconstructedPat::wildcard(scrut_ty, DUMMY_SP));
    let v = PatStack::from_pattern(wild_pattern);
    let usefulness = is_useful(cx, &matrix, &v, FakeExtraWildcard, lint_root, false, true);
    let non_exhaustiveness_witnesses: Vec<_> = match usefulness {
        WithWitnesses(pats) => pats.into_iter().map(|w| w.single_pattern()).collect(),
        NoWitnesses { .. } => bug!(),
    };

    // Run the non_exhaustive_omitted_patterns lint. Only run on refutable patterns to avoid hitting
    // `if let`s. Only run if the match is exhaustive otherwise the error is redundant.
    if cx.refutable
        && non_exhaustiveness_witnesses.is_empty()
        && !matches!(
            cx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, lint_root).0,
            rustc_session::lint::Level::Allow
        )
    {
        let pat_column = arms.iter().flat_map(|arm| arm.pat.flatten_or_pat()).collect::<Vec<_>>();
        let witnesses = collect_nonexhaustive_missing_variants(cx, &pat_column);

        if !witnesses.is_empty() {
            // Report that a match of a `non_exhaustive` enum marked with `non_exhaustive_omitted_patterns`
            // is not exhaustive enough.
            //
            // NB: The partner lint for structs lives in `compiler/rustc_hir_analysis/src/check/pat.rs`.
            cx.tcx.emit_spanned_lint(
                NON_EXHAUSTIVE_OMITTED_PATTERNS,
                lint_root,
                scrut_span,
                NonExhaustiveOmittedPattern {
                    scrut_ty,
                    uncovered: Uncovered::new(scrut_span, cx, witnesses),
                },
            );
        }
    }

    UsefulnessReport { arm_usefulness, non_exhaustiveness_witnesses }
}
