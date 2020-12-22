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
//! match x {
//!     Some(_) => ...,
//!     None => ..., // reachable: `None` is matched by this but not the branch above
//!     Some(0) => ..., // unreachable: all the values this matches are already matched by
//!                     // `Some(_)` above
//! }
//! ```
//!
//! This is also enough to compute exhaustiveness: a match is exhaustive iff the wildcard `_`
//! pattern is _not_ useful w.r.t. the patterns in the match. The values returned by `usefulness`
//! are used to tell the user which values are missing.
//! ```rust
//! match x {
//!     Some(0) => ...,
//!     None => ...,
//!     // not exhaustive: `_` is useful because it matches `Some(1)`
//! }
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
//! The idea that powers everything that is done in this file is the following: a (matcheable)
//! value is made from a constructor applied to a number of subvalues. Examples of constructors are
//! `Some`, `None`, `(,)` (the 2-tuple constructor), `Foo {..}` (the constructor for a struct
//! `Foo`), and `2` (the constructor for the number `2`). This is natural when we think of
//! pattern-matching, and this is the basis for what follows.
//!
//! Some of the ctors listed above might feel weird: `None` and `2` don't take any arguments.
//! That's ok: those are ctors that take a list of 0 arguments; they are the simplest case of
//! ctors. We treat `2` as a ctor because `u64` and other number types behave exactly like a huge
//! `enum`, with one variant for each number. This allows us to see any matcheable value as made up
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
//! ```
//! match x {
//!     Enum::Variant1(_) => {} // `p1`
//!     Enum::Variant2(None, 0) => {} // `p2`
//!     Enum::Variant2(Some(_), 0) => {} // `q`
//! }
//! ```
//!
//! We can easily see that if our candidate value `v` starts with `Variant1` it will not match `q`.
//! If `v = Variant2(v0, v1)` however, whether or not it matches `p2` and `q` will depend on `v0`
//! and `v1`. In fact, such a `v` will be a witness of usefulness of `q` exactly when the tuple
//! `(v0, v1)` is a witness of usefulness of `q'` in the following reduced match:
//!
//! ```
//! match x {
//!     (None, 0) => {} // `p2'`
//!     (Some(_), 0) => {} // `q'`
//! }
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
//! ```
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
//! ```
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

use self::Usefulness::*;
use self::WitnessPreference::*;

use super::deconstruct_pat::{Constructor, Fields, SplitWildcard};
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
            ctor_wild_subpatterns.replace_with_pattern_arguments(self.head()).into_patterns();
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
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'p> + Clone {
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
        matrix: &Matrix<'p, 'tcx>, // used to compute missing ctors
        ctor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) -> Self {
        match self {
            UsefulWithWitness(witnesses) => {
                let new_witnesses = if matches!(ctor, Constructor::Missing) {
                    let mut split_wildcard = SplitWildcard::new(pcx);
                    split_wildcard.split(pcx, matrix.head_ctors(pcx.cx));
                    // Construct for each missing constructor a "wild" version of this
                    // constructor, that matches everything that can be built with
                    // it. For example, if `ctor` is a `Constructor::Variant` for
                    // `Option::Some`, we get the pattern `Some(_)`.
                    let new_patterns: Vec<_> = split_wildcard
                        .iter_missing(pcx)
                        .map(|missing_ctor| {
                            Fields::wildcards(pcx, missing_ctor).apply(pcx, missing_ctor)
                        })
                        .collect();
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
/// 2. Push a witness `true` against the `false`
///     `Witness(vec![true])`
/// 3. Push a witness `Some(_)` against the `None`
///     `Witness(vec![true, Some(_)])`
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
    let pcx = PatCtxt { cx, ty, span: v.head().span, is_top_level };

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
        let v_ctor = v.head_ctor(cx);
        if let Constructor::IntRange(ctor_range) = &v_ctor {
            // Lint on likely incorrect range patterns (#63987)
            ctor_range.lint_overlapping_range_endpoints(
                pcx,
                matrix.head_ctors_and_spans(cx),
                matrix.column_count().unwrap_or(0),
                hir_id,
            )
        }
        // We split the head constructor of `v`.
        let split_ctors = v_ctor.split(pcx, matrix.head_ctors(cx));
        // For each constructor, we compute whether there's a value that starts with it that would
        // witness the usefulness of `v`.
        let start_matrix = &matrix;
        let usefulnesses = split_ctors.into_iter().map(|ctor| {
            // We cache the result of `Fields::wildcards` because it is used a lot.
            let ctor_wild_subpatterns = Fields::wildcards(pcx, &ctor);
            let spec_matrix =
                start_matrix.specialize_constructor(pcx, &ctor, &ctor_wild_subpatterns);
            let v = v.pop_head_constructor(&ctor_wild_subpatterns);
            let usefulness =
                is_useful(cx, &spec_matrix, &v, witness_preference, hir_id, is_under_guard, false);
            usefulness.apply_constructor(pcx, start_matrix, &ctor, &ctor_wild_subpatterns)
        });
        Usefulness::merge(usefulnesses)
    };
    debug!("is_useful::returns({:#?}, {:#?}) = {:?}", matrix, v, ret);
    ret
}

/// The arm of a match expression.
#[derive(Clone, Copy)]
crate struct MatchArm<'p, 'tcx> {
    /// The pattern must have been lowered through `check_match::MatchVisitor::lower_pattern`.
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
/// Note: the input patterns must have been lowered through
/// `check_match::MatchVisitor::lower_pattern`.
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
