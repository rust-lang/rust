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
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::OnceCell;

use rustc_arena::TypedArena;
use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

use smallvec::{smallvec, SmallVec};
use std::fmt;
use std::iter::IntoIterator;
use std::ops::Index;

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

impl<'a, 'p, 'tcx> fmt::Debug for PatCtxt<'a, 'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PatCtxt").field("ty", &self.ty).finish()
    }
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

    fn is_or_pat(&self) -> bool {
        matches!(*self.kind, PatKind::Or { .. })
    }

    /// Recursively expand this pattern into its subpatterns. Only useful for or-patterns.
    fn expand_or_pat(&self) -> Vec<&Self> {
        fn expand<'p, 'tcx>(pat: &'p Pat<'tcx>, vec: &mut Vec<&'p Pat<'tcx>>) {
            if let PatKind::Or { pats } = pat.kind.as_ref() {
                for pat in pats {
                    expand(pat, vec);
                }
            } else {
                vec.push(pat)
            }
        }

        if self.is_or_pat() {
            let mut pats = Vec::new();
            expand(self, &mut pats);
            pats
        } else {
            vec![self]
        }
    }
}

/// Behaves like `Vec<Vec<T>>`, but uses a single allocation. Only the last vector can be
/// extended/shrunk.
#[derive(Debug, Clone)]
struct VecVec<T> {
    data: Vec<T>,
    /// Indices into `data`. Each index is the start of a contained slice.
    indices: Vec<usize>,
    /// Remember the last index because we use it often.
    last_index_cache: Option<usize>,
}

impl<T> VecVec<T> {
    fn new() -> Self {
        VecVec { data: Vec::new(), indices: vec![], last_index_cache: None }
    }

    #[inline]
    fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn last(&self) -> Option<&[T]> {
        if let Some(i) = self.last_index_cache {
            Some(&self.data[i..self.data.len()])
        } else {
            None
        }
    }

    /// Pushes a new `Vec` at the end.
    fn push_slice(&mut self, items: &[T])
    where
        T: Clone,
    {
        self.last_index_cache = Some(self.data.len());
        self.indices.push(self.data.len());
        self.data.extend_from_slice(items);
    }

    /// Pushes a new `Vec` at the end.
    fn push_vec(&mut self, mut items: Vec<T>) {
        self.last_index_cache = Some(self.data.len());
        self.indices.push(self.data.len());
        self.data.append(&mut items);
    }

    fn pop_iter<'a>(&'a mut self) -> Option<impl Iterator<Item = T> + 'a> {
        let l = self.indices.pop()?;
        self.last_index_cache = self.indices.last().copied();
        Some(self.data.drain(l..))
    }

    fn pop_vec(&mut self) -> Option<Vec<T>> {
        let l = self.indices.pop()?;
        self.last_index_cache = self.indices.last().copied();
        Some(self.data.split_off(l))
    }
}

impl<T> Index<usize> for VecVec<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len());
        let lo = self.indices[index];
        let hi = *self.indices.get(index + 1).unwrap_or(&self.data.len());
        &self.data.index(lo..hi)
    }
}

impl<T> Default for VecVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A row of a matrix. Rows of len 1 are very common, which is why `SmallVec[_; 2]`
/// works well.
#[derive(Clone)]
struct PatStack<'p, 'tcx> {
    pats: SmallVec<[&'p Pat<'tcx>; 2]>,
    row_data: RowData<'p, 'tcx>,
    /// Cache for the constructor of the head
    head_ctor: OnceCell<Constructor<'tcx>>,
}

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    fn from_pattern(pat: &'p Pat<'tcx>, row_data: RowData<'p, 'tcx>) -> Self {
        Self::from_vec(smallvec![pat], row_data)
    }

    fn from_vec(vec: SmallVec<[&'p Pat<'tcx>; 2]>, row_data: RowData<'p, 'tcx>) -> Self {
        PatStack { pats: vec, row_data, head_ctor: OnceCell::new() }
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
        PatStack::from_vec(new_fields, self.row_data.clone())
    }
}

/// Pretty-printing for matrix row.
impl<'p, 'tcx> fmt::Debug for PatStack<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for pat in self.iter() {
            write!(f, " {} +", pat)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum Origin<'p, 'tcx> {
    Pushed,
    Specialized { col: usize, row: usize, ctor_arity: usize },
    OrPat { col: usize, row: usize, alt_id: usize, alt_count: usize, pat: &'p Pat<'tcx> },
}

#[derive(Clone)]
struct MatrixEntry<'p, 'tcx> {
    pat: &'p Pat<'tcx>,
    /// The current row continues in the following column at index `next_row_id`. This is needed
    /// because or-patterns and filtering disalign the columns. In the last column this is `0`.
    next_row_id: usize,
    /// Cache the constructor for this pattern.
    ctor: OnceCell<Constructor<'tcx>>,
    /// Whether this pattern is under a guard and thus should be ignored by patterns below.
    is_under_guard: bool,
}

#[derive(Clone, Copy)]
struct SelectedRow<'p, 'tcx> {
    id: usize,
    /// Where the pointed row item came from.
    origin: Origin<'p, 'tcx>,
}

#[derive(Clone, Debug)]
struct RowData<'p, 'tcx> {
    is_under_guard: bool,
    hir_id: HirId,
    usefulness: Usefulness<'p, 'tcx>,
}

impl<'p, 'tcx> MatrixEntry<'p, 'tcx> {
    fn head_ctor<'a>(&'a self, cx: &MatchCheckCtxt<'p, 'tcx>) -> &'a Constructor<'tcx> {
        self.ctor.get_or_init(|| Constructor::from_pat(cx, self.pat))
    }
}

#[derive(Clone, Copy)]
enum UndoKind {
    Specialize { ctor_arity: usize },
    OrPatsExp { any_or_pats: bool },
}

/// A 2D matrix.
#[derive(Clone, Default)]
struct Matrix<'p, 'tcx> {
    /// Stores the patterns by column. The leftmost column is stored last.
    columns: Vec<Vec<MatrixEntry<'p, 'tcx>>>,
    /// In order to not rebuild the matrix every time, we may keep patterns around even when their
    /// rows have been filtered out. The actual contents of the first column are those indices in
    /// `selected_rows`, and the contents in subsequent columns can be found by following
    /// `next_row_id`s.
    selected_rows: Vec<SelectedRow<'p, 'tcx>>,
    /// Stores data for each of the original rows. The rightmost column's `next_row_id` points into
    /// this.
    row_data: Vec<RowData<'p, 'tcx>>,
    /// Stores the history of operations done on this matrix, so that we can undo them in-place.
    history: Vec<UndoKind>,
    last_col_history: VecVec<MatrixEntry<'p, 'tcx>>,
    selected_rows_history: VecVec<SelectedRow<'p, 'tcx>>,
}

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    /// A matrix is created from a list of match arms.
    fn new(arms: &[MatchArm<'p, 'tcx>]) -> Self {
        let mut matrix = Matrix::default();
        let mut column = Vec::new();
        for (id, arm) in arms.iter().enumerate() {
            matrix.selected_rows.push(SelectedRow { id, origin: Origin::Pushed });
            matrix.row_data.push(RowData {
                is_under_guard: arm.has_guard,
                hir_id: arm.hir_id,
                usefulness: Usefulness::new_not_useful(LeaveOutWitness),
            });
            column.push(MatrixEntry {
                pat: arm.pat,
                next_row_id: id,
                ctor: OnceCell::new(),
                is_under_guard: arm.has_guard,
            });
        }
        matrix.columns.push(column);
        matrix
    }

    /// Number of columns of this matrix.
    fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns the type of the first column, if any.
    fn ty_of_last_col(&self) -> Option<Ty<'tcx>> {
        let last_col = self.columns.last()?;
        let row = self.selected_rows.first()?;
        Some(last_col[row.id].pat.ty)
    }

    /// Iterate over the last column; panics if no columns.
    fn last_col<'a>(&'a self) -> impl Iterator<Item = &'a MatrixEntry<'p, 'tcx>> + Clone {
        let last_col = self.columns.last().unwrap();
        self.selected_rows.iter().map(move |row| &last_col[row.id])
    }

    /// Iterate over the first constructor of each row.
    fn head_ctors<'a>(
        &'a self,
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
    ) -> impl Iterator<Item = &'a Constructor<'tcx>> + Captures<'p> + Clone {
        self.head_ctors_and_spans(cx).map(|(ctor, _)| ctor)
    }

    /// Iterate over the first constructor and the corresponding span of each row.
    fn head_ctors_and_spans<'a>(
        &'a self,
        cx: &'a MatchCheckCtxt<'p, 'tcx>,
    ) -> impl Iterator<Item = (&'a Constructor<'tcx>, Span)> + Captures<'p> + Clone {
        self.last_col().map(move |entry| {
            // // TODO: breaks diagnostics
            // if entry.is_under_guard {
            //     None
            // } else {
            //     Some((entry.head_ctor(cx), entry.pat.span))
            // }
            (entry.head_ctor(cx), entry.pat.span)
        })
    }

    /// Iterate over the entries of the selected row.
    fn row<'a>(&'a self, mut row_id: usize) -> impl Iterator<Item = &'a MatrixEntry<'p, 'tcx>> {
        // Starting from the last column, follow the `next_row_id`s to explore the row.
        (0..self.columns.len()).rev().map(move |col_id| {
            let entry = &self.columns[col_id][row_id];
            row_id = entry.next_row_id;
            entry
        })
    }

    /// Stores the last column and the currently selected rows into history. The last column is
    /// removed from `self.columns`. `selected_rows` is kept as is.
    fn save_last_col(&mut self) {
        self.last_col_history.push_vec(self.columns.pop().unwrap());
        self.selected_rows_history.push_slice(&self.selected_rows);
    }
    /// Restores the last column and the selected rows from history.
    fn restore_last_col(&mut self) {
        self.columns.push(self.last_col_history.pop_vec().unwrap());
        self.selected_rows.clear();
        self.selected_rows.extend_from_slice(self.selected_rows_history.last().unwrap());
        self.selected_rows_history.pop_iter();
    }

    /// This computes `S(constructor, self)`. See top of the file for explanations.
    fn specialize<'a>(
        &'a mut self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
    ) {
        assert!(self.column_count() >= 1);

        // Remove and save the last column and its selected rows.
        self.save_last_col();
        let old_last_col = self.last_col_history.last().unwrap();
        let old_selected_rows = self.selected_rows_history.last().unwrap();
        let sel_rows_origin = self.selected_rows_history.len() - 1;

        // Keep only rows that match this ctor.
        self.selected_rows.clear();
        self.selected_rows.extend(
            old_selected_rows
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, row)| ctor.is_covered_by(pcx, old_last_col[row.id].head_ctor(pcx.cx)))
                .map(|(sel_row_id, row)| SelectedRow {
                    id: row.id,
                    origin: Origin::Specialized {
                        col: sel_rows_origin,
                        row: sel_row_id,
                        ctor_arity: ctor_wild_subpatterns.len(),
                    },
                }),
        );

        // Prepare new columns for the arguments of the patterns we are specializing.
        let old_col_count = self.column_count();
        self.columns.reserve(ctor_wild_subpatterns.len());

        // Add new columns filled with wildcards.
        let new_fields = ctor_wild_subpatterns.clone().into_patterns();
        // Note: the fields are in the natural left-to-right order but the columns are not.
        // We need to start from the last field to get the columns right.
        for &pat in new_fields.iter().rev() {
            let mut col = Vec::with_capacity(self.selected_rows.len());
            for _ in 0..self.selected_rows.len() {
                // dummy entry
                col.push(MatrixEntry {
                    pat,
                    next_row_id: 0,
                    ctor: OnceCell::new(),
                    is_under_guard: false,
                });
            }
            self.columns.push(col);
        }
        let new_columns = &mut self.columns[old_col_count..];

        // Fill the columns with the patterns we want.
        for (new_id, row) in self.selected_rows.iter_mut().enumerate() {
            let head_entry = &old_last_col[row.id];
            // Extract fields from the arguments of the head of each row.
            let new_fields = ctor_wild_subpatterns
                .replace_with_pattern_arguments(head_entry.pat)
                .into_patterns();
            // The rest of the row starts at `row_id`.
            row.id = head_entry.next_row_id;
            // Note: the fields are in the natural left-to-right order but the columns are not.
            // We need to start from the last field to get the `next_row_id`s right.
            for (col, &pat) in new_columns.iter_mut().zip(new_fields.iter().rev()) {
                col[new_id].pat = pat;
                col[new_id].next_row_id = row.id;
                col[new_id].is_under_guard = head_entry.is_under_guard;
                // Make `row_id` point to the entry just added.
                row.id = new_id;
            }
        }

        self.history.push(UndoKind::Specialize { ctor_arity: ctor_wild_subpatterns.len() });
    }

    /// Expands or-patterns in the last column of the matrix. Does nothing if the first column is
    /// missing or has no or-patterns.
    fn expand_or_patterns(&mut self) {
        let any_or_pats = !self.columns.is_empty() && self.last_col().any(|e| e.pat.is_or_pat());
        self.history.push(UndoKind::OrPatsExp { any_or_pats });
        if !any_or_pats {
            return;
        }

        self.save_last_col();
        self.selected_rows.clear();
        let old_last_col = self.last_col_history.last().unwrap();
        let old_selected_rows = self.selected_rows_history.last().unwrap();
        let mut new_last_col = Vec::new();
        for (sel_row_id, row) in old_selected_rows.iter().enumerate() {
            let entry = &old_last_col[row.id];
            let orig_pat = entry.pat;
            if orig_pat.is_or_pat() {
                let subpats = entry.pat.expand_or_pat();
                let alt_count = subpats.len();
                for (alt_id, pat) in subpats.into_iter().enumerate() {
                    // All subpatterns point to the same row in the next column.
                    let entry = MatrixEntry {
                        pat,
                        next_row_id: entry.next_row_id,
                        ctor: OnceCell::new(),
                        is_under_guard: entry.is_under_guard,
                    };
                    new_last_col.push(entry);
                    self.selected_rows.push(SelectedRow {
                        id: new_last_col.len() - 1,
                        origin: Origin::OrPat {
                            col: self.selected_rows_history.len() - 1,
                            row: sel_row_id,
                            alt_id,
                            alt_count,
                            pat: orig_pat,
                        },
                        any_or_pats: true,
                    });
                }
            } else {
                new_last_col.push(entry.clone());
                self.selected_rows.push(SelectedRow { id: new_last_col.len() - 1, ..*row });
            }
        }
        self.columns.push(new_last_col);
    }

    /// Undo either the last specialization or the last manual or-pattern expansion.
    fn undo(&mut self) -> Option<UndoKind> {
        let undo = self.history.pop()?;
        match undo {
            UndoKind::Specialize { ctor_arity, .. } => {
                for _ in 0..ctor_arity {
                    self.columns.pop().unwrap();
                }
                self.restore_last_col();
            }
            UndoKind::OrPatsExp { any_or_pats } => {
                if any_or_pats {
                    self.columns.pop().unwrap();
                    self.restore_last_col();
                }
            }
        };
        Some(undo)
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

        let pretty_printed_matrix: Vec<Vec<String>> = self
            .selected_rows
            .iter()
            .map(|row| self.row(row.id).map(|e| format!("{}", e.pat)).collect())
            .collect();

        let column_count = self.column_count();
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

/// Given a pattern or a pattern-stack, this struct captures a set of its subpattern branches. We
/// use that to track unreachable sub-patterns arising from or-patterns. In the absence of
/// or-patterns this will always be either `Empty` or `Full`.
/// We support a limited set of operations, so not all possible sets of subpatterns can be
/// represented. That's ok, we only want the ones that make sense to capture unreachable
/// subpatterns.
/// What we're trying to do is illustrated by this:
/// ```
/// match (true, true) {
///     (true, true) => {}
///     (true | false, true | false) => {}
/// }
/// ```
/// When we try the alternatives of the first or-pattern, the last `true` is unreachable in the
/// first alternative but no the other. So we don't want to report it as unreachable. Therefore we
/// intersect sets of unreachable patterns coming from different alternatives in order to figure
/// out which subpatterns are overall unreachable.
#[derive(Debug, Clone)]
enum SubPatSet<'p, 'tcx> {
    /// The set containing the full pattern.
    Full,
    /// The empty set.
    Empty,
    /// If the pattern is a pattern with a constructor or a pattern-stack, we store a set for each
    /// of its subpatterns. Missing entries in the map are implicitly empty.
    Seq { subpats: FxHashMap<usize, SubPatSet<'p, 'tcx>> },
    /// If the pattern is an or-pattern, we store a set for each of its alternatives. Missing
    /// entries in the map are implicitly full. Note: we always flatten nested or-patterns.
    Alt {
        subpats: FxHashMap<usize, SubPatSet<'p, 'tcx>>,
        /// Counts the total number of alternatives in the pattern
        alt_count: usize,
        /// We keep the pattern around to retrieve spans.
        pat: &'p Pat<'tcx>,
    },
}

impl<'p, 'tcx> SubPatSet<'p, 'tcx> {
    fn empty() -> Self {
        SubPatSet::Empty
    }
    fn full() -> Self {
        SubPatSet::Full
    }

    fn is_full(&self) -> bool {
        match self {
            SubPatSet::Full => true,
            SubPatSet::Empty => false,
            // If any subpattern in a sequence is unreachable, the whole pattern is unreachable.
            SubPatSet::Seq { subpats } => subpats.values().any(|set| set.is_full()),
            SubPatSet::Alt { subpats, .. } => subpats.values().all(|set| set.is_full()),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            SubPatSet::Full => false,
            SubPatSet::Empty => true,
            SubPatSet::Seq { subpats } => subpats.values().all(|sub_set| sub_set.is_empty()),
            SubPatSet::Alt { subpats, alt_count, .. } => {
                subpats.len() == *alt_count && subpats.values().all(|set| set.is_empty())
            }
        }
    }

    /// Intersect `self` with `other`, mutating `self`.
    fn intersect(&mut self, other: Self) {
        use SubPatSet::*;
        // Intersecting with empty stays empty; intersecting with full changes nothing.
        if self.is_empty() || other.is_full() {
            return;
        } else if self.is_full() {
            *self = other;
            return;
        } else if other.is_empty() {
            *self = Empty;
            return;
        }

        match (&mut *self, other) {
            (Seq { subpats: s_set }, Seq { subpats: mut o_set }) => {
                s_set.retain(|i, s_sub_set| {
                    // Missing entries count as empty.
                    let o_sub_set = o_set.remove(&i).unwrap_or(Empty);
                    s_sub_set.intersect(o_sub_set);
                    // We drop empty entries.
                    !s_sub_set.is_empty()
                });
                // Everything left in `o_set` is missing from `s_set`, i.e. counts as empty. Since
                // intersecting with empty returns empty, we can drop those entries.
            }
            (Alt { subpats: s_set, .. }, Alt { subpats: mut o_set, .. }) => {
                s_set.retain(|i, s_sub_set| {
                    // Missing entries count as full.
                    let o_sub_set = o_set.remove(&i).unwrap_or(Full);
                    s_sub_set.intersect(o_sub_set);
                    // We drop full entries.
                    !s_sub_set.is_full()
                });
                // Everything left in `o_set` is missing from `s_set`, i.e. counts as full. Since
                // intersecting with full changes nothing, we can take those entries as is.
                s_set.extend(o_set);
            }
            _ => bug!(),
        }

        if self.is_empty() {
            *self = Empty;
        }
    }

    /// Returns a list of the spans of the unreachable subpatterns. If `self` is full we return
    /// `None`.
    fn to_spans(&self) -> Option<Vec<Span>> {
        /// Panics if `set.is_full()`.
        fn fill_spans(set: &SubPatSet<'_, '_>, spans: &mut Vec<Span>) {
            match set {
                SubPatSet::Full => bug!(),
                SubPatSet::Empty => {}
                SubPatSet::Seq { subpats } => {
                    for (_, sub_set) in subpats {
                        fill_spans(sub_set, spans);
                    }
                }
                SubPatSet::Alt { subpats, pat, alt_count, .. } => {
                    let expanded = pat.expand_or_pat();
                    for i in 0..*alt_count {
                        let sub_set = subpats.get(&i).unwrap_or(&SubPatSet::Full);
                        if sub_set.is_full() {
                            spans.push(expanded[i].span);
                        } else {
                            fill_spans(sub_set, spans);
                        }
                    }
                }
            }
        }

        if self.is_full() {
            return None;
        }
        if self.is_empty() {
            return Some(Vec::new());
        }
        let mut spans = Vec::new();
        fill_spans(self, &mut spans);
        Some(spans)
    }

    /// When `self` refers to a patstack that was obtained from specialization, after running
    /// `unspecialize` it will refer to the original patstack before specialization.
    fn unspecialize(self, arity: usize) -> Self {
        use SubPatSet::*;
        match self {
            Full => Full,
            Empty => Empty,
            Seq { subpats } => {
                // We gather the first `arity` subpatterns together and shift the remaining ones.
                let mut new_subpats = FxHashMap::default();
                let mut new_subpats_first_col = FxHashMap::default();
                for (i, sub_set) in subpats {
                    if i < arity {
                        // The first `arity` indices are now part of the pattern in the first
                        // column.
                        new_subpats_first_col.insert(i, sub_set);
                    } else {
                        // Indices after `arity` are simply shifted
                        new_subpats.insert(i - arity + 1, sub_set);
                    }
                }
                if !new_subpats_first_col.is_empty() {
                    new_subpats.insert(0, Seq { subpats: new_subpats_first_col });
                }
                Seq { subpats: new_subpats }
            }
            Alt { .. } => bug!(), // `self` is a patstack
        }
    }

    /// When `self` refers to a patstack that was obtained from splitting an or-pattern, after
    /// running `unspecialize` it will refer to the original patstack before splitting.
    ///
    /// This case is subtle. Consider:
    /// ```
    /// match Some(true) {
    ///     Some(true) => {}
    ///     None | Some(true | false) => {}
    /// }
    /// ```
    /// Imagine we naively preserved the sets of unreachable subpatterns. Here `None` would return
    /// the empty set and `Some(true | false)` would return the set containing `true`. Intersecting
    /// those two would return the empty set, so we'd miss that the last `true` is unreachable.
    /// To fix that, when specializing a given alternative of an or-pattern, we consider all other
    /// alternatives as unreachable. That way, intersecting the results will not unduly discard
    /// unreachable subpatterns coming from the other alternatives. This is what this function does
    /// (remember that missing entries in the `Alt` case count as full; in other words alternatives
    /// other than `alt_id` count as unreachable).
    fn unsplit_or_pat(mut self, alt_id: usize, alt_count: usize, pat: &'p Pat<'tcx>) -> Self {
        use SubPatSet::*;
        if self.is_full() {
            return Full;
        }

        let set_first_col = match &mut self {
            Empty => Empty,
            Seq { subpats } => subpats.remove(&0).unwrap_or(Empty),
            Full => unreachable!(),
            Alt { .. } => bug!(), // `self` is a patstack
        };
        let mut subpats_first_col = FxHashMap::default();
        subpats_first_col.insert(alt_id, set_first_col);
        let set_first_col = Alt { subpats: subpats_first_col, pat, alt_count };

        let mut subpats = match self {
            Empty => FxHashMap::default(),
            Seq { subpats } => subpats,
            Full => unreachable!(),
            Alt { .. } => bug!(), // `self` is a patstack
        };
        subpats.insert(0, set_first_col);
        Seq { subpats }
    }
}

#[derive(Clone, Debug)]
enum Usefulness<'p, 'tcx> {
    /// Carries a set of subpatterns that have been found to be unreachable. If full, this
    /// indicates the whole pattern is unreachable. If not, this indicates that the pattern is
    /// reachable but has some unreachable sub-patterns (due to or-patterns). In the absence of
    /// or-patterns, this is either `Empty` or `Full`.
    NoWitnesses(SubPatSet<'p, 'tcx>),
    /// Carries a list of witnesses of non-exhaustiveness. If empty, indicates that the whole
    /// pattern is unreachable.
    WithWitnesses(Vec<Witness<'tcx>>),
}

impl<'p, 'tcx> Usefulness<'p, 'tcx> {
    fn new_useful(preference: WitnessPreference) -> Self {
        match preference {
            ConstructWitness => WithWitnesses(vec![Witness(vec![])]),
            LeaveOutWitness => NoWitnesses(SubPatSet::empty()),
        }
    }
    fn new_not_useful(preference: WitnessPreference) -> Self {
        match preference {
            ConstructWitness => WithWitnesses(vec![]),
            LeaveOutWitness => NoWitnesses(SubPatSet::full()),
        }
    }

    /// Combine usefulnesses from two branches. This is an associative operation.
    fn extend(&mut self, other: Self) {
        match (&mut *self, other) {
            (WithWitnesses(_), WithWitnesses(o)) if o.is_empty() => {}
            (WithWitnesses(s), WithWitnesses(o)) if s.is_empty() => *self = WithWitnesses(o),
            (WithWitnesses(s), WithWitnesses(o)) => s.extend(o),
            (NoWitnesses(s), NoWitnesses(o)) => s.intersect(o),
            _ => unreachable!(),
        }
    }

    /// After calculating usefulness after a specialization, call this to recontruct a usefulness
    /// that makes sense for the matrix pre-specialization. This new usefulness can then be merged
    /// with the results of specializing with the other constructors.
    fn apply_constructor<'a>(
        self,
        pcx: PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        ctor_wild_subpatterns: &Fields<'p, 'tcx>,
        head_ctors: impl Iterator<Item = &'a Constructor<'tcx>> + Clone,
    ) -> Self
    where
        'tcx: 'a,
    {
        match self {
            WithWitnesses(witnesses) if witnesses.is_empty() => WithWitnesses(witnesses),
            WithWitnesses(witnesses) => {
                let new_witnesses = if matches!(ctor, Constructor::Missing) {
                    let mut split_wildcard = SplitWildcard::new(pcx);
                    split_wildcard.split(pcx, head_ctors);
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
                WithWitnesses(new_witnesses)
            }
            NoWitnesses(subpats) => NoWitnesses(subpats.unspecialize(ctor_wild_subpatterns.len())),
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
#[instrument(skip(cx, matrix, is_top_level))]
fn compute_matrix_usefulness<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    matrix: &mut Matrix<'p, 'tcx>,
    v: &PatStack<'p, 'tcx>,
    is_top_level: bool,
) -> Usefulness<'p, 'tcx> {
    debug!("matrix,v={:?}{:?}", matrix, v);

    // The base case. We are pattern-matching on () and the return value is
    // based on whether our matrix has a row or not.
    // NOTE: This could potentially be optimized by checking rows.is_empty()
    // first and then, if v is non-empty, the return value is based on whether
    // the type of the tuple we're checking is inhabited or not.
    if matrix.column_count() == 0 {
        let mut is_covered = false;
        for row in &matrix.selected_rows {
            if !is_covered {
                let mut set = SubPatSet::empty();
                let mut origin = row.origin;
                loop {
                    match origin {
                        Origin::Pushed => break,
                        Origin::Specialized { col, row, ctor_arity } => {
                            set = set.unspecialize(ctor_arity);
                            origin = matrix.selected_rows_history[col][row].origin;
                        }
                        Origin::OrPat { col, row, alt_id, alt_count, pat } => {
                            set = set.unsplit_or_pat(alt_id, alt_count, pat);
                            origin = matrix.selected_rows_history[col][row].origin;
                        }
                    }
                }
                debug!("row {} found useful {:?}", row.id, set);
                matrix.row_data[row.id].usefulness.extend(NoWitnesses(set));
            }
            if !matrix.row_data[row.id].is_under_guard {
                is_covered = true;
                break;
            }
        }
        let usefulness = if is_covered {
            Usefulness::new_not_useful(ConstructWitness)
        } else {
            Usefulness::new_useful(ConstructWitness)
        };
        debug!(?usefulness);
        return usefulness;
    }

    let pat = v.head();
    // FIXME(Nadrieril): Hack to work around type normalization issues (see #72476).
    let ty = matrix.ty_of_last_col().unwrap_or(pat.ty);
    let pcx = PatCtxt { cx, ty, span: pat.span, is_top_level };

    if super::deconstruct_pat::IntRange::is_integral(ty) {
        let mut seen = Vec::new();
        for entry in matrix.last_col() {
            let ctor = entry.head_ctor(cx);
            let span = entry.pat.span;
            if let Constructor::IntRange(ctor_range) = &ctor {
                let pcx = PatCtxt { cx, ty, span, is_top_level };
                // Lint on likely incorrect range patterns (#63987)
                ctor_range.lint_overlapping_range_endpoints(
                    pcx,
                    seen.iter().copied(),
                    matrix.column_count(),
                    v.row_data.hir_id,
                )
            }
            if !entry.is_under_guard {
                seen.push((ctor, span));
            }
        }
    }
    let v_ctor = v.head_ctor(cx);
    // We split the head constructor of `v`.
    let split_ctors = v_ctor.split(pcx, matrix.head_ctors(cx));
    // For each constructor, we compute whether there's a value that starts with it that would
    // witness the usefulness of `v`.
    let mut usefulness = Usefulness::new_not_useful(ConstructWitness);
    let mut any_missing = false;
    for ctor in split_ctors.into_iter() {
        debug!("specialize({:?})", ctor);
        // We cache the result of `Fields::wildcards` because it is used a lot.
        let ctor_wild_subpatterns = Fields::wildcards(pcx, &ctor);
        let v = v.pop_head_constructor(&ctor_wild_subpatterns);
        matrix.specialize(pcx, &ctor, &ctor_wild_subpatterns);
        // Expand any or-patterns present in the new last column.
        matrix.expand_or_patterns();
        let u = compute_matrix_usefulness(cx, matrix, &v, false);
        matrix.undo();
        matrix.undo();
        if !any_missing {
            // If we've seen the `Missing` constructor already, we don't further accumulate
            // witnesses.
            let u = u.apply_constructor(pcx, &ctor, &ctor_wild_subpatterns, matrix.head_ctors(cx));
            usefulness.extend(u);
        }
        any_missing = any_missing || matches!(&ctor, Constructor::Missing);
    }
    debug!(?usefulness);
    usefulness
}

/// The arm of a match expression.
#[derive(Clone, Copy)]
crate struct MatchArm<'p, 'tcx> {
    /// The pattern must have been lowered through `check_match::MatchVisitor::lower_pattern`.
    crate pat: &'p super::Pat<'tcx>,
    crate hir_id: HirId,
    crate has_guard: bool,
}

#[derive(Clone, Debug)]
crate enum Reachability {
    /// Potentially carries a set of sub-branches that have been found to be unreachable. Used only
    /// in the presence of or-patterns, otherwise it stays empty.
    Reachable(Vec<Span>),
    Unreachable,
}

/// The output of checking a match for exhaustiveness and arm reachability.
crate struct UsefulnessReport<'p, 'tcx> {
    /// For each arm of the input, whether that arm is reachable after the arms above it.
    crate arm_usefulness: Vec<(MatchArm<'p, 'tcx>, Reachability)>,
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
    let mut matrix = Matrix::new(arms);

    let wild_pattern = cx.pattern_arena.alloc(super::Pat::wildcard_from_ty(scrut_ty));
    let row_data = RowData {
        is_under_guard: false,
        hir_id: scrut_hir_id,
        usefulness: Usefulness::new_not_useful(ConstructWitness),
    };
    let v = PatStack::from_pattern(wild_pattern, row_data);
    matrix.expand_or_patterns();
    let usefulness = compute_matrix_usefulness(cx, &mut matrix, &v, true);
    let non_exhaustiveness_witnesses = match usefulness {
        WithWitnesses(pats) => pats.into_iter().map(|w| w.single_pattern()).collect(),
        NoWitnesses(_) => bug!(),
    };

    let arm_usefulness: Vec<_> = matrix
        .row_data
        .iter()
        .zip(arms.iter().copied())
        .map(|(data, arm)| {
            let reachability = match &data.usefulness {
                NoWitnesses(subpats) if subpats.is_full() => Reachability::Unreachable,
                NoWitnesses(subpats) => Reachability::Reachable(subpats.to_spans().unwrap()),
                WithWitnesses(..) => bug!(),
            };
            (arm, reachability)
        })
        .collect();

    UsefulnessReport { arm_usefulness, non_exhaustiveness_witnesses }
}
