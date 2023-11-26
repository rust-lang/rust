//! # Match exhaustiveness and reachability algorithm
//!
//! This file contains the logic for exhaustiveness and reachability checking for pattern-matching.
//! Specifically, given a list of patterns in a match, we can tell whether:
//! (a) a given pattern is reachable (reachability)
//! (b) the patterns cover every possible value for the type (exhaustiveness)
//!
//! The algorithm implemented here is inspired from the one described in [this
//! paper](http://moscova.inria.fr/~maranget/papers/warn/index.html). We have however changed it in
//! various ways to accommodate the variety of patterns that Rust supports. We thus explain our
//! version here, without being as precise.
//!
//! Fun fact: computing exhaustiveness is NP-complete, because we can encode a SAT problem as an
//! exhaustiveness problem. See [here](https://niedzejkob.p4.team/rust-np) for the fun details.
//!
//!
//! # Summary
//!
//! The algorithm is given as input a list of patterns, one for each arm of a match, and computes
//! the following:
//! - a set of values that match none of the patterns (if any),
//! - for each subpattern (taking into account or-patterns), whether it would catch any value that
//!     isn't caught by a pattern before it, i.e. whether it is reachable.
//!
//! To a first approximation, the algorithm works by exploring all possible values for the type
//! being matched on, and determining which arm(s) catch which value. To make this tractable we
//! cleverly group together values, as we'll see below.
//!
//! The entrypoint of this file is the [`compute_match_usefulness`] function, which computes
//! reachability for each subpattern and exhaustiveness for the whole match.
//!
//! In this page we explain the necessary concepts to understand how the algorithm works.
//!
//!
//! # Usefulness
//!
//! The central concept of this file is the notion of "usefulness". Given some patterns `p_1 ..
//! p_n`, a pattern `q` is said to be *useful* if there is a value that is matched by `q` and by
//! none of the `p_i`. We write `usefulness(p_1 .. p_n, q)` for a function that returns a list of
//! such values. The aim of this file is to compute it efficiently.
//!
//! This is enough to compute reachability: a pattern in a `match` expression is reachable iff it is
//! useful w.r.t. the patterns above it:
//! ```compile_fail,E0004
//! # #![feature(exclusive_range_pattern)]
//! # fn foo() {
//! match Some(0u32) {
//!     Some(0..100) => {},
//!     Some(90..190) => {}, // reachable: `Some(150)` is matched by this but not the branch above
//!     Some(50..150) => {}, // unreachable: all the values this matches are already matched by
//!                          //   the branches above
//!     None => {},          // reachable: `None` is matched by this but not the branches above
//! }
//! # }
//! ```
//!
//! This is also enough to compute exhaustiveness: a match is exhaustive iff the wildcard `_`
//! pattern is _not_ useful w.r.t. the patterns in the match. The values returned by `usefulness`
//! are used to tell the user which values are missing.
//! ```compile_fail,E0004
//! # fn foo(x: Option<u32>) {
//! match x {
//!     None => {},
//!     Some(0) => {},
//!     // not exhaustive: `_` is useful because it matches `Some(1)`
//! }
//! # }
//! ```
//!
//!
//! # Constructors and fields
//!
//! In the value `Pair(Some(0), true)`, `Pair` is called the constructor of the value, and `Some(0)`
//! and `true` are its fields. Every matcheable value can be decomposed in this way. Examples of
//! constructors are: `Some`, `None`, `(,)` (the 2-tuple constructor), `Foo {..}` (the constructor
//! for a struct `Foo`), and `2` (the constructor for the number `2`).
//!
//! Each constructor takes a fixed number of fields; this is called its arity. `Pair` and `(,)` have
//! arity 2, `Some` has arity 1, `None` and `42` have arity 0. Each type has a known set of
//! constructors. Some types have many constructors (like `u64`) or even an infinitely many (like
//! `&str` and `&[T]`).
//!
//! Patterns are similar: `Pair(Some(_), _)` has constructor `Pair` and two fields. The difference
//! is that we get some extra pattern-only constructors, namely: the wildcard `_`, variable
//! bindings, integer ranges like `0..=10`, and variable-length slices like `[_, .., _]`. We treat
//! or-patterns separately, see the dedicated section below.
//!
//! Now to check if a value `v` matches a pattern `p`, we check if `v`'s constructor matches `p`'s
//! constructor, then recursively compare their fields if necessary. A few representative examples:
//!
//! - `matches!(v, _) := true`
//! - `matches!((v0,  v1), (p0,  p1)) := matches!(v0, p0) && matches!(v1, p1)`
//! - `matches!(Foo { bar: v0, baz: v1 }, Foo { bar: p0, baz: p1 }) := matches!(v0, p0) && matches!(v1, p1)`
//! - `matches!(Ok(v0), Ok(p0)) := matches!(v0, p0)`
//! - `matches!(Ok(v0), Err(p0)) := false` (incompatible variants)
//! - `matches!(v, 1..=100) := matches!(v, 1) || ... || matches!(v, 100)`
//! - `matches!([v0], [p0, .., p1]) := false` (incompatible lengths)
//! - `matches!([v0, v1, v2], [p0, .., p1]) := matches!(v0, p0) && matches!(v2, p1)`
//!
//! Constructors, fields and relevant operations are defined in the [`super::deconstruct_pat`]
//! module. The question of whether a constructor is matched by another one is answered by
//! [`Constructor::is_covered_by`].
//!
//! Note 1: variable bindings (like the `x` in `Some(x)`) match anything, so we treat them as wildcards.
//! Note 2: this only applies to matcheable values. For example a value of type `Rc<u64>` can't be
//! deconstructed that way.
//!
//!
//!
//! # Specialization
//!
//! The examples in the previous section motivate the operation at the heart of the algorithm:
//! "specialization". It captures this idea of "removing one layer of constructor".
//!
//! `specialize(c, p)` takes a value-only constructor `c` and a pattern `p`, and returns a
//! pattern-tuple or nothing. It works as follows:
//!
//! - Specializing for the wrong constructor returns nothing
//!
//!   - `specialize(None, Some(p0)) := <nothing>`
//!   - `specialize([,,,], [p0]) := <nothing>`
//!
//! - Specializing for the correct constructor returns a tuple of the fields
//!
//!   - `specialize(Variant1, Variant1(p0, p1, p2)) := (p0, p1, p2)`
//!   - `specialize(Foo{ bar, baz, quz }, Foo { bar: p0, baz: p1, .. }) := (p0, p1, _)`
//!   - `specialize([,,,], [p0, .., p1]) := (p0, _, _, p1)`
//!
//! We get the following property: for any values `v_1, .., v_n` of appropriate types, we have:
//! ```text
//! matches!(c(v_1, .., v_n), p)
//! <=> specialize(c, p) returns something
//!     && matches!((v_1, .., v_n), specialize(c, p))
//! ```
//!
//! We also extend specialization to pattern-tuples by applying it to the first pattern:
//! `specialize(c, (p_0, .., p_n)) := specialize(c, p_0) ++ (p_1, .., p_m)`
//! where `++` is concatenation of tuples.
//!
//!
//! The previous property extends to pattern-tuples:
//! ```text
//! matches!((c(v_1, .., v_n), w_1, .., w_m), (p_0, p_1, .., p_m))
//! <=> specialize(c, p_0) does not error
//!     && matches!((v_1, .., v_n, w_1, .., w_m), specialize(c, (p_0, p_1, .., p_m)))
//! ```
//!
//! Whether specialization returns something or not is given by [`Constructor::is_covered_by`].
//! Specialization of a pattern is computed in [`DeconstructedPat::specialize`]. Specialization for
//! a pattern-tuple is computed in [`PatStack::pop_head_constructor`]. Finally, specialization for a
//! set of pattern-tuples is computed in [`Matrix::specialize_constructor`].
//!
//!
//!
//! # Undoing specialization
//!
//! To construct witnesses we will need an inverse of specialization. If `c` is a constructor of
//! arity `n`, we define `unspecialize` as:
//! `unspecialize(c, (p_1, .., p_n, q_1, .., q_m)) := (c(p_1, .., p_n), q_1, .., q_m)`.
//!
//! This is done for a single witness-tuple in [`WitnessStack::apply_constructor`], and for a set of
//! witness-tuples in [`WitnessMatrix::apply_constructor`].
//!
//!
//!
//! # Computing usefulness
//!
//! We now present a naive version of the algorithm for computing usefulness. From now on we operate
//! on pattern-tuples.
//!
//! Let `pt_1, .., pt_n` and `qt` be length-m tuples of patterns for the same type `(T_1, .., T_m)`.
//! We compute `usefulness(tp_1, .., tp_n, tq)` as follows:
//!
//! - Base case: `m == 0`.
//!     The pattern-tuples are all empty, i.e. they're all `()`. Thus `tq` is useful iff there are
//!     no rows above it, i.e. if `n == 0`. In that case we return `()` as a witness-tuple of
//!     usefulness of `tq`.
//!
//! - Inductive case: `m > 0`.
//!     In this naive version, we list all the possible constructors for values of type `T1` (we
//!     will be more clever in the next section).
//!
//!     - For each such constructor `c` for which `specialize(c, tq)` is not nothing:
//!         - We recursively compute `usefulness(specialize(c, tp_1) ... specialize(c, tp_n), specialize(c, tq))`,
//!             where we discard any `specialize(c, p_i)` that returns nothing.
//!         - For each witness-tuple `w` found, we apply `unspecialize(c, w)` to it.
//!
//!     - We return the all the witnesses found, if any.
//!
//!
//! Let's take the following example:
//! ```compile_fail,E0004
//! # enum Enum { Variant1(()), Variant2(Option<bool>, u32)}
//! # use Enum::*;
//! # fn foo(x: Enum) {
//! match x {
//!     Variant1(_) => {} // `p1`
//!     Variant2(None, 0) => {} // `p2`
//!     Variant2(Some(_), 0) => {} // `q`
//! }
//! # }
//! ```
//!
//! To compute the usefulness of `q`, we would proceed as follows:
//! ```text
//! Start:
//!   `tp1 = [Variant1(_)]`
//!   `tp2 = [Variant2(None, 0)]`
//!   `tq  = [Variant2(Some(true), 0)]`
//!
//!   Constructors are `Variant1` and `Variant2`. Only `Variant2` can specialize `tq`.
//!   Specialize with `Variant2`:
//!     `tp2 = [None, 0]`
//!     `tq  = [Some(true), 0]`
//!
//!     Constructors are `None` and `Some`. Only `Some` can specialize `tq`.
//!     Specialize with `Some`:
//!       `tq  = [true, 0]`
//!
//!       Constructors are `false` and `true`. Only `true` can specialize `tq`.
//!       Specialize with `true`:
//!         `tq  = [0]`
//!
//!         Constructors are `0`, `1`, .. up to infinity. Only `0` can specialize `tq`.
//!         Specialize with `0`:
//!           `tq  = []`
//!
//!           m == 0 and n == 0, so `tq` is useful with witness `[]`.
//!             `witness  = []`
//!
//!         Unspecialize with `0`:
//!           `witness  = [0]`
//!       Unspecialize with `true`:
//!         `witness  = [true, 0]`
//!     Unspecialize with `Some`:
//!       `witness  = [Some(true), 0]`
//!   Unspecialize with `Variant2`:
//!     `witness  = [Variant2(Some(true), 0)]`
//! ```
//!
//! Therefore `usefulness(tp_1, tp_2, tq)` returns the single witness-tuple `[Variant2(Some(true), 0)]`.
//!
//!
//! Computing the set of constructors for a type is done in [`ConstructorSet::for_ty`]. See the
//! following sections for more accurate versions of the algorithm and corresponding links.
//!
//!
//!
//! # Computing reachability and exhaustiveness in one go
//!
//! The algorithm we have described so far computes usefulness of each pattern in turn to check if
//! it is reachable, and ends by checking if `_` is useful to determine exhaustiveness of the whole
//! match. In practice, instead of doing "for each pattern { for each constructor { ... } }", we do
//! "for each constructor { for each pattern { ... } }". This allows us to compute everything in one
//! go.
//!
//! [`Matrix`] stores the set of pattern-tuples under consideration. We track reachability of each
//! row mutably in the matrix as we go along. We ignore witnesses of usefulness of the match rows.
//! We gather witnesses of the usefulness of `_` in [`WitnessMatrix`]. The algorithm that computes
//! all this is in [`compute_exhaustiveness_and_reachability`].
//!
//! See the full example at the bottom of this documentation.
//!
//!
//!
//! # Making usefulness tractable: constructor splitting
//!
//! We're missing one last detail: which constructors do we list? Naively listing all value
//! constructors cannot work for types like `u64` or `&str`, so we need to be more clever. The final
//! clever idea for this algorithm is that we can group together constructors that behave the same.
//!
//! Examples:
//! ```compile_fail,E0004
//! match (0, false) {
//!     (0 ..=100, true) => {}
//!     (50..=150, false) => {}
//!     (0 ..=200, _) => {}
//! }
//! ```
//!
//! In this example, trying any of `0`, `1`, .., `49` will give the same specialized matrix, and
//! thus the same reachability/exhaustiveness results. We can thus accelerate the algorithm by
//! trying them all at once. Here in fact, the only cases we need to consider are: `0..50`,
//! `50..=100`, `101..=150`,`151..=200` and `201..`.
//!
//! ```
//! enum Direction { North, South, East, West }
//! # let wind = (Direction::North, 0u8);
//! match wind {
//!     (Direction::North, 50..) => {}
//!     (_, _) => {}
//! }
//! ```
//!
//! In this example, trying any of `South`, `East`, `West` will give the same specialized matrix. By
//! the same reasoning, we only need to try two cases: `North`, and "everything else".
//!
//! We call _constructor splitting_ the operation that computes such a minimal set of cases to try.
//! This is done in [`ConstructorSet::split`] and explained in [`super::deconstruct_pat`].
//!
//!
//! # Or-patterns
//!
//! What we have described so far works well if there are no or-patterns. To handle them, if the
//! first pattern of a row in the matrix is an or-pattern, we expand it by duplicating the rest of
//! the row as necessary. This is handled automatically in [`Matrix`].
//!
//! This makes reachability tracking subtle, because we also want to compute whether an alternative
//! of an or-pattern is unreachable, e.g. in `Some(_) | Some(0)`. We track reachability of each
//! subpattern by interior mutability in [`DeconstructedPat`] with `set_reachable`/`is_reachable`.
//!
//! It's unfortunate that we have to use interior mutability, but believe me (Nadrieril), I have
//! tried [other](https://github.com/rust-lang/rust/pull/80104)
//! [solutions](https://github.com/rust-lang/rust/pull/80632) and nothing is remotely as simple.
//!
//!
//!
//! # Constants and opaques
//!
//! There are two kinds of constants in patterns:
//!
//! * literals (`1`, `true`, `"foo"`)
//! * named or inline consts (`FOO`, `const { 5 + 6 }`)
//!
//! The latter are converted into the corresponding patterns by a previous phase. For example
//! `const_to_pat(const { [1, 2, 3] })` becomes an `Array(vec![Const(1), Const(2), Const(3)])`
//! pattern. This gets problematic when comparing the constant via `==` would behave differently
//! from matching on the constant converted to a pattern. The situation around this is currently
//! unclear and the lang team is working on clarifying what we want to do there. In any case, there
//! are constants we will not turn into patterns. We capture these with `Constructor::Opaque`. These
//! `Opaque` patterns do not participate in exhaustiveness, specialization or overlap checking.
//!
//!
//!
//! # Full example
//!
//! We illustrate a full run of the algorithm on the following match.
//!
//! ```compile_fail,E0004
//! # struct Pair(Option<u32>, bool);
//! # fn foo(x: Pair) -> u32 {
//! match x {
//!     Pair(Some(0), _) => 1,
//!     Pair(_, false) => 2,
//!     Pair(Some(0), false) => 3,
//! }
//! # }
//! ```
//!
//! We keep track of the original row for illustration purposes, this is not what the algorithm
//! actually does (it tracks reachability as a boolean on each row).
//!
//! ```text
//!  ┐ Patterns:
//!  │   1. `[Pair(Some(0), _)]`
//!  │   2. `[Pair(_, false)]`
//!  │   3. `[Pair(Some(0), false)]`
//!  │
//!  │ Specialize with `Pair`:
//!  ├─┐ Patterns:
//!  │ │   1. `[Some(0), _]`
//!  │ │   2. `[_, false]`
//!  │ │   3. `[Some(0), false]`
//!  │ │
//!  │ │ Specialize with `Some`:
//!  │ ├─┐ Patterns:
//!  │ │ │   1. `[0, _]`
//!  │ │ │   2. `[_, false]`
//!  │ │ │   3. `[0, false]`
//!  │ │ │
//!  │ │ │ Specialize with `0`:
//!  │ │ ├─┐ Patterns:
//!  │ │ │ │   1. `[_]`
//!  │ │ │ │   3. `[false]`
//!  │ │ │ │
//!  │ │ │ │ Specialize with `true`:
//!  │ │ │ ├─┐ Patterns:
//!  │ │ │ │ │   1. `[]`
//!  │ │ │ │ │
//!  │ │ │ │ │ We note arm 1 is reachable (by `Pair(Some(0), true)`).
//!  │ │ │ ├─┘
//!  │ │ │ │
//!  │ │ │ │ Specialize with `false`:
//!  │ │ │ ├─┐ Patterns:
//!  │ │ │ │ │   1. `[]`
//!  │ │ │ │ │   3. `[]`
//!  │ │ │ │ │
//!  │ │ │ │ │ We note arm 1 is reachable (by `Pair(Some(0), false)`).
//!  │ │ │ ├─┘
//!  │ │ ├─┘
//!  │ │ │
//!  │ │ │ Specialize with `1..`:
//!  │ │ ├─┐ Patterns:
//!  │ │ │ │   2. `[false]`
//!  │ │ │ │
//!  │ │ │ │ Specialize with `true`:
//!  │ │ │ ├─┐ Patterns:
//!  │ │ │ │ │   // no rows left
//!  │ │ │ │ │
//!  │ │ │ │ │ We have found an unmatched value (`Pair(Some(1..), true)`)! This gives us a witness.
//!  │ │ │ │ │ New witnesses:
//!  │ │ │ │ │   `[]`
//!  │ │ │ ├─┘
//!  │ │ │ │ Unspecialize new witnesses with `true`:
//!  │ │ │ │   `[true]`
//!  │ │ │ │
//!  │ │ │ │ Specialize with `false`:
//!  │ │ │ ├─┐ Patterns:
//!  │ │ │ │ │   2. `[]`
//!  │ │ │ │ │
//!  │ │ │ │ │ We note arm 2 is reachable (by `Pair(Some(1..), false)`).
//!  │ │ │ ├─┘
//!  │ │ │ │
//!  │ │ │ │ Total witnesses for `1..`:
//!  │ │ │ │   `[true]`
//!  │ │ ├─┘
//!  │ │ │ Unspecialize new witnesses with `1..`:
//!  │ │ │   `[1.., true]`
//!  │ │ │
//!  │ │ │ Total witnesses for `Some`:
//!  │ │ │   `[1.., true]`
//!  │ ├─┘
//!  │ │ Unspecialize new witnesses with `Some`:
//!  │ │   `[Some(1..), true]`
//!  │ │
//!  │ │ Specialize with `None`:
//!  │ ├─┐ Patterns:
//!  │ │ │   2. `[false]`
//!  │ │ │
//!  │ │ │ Specialize with `true`:
//!  │ │ ├─┐ Patterns:
//!  │ │ │ │   // no rows left
//!  │ │ │ │
//!  │ │ │ │ We have found an unmatched value (`Pair(None, true)`)! This gives us a witness.
//!  │ │ │ │ New witnesses:
//!  │ │ │ │   `[]`
//!  │ │ ├─┘
//!  │ │ │ Unspecialize new witnesses with `true`:
//!  │ │ │   `[true]`
//!  │ │ │
//!  │ │ │ Specialize with `false`:
//!  │ │ ├─┐ Patterns:
//!  │ │ │ │   2. `[]`
//!  │ │ │ │
//!  │ │ │ │ We note arm 2 is reachable (by `Pair(None, false)`).
//!  │ │ ├─┘
//!  │ │ │
//!  │ │ │ Total witnesses for `None`:
//!  │ │ │   `[true]`
//!  │ ├─┘
//!  │ │ Unspecialize new witnesses with `None`:
//!  │ │   `[None, true]`
//!  │ │
//!  │ │ Total witnesses for `Pair`:
//!  │ │   `[Some(1..), true]`
//!  │ │   `[None, true]`
//!  ├─┘
//!  │ Unspecialize new witnesses with `Pair`:
//!  │   `[Pair(Some(1..), true)]`
//!  │   `[Pair(None, true)]`
//!  │
//!  │ Final witnesses:
//!  │   `[Pair(Some(1..), true)]`
//!  │   `[Pair(None, true)]`
//!  ┘
//! ```
//!
//! We conclude:
//! - Arm 3 is unreachable (it was never marked as reachable);
//! - The match is not exhaustive;
//! - Adding arms with `Pair(Some(1..), true)` and `Pair(None, true)` would make the match exhaustive.
//!
//! Note that when we're deep in the algorithm, we don't know what specialization steps got us here.
//! We can only figure out what our witnesses correspond to by unspecializing back up the stack.
//!
//!
//! # Tests
//!
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
//! reason not to, for example if they crucially depend on a particular feature like `or_patterns`.

use super::deconstruct_pat::{
    Constructor, ConstructorSet, DeconstructedPat, IntRange, MaybeInfiniteInt, SplitConstructorSet,
    WitnessPat,
};
use crate::errors::{
    NonExhaustiveOmittedPattern, NonExhaustiveOmittedPatternLintOnArm, Overlap,
    OverlappingRangeEndpoints, Uncovered,
};

use rustc_data_structures::captures::Captures;

use rustc_arena::TypedArena;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def_id::DefId;
use rustc_hir::HirId;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint;
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
    /// Lint level at the match.
    pub(crate) match_lint_level: HirId,
    /// The span of the whole match, if applicable.
    pub(crate) match_span: Option<Span>,
    /// Span of the scrutinee.
    pub(crate) scrut_span: Span,
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
    /// Whether the current pattern is the whole pattern as found in a match arm, or if it's a
    /// subpattern.
    pub(super) is_top_level: bool,
}

impl<'a, 'p, 'tcx> fmt::Debug for PatCtxt<'a, 'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PatCtxt").field("ty", &self.ty).finish()
    }
}

/// Represents a pattern-tuple under investigation.
#[derive(Clone)]
struct PatStack<'p, 'tcx> {
    // Rows of len 1 are very common, which is why `SmallVec[_; 2]` works well.
    pats: SmallVec<[&'p DeconstructedPat<'p, 'tcx>; 2]>,
}

impl<'p, 'tcx> PatStack<'p, 'tcx> {
    fn from_pattern(pat: &'p DeconstructedPat<'p, 'tcx>) -> Self {
        PatStack { pats: smallvec![pat] }
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

    // Recursively expand the first or-pattern into its subpatterns. Only useful if the pattern is
    // an or-pattern. Panics if `self` is empty.
    fn expand_or_pat<'a>(&'a self) -> impl Iterator<Item = PatStack<'p, 'tcx>> + Captures<'a> {
        self.head().flatten_or_pat().into_iter().map(move |pat| {
            let mut new_pats = smallvec![pat];
            new_pats.extend_from_slice(&self.pats[1..]);
            PatStack { pats: new_pats }
        })
    }

    /// This computes `specialize(ctor, self)`. See top of the file for explanations.
    /// Only call if `ctor.is_covered_by(self.head().ctor())` is true.
    fn pop_head_constructor(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
    ) -> PatStack<'p, 'tcx> {
        // We pop the head pattern and push the new fields extracted from the arguments of
        // `self.head()`.
        let mut new_pats = self.head().specialize(pcx, ctor);
        new_pats.extend_from_slice(&self.pats[1..]);
        PatStack { pats: new_pats }
    }
}

impl<'p, 'tcx> fmt::Debug for PatStack<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We pretty-print similarly to the `Debug` impl of `Matrix`.
        write!(f, "+")?;
        for pat in self.iter() {
            write!(f, " {pat:?} +")?;
        }
        Ok(())
    }
}

/// A row of the matrix.
#[derive(Clone)]
struct MatrixRow<'p, 'tcx> {
    // The patterns in the row.
    pats: PatStack<'p, 'tcx>,
    /// Whether the original arm had a guard. This is inherited when specializing.
    is_under_guard: bool,
    /// When we specialize, we remember which row of the original matrix produced a given row of the
    /// specialized matrix. When we unspecialize, we use this to propagate reachability back up the
    /// callstack.
    parent_row: usize,
    /// False when the matrix is just built. This is set to `true` by
    /// [`compute_exhaustiveness_and_reachability`] if the arm is found to be reachable.
    /// This is reset to `false` when specializing.
    reachable: bool,
}

impl<'p, 'tcx> MatrixRow<'p, 'tcx> {
    fn is_empty(&self) -> bool {
        self.pats.is_empty()
    }

    fn len(&self) -> usize {
        self.pats.len()
    }

    fn head(&self) -> &'p DeconstructedPat<'p, 'tcx> {
        self.pats.head()
    }

    fn iter(&self) -> impl Iterator<Item = &DeconstructedPat<'p, 'tcx>> {
        self.pats.iter()
    }

    // Recursively expand the first or-pattern into its subpatterns. Only useful if the pattern is
    // an or-pattern. Panics if `self` is empty.
    fn expand_or_pat<'a>(&'a self) -> impl Iterator<Item = MatrixRow<'p, 'tcx>> + Captures<'a> {
        self.pats.expand_or_pat().map(|patstack| MatrixRow {
            pats: patstack,
            parent_row: self.parent_row,
            is_under_guard: self.is_under_guard,
            reachable: false,
        })
    }

    /// This computes `specialize(ctor, self)`. See top of the file for explanations.
    /// Only call if `ctor.is_covered_by(self.head().ctor())` is true.
    fn pop_head_constructor(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
        parent_row: usize,
    ) -> MatrixRow<'p, 'tcx> {
        MatrixRow {
            pats: self.pats.pop_head_constructor(pcx, ctor),
            parent_row,
            is_under_guard: self.is_under_guard,
            reachable: false,
        }
    }
}

impl<'p, 'tcx> fmt::Debug for MatrixRow<'p, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pats.fmt(f)
    }
}

/// A 2D matrix. Represents a list of pattern-tuples under investigation.
///
/// Invariant: each row must have the same length, and each column must have the same type.
///
/// Invariant: the first column must not contain or-patterns. This is handled by
/// [`Matrix::expand_and_push`].
///
/// In fact each column corresponds to a place inside the scrutinee of the match. E.g. after
/// specializing `(,)` and `Some` on a pattern of type `(Option<u32>, bool)`, the first column of
/// the matrix will correspond to `scrutinee.0.Some.0` and the second column to `scrutinee.1`.
#[derive(Clone)]
struct Matrix<'p, 'tcx> {
    rows: Vec<MatrixRow<'p, 'tcx>>,
    /// Stores an extra fictitious row full of wildcards. Mostly used to keep track of the type of
    /// each column. This must obey the same invariants as the real rows.
    wildcard_row: PatStack<'p, 'tcx>,
}

impl<'p, 'tcx> Matrix<'p, 'tcx> {
    /// Pushes a new row to the matrix. If the row starts with an or-pattern, this recursively
    /// expands it. Internal method, prefer [`Matrix::new`].
    fn expand_and_push(&mut self, row: MatrixRow<'p, 'tcx>) {
        if !row.is_empty() && row.head().is_or_pat() {
            // Expand nested or-patterns.
            for new_row in row.expand_or_pat() {
                self.rows.push(new_row);
            }
        } else {
            self.rows.push(row);
        }
    }

    /// Build a new matrix from an iterator of `MatchArm`s.
    fn new<'a>(
        cx: &MatchCheckCtxt<'p, 'tcx>,
        iter: impl Iterator<Item = &'a MatchArm<'p, 'tcx>>,
        scrut_ty: Ty<'tcx>,
    ) -> Self
    where
        'p: 'a,
    {
        let wild_pattern = cx.pattern_arena.alloc(DeconstructedPat::wildcard(scrut_ty, DUMMY_SP));
        let wildcard_row = PatStack::from_pattern(wild_pattern);
        let mut matrix = Matrix { rows: vec![], wildcard_row };
        for (row_id, arm) in iter.enumerate() {
            let v = MatrixRow {
                pats: PatStack::from_pattern(arm.pat),
                parent_row: row_id, // dummy, we won't read it
                is_under_guard: arm.has_guard,
                reachable: false,
            };
            matrix.expand_and_push(v);
        }
        matrix
    }

    fn head_ty(&self) -> Option<Ty<'tcx>> {
        if self.column_count() == 0 {
            return None;
        }

        let mut ty = self.wildcard_row.head().ty();
        // If the type is opaque and it is revealed anywhere in the column, we take the revealed
        // version. Otherwise we could encounter constructors for the revealed type and crash.
        let is_opaque = |ty: Ty<'tcx>| matches!(ty.kind(), ty::Alias(ty::Opaque, ..));
        if is_opaque(ty) {
            for pat in self.heads() {
                let pat_ty = pat.ty();
                if !is_opaque(pat_ty) {
                    ty = pat_ty;
                    break;
                }
            }
        }
        Some(ty)
    }
    fn column_count(&self) -> usize {
        self.wildcard_row.len()
    }

    fn rows<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'a MatrixRow<'p, 'tcx>> + Clone + DoubleEndedIterator + ExactSizeIterator
    {
        self.rows.iter()
    }
    fn rows_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = &'a mut MatrixRow<'p, 'tcx>> + DoubleEndedIterator + ExactSizeIterator
    {
        self.rows.iter_mut()
    }

    /// Iterate over the first pattern of each row.
    fn heads<'a>(
        &'a self,
    ) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Clone + Captures<'a> {
        self.rows().map(|r| r.head())
    }

    /// This computes `specialize(ctor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        pcx: &PatCtxt<'_, 'p, 'tcx>,
        ctor: &Constructor<'tcx>,
    ) -> Matrix<'p, 'tcx> {
        let wildcard_row = self.wildcard_row.pop_head_constructor(pcx, ctor);
        let mut matrix = Matrix { rows: vec![], wildcard_row };
        for (i, row) in self.rows().enumerate() {
            if ctor.is_covered_by(pcx, row.head().ctor()) {
                let new_row = row.pop_head_constructor(pcx, ctor, i);
                matrix.expand_and_push(new_row);
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

        let Matrix { rows, .. } = self;
        let pretty_printed_matrix: Vec<Vec<String>> =
            rows.iter().map(|row| row.iter().map(|pat| format!("{pat:?}")).collect()).collect();

        let column_count = rows.iter().map(|row| row.len()).next().unwrap_or(0);
        assert!(rows.iter().all(|row| row.len() == column_count));
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

/// A witness-tuple of non-exhaustiveness for error reporting, represented as a list of patterns (in
/// reverse order of construction).
///
/// This mirrors `PatStack`: they function similarly, except `PatStack` contains user patterns we
/// are inspecting, and `WitnessStack` contains witnesses we are constructing.
/// FIXME(Nadrieril): use the same order of patterns for both.
///
/// A `WitnessStack` should have the same types and length as the `PatStack`s we are inspecting
/// (except we store the patterns in reverse order). The same way `PatStack` starts with length 1,
/// at the end of the algorithm this will have length 1. In the middle of the algorithm, it can
/// contain multiple patterns.
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
/// ```text
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
/// ```
///
/// The final `Pair(Some(_), true)` is then the resulting witness.
///
/// See the top of the file for more detailed explanations and examples.
#[derive(Debug, Clone)]
pub(crate) struct WitnessStack<'tcx>(Vec<WitnessPat<'tcx>>);

impl<'tcx> WitnessStack<'tcx> {
    /// Asserts that the witness contains a single pattern, and returns it.
    fn single_pattern(self) -> WitnessPat<'tcx> {
        assert_eq!(self.0.len(), 1);
        self.0.into_iter().next().unwrap()
    }

    /// Reverses specialization by the `Missing` constructor by pushing a whole new pattern.
    fn push_pattern(&mut self, pat: WitnessPat<'tcx>) {
        self.0.push(pat);
    }

    /// Reverses specialization. Given a witness obtained after specialization, this constructs a
    /// new witness valid for before specialization. See the section on `unspecialize` at the top of
    /// the file.
    ///
    /// Examples:
    /// ```text
    /// ctor: tuple of 2 elements
    /// pats: [false, "foo", _, true]
    /// result: [(false, "foo"), _, true]
    ///
    /// ctor: Enum::Variant { a: (bool, &'static str), b: usize}
    /// pats: [(false, "foo"), _, true]
    /// result: [Enum::Variant { a: (false, "foo"), b: _ }, true]
    /// ```
    fn apply_constructor(&mut self, pcx: &PatCtxt<'_, '_, 'tcx>, ctor: &Constructor<'tcx>) {
        let len = self.0.len();
        let arity = ctor.arity(pcx);
        let fields = self.0.drain((len - arity)..).rev().collect();
        let pat = WitnessPat::new(ctor.clone(), fields, pcx.ty);
        self.0.push(pat);
    }
}

/// Represents a set of pattern-tuples that are witnesses of non-exhaustiveness for error
/// reporting. This has similar invariants as `Matrix` does.
///
/// The `WitnessMatrix` returned by [`compute_exhaustiveness_and_reachability`] obeys the invariant
/// that the union of the input `Matrix` and the output `WitnessMatrix` together matches the type
/// exhaustively.
///
/// Just as the `Matrix` starts with a single column, by the end of the algorithm, this has a single
/// column, which contains the patterns that are missing for the match to be exhaustive.
#[derive(Debug, Clone)]
pub struct WitnessMatrix<'tcx>(Vec<WitnessStack<'tcx>>);

impl<'tcx> WitnessMatrix<'tcx> {
    /// New matrix with no witnesses.
    fn empty() -> Self {
        WitnessMatrix(vec![])
    }
    /// New matrix with one `()` witness, i.e. with no columns.
    fn unit_witness() -> Self {
        WitnessMatrix(vec![WitnessStack(vec![])])
    }

    /// Whether this has any witnesses.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Asserts that there is a single column and returns the patterns in it.
    fn single_column(self) -> Vec<WitnessPat<'tcx>> {
        self.0.into_iter().map(|w| w.single_pattern()).collect()
    }

    /// Reverses specialization by the `Missing` constructor by pushing a whole new pattern.
    fn push_pattern(&mut self, pat: WitnessPat<'tcx>) {
        for witness in self.0.iter_mut() {
            witness.push_pattern(pat.clone())
        }
    }

    /// Reverses specialization by `ctor`. See the section on `unspecialize` at the top of the file.
    fn apply_constructor(
        &mut self,
        pcx: &PatCtxt<'_, '_, 'tcx>,
        missing_ctors: &[Constructor<'tcx>],
        ctor: &Constructor<'tcx>,
        report_individual_missing_ctors: bool,
    ) {
        if self.is_empty() {
            return;
        }
        if matches!(ctor, Constructor::Missing) {
            // We got the special `Missing` constructor that stands for the constructors not present
            // in the match.
            if !report_individual_missing_ctors {
                // Report `_` as missing.
                let pat = WitnessPat::wild_from_ctor(pcx, Constructor::Wildcard);
                self.push_pattern(pat);
            } else if missing_ctors.iter().any(|c| c.is_non_exhaustive()) {
                // We need to report a `_` anyway, so listing other constructors would be redundant.
                // `NonExhaustive` is displayed as `_` just like `Wildcard`, but it will be picked
                // up by diagnostics to add a note about why `_` is required here.
                let pat = WitnessPat::wild_from_ctor(pcx, Constructor::NonExhaustive);
                self.push_pattern(pat);
            } else {
                // For each missing constructor `c`, we add a `c(_, _, _)` witness appropriately
                // filled with wildcards.
                let mut ret = Self::empty();
                for ctor in missing_ctors {
                    let pat = WitnessPat::wild_from_ctor(pcx, ctor.clone());
                    // Clone `self` and add `c(_, _, _)` to each of its witnesses.
                    let mut wit_matrix = self.clone();
                    wit_matrix.push_pattern(pat);
                    ret.extend(wit_matrix);
                }
                *self = ret;
            }
        } else {
            // Any other constructor we unspecialize as expected.
            for witness in self.0.iter_mut() {
                witness.apply_constructor(pcx, ctor)
            }
        }
    }

    /// Merges the witnesses of two matrices. Their column types must match.
    fn extend(&mut self, other: Self) {
        self.0.extend(other.0)
    }
}

/// The core of the algorithm.
///
/// This recursively computes witnesses of the non-exhaustiveness of `matrix` (if any). Also tracks
/// usefulness of each row in the matrix (in `row.reachable`). We track reachability of each
/// subpattern using interior mutability in `DeconstructedPat`.
///
/// The input `Matrix` and the output `WitnessMatrix` together match the type exhaustively.
///
/// The key steps are:
/// - specialization, where we dig into the rows that have a specific constructor and call ourselves
///     recursively;
/// - unspecialization, where we lift the results from the previous step into results for this step
///     (using `apply_constructor` and by updating `row.reachable` for each parent row).
/// This is all explained at the top of the file.
#[instrument(level = "debug", skip(cx, is_top_level), ret)]
fn compute_exhaustiveness_and_reachability<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    matrix: &mut Matrix<'p, 'tcx>,
    is_top_level: bool,
) -> WitnessMatrix<'tcx> {
    debug_assert!(matrix.rows().all(|r| r.len() == matrix.column_count()));

    let Some(ty) = matrix.head_ty() else {
        // The base case: there are no columns in the matrix. We are morally pattern-matching on ().
        // A row is reachable iff it has no (unguarded) rows above it.
        for row in matrix.rows_mut() {
            // All rows are reachable until we find one without a guard.
            row.reachable = true;
            if !row.is_under_guard {
                // There's an unguarded row, so the match is exhaustive, and any subsequent row is
                // unreachable.
                return WitnessMatrix::empty();
            }
        }
        // No (unguarded) rows, so the match is not exhaustive. We return a new witness.
        return WitnessMatrix::unit_witness();
    };

    debug!("ty: {ty:?}");
    let pcx = &PatCtxt { cx, ty, is_top_level };

    // Analyze the constructors present in this column.
    let ctors = matrix.heads().map(|p| p.ctor());
    let split_set = ConstructorSet::for_ty(pcx.cx, pcx.ty).split(pcx, ctors);

    let all_missing = split_set.present.is_empty();
    let always_report_all = is_top_level && !IntRange::is_integral(pcx.ty);
    // Whether we should report "Enum::A and Enum::C are missing" or "_ is missing".
    let report_individual_missing_ctors = always_report_all || !all_missing;

    let mut split_ctors = split_set.present;
    let mut only_report_missing = false;
    if !split_set.missing.is_empty() {
        // We need to iterate over a full set of constructors, so we add `Missing` to represent the
        // missing ones. This is explained under "Constructor Splitting" at the top of this file.
        split_ctors.push(Constructor::Missing);
        // For diagnostic purposes we choose to only report the constructors that are missing. Since
        // `Missing` matches only the wildcard rows, it matches fewer rows than any normal
        // constructor and is therefore guaranteed to result in more witnesses. So skipping the
        // other constructors does not jeopardize correctness.
        only_report_missing = true;
    }

    let mut ret = WitnessMatrix::empty();
    for ctor in split_ctors {
        debug!("specialize({:?})", ctor);
        // Dig into rows that match `ctor`.
        let mut spec_matrix = matrix.specialize_constructor(pcx, &ctor);
        let mut witnesses = ensure_sufficient_stack(|| {
            compute_exhaustiveness_and_reachability(cx, &mut spec_matrix, false)
        });

        if !only_report_missing || matches!(ctor, Constructor::Missing) {
            // Transform witnesses for `spec_matrix` into witnesses for `matrix`.
            witnesses.apply_constructor(
                pcx,
                &split_set.missing,
                &ctor,
                report_individual_missing_ctors,
            );
            // Accumulate the found witnesses.
            ret.extend(witnesses);
        }

        // A parent row is useful if any of its children is.
        for child_row in spec_matrix.rows() {
            let parent_row = &mut matrix.rows[child_row.parent_row];
            parent_row.reachable = parent_row.reachable || child_row.reachable;
        }
    }

    // Record that the subpattern is reachable.
    for row in matrix.rows() {
        if row.reachable {
            row.head().set_reachable();
        }
    }

    ret
}

/// A column of patterns in the matrix, where a column is the intuitive notion of "subpatterns that
/// inspect the same subvalue/place".
/// This is used to traverse patterns column-by-column for lints. Despite similarities with
/// [`compute_exhaustiveness_and_reachability`], this does a different traversal. Notably this is
/// linear in the depth of patterns, whereas `compute_exhaustiveness_and_reachability` is worst-case
/// exponential (exhaustiveness is NP-complete). The core difference is that we treat sub-columns
/// separately.
///
/// This must not contain an or-pattern. `specialize` takes care to expand them.
///
/// This is not used in the main algorithm; only in lints.
#[derive(Debug)]
struct PatternColumn<'p, 'tcx> {
    patterns: Vec<&'p DeconstructedPat<'p, 'tcx>>,
}

impl<'p, 'tcx> PatternColumn<'p, 'tcx> {
    fn new(patterns: Vec<&'p DeconstructedPat<'p, 'tcx>>) -> Self {
        Self { patterns }
    }

    fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
    fn head_ty(&self) -> Option<Ty<'tcx>> {
        if self.patterns.len() == 0 {
            return None;
        }
        // If the type is opaque and it is revealed anywhere in the column, we take the revealed
        // version. Otherwise we could encounter constructors for the revealed type and crash.
        let is_opaque = |ty: Ty<'tcx>| matches!(ty.kind(), ty::Alias(ty::Opaque, ..));
        let first_ty = self.patterns[0].ty();
        if is_opaque(first_ty) {
            for pat in &self.patterns {
                let ty = pat.ty();
                if !is_opaque(ty) {
                    return Some(ty);
                }
            }
        }
        Some(first_ty)
    }

    /// Do constructor splitting on the constructors of the column.
    fn analyze_ctors(&self, pcx: &PatCtxt<'_, 'p, 'tcx>) -> SplitConstructorSet<'tcx> {
        let column_ctors = self.patterns.iter().map(|p| p.ctor());
        ConstructorSet::for_ty(pcx.cx, pcx.ty).split(pcx, column_ctors)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'p DeconstructedPat<'p, 'tcx>> + Captures<'a> {
        self.patterns.iter().copied()
    }

    /// Does specialization: given a constructor, this takes the patterns from the column that match
    /// the constructor, and outputs their fields.
    /// This returns one column per field of the constructor. They usually all have the same length
    /// (the number of patterns in `self` that matched `ctor`), except that we expand or-patterns
    /// which may change the lengths.
    fn specialize(&self, pcx: &PatCtxt<'_, 'p, 'tcx>, ctor: &Constructor<'tcx>) -> Vec<Self> {
        let arity = ctor.arity(pcx);
        if arity == 0 {
            return Vec::new();
        }

        // We specialize the column by `ctor`. This gives us `arity`-many columns of patterns. These
        // columns may have different lengths in the presence of or-patterns (this is why we can't
        // reuse `Matrix`).
        let mut specialized_columns: Vec<_> =
            (0..arity).map(|_| Self { patterns: Vec::new() }).collect();
        let relevant_patterns =
            self.patterns.iter().filter(|pat| ctor.is_covered_by(pcx, pat.ctor()));
        for pat in relevant_patterns {
            let specialized = pat.specialize(pcx, ctor);
            for (subpat, column) in specialized.iter().zip(&mut specialized_columns) {
                if subpat.is_or_pat() {
                    column.patterns.extend(subpat.flatten_or_pat())
                } else {
                    column.patterns.push(subpat)
                }
            }
        }

        assert!(
            !specialized_columns[0].is_empty(),
            "ctor {ctor:?} was listed as present but isn't;
            there is an inconsistency between `Constructor::is_covered_by` and `ConstructorSet::split`"
        );
        specialized_columns
    }
}

/// Traverse the patterns to collect any variants of a non_exhaustive enum that fail to be mentioned
/// in a given column.
#[instrument(level = "debug", skip(cx), ret)]
fn collect_nonexhaustive_missing_variants<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    column: &PatternColumn<'p, 'tcx>,
) -> Vec<WitnessPat<'tcx>> {
    let Some(ty) = column.head_ty() else {
        return Vec::new();
    };
    let pcx = &PatCtxt { cx, ty, is_top_level: false };

    let set = column.analyze_ctors(pcx);
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
        let specialized_columns = column.specialize(pcx, &ctor);
        let wild_pat = WitnessPat::wild_from_ctor(pcx, ctor);
        for (i, col_i) in specialized_columns.iter().enumerate() {
            // Compute witnesses for each column.
            let wits_for_col_i = collect_nonexhaustive_missing_variants(cx, col_i);
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

/// Traverse the patterns to warn the user about ranges that overlap on their endpoints.
#[instrument(level = "debug", skip(cx))]
fn lint_overlapping_range_endpoints<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    column: &PatternColumn<'p, 'tcx>,
) {
    let Some(ty) = column.head_ty() else {
        return;
    };
    let pcx = &PatCtxt { cx, ty, is_top_level: false };

    let set = column.analyze_ctors(pcx);

    if IntRange::is_integral(ty) {
        let emit_lint = |overlap: &IntRange, this_span: Span, overlapped_spans: &[Span]| {
            let overlap_as_pat = overlap.to_diagnostic_pat(ty, cx.tcx);
            let overlaps: Vec<_> = overlapped_spans
                .iter()
                .copied()
                .map(|span| Overlap { range: overlap_as_pat.clone(), span })
                .collect();
            cx.tcx.emit_spanned_lint(
                lint::builtin::OVERLAPPING_RANGE_ENDPOINTS,
                cx.match_lint_level,
                this_span,
                OverlappingRangeEndpoints { overlap: overlaps, range: this_span },
            );
        };

        // If two ranges overlapped, the split set will contain their intersection as a singleton.
        let split_int_ranges = set.present.iter().filter_map(|c| c.as_int_range());
        for overlap_range in split_int_ranges.clone() {
            if overlap_range.is_singleton() {
                let overlap: MaybeInfiniteInt = overlap_range.lo;
                // Ranges that look like `lo..=overlap`.
                let mut prefixes: SmallVec<[_; 1]> = Default::default();
                // Ranges that look like `overlap..=hi`.
                let mut suffixes: SmallVec<[_; 1]> = Default::default();
                // Iterate on patterns that contained `overlap`.
                for pat in column.iter() {
                    let this_span = pat.span();
                    let Constructor::IntRange(this_range) = pat.ctor() else { continue };
                    if this_range.is_singleton() {
                        // Don't lint when one of the ranges is a singleton.
                        continue;
                    }
                    if this_range.lo == overlap {
                        // `this_range` looks like `overlap..=this_range.hi`; it overlaps with any
                        // ranges that look like `lo..=overlap`.
                        if !prefixes.is_empty() {
                            emit_lint(overlap_range, this_span, &prefixes);
                        }
                        suffixes.push(this_span)
                    } else if this_range.hi == overlap.plus_one() {
                        // `this_range` looks like `this_range.lo..=overlap`; it overlaps with any
                        // ranges that look like `overlap..=hi`.
                        if !suffixes.is_empty() {
                            emit_lint(overlap_range, this_span, &suffixes);
                        }
                        prefixes.push(this_span)
                    }
                }
            }
        }
    } else {
        // Recurse into the fields.
        for ctor in set.present {
            for col in column.specialize(pcx, &ctor) {
                lint_overlapping_range_endpoints(cx, &col);
            }
        }
    }
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

/// The entrypoint for this file. Computes whether a match is exhaustive and which of its arms are
/// reachable.
#[instrument(skip(cx, arms), level = "debug")]
pub(crate) fn compute_match_usefulness<'p, 'tcx>(
    cx: &MatchCheckCtxt<'p, 'tcx>,
    arms: &[MatchArm<'p, 'tcx>],
    scrut_ty: Ty<'tcx>,
) -> UsefulnessReport<'p, 'tcx> {
    let mut matrix = Matrix::new(cx, arms.iter(), scrut_ty);
    let non_exhaustiveness_witnesses =
        compute_exhaustiveness_and_reachability(cx, &mut matrix, true);

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
    let report = UsefulnessReport { arm_usefulness, non_exhaustiveness_witnesses };

    let pat_column = PatternColumn::new(matrix.heads().collect());
    // Lint on ranges that overlap on their endpoints, which is likely a mistake.
    lint_overlapping_range_endpoints(cx, &pat_column);

    // Run the non_exhaustive_omitted_patterns lint. Only run on refutable patterns to avoid hitting
    // `if let`s. Only run if the match is exhaustive otherwise the error is redundant.
    if cx.refutable && report.non_exhaustiveness_witnesses.is_empty() {
        if !matches!(
            cx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, cx.match_lint_level).0,
            rustc_session::lint::Level::Allow
        ) {
            let witnesses = collect_nonexhaustive_missing_variants(cx, &pat_column);
            if !witnesses.is_empty() {
                // Report that a match of a `non_exhaustive` enum marked with `non_exhaustive_omitted_patterns`
                // is not exhaustive enough.
                //
                // NB: The partner lint for structs lives in `compiler/rustc_hir_analysis/src/check/pat.rs`.
                cx.tcx.emit_spanned_lint(
                    NON_EXHAUSTIVE_OMITTED_PATTERNS,
                    cx.match_lint_level,
                    cx.scrut_span,
                    NonExhaustiveOmittedPattern {
                        scrut_ty,
                        uncovered: Uncovered::new(cx.scrut_span, cx, witnesses),
                    },
                );
            }
        } else {
            // We used to allow putting the `#[allow(non_exhaustive_omitted_patterns)]` on a match
            // arm. This no longer makes sense so we warn users, to avoid silently breaking their
            // usage of the lint.
            for arm in arms {
                let (lint_level, lint_level_source) =
                    cx.tcx.lint_level_at_node(NON_EXHAUSTIVE_OMITTED_PATTERNS, arm.hir_id);
                if !matches!(lint_level, rustc_session::lint::Level::Allow) {
                    let decorator = NonExhaustiveOmittedPatternLintOnArm {
                        lint_span: lint_level_source.span(),
                        suggest_lint_on_match: cx.match_span.map(|span| span.shrink_to_lo()),
                        lint_level: lint_level.as_str(),
                        lint_name: "non_exhaustive_omitted_patterns",
                    };

                    use rustc_errors::DecorateLint;
                    let mut err = cx.tcx.sess.struct_span_warn(arm.pat.span(), "");
                    err.set_primary_message(decorator.msg());
                    decorator.decorate_lint(&mut err);
                    err.emit();
                }
            }
        }
    }

    report
}
