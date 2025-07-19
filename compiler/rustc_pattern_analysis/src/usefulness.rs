//! # Match exhaustiveness and redundancy algorithm
//!
//! This file contains the logic for exhaustiveness and usefulness checking for pattern-matching.
//! Specifically, given a list of patterns in a match, we can tell whether:
//! (a) a given pattern is redundant
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
//! - for each subpattern (taking into account or-patterns), whether removing it would change
//!     anything about how the match executes, i.e. whether it is useful/not redundant.
//!
//! To a first approximation, the algorithm works by exploring all possible values for the type
//! being matched on, and determining which arm(s) catch which value. To make this tractable we
//! cleverly group together values, as we'll see below.
//!
//! The entrypoint of this file is the [`compute_match_usefulness`] function, which computes
//! usefulness for each subpattern and exhaustiveness for the whole match.
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
//! This is enough to compute usefulness: a pattern in a `match` expression is redundant iff it is
//! not useful w.r.t. the patterns above it:
//! ```compile_fail,E0004
//! # fn foo() {
//! match Some(0u32) {
//!     Some(0..100) => {},
//!     Some(90..190) => {}, // useful: `Some(150)` is matched by this but not the branch above
//!     Some(50..150) => {}, // redundant: all the values this matches are already matched by
//!                          //   the branches above
//!     None => {},          // useful: `None` is matched by this but not the branches above
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
//! and `true` are its fields. Every matchable value can be decomposed in this way. Examples of
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
//! Constructors and relevant operations are defined in the [`crate::constructor`] module. A
//! representation of patterns that uses constructors is available in [`crate::pat`]. The question
//! of whether a constructor is matched by another one is answered by
//! [`Constructor::is_covered_by`].
//!
//! Note 1: variable bindings (like the `x` in `Some(x)`) match anything, so we treat them as wildcards.
//! Note 2: this only applies to matchable values. For example a value of type `Rc<u64>` can't be
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
//! Computing the set of constructors for a type is done in [`PatCx::ctors_for_ty`]. See
//! the following sections for more accurate versions of the algorithm and corresponding links.
//!
//!
//!
//! # Computing usefulness and exhaustiveness in one go
//!
//! The algorithm we have described so far computes usefulness of each pattern in turn, and ends by
//! checking if `_` is useful to determine exhaustiveness of the whole match. In practice, instead
//! of doing "for each pattern { for each constructor { ... } }", we do "for each constructor { for
//! each pattern { ... } }". This allows us to compute everything in one go.
//!
//! [`Matrix`] stores the set of pattern-tuples under consideration. We track usefulness of each
//! row mutably in the matrix as we go along. We ignore witnesses of usefulness of the match rows.
//! We gather witnesses of the usefulness of `_` in [`WitnessMatrix`]. The algorithm that computes
//! all this is in [`compute_exhaustiveness_and_usefulness`].
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
//! thus the same usefulness/exhaustiveness results. We can thus accelerate the algorithm by
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
//! This is done in [`ConstructorSet::split`] and explained in [`crate::constructor`].
//!
//!
//!
//! # `Missing` and relevancy
//!
//! ## Relevant values
//!
//! Take the following example:
//!
//! ```compile_fail,E0004
//! # let foo = (true, true);
//! match foo {
//!     (true, _) => 1,
//!     (_, true) => 2,
//! };
//! ```
//!
//! Consider the value `(true, true)`:
//! - Row 2 does not distinguish `(true, true)` and `(false, true)`;
//! - `false` does not show up in the first column of the match, so without knowing anything else we
//!     can deduce that `(false, true)` matches the same or fewer rows than `(true, true)`.
//!
//! Using those two facts together, we deduce that `(true, true)` will not give us more usefulness
//! information about row 2 than `(false, true)` would. We say that "`(true, true)` is made
//! irrelevant for row 2 by `(false, true)`". We will use this idea to prune the search tree.
//!
//!
//! ## Computing relevancy
//!
//! We now generalize from the above example to approximate relevancy in a simple way. Note that we
//! will only compute an approximation: we can sometimes determine when a case is irrelevant, but
//! computing this precisely is at least as hard as computing usefulness.
//!
//! Our computation of relevancy relies on the `Missing` constructor. As explained in
//! [`crate::constructor`], `Missing` represents the constructors not present in a given column. For
//! example in the following:
//!
//! ```compile_fail,E0004
//! enum Direction { North, South, East, West }
//! # let wind = (Direction::North, 0u8);
//! match wind {
//!     (Direction::North, _) => 1,
//!     (_, 50..) => 2,
//! };
//! ```
//!
//! Here `South`, `East` and `West` are missing in the first column, and `0..50`  is missing in the
//! second. Both of these sets are represented by `Constructor::Missing` in their corresponding
//! column.
//!
//! We then compute relevancy as follows: during the course of the algorithm, for a row `r`:
//! - if `r` has a wildcard in the first column;
//! - and some constructors are missing in that column;
//! - then any `c != Missing` is considered irrelevant for row `r`.
//!
//! By this we mean that continuing the algorithm by specializing with `c` is guaranteed not to
//! contribute more information about the usefulness of row `r` than what we would get by
//! specializing with `Missing`. The argument is the same as in the previous subsection.
//!
//! Once we've specialized by a constructor `c` that is irrelevant for row `r`, we're guaranteed to
//! only explore values irrelevant for `r`. If we then ever reach a point where we're only exploring
//! values that are irrelevant to all of the rows (including the virtual wildcard row used for
//! exhaustiveness), we skip that case entirely.
//!
//!
//! ## Example
//!
//! Let's go through a variation on the first example:
//!
//! ```compile_fail,E0004
//! # let foo = (true, true, true);
//! match foo {
//!     (true, _, true) => 1,
//!     (_, true, _) => 2,
//! };
//! ```
//!
//! ```text
//!  ┐ Patterns:
//!  │   1. `[(true, _, true)]`
//!  │   2. `[(_, true, _)]`
//!  │   3. `[_]` // virtual extra wildcard row
//!  │
//!  │ Specialize with `(,,)`:
//!  ├─┐ Patterns:
//!  │ │   1. `[true, _, true]`
//!  │ │   2. `[_, true, _]`
//!  │ │   3. `[_, _, _]`
//!  │ │
//!  │ │ There are missing constructors in the first column (namely `false`), hence
//!  │ │ `true` is irrelevant for rows 2 and 3.
//!  │ │
//!  │ │ Specialize with `true`:
//!  │ ├─┐ Patterns:
//!  │ │ │   1. `[_, true]`
//!  │ │ │   2. `[true, _]` // now exploring irrelevant cases
//!  │ │ │   3. `[_, _]`    // now exploring irrelevant cases
//!  │ │ │
//!  │ │ │ There are missing constructors in the first column (namely `false`), hence
//!  │ │ │ `true` is irrelevant for rows 1 and 3.
//!  │ │ │
//!  │ │ │ Specialize with `true`:
//!  │ │ ├─┐ Patterns:
//!  │ │ │ │   1. `[true]` // now exploring irrelevant cases
//!  │ │ │ │   2. `[_]`    // now exploring irrelevant cases
//!  │ │ │ │   3. `[_]`    // now exploring irrelevant cases
//!  │ │ │ │
//!  │ │ │ │ The current case is irrelevant for all rows: we backtrack immediately.
//!  │ │ ├─┘
//!  │ │ │
//!  │ │ │ Specialize with `false`:
//!  │ │ ├─┐ Patterns:
//!  │ │ │ │   1. `[true]`
//!  │ │ │ │   3. `[_]`    // now exploring irrelevant cases
//!  │ │ │ │
//!  │ │ │ │ Specialize with `true`:
//!  │ │ │ ├─┐ Patterns:
//!  │ │ │ │ │   1. `[]`
//!  │ │ │ │ │   3. `[]`    // now exploring irrelevant cases
//!  │ │ │ │ │
//!  │ │ │ │ │ Row 1 is therefore useful.
//!  │ │ │ ├─┘
//! <etc...>
//! ```
//!
//! Relevancy allowed us to skip the case `(true, true, _)` entirely. In some cases this pruning can
//! give drastic speedups. The case this was built for is the following (#118437):
//!
//! ```ignore(illustrative)
//! match foo {
//!     (true, _, _, _, ..) => 1,
//!     (_, true, _, _, ..) => 2,
//!     (_, _, true, _, ..) => 3,
//!     (_, _, _, true, ..) => 4,
//!     ...
//! }
//! ```
//!
//! Without considering relevancy, we would explore all 2^n combinations of the `true` and `Missing`
//! constructors. Relevancy tells us that e.g. `(true, true, false, false, false, ...)` is
//! irrelevant for all the rows. This allows us to skip all cases with more than one `true`
//! constructor, changing the runtime from exponential to linear.
//!
//!
//! ## Relevancy and exhaustiveness
//!
//! For exhaustiveness, we do something slightly different w.r.t relevancy: we do not report
//! witnesses of non-exhaustiveness that are irrelevant for the virtual wildcard row. For example,
//! in:
//!
//! ```ignore(illustrative)
//! match foo {
//!     (true, true) => {}
//! }
//! ```
//!
//! we only report `(false, _)` as missing. This was a deliberate choice made early in the
//! development of rust, for diagnostic and performance purposes. As showed in the previous section,
//! ignoring irrelevant cases preserves usefulness, so this choice still correctly computes whether
//! a match is exhaustive.
//!
//!
//!
//! # Or-patterns
//!
//! What we have described so far works well if there are no or-patterns. To handle them, if the
//! first pattern of any row in the matrix is an or-pattern, we expand it by duplicating the rest of
//! the row as necessary. For code reuse, this is implemented as "specializing with the `Or`
//! constructor".
//!
//! This makes usefulness tracking subtle, because we also want to compute whether an alternative of
//! an or-pattern is redundant, e.g. in `Some(_) | Some(0)`. We therefore track usefulness of each
//! subpattern of the match.
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
//! # Usefulness vs reachability, validity, and empty patterns
//!
//! This is likely the subtlest aspect of the algorithm. To be fully precise, a match doesn't
//! operate on a value, it operates on a place. In certain unsafe circumstances, it is possible for
//! a place to not contain valid data for its type. This has subtle consequences for empty types.
//! Take the following:
//!
//! ```rust
//! enum Void {}
//! let x: u8 = 0;
//! let ptr: *const Void = &x as *const u8 as *const Void;
//! unsafe {
//!     match *ptr {
//!         _ => println!("Reachable!"),
//!     }
//! }
//! ```
//!
//! In this example, `ptr` is a valid pointer pointing to a place with invalid data. The `_` pattern
//! does not look at the contents of `*ptr`, so this is ok and the arm is taken. In other words,
//! despite the place we are inspecting being of type `Void`, there is a reachable arm. If the
//! arm had a binding however:
//!
//! ```rust
//! # #[derive(Copy, Clone)]
//! # enum Void {}
//! # let x: u8 = 0;
//! # let ptr: *const Void = &x as *const u8 as *const Void;
//! # unsafe {
//! match *ptr {
//!     _a => println!("Unreachable!"),
//! }
//! # }
//! ```
//!
//! Here the binding loads the value of type `Void` from the `*ptr` place. In this example, this
//! causes UB since the data is not valid. In the general case, this asserts validity of the data at
//! `*ptr`. Either way, this arm will never be taken.
//!
//! Finally, let's consider the empty match `match *ptr {}`. If we consider this exhaustive, then
//! having invalid data at `*ptr` is invalid. In other words, the empty match is semantically
//! equivalent to the `_a => ...` match. In the interest of explicitness, we prefer the case with an
//! arm, hence we won't tell the user to remove the `_a` arm. In other words, the `_a` arm is
//! unreachable yet not redundant. This is why we lint on redundant arms rather than unreachable
//! arms, despite the fact that the lint says "unreachable".
//!
//! These considerations only affects certain places, namely those that can contain non-valid data
//! without UB. These are: pointer dereferences, reference dereferences, and union field accesses.
//! We track in the algorithm whether a given place is known to contain valid data. This is done
//! first by inspecting the scrutinee syntactically (which gives us `cx.known_valid_scrutinee`), and
//! then by tracking validity of each column of the matrix (which correspond to places) as we
//! recurse into subpatterns. That second part is done through [`PlaceValidity`], most notably
//! [`PlaceValidity::specialize`].
//!
//! Having said all that, we don't fully follow what's been presented in this section. For
//! backwards-compatibility, we ignore place validity when checking whether a pattern is required
//! for exhaustiveness in two cases: when the `exhaustive_patterns` feature gate is on, or when the
//! match scrutinee itself has type `!` or `EmptyEnum`. I (Nadrieril) hope to deprecate this
//! exception.
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
//! actually does (it tracks usefulness as a boolean on each row).
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
//!  │ │ │ │ │ We note arm 1 is useful (by `Pair(Some(0), true)`).
//!  │ │ │ ├─┘
//!  │ │ │ │
//!  │ │ │ │ Specialize with `false`:
//!  │ │ │ ├─┐ Patterns:
//!  │ │ │ │ │   1. `[]`
//!  │ │ │ │ │   3. `[]`
//!  │ │ │ │ │
//!  │ │ │ │ │ We note arm 1 is useful (by `Pair(Some(0), false)`).
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
//!  │ │ │ │ │ We note arm 2 is useful (by `Pair(Some(1..), false)`).
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
//!  │ │ │ │ We note arm 2 is useful (by `Pair(None, false)`).
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
//! - Arm 3 is redundant (it was never marked as useful);
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
//!   - `ui/pattern/deref-patterns`
//!   - probably many others
//!
//! I (Nadrieril) prefer to put new tests in `ui/pattern/usefulness` unless there's a specific
//! reason not to, for example if they crucially depend on a particular feature like `or_patterns`.

use std::fmt;

#[cfg(feature = "rustc")]
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_index::bit_set::DenseBitSet;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use self::PlaceValidity::*;
use crate::constructor::{Constructor, ConstructorSet, IntRange};
use crate::pat::{DeconstructedPat, PatId, PatOrWild, WitnessPat};
use crate::{MatchArm, PatCx, PrivateUninhabitedField, checks};
#[cfg(not(feature = "rustc"))]
pub fn ensure_sufficient_stack<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// A pattern is a "branch" if it is the immediate child of an or-pattern, or if it is the whole
/// pattern of a match arm. These are the patterns that can be meaningfully considered "redundant",
/// since e.g. `0` in `(0, 1)` cannot be redundant on its own.
///
/// We track for each branch pattern whether it is useful, and if not why.
struct BranchPatUsefulness<'p, Cx: PatCx> {
    /// Whether this pattern is useful.
    useful: bool,
    /// A set of patterns that:
    /// - come before this one in the match;
    /// - intersect this one;
    /// - at the end of the algorithm, if `!self.useful`, their union covers this pattern.
    covered_by: FxHashSet<&'p DeconstructedPat<Cx>>,
}

impl<'p, Cx: PatCx> BranchPatUsefulness<'p, Cx> {
    /// Update `self` with the usefulness information found in `row`.
    fn update(&mut self, row: &MatrixRow<'p, Cx>, matrix: &Matrix<'p, Cx>) {
        self.useful |= row.useful;
        // This deserves an explanation: `intersects_at_least` does not contain all intersections
        // because we skip irrelevant values (see the docs for `intersects_at_least` for an
        // example). Yet we claim this suffices to build a covering set.
        //
        // Let `p` be our pattern. Assume it is found not useful. For a value `v`, if the value was
        // relevant then we explored that value and found that there was another pattern `q` before
        // `p` that matches it too. We therefore recorded an intersection with `q`. If `v` was
        // irrelevant, we know there's another value `v2` that matches strictly fewer rows (while
        // still matching our row) and is relevant. Since `p` is not useful, there must have been a
        // `q` before `p` that matches `v2`, and we recorded that intersection. Since `v2` matches
        // strictly fewer rows than `v`, `q` also matches `v`. In either case, we recorded in
        // `intersects_at_least` a pattern that matches `v`. Hence using `intersects_at_least` is
        // sufficient to build a covering set.
        for row_id in row.intersects_at_least.iter() {
            let row = &matrix.rows[row_id];
            if row.useful && !row.is_under_guard {
                if let PatOrWild::Pat(intersecting) = row.head() {
                    self.covered_by.insert(intersecting);
                }
            }
        }
    }

    /// Check whether this pattern is redundant, and if so explain why.
    fn is_redundant(&self) -> Option<RedundancyExplanation<'p, Cx>> {
        if self.useful {
            None
        } else {
            // We avoid instability by sorting by `uid`. The order of `uid`s only depends on the
            // pattern structure.
            #[cfg_attr(feature = "rustc", allow(rustc::potential_query_instability))]
            let mut covered_by: Vec<_> = self.covered_by.iter().copied().collect();
            covered_by.sort_by_key(|pat| pat.uid); // sort to avoid instability
            Some(RedundancyExplanation { covered_by })
        }
    }
}

impl<'p, Cx: PatCx> Default for BranchPatUsefulness<'p, Cx> {
    fn default() -> Self {
        Self { useful: Default::default(), covered_by: Default::default() }
    }
}

/// Context that provides information for usefulness checking.
struct UsefulnessCtxt<'a, 'p, Cx: PatCx> {
    /// The context for type information.
    tycx: &'a Cx,
    /// Track information about the usefulness of branch patterns (see definition of "branch
    /// pattern" at [`BranchPatUsefulness`]).
    branch_usefulness: FxHashMap<PatId, BranchPatUsefulness<'p, Cx>>,
    // Ideally this field would have type `Limit`, but this crate is used by
    // rust-analyzer which cannot have a dependency on `Limit`, because `Limit`
    // is from crate `rustc_session` which uses unstable Rust features.
    complexity_limit: usize,
    complexity_level: usize,
}

impl<'a, 'p, Cx: PatCx> UsefulnessCtxt<'a, 'p, Cx> {
    fn increase_complexity_level(&mut self, complexity_add: usize) -> Result<(), Cx::Error> {
        self.complexity_level += complexity_add;
        if self.complexity_level <= self.complexity_limit {
            Ok(())
        } else {
            self.tycx.complexity_exceeded()
        }
    }
}

/// Context that provides information local to a place under investigation.
struct PlaceCtxt<'a, Cx: PatCx> {
    cx: &'a Cx,
    /// Type of the place under investigation.
    ty: &'a Cx::Ty,
}

impl<'a, Cx: PatCx> Copy for PlaceCtxt<'a, Cx> {}
impl<'a, Cx: PatCx> Clone for PlaceCtxt<'a, Cx> {
    fn clone(&self) -> Self {
        Self { cx: self.cx, ty: self.ty }
    }
}

impl<'a, Cx: PatCx> fmt::Debug for PlaceCtxt<'a, Cx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("PlaceCtxt").field("ty", self.ty).finish()
    }
}

impl<'a, Cx: PatCx> PlaceCtxt<'a, Cx> {
    fn ctor_arity(&self, ctor: &Constructor<Cx>) -> usize {
        self.cx.ctor_arity(ctor, self.ty)
    }
    fn wild_from_ctor(&self, ctor: Constructor<Cx>) -> WitnessPat<Cx> {
        WitnessPat::wild_from_ctor(self.cx, ctor, self.ty.clone())
    }
}

/// Track whether a given place (aka column) is known to contain a valid value or not.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PlaceValidity {
    ValidOnly,
    MaybeInvalid,
}

impl PlaceValidity {
    pub fn from_bool(is_valid_only: bool) -> Self {
        if is_valid_only { ValidOnly } else { MaybeInvalid }
    }

    fn is_known_valid(self) -> bool {
        matches!(self, ValidOnly)
    }

    /// If the place has validity given by `self` and we read that the value at the place has
    /// constructor `ctor`, this computes what we can assume about the validity of the constructor
    /// fields.
    ///
    /// Pending further opsem decisions, the current behavior is: validity is preserved, except
    /// inside `&` and union fields where validity is reset to `MaybeInvalid`.
    fn specialize<Cx: PatCx>(self, ctor: &Constructor<Cx>) -> Self {
        // We preserve validity except when we go inside a reference or a union field.
        if matches!(ctor, Constructor::Ref | Constructor::DerefPattern(_) | Constructor::UnionField)
        {
            // Validity of `x: &T` does not imply validity of `*x: T`.
            MaybeInvalid
        } else {
            self
        }
    }
}

impl fmt::Display for PlaceValidity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ValidOnly => "✓",
            MaybeInvalid => "?",
        };
        write!(f, "{s}")
    }
}

/// Data about a place under investigation. Its methods contain a lot of the logic used to analyze
/// the constructors in the matrix.
struct PlaceInfo<Cx: PatCx> {
    /// The type of the place.
    ty: Cx::Ty,
    /// Whether the place is a private uninhabited field. If so we skip this field during analysis
    /// so that we don't observe its emptiness.
    private_uninhabited: bool,
    /// Whether the place is known to contain valid data.
    validity: PlaceValidity,
    /// Whether the place is the scrutinee itself or a subplace of it.
    is_scrutinee: bool,
}

impl<Cx: PatCx> PlaceInfo<Cx> {
    /// Given a constructor for the current place, we return one `PlaceInfo` for each field of the
    /// constructor.
    fn specialize(
        &self,
        cx: &Cx,
        ctor: &Constructor<Cx>,
    ) -> impl Iterator<Item = Self> + ExactSizeIterator {
        let ctor_sub_tys = cx.ctor_sub_tys(ctor, &self.ty);
        let ctor_sub_validity = self.validity.specialize(ctor);
        ctor_sub_tys.map(move |(ty, PrivateUninhabitedField(private_uninhabited))| PlaceInfo {
            ty,
            private_uninhabited,
            validity: ctor_sub_validity,
            is_scrutinee: false,
        })
    }

    /// This analyzes a column of constructors corresponding to the current place. It returns a pair
    /// `(split_ctors, missing_ctors)`.
    ///
    /// `split_ctors` is a splitted list of constructors that cover the whole type. This will be
    /// used to specialize the matrix.
    ///
    /// `missing_ctors` is a list of the constructors not found in the column, for reporting
    /// purposes.
    fn split_column_ctors<'a>(
        &self,
        cx: &Cx,
        ctors: impl Iterator<Item = &'a Constructor<Cx>> + Clone,
    ) -> Result<(SmallVec<[Constructor<Cx>; 1]>, Vec<Constructor<Cx>>), Cx::Error>
    where
        Cx: 'a,
    {
        debug!(?self.ty);
        if self.private_uninhabited {
            // Skip the whole column
            return Ok((smallvec![Constructor::PrivateUninhabited], vec![]));
        }

        if ctors.clone().any(|c| matches!(c, Constructor::Or)) {
            // If any constructor is `Or`, we expand or-patterns.
            return Ok((smallvec![Constructor::Or], vec![]));
        }

        let ctors_for_ty = cx.ctors_for_ty(&self.ty)?;
        debug!(?ctors_for_ty);

        // We treat match scrutinees of type `!` or `EmptyEnum` differently.
        let is_toplevel_exception =
            self.is_scrutinee && matches!(ctors_for_ty, ConstructorSet::NoConstructors);
        // Whether empty patterns are counted as useful or not. We only warn an empty arm unreachable if
        // it is guaranteed unreachable by the opsem (i.e. if the place is `known_valid`).
        // We don't want to warn empty patterns as unreachable by default just yet. We will in a
        // later version of rust or under a different lint name, see
        // https://github.com/rust-lang/rust/pull/129103.
        let empty_arms_are_unreachable = self.validity.is_known_valid()
            && (is_toplevel_exception || cx.is_exhaustive_patterns_feature_on());
        // Whether empty patterns can be omitted for exhaustiveness. We ignore place validity in the
        // toplevel exception and `exhaustive_patterns` cases for backwards compatibility.
        let can_omit_empty_arms = self.validity.is_known_valid()
            || is_toplevel_exception
            || cx.is_exhaustive_patterns_feature_on();

        // Analyze the constructors present in this column.
        let mut split_set = ctors_for_ty.split(ctors);
        debug!(?split_set);
        let all_missing = split_set.present.is_empty();

        // Build the set of constructors we will specialize with. It must cover the whole type, so
        // we add `Missing` to represent the missing ones. This is explained under "Constructor
        // Splitting" at the top of this file.
        let mut split_ctors = split_set.present;
        if !(split_set.missing.is_empty()
            && (split_set.missing_empty.is_empty() || empty_arms_are_unreachable))
        {
            split_ctors.push(Constructor::Missing);
        }

        // Which empty constructors are considered missing. We ensure that
        // `!missing_ctors.is_empty() => split_ctors.contains(Missing)`. The converse usually holds
        // except when `!self.validity.is_known_valid()`.
        let mut missing_ctors = split_set.missing;
        if !can_omit_empty_arms {
            missing_ctors.append(&mut split_set.missing_empty);
        }

        // Whether we should report "Enum::A and Enum::C are missing" or "_ is missing". At the top
        // level we prefer to list all constructors.
        let report_individual_missing_ctors = self.is_scrutinee || !all_missing;
        if !missing_ctors.is_empty() && !report_individual_missing_ctors {
            // Report `_` as missing.
            missing_ctors = vec![Constructor::Wildcard];
        } else if missing_ctors.iter().any(|c| c.is_non_exhaustive()) && !cx.exhaustive_witnesses()
        {
            // We need to report a `_` anyway, so listing other constructors would be redundant.
            // `NonExhaustive` is displayed as `_` just like `Wildcard`, but it will be picked
            // up by diagnostics to add a note about why `_` is required here.
            missing_ctors = vec![Constructor::NonExhaustive];
        }

        Ok((split_ctors, missing_ctors))
    }
}

impl<Cx: PatCx> Clone for PlaceInfo<Cx> {
    fn clone(&self) -> Self {
        Self {
            ty: self.ty.clone(),
            private_uninhabited: self.private_uninhabited,
            validity: self.validity,
            is_scrutinee: self.is_scrutinee,
        }
    }
}

/// Represents a pattern-tuple under investigation.
// The three lifetimes are:
// - 'p coming from the input
// - Cx global compilation context
struct PatStack<'p, Cx: PatCx> {
    // Rows of len 1 are very common, which is why `SmallVec[_; 2]` works well.
    pats: SmallVec<[PatOrWild<'p, Cx>; 2]>,
    /// Sometimes we know that as far as this row is concerned, the current case is already handled
    /// by a different, more general, case. When the case is irrelevant for all rows this allows us
    /// to skip a case entirely. This is purely an optimization. See at the top for details.
    relevant: bool,
}

impl<'p, Cx: PatCx> Clone for PatStack<'p, Cx> {
    fn clone(&self) -> Self {
        Self { pats: self.pats.clone(), relevant: self.relevant }
    }
}

impl<'p, Cx: PatCx> PatStack<'p, Cx> {
    fn from_pattern(pat: &'p DeconstructedPat<Cx>) -> Self {
        PatStack { pats: smallvec![PatOrWild::Pat(pat)], relevant: true }
    }

    fn len(&self) -> usize {
        self.pats.len()
    }

    fn head(&self) -> PatOrWild<'p, Cx> {
        self.pats[0]
    }

    fn iter(&self) -> impl Iterator<Item = PatOrWild<'p, Cx>> {
        self.pats.iter().copied()
    }

    // Expand the first or-pattern into its subpatterns. Only useful if the pattern is an
    // or-pattern. Panics if `self` is empty.
    fn expand_or_pat(&self) -> impl Iterator<Item = PatStack<'p, Cx>> {
        self.head().expand_or_pat().into_iter().map(move |pat| {
            let mut new = self.clone();
            new.pats[0] = pat;
            new
        })
    }

    /// This computes `specialize(ctor, self)`. See top of the file for explanations.
    /// Only call if `ctor.is_covered_by(self.head().ctor())` is true.
    fn pop_head_constructor(
        &self,
        cx: &Cx,
        ctor: &Constructor<Cx>,
        ctor_arity: usize,
        ctor_is_relevant: bool,
    ) -> Result<PatStack<'p, Cx>, Cx::Error> {
        let head_pat = self.head();
        if head_pat.as_pat().is_some_and(|pat| pat.arity() > ctor_arity) {
            // Arity can be smaller in case of variable-length slices, but mustn't be larger.
            return Err(cx.bug(format_args!(
                "uncaught type error: pattern {:?} has inconsistent arity (expected arity <= {ctor_arity})",
                head_pat.as_pat().unwrap()
            )));
        }
        // We pop the head pattern and push the new fields extracted from the arguments of
        // `self.head()`.
        let mut new_pats = head_pat.specialize(ctor, ctor_arity);
        new_pats.extend_from_slice(&self.pats[1..]);
        // `ctor` is relevant for this row if it is the actual constructor of this row, or if the
        // row has a wildcard and `ctor` is relevant for wildcards.
        let ctor_is_relevant =
            !matches!(self.head().ctor(), Constructor::Wildcard) || ctor_is_relevant;
        Ok(PatStack { pats: new_pats, relevant: self.relevant && ctor_is_relevant })
    }
}

impl<'p, Cx: PatCx> fmt::Debug for PatStack<'p, Cx> {
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
struct MatrixRow<'p, Cx: PatCx> {
    // The patterns in the row.
    pats: PatStack<'p, Cx>,
    /// Whether the original arm had a guard. This is inherited when specializing.
    is_under_guard: bool,
    /// When we specialize, we remember which row of the original matrix produced a given row of the
    /// specialized matrix. When we unspecialize, we use this to propagate usefulness back up the
    /// callstack. On creation, this stores the index of the original match arm.
    parent_row: usize,
    /// False when the matrix is just built. This is set to `true` by
    /// [`compute_exhaustiveness_and_usefulness`] if the arm is found to be useful.
    /// This is reset to `false` when specializing.
    useful: bool,
    /// Tracks some rows above this one that have an intersection with this one, i.e. such that
    /// there is a value that matches both rows.
    /// Because of relevancy we may miss some intersections. The intersections we do find are
    /// correct. In other words, this is an underapproximation of the real set of intersections.
    ///
    /// For example:
    /// ```rust,ignore(illustrative)
    /// match ... {
    ///     (true, _, _) => {} // `intersects_at_least = []`
    ///     (_, true, 0..=10) => {} // `intersects_at_least = []`
    ///     (_, true, 5..15) => {} // `intersects_at_least = [1]`
    /// }
    /// ```
    /// Here the `(true, true)` case is irrelevant. Since we skip it, we will not detect that row 0
    /// intersects rows 1 and 2.
    intersects_at_least: DenseBitSet<usize>,
    /// Whether the head pattern is a branch (see definition of "branch pattern" at
    /// [`BranchPatUsefulness`])
    head_is_branch: bool,
}

impl<'p, Cx: PatCx> MatrixRow<'p, Cx> {
    fn new(arm: &MatchArm<'p, Cx>, arm_id: usize) -> Self {
        MatrixRow {
            pats: PatStack::from_pattern(arm.pat),
            parent_row: arm_id,
            is_under_guard: arm.has_guard,
            useful: false,
            intersects_at_least: DenseBitSet::new_empty(0), // Initialized in `Matrix::push`.
            // This pattern is a branch because it comes from a match arm.
            head_is_branch: true,
        }
    }

    fn len(&self) -> usize {
        self.pats.len()
    }

    fn head(&self) -> PatOrWild<'p, Cx> {
        self.pats.head()
    }

    fn iter(&self) -> impl Iterator<Item = PatOrWild<'p, Cx>> {
        self.pats.iter()
    }

    // Expand the first or-pattern (if any) into its subpatterns. Panics if `self` is empty.
    fn expand_or_pat(&self, parent_row: usize) -> impl Iterator<Item = MatrixRow<'p, Cx>> {
        let is_or_pat = self.pats.head().is_or_pat();
        self.pats.expand_or_pat().map(move |patstack| MatrixRow {
            pats: patstack,
            parent_row,
            is_under_guard: self.is_under_guard,
            useful: false,
            intersects_at_least: DenseBitSet::new_empty(0), // Initialized in `Matrix::push`.
            head_is_branch: is_or_pat,
        })
    }

    /// This computes `specialize(ctor, self)`. See top of the file for explanations.
    /// Only call if `ctor.is_covered_by(self.head().ctor())` is true.
    fn pop_head_constructor(
        &self,
        cx: &Cx,
        ctor: &Constructor<Cx>,
        ctor_arity: usize,
        ctor_is_relevant: bool,
        parent_row: usize,
    ) -> Result<MatrixRow<'p, Cx>, Cx::Error> {
        Ok(MatrixRow {
            pats: self.pats.pop_head_constructor(cx, ctor, ctor_arity, ctor_is_relevant)?,
            parent_row,
            is_under_guard: self.is_under_guard,
            useful: false,
            intersects_at_least: DenseBitSet::new_empty(0), // Initialized in `Matrix::push`.
            head_is_branch: false,
        })
    }
}

impl<'p, Cx: PatCx> fmt::Debug for MatrixRow<'p, Cx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.pats.fmt(f)
    }
}

/// A 2D matrix. Represents a list of pattern-tuples under investigation.
///
/// Invariant: each row must have the same length, and each column must have the same type.
///
/// Invariant: the first column must not contain or-patterns. This is handled by
/// [`Matrix::push`].
///
/// In fact each column corresponds to a place inside the scrutinee of the match. E.g. after
/// specializing `(,)` and `Some` on a pattern of type `(Option<u32>, bool)`, the first column of
/// the matrix will correspond to `scrutinee.0.Some.0` and the second column to `scrutinee.1`.
#[derive(Clone)]
struct Matrix<'p, Cx: PatCx> {
    /// Vector of rows. The rows must form a rectangular 2D array. Moreover, all the patterns of
    /// each column must have the same type. Each column corresponds to a place within the
    /// scrutinee.
    rows: Vec<MatrixRow<'p, Cx>>,
    /// Track info about each place. Each place corresponds to a column in `rows`, and their types
    /// must match.
    place_info: SmallVec<[PlaceInfo<Cx>; 2]>,
    /// Track whether the virtual wildcard row used to compute exhaustiveness is relevant. See top
    /// of the file for details on relevancy.
    wildcard_row_is_relevant: bool,
}

impl<'p, Cx: PatCx> Matrix<'p, Cx> {
    /// Pushes a new row to the matrix. Internal method, prefer [`Matrix::new`].
    fn push(&mut self, mut row: MatrixRow<'p, Cx>) {
        row.intersects_at_least = DenseBitSet::new_empty(self.rows.len());
        self.rows.push(row);
    }

    /// Build a new matrix from an iterator of `MatchArm`s.
    fn new(arms: &[MatchArm<'p, Cx>], scrut_ty: Cx::Ty, scrut_validity: PlaceValidity) -> Self {
        let place_info = PlaceInfo {
            ty: scrut_ty,
            private_uninhabited: false,
            validity: scrut_validity,
            is_scrutinee: true,
        };
        let mut matrix = Matrix {
            rows: Vec::with_capacity(arms.len()),
            place_info: smallvec![place_info],
            wildcard_row_is_relevant: true,
        };
        for (arm_id, arm) in arms.iter().enumerate() {
            matrix.push(MatrixRow::new(arm, arm_id));
        }
        matrix
    }

    fn head_place(&self) -> Option<&PlaceInfo<Cx>> {
        self.place_info.first()
    }
    fn column_count(&self) -> usize {
        self.place_info.len()
    }

    fn rows(
        &self,
    ) -> impl Iterator<Item = &MatrixRow<'p, Cx>> + Clone + DoubleEndedIterator + ExactSizeIterator
    {
        self.rows.iter()
    }
    fn rows_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut MatrixRow<'p, Cx>> + DoubleEndedIterator + ExactSizeIterator
    {
        self.rows.iter_mut()
    }

    /// Iterate over the first pattern of each row.
    fn heads(&self) -> impl Iterator<Item = PatOrWild<'p, Cx>> + Clone {
        self.rows().map(|r| r.head())
    }

    /// This computes `specialize(ctor, self)`. See top of the file for explanations.
    fn specialize_constructor(
        &self,
        pcx: &PlaceCtxt<'_, Cx>,
        ctor: &Constructor<Cx>,
        ctor_is_relevant: bool,
    ) -> Result<Matrix<'p, Cx>, Cx::Error> {
        if matches!(ctor, Constructor::Or) {
            // Specializing with `Or` means expanding rows with or-patterns.
            let mut matrix = Matrix {
                rows: Vec::new(),
                place_info: self.place_info.clone(),
                wildcard_row_is_relevant: self.wildcard_row_is_relevant,
            };
            for (i, row) in self.rows().enumerate() {
                for new_row in row.expand_or_pat(i) {
                    matrix.push(new_row);
                }
            }
            Ok(matrix)
        } else {
            let subfield_place_info = self.place_info[0].specialize(pcx.cx, ctor);
            let arity = subfield_place_info.len();
            let specialized_place_info =
                subfield_place_info.chain(self.place_info[1..].iter().cloned()).collect();
            let mut matrix = Matrix {
                rows: Vec::new(),
                place_info: specialized_place_info,
                wildcard_row_is_relevant: self.wildcard_row_is_relevant && ctor_is_relevant,
            };
            for (i, row) in self.rows().enumerate() {
                if ctor.is_covered_by(pcx.cx, row.head().ctor())? {
                    let new_row =
                        row.pop_head_constructor(pcx.cx, ctor, arity, ctor_is_relevant, i)?;
                    matrix.push(new_row);
                }
            }
            Ok(matrix)
        }
    }

    /// Recover row usefulness and intersection information from a processed specialized matrix.
    /// `specialized` must come from `self.specialize_constructor`.
    fn unspecialize(&mut self, specialized: Self) {
        for child_row in specialized.rows() {
            let parent_row_id = child_row.parent_row;
            let parent_row = &mut self.rows[parent_row_id];
            // A parent row is useful if any of its children is.
            parent_row.useful |= child_row.useful;
            for child_intersection in child_row.intersects_at_least.iter() {
                // Convert the intersecting ids into ids for the parent matrix.
                let parent_intersection = specialized.rows[child_intersection].parent_row;
                // Note: self-intersection can happen with or-patterns.
                if parent_intersection != parent_row_id {
                    parent_row.intersects_at_least.insert(parent_intersection);
                }
            }
        }
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
/// | ✓     | ?                 | // validity
/// ```
impl<'p, Cx: PatCx> fmt::Debug for Matrix<'p, Cx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n")?;

        let mut pretty_printed_matrix: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(|pat| format!("{pat:?}")).collect())
            .collect();
        pretty_printed_matrix
            .push(self.place_info.iter().map(|place| format!("{}", place.validity)).collect());

        let column_count = self.column_count();
        assert!(self.rows.iter().all(|row| row.len() == column_count));
        assert!(self.place_info.len() == column_count);
        let column_widths: Vec<usize> = (0..column_count)
            .map(|col| pretty_printed_matrix.iter().map(|row| row[col].len()).max().unwrap_or(0))
            .collect();

        for (row_i, row) in pretty_printed_matrix.into_iter().enumerate() {
            let is_validity_row = row_i == self.rows.len();
            let sep = if is_validity_row { "|" } else { "+" };
            write!(f, "{sep}")?;
            for (column, pat_str) in row.into_iter().enumerate() {
                write!(f, " ")?;
                write!(f, "{:1$}", pat_str, column_widths[column])?;
                write!(f, " {sep}")?;
            }
            if is_validity_row {
                write!(f, " // validity")?;
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
#[derive(Debug)]
struct WitnessStack<Cx: PatCx>(Vec<WitnessPat<Cx>>);

impl<Cx: PatCx> Clone for WitnessStack<Cx> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<Cx: PatCx> WitnessStack<Cx> {
    /// Asserts that the witness contains a single pattern, and returns it.
    fn single_pattern(self) -> WitnessPat<Cx> {
        assert_eq!(self.0.len(), 1);
        self.0.into_iter().next().unwrap()
    }

    /// Reverses specialization by the `Missing` constructor by pushing a whole new pattern.
    fn push_pattern(&mut self, pat: WitnessPat<Cx>) {
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
    fn apply_constructor(
        mut self,
        pcx: &PlaceCtxt<'_, Cx>,
        ctor: &Constructor<Cx>,
    ) -> SmallVec<[Self; 1]> {
        let len = self.0.len();
        let arity = pcx.ctor_arity(ctor);
        let fields: Vec<_> = self.0.drain((len - arity)..).rev().collect();
        if matches!(ctor, Constructor::UnionField)
            && fields.iter().filter(|p| !matches!(p.ctor(), Constructor::Wildcard)).count() >= 2
        {
            // Convert a `Union { a: p, b: q }` witness into `Union { a: p }` and `Union { b: q }`.
            // First add `Union { .. }` to `self`.
            self.0.push(WitnessPat::wild_from_ctor(pcx.cx, ctor.clone(), pcx.ty.clone()));
            fields
                .into_iter()
                .enumerate()
                .filter(|(_, p)| !matches!(p.ctor(), Constructor::Wildcard))
                .map(|(i, p)| {
                    let mut ret = self.clone();
                    // Fill the `i`th field of the union with `p`.
                    ret.0.last_mut().unwrap().fields[i] = p;
                    ret
                })
                .collect()
        } else {
            self.0.push(WitnessPat::new(ctor.clone(), fields, pcx.ty.clone()));
            smallvec![self]
        }
    }
}

/// Represents a set of pattern-tuples that are witnesses of non-exhaustiveness for error
/// reporting. This has similar invariants as `Matrix` does.
///
/// The `WitnessMatrix` returned by [`compute_exhaustiveness_and_usefulness`] obeys the invariant
/// that the union of the input `Matrix` and the output `WitnessMatrix` together matches the type
/// exhaustively.
///
/// Just as the `Matrix` starts with a single column, by the end of the algorithm, this has a single
/// column, which contains the patterns that are missing for the match to be exhaustive.
#[derive(Debug)]
struct WitnessMatrix<Cx: PatCx>(Vec<WitnessStack<Cx>>);

impl<Cx: PatCx> Clone for WitnessMatrix<Cx> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<Cx: PatCx> WitnessMatrix<Cx> {
    /// New matrix with no witnesses.
    fn empty() -> Self {
        WitnessMatrix(Vec::new())
    }
    /// New matrix with one `()` witness, i.e. with no columns.
    fn unit_witness() -> Self {
        WitnessMatrix(vec![WitnessStack(Vec::new())])
    }

    /// Whether this has any witnesses.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Asserts that there is a single column and returns the patterns in it.
    fn single_column(self) -> Vec<WitnessPat<Cx>> {
        self.0.into_iter().map(|w| w.single_pattern()).collect()
    }

    /// Reverses specialization by the `Missing` constructor by pushing a whole new pattern.
    fn push_pattern(&mut self, pat: WitnessPat<Cx>) {
        for witness in self.0.iter_mut() {
            witness.push_pattern(pat.clone())
        }
    }

    /// Reverses specialization by `ctor`. See the section on `unspecialize` at the top of the file.
    fn apply_constructor(
        &mut self,
        pcx: &PlaceCtxt<'_, Cx>,
        missing_ctors: &[Constructor<Cx>],
        ctor: &Constructor<Cx>,
    ) {
        // The `Or` constructor indicates that we expanded or-patterns. This doesn't affect
        // witnesses.
        if self.is_empty() || matches!(ctor, Constructor::Or) {
            return;
        }
        if matches!(ctor, Constructor::Missing) {
            // We got the special `Missing` constructor that stands for the constructors not present
            // in the match. For each missing constructor `c`, we add a `c(_, _, _)` witness
            // appropriately filled with wildcards.
            let mut ret = Self::empty();
            for ctor in missing_ctors {
                let pat = pcx.wild_from_ctor(ctor.clone());
                // Clone `self` and add `c(_, _, _)` to each of its witnesses.
                let mut wit_matrix = self.clone();
                wit_matrix.push_pattern(pat);
                ret.extend(wit_matrix);
            }
            *self = ret;
        } else {
            // Any other constructor we unspecialize as expected.
            for witness in std::mem::take(&mut self.0) {
                self.0.extend(witness.apply_constructor(pcx, ctor));
            }
        }
    }

    /// Merges the witnesses of two matrices. Their column types must match.
    fn extend(&mut self, other: Self) {
        self.0.extend(other.0)
    }
}

/// Collect ranges that overlap like `lo..=overlap`/`overlap..=hi`. Must be called during
/// exhaustiveness checking, if we find a singleton range after constructor splitting. This reuses
/// row intersection information to only detect ranges that truly overlap.
///
/// If two ranges overlapped, the split set will contain their intersection as a singleton.
/// Specialization will then select rows that match the overlap, and exhaustiveness will compute
/// which rows have an intersection that includes the overlap. That gives us all the info we need to
/// compute overlapping ranges without false positives.
///
/// We can however get false negatives because exhaustiveness does not explore all cases. See the
/// section on relevancy at the top of the file.
fn collect_overlapping_range_endpoints<'p, Cx: PatCx>(
    cx: &Cx,
    overlap_range: IntRange,
    matrix: &Matrix<'p, Cx>,
    specialized_matrix: &Matrix<'p, Cx>,
) {
    let overlap = overlap_range.lo;
    // Ranges that look like `lo..=overlap`.
    let mut prefixes: SmallVec<[_; 1]> = Default::default();
    // Ranges that look like `overlap..=hi`.
    let mut suffixes: SmallVec<[_; 1]> = Default::default();
    // Iterate on patterns that contained `overlap`. We iterate on `specialized_matrix` which
    // contains only rows that matched the current `ctor` as well as accurate intersection
    // information. It doesn't contain the column that contains the range; that can be found in
    // `matrix`.
    for (child_row_id, child_row) in specialized_matrix.rows().enumerate() {
        let PatOrWild::Pat(pat) = matrix.rows[child_row.parent_row].head() else { continue };
        let Constructor::IntRange(this_range) = pat.ctor() else { continue };
        // Don't lint when one of the ranges is a singleton.
        if this_range.is_singleton() {
            continue;
        }
        if this_range.lo == overlap {
            // `this_range` looks like `overlap..=this_range.hi`; it overlaps with any
            // ranges that look like `lo..=overlap`.
            if !prefixes.is_empty() {
                let overlaps_with: Vec<_> = prefixes
                    .iter()
                    .filter(|&&(other_child_row_id, _)| {
                        child_row.intersects_at_least.contains(other_child_row_id)
                    })
                    .map(|&(_, pat)| pat)
                    .collect();
                if !overlaps_with.is_empty() {
                    cx.lint_overlapping_range_endpoints(pat, overlap_range, &overlaps_with);
                }
            }
            suffixes.push((child_row_id, pat))
        } else if Some(this_range.hi) == overlap.plus_one() {
            // `this_range` looks like `this_range.lo..=overlap`; it overlaps with any
            // ranges that look like `overlap..=hi`.
            if !suffixes.is_empty() {
                let overlaps_with: Vec<_> = suffixes
                    .iter()
                    .filter(|&&(other_child_row_id, _)| {
                        child_row.intersects_at_least.contains(other_child_row_id)
                    })
                    .map(|&(_, pat)| pat)
                    .collect();
                if !overlaps_with.is_empty() {
                    cx.lint_overlapping_range_endpoints(pat, overlap_range, &overlaps_with);
                }
            }
            prefixes.push((child_row_id, pat))
        }
    }
}

/// Collect ranges that have a singleton gap between them.
fn collect_non_contiguous_range_endpoints<'p, Cx: PatCx>(
    cx: &Cx,
    gap_range: &IntRange,
    matrix: &Matrix<'p, Cx>,
) {
    let gap = gap_range.lo;
    // Ranges that look like `lo..gap`.
    let mut onebefore: SmallVec<[_; 1]> = Default::default();
    // Ranges that start on `gap+1` or singletons `gap+1`.
    let mut oneafter: SmallVec<[_; 1]> = Default::default();
    // Look through the column for ranges near the gap.
    for pat in matrix.heads() {
        let PatOrWild::Pat(pat) = pat else { continue };
        let Constructor::IntRange(this_range) = pat.ctor() else { continue };
        if gap == this_range.hi {
            onebefore.push(pat)
        } else if gap.plus_one() == Some(this_range.lo) {
            oneafter.push(pat)
        }
    }

    for pat_before in onebefore {
        cx.lint_non_contiguous_range_endpoints(pat_before, *gap_range, oneafter.as_slice());
    }
}

/// The core of the algorithm.
///
/// This recursively computes witnesses of the non-exhaustiveness of `matrix` (if any). Also tracks
/// usefulness of each row in the matrix (in `row.useful`). We track usefulness of subpatterns in
/// `mcx.branch_usefulness`.
///
/// The input `Matrix` and the output `WitnessMatrix` together match the type exhaustively.
///
/// The key steps are:
/// - specialization, where we dig into the rows that have a specific constructor and call ourselves
///     recursively;
/// - unspecialization, where we lift the results from the previous step into results for this step
///     (using `apply_constructor` and by updating `row.useful` for each parent row).
/// This is all explained at the top of the file.
#[instrument(level = "debug", skip(mcx), ret)]
fn compute_exhaustiveness_and_usefulness<'a, 'p, Cx: PatCx>(
    mcx: &mut UsefulnessCtxt<'a, 'p, Cx>,
    matrix: &mut Matrix<'p, Cx>,
) -> Result<WitnessMatrix<Cx>, Cx::Error> {
    debug_assert!(matrix.rows().all(|r| r.len() == matrix.column_count()));

    if !matrix.wildcard_row_is_relevant && matrix.rows().all(|r| !r.pats.relevant) {
        // Here we know that nothing will contribute further to exhaustiveness or usefulness. This
        // is purely an optimization: skipping this check doesn't affect correctness. See the top of
        // the file for details.
        return Ok(WitnessMatrix::empty());
    }

    let Some(place) = matrix.head_place() else {
        mcx.increase_complexity_level(matrix.rows().len())?;
        // The base case: there are no columns in the matrix. We are morally pattern-matching on ().
        // A row is useful iff it has no (unguarded) rows above it.
        let mut useful = true; // Whether the next row is useful.
        for (i, row) in matrix.rows_mut().enumerate() {
            row.useful = useful;
            row.intersects_at_least.insert_range(0..i);
            // The next rows stays useful if this one is under a guard.
            useful &= row.is_under_guard;
        }
        return if useful && matrix.wildcard_row_is_relevant {
            // The wildcard row is useful; the match is non-exhaustive.
            Ok(WitnessMatrix::unit_witness())
        } else {
            // Either the match is exhaustive, or we choose not to report anything because of
            // relevancy. See at the top for details.
            Ok(WitnessMatrix::empty())
        };
    };

    // Analyze the constructors present in this column.
    let ctors = matrix.heads().map(|p| p.ctor());
    let (split_ctors, missing_ctors) = place.split_column_ctors(mcx.tycx, ctors)?;

    let ty = &place.ty.clone(); // Clone it out so we can mutate `matrix` later.
    let pcx = &PlaceCtxt { cx: mcx.tycx, ty };
    let mut ret = WitnessMatrix::empty();
    for ctor in split_ctors {
        // Dig into rows that match `ctor`.
        debug!("specialize({:?})", ctor);
        // `ctor` is *irrelevant* if there's another constructor in `split_ctors` that matches
        // strictly fewer rows. In that case we can sometimes skip it. See the top of the file for
        // details.
        let ctor_is_relevant = matches!(ctor, Constructor::Missing)
            || missing_ctors.is_empty()
            || mcx.tycx.exhaustive_witnesses();
        let mut spec_matrix = matrix.specialize_constructor(pcx, &ctor, ctor_is_relevant)?;
        let mut witnesses = ensure_sufficient_stack(|| {
            compute_exhaustiveness_and_usefulness(mcx, &mut spec_matrix)
        })?;

        // Transform witnesses for `spec_matrix` into witnesses for `matrix`.
        witnesses.apply_constructor(pcx, &missing_ctors, &ctor);
        // Accumulate the found witnesses.
        ret.extend(witnesses);

        // Detect ranges that overlap on their endpoints.
        if let Constructor::IntRange(overlap_range) = ctor {
            if overlap_range.is_singleton()
                && spec_matrix.rows.len() >= 2
                && spec_matrix.rows.iter().any(|row| !row.intersects_at_least.is_empty())
            {
                collect_overlapping_range_endpoints(mcx.tycx, overlap_range, matrix, &spec_matrix);
            }
        }

        matrix.unspecialize(spec_matrix);
    }

    // Detect singleton gaps between ranges.
    if missing_ctors.iter().any(|c| matches!(c, Constructor::IntRange(..))) {
        for missing in &missing_ctors {
            if let Constructor::IntRange(gap) = missing {
                if gap.is_singleton() {
                    collect_non_contiguous_range_endpoints(mcx.tycx, gap, matrix);
                }
            }
        }
    }

    // Record usefulness of the branch patterns.
    for row in matrix.rows() {
        if row.head_is_branch {
            if let PatOrWild::Pat(pat) = row.head() {
                mcx.branch_usefulness.entry(pat.uid).or_default().update(row, matrix);
            }
        }
    }

    Ok(ret)
}

/// Indicates why a given pattern is considered redundant.
#[derive(Clone, Debug)]
pub struct RedundancyExplanation<'p, Cx: PatCx> {
    /// All the values matched by this pattern are already matched by the given set of patterns.
    /// This list is not guaranteed to be minimal but the contained patterns are at least guaranteed
    /// to intersect this pattern.
    pub covered_by: Vec<&'p DeconstructedPat<Cx>>,
}

/// Indicates whether or not a given arm is useful.
#[derive(Clone, Debug)]
pub enum Usefulness<'p, Cx: PatCx> {
    /// The arm is useful. This additionally carries a set of or-pattern branches that have been
    /// found to be redundant despite the overall arm being useful. Used only in the presence of
    /// or-patterns, otherwise it stays empty.
    Useful(Vec<(&'p DeconstructedPat<Cx>, RedundancyExplanation<'p, Cx>)>),
    /// The arm is redundant and can be removed without changing the behavior of the match
    /// expression.
    Redundant(RedundancyExplanation<'p, Cx>),
}

/// The output of checking a match for exhaustiveness and arm usefulness.
pub struct UsefulnessReport<'p, Cx: PatCx> {
    /// For each arm of the input, whether that arm is useful after the arms above it.
    pub arm_usefulness: Vec<(MatchArm<'p, Cx>, Usefulness<'p, Cx>)>,
    /// If the match is exhaustive, this is empty. If not, this contains witnesses for the lack of
    /// exhaustiveness.
    pub non_exhaustiveness_witnesses: Vec<WitnessPat<Cx>>,
    /// For each arm, a set of indices of arms above it that have non-empty intersection, i.e. there
    /// is a value matched by both arms. This may miss real intersections.
    pub arm_intersections: Vec<DenseBitSet<usize>>,
}

/// Computes whether a match is exhaustive and which of its arms are useful.
#[instrument(skip(tycx, arms), level = "debug")]
pub fn compute_match_usefulness<'p, Cx: PatCx>(
    tycx: &Cx,
    arms: &[MatchArm<'p, Cx>],
    scrut_ty: Cx::Ty,
    scrut_validity: PlaceValidity,
    complexity_limit: usize,
) -> Result<UsefulnessReport<'p, Cx>, Cx::Error> {
    // The analysis doesn't support deref patterns mixed with normal constructors; error if present.
    if tycx.match_may_contain_deref_pats() {
        checks::detect_mixed_deref_pat_ctors(tycx, arms)?;
    }

    let mut cx = UsefulnessCtxt {
        tycx,
        branch_usefulness: FxHashMap::default(),
        complexity_limit,
        complexity_level: 0,
    };
    let mut matrix = Matrix::new(arms, scrut_ty, scrut_validity);
    let non_exhaustiveness_witnesses = compute_exhaustiveness_and_usefulness(&mut cx, &mut matrix)?;

    let non_exhaustiveness_witnesses: Vec<_> = non_exhaustiveness_witnesses.single_column();
    let arm_usefulness: Vec<_> = arms
        .iter()
        .copied()
        .map(|arm| {
            debug!(?arm);
            let usefulness = cx.branch_usefulness.get(&arm.pat.uid).unwrap();
            let usefulness = if let Some(explanation) = usefulness.is_redundant() {
                Usefulness::Redundant(explanation)
            } else {
                let mut redundant_subpats = Vec::new();
                arm.pat.walk(&mut |subpat| {
                    if let Some(u) = cx.branch_usefulness.get(&subpat.uid) {
                        if let Some(explanation) = u.is_redundant() {
                            redundant_subpats.push((subpat, explanation));
                            false // stop recursing
                        } else {
                            true // keep recursing
                        }
                    } else {
                        true // keep recursing
                    }
                });
                Usefulness::Useful(redundant_subpats)
            };
            debug!(?usefulness);
            (arm, usefulness)
        })
        .collect();

    let arm_intersections: Vec<_> =
        matrix.rows().map(|row| row.intersects_at_least.clone()).collect();

    Ok(UsefulnessReport { arm_usefulness, non_exhaustiveness_witnesses, arm_intersections })
}
