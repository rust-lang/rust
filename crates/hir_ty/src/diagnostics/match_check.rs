//! This module implements match statement exhaustiveness checking and usefulness checking
//! for match arms.
//!
//! It is modeled on the rustc module `librustc_mir_build::hair::pattern::_match`, which
//! contains very detailed documentation about the algorithms used here. I've duplicated
//! most of that documentation below.
//!
//! This file includes the logic for exhaustiveness and usefulness checking for
//! pattern-matching. Specifically, given a list of patterns for a type, we can
//! tell whether:
//! - (a) the patterns cover every possible constructor for the type (exhaustiveness).
//! - (b) each pattern is necessary (usefulness).
//!
//! The algorithm implemented here is a modified version of the one described in
//! <http://moscova.inria.fr/~maranget/papers/warn/index.html>.
//! However, to save future implementors from reading the original paper, we
//! summarize the algorithm here to hopefully save time and be a little clearer
//! (without being so rigorous).
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
//! During the course of the algorithm, the rows of the matrix won't just be individual patterns,
//! but rather partially-deconstructed patterns in the form of a list of patterns. The paper
//! calls those pattern-vectors, and we will call them pattern-stacks. The same holds for the
//! new pattern `p`.
//!
//! For example, say we have the following:
//!
//! ```ignore
//! // x: (Option<bool>, Result<()>)
//! match x {
//!     (Some(true), _) => (),
//!     (None, Err(())) => (),
//!     (None, Err(_)) => (),
//! }
//! ```
//!
//! Here, the matrix `P` starts as:
//!
//! ```text
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
//!    * 1.1. `p_1 = c(r_1, .., r_a)`, i.e. the top of the stack has constructor `c`. We push onto
//!           the stack the arguments of this constructor, and return the result:
//!
//!          r_1, .., r_a, p_2, .., p_n
//!
//!    * 1.2. `p_1 = c'(r_1, .., r_a')` where `c ≠ c'`. We discard the current stack and return
//!           nothing.
//!    * 1.3. `p_1 = _`. We push onto the stack as many wildcards as the constructor `c` has
//!           arguments (its arity), and return the resulting stack:
//!
//!          _, .., _, p_2, .., p_n
//!
//!    * 1.4. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting stack:
//!
//!          S(c, (r_1, p_2, .., p_n))
//!          S(c, (r_2, p_2, .., p_n))
//!
//! 2. We can pop a wildcard off the top of the stack. This is called `D(p)`, where `p` is
//!    a pattern-stack.
//!    This is used when we know there are missing constructor cases, but there might be
//!    existing wildcard patterns, so to check the usefulness of the matrix, we have to check
//!    all its *other* components.
//!
//!    It is computed as follows. We look at the pattern `p_1` on top of the stack,
//!    and we have three cases:
//!    * 1.1. `p_1 = c(r_1, .., r_a)`. We discard the current stack and return nothing.
//!    * 1.2. `p_1 = _`. We return the rest of the stack:
//!
//!          p_2, .., p_n
//!
//!    * 1.3. `p_1 = r_1 | r_2`. We expand the OR-pattern and then recurse on each resulting stack:
//!
//!          D((r_1, p_2, .., p_n))
//!          D((r_2, p_2, .., p_n))
//!
//!    Note that the OR-patterns are not always used directly in Rust, but are used to derive the
//!    exhaustive integer matching rules, so they're written here for posterity.
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
//! This algorithm is realized in the `is_useful` function.
//!
//! Base case (`n = 0`, i.e., an empty tuple pattern):
//! - If `P` already contains an empty pattern (i.e., if the number of patterns `m > 0`), then
//!   `U(P, p)` is false.
//! - Otherwise, `P` must be empty, so `U(P, p)` is true.
//!
//! Inductive step (`n > 0`, i.e., whether there's at least one column [which may then be expanded
//! into further columns later]). We're going to match on the top of the new pattern-stack, `p_1`:
//!
//! - If `p_1 == c(r_1, .., r_a)`, i.e. we have a constructor pattern.
//!   Then, the usefulness of `p_1` can be reduced to whether it is useful when
//!   we ignore all the patterns in the first column of `P` that involve other constructors.
//!   This is where `S(c, P)` comes in:
//!
//!   ```text
//!   U(P, p) := U(S(c, P), S(c, p))
//!   ```
//!
//!   This special case is handled in `is_useful_specialized`.
//!
//!   For example, if `P` is:
//!
//!   ```text
//!   [
//!       [Some(true), _],
//!       [None, 0],
//!   ]
//!   ```
//!
//!   and `p` is `[Some(false), 0]`, then we don't care about row 2 since we know `p` only
//!   matches values that row 2 doesn't. For row 1 however, we need to dig into the
//!   arguments of `Some` to know whether some new value is covered. So we compute
//!   `U([[true, _]], [false, 0])`.
//!
//! - If `p_1 == _`, then we look at the list of constructors that appear in the first component of
//!   the rows of `P`:
//!     - If there are some constructors that aren't present, then we might think that the
//!       wildcard `_` is useful, since it covers those constructors that weren't covered
//!       before.
//!       That's almost correct, but only works if there were no wildcards in those first
//!       components. So we need to check that `p` is useful with respect to the rows that
//!       start with a wildcard, if there are any. This is where `D` comes in:
//!       `U(P, p) := U(D(P), D(p))`
//!
//!       For example, if `P` is:
//!       ```text
//!       [
//!           [_, true, _],
//!           [None, false, 1],
//!       ]
//!       ```
//!       and `p` is `[_, false, _]`, the `Some` constructor doesn't appear in `P`. So if we
//!       only had row 2, we'd know that `p` is useful. However row 1 starts with a
//!       wildcard, so we need to check whether `U([[true, _]], [false, 1])`.
//!
//!     - Otherwise, all possible constructors (for the relevant type) are present. In this
//!       case we must check whether the wildcard pattern covers any unmatched value. For
//!       that, we can think of the `_` pattern as a big OR-pattern that covers all
//!       possible constructors. For `Option`, that would mean `_ = None | Some(_)` for
//!       example. The wildcard pattern is useful in this case if it is useful when
//!       specialized to one of the possible constructors. So we compute:
//!       `U(P, p) := ∃(k ϵ constructors) U(S(k, P), S(k, p))`
//!
//!       For example, if `P` is:
//!       ```text
//!       [
//!           [Some(true), _],
//!           [None, false],
//!       ]
//!       ```
//!       and `p` is `[_, false]`, both `None` and `Some` constructors appear in the first
//!       components of `P`. We will therefore try popping both constructors in turn: we
//!       compute `U([[true, _]], [_, false])` for the `Some` constructor, and `U([[false]],
//!       [false])` for the `None` constructor. The first case returns true, so we know that
//!       `p` is useful for `P`. Indeed, it matches `[Some(false), _]` that wasn't matched
//!       before.
//!
//! - If `p_1 == r_1 | r_2`, then the usefulness depends on each `r_i` separately:
//!
//!   ```text
//!   U(P, p) := U(P, (r_1, p_2, .., p_n))
//!            || U(P, (r_2, p_2, .., p_n))
//!   ```
use std::{iter, sync::Arc};

use hir_def::{
    adt::VariantData,
    body::Body,
    expr::{Expr, Literal, Pat, PatId},
    EnumVariantId, StructId, VariantId,
};
use la_arena::Idx;
use smallvec::{smallvec, SmallVec};

use crate::{db::HirDatabase, AdtId, InferenceResult, Interner, TyKind};

#[derive(Debug, Clone, Copy)]
/// Either a pattern from the source code being analyzed, represented as
/// as `PatId`, or a `Wild` pattern which is created as an intermediate
/// step in the match checking algorithm and thus is not backed by a
/// real `PatId`.
///
/// Note that it is totally valid for the `PatId` variant to contain
/// a `PatId` which resolves to a `Wild` pattern, if that wild pattern
/// exists in the source code being analyzed.
enum PatIdOrWild {
    PatId(PatId),
    Wild,
}

impl PatIdOrWild {
    fn as_pat(self, cx: &MatchCheckCtx) -> Pat {
        match self {
            PatIdOrWild::PatId(id) => cx.body.pats[id].clone(),
            PatIdOrWild::Wild => Pat::Wild,
        }
    }

    fn as_id(self) -> Option<PatId> {
        match self {
            PatIdOrWild::PatId(id) => Some(id),
            PatIdOrWild::Wild => None,
        }
    }
}

impl From<PatId> for PatIdOrWild {
    fn from(pat_id: PatId) -> Self {
        Self::PatId(pat_id)
    }
}

impl From<&PatId> for PatIdOrWild {
    fn from(pat_id: &PatId) -> Self {
        Self::PatId(*pat_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum MatchCheckErr {
    NotImplemented,
    MalformedMatchArm,
    /// Used when type inference cannot resolve the type of
    /// a pattern or expression.
    Unknown,
}

/// The return type of `is_useful` is either an indication of usefulness
/// of the match arm, or an error in the case the match statement
/// is made up of types for which exhaustiveness checking is currently
/// not completely implemented.
///
/// The `std::result::Result` type is used here rather than a custom enum
/// to allow the use of `?`.
pub(super) type MatchCheckResult<T> = Result<T, MatchCheckErr>;

#[derive(Debug)]
/// A row in a Matrix.
///
/// This type is modeled from the struct of the same name in `rustc`.
pub(super) struct PatStack(PatStackInner);
type PatStackInner = SmallVec<[PatIdOrWild; 2]>;

impl PatStack {
    pub(super) fn from_pattern(pat_id: PatId) -> PatStack {
        Self(smallvec!(pat_id.into()))
    }

    pub(super) fn from_wild() -> PatStack {
        Self(smallvec!(PatIdOrWild::Wild))
    }

    fn from_slice(slice: &[PatIdOrWild]) -> PatStack {
        Self(SmallVec::from_slice(slice))
    }

    fn from_vec(v: PatStackInner) -> PatStack {
        Self(v)
    }

    fn get_head(&self) -> Option<PatIdOrWild> {
        self.0.first().copied()
    }

    fn tail(&self) -> &[PatIdOrWild] {
        self.0.get(1..).unwrap_or(&[])
    }

    fn to_tail(&self) -> PatStack {
        Self::from_slice(self.tail())
    }

    fn replace_head_with<I, T>(&self, pats: I) -> PatStack
    where
        I: Iterator<Item = T>,
        T: Into<PatIdOrWild>,
    {
        let mut patterns: PatStackInner = smallvec![];
        for pat in pats {
            patterns.push(pat.into());
        }
        for pat in &self.0[1..] {
            patterns.push(*pat);
        }
        PatStack::from_vec(patterns)
    }

    /// Computes `D(self)`.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_wildcard(&self, cx: &MatchCheckCtx) -> Option<PatStack> {
        if matches!(self.get_head()?.as_pat(cx), Pat::Wild) {
            Some(self.to_tail())
        } else {
            None
        }
    }

    /// Computes `S(constructor, self)`.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_constructor(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> MatchCheckResult<Option<PatStack>> {
        let head = match self.get_head() {
            Some(head) => head,
            None => return Ok(None),
        };

        let head_pat = head.as_pat(cx);
        let result = match (head_pat, constructor) {
            (Pat::Tuple { args: pat_ids, ellipsis }, &Constructor::Tuple { arity }) => {
                if let Some(ellipsis) = ellipsis {
                    let (pre, post) = pat_ids.split_at(ellipsis);
                    let n_wild_pats = arity.saturating_sub(pat_ids.len());
                    let pre_iter = pre.iter().map(Into::into);
                    let wildcards = iter::repeat(PatIdOrWild::Wild).take(n_wild_pats);
                    let post_iter = post.iter().map(Into::into);
                    Some(self.replace_head_with(pre_iter.chain(wildcards).chain(post_iter)))
                } else {
                    Some(self.replace_head_with(pat_ids.iter()))
                }
            }
            (Pat::Lit(lit_expr), Constructor::Bool(constructor_val)) => {
                match cx.body.exprs[lit_expr] {
                    Expr::Literal(Literal::Bool(pat_val)) if *constructor_val == pat_val => {
                        Some(self.to_tail())
                    }
                    // it was a bool but the value doesn't match
                    Expr::Literal(Literal::Bool(_)) => None,
                    // perhaps this is actually unreachable given we have
                    // already checked that these match arms have the appropriate type?
                    _ => return Err(MatchCheckErr::NotImplemented),
                }
            }
            (Pat::Wild, constructor) => Some(self.expand_wildcard(cx, constructor)?),
            (Pat::Path(_), constructor) => {
                // unit enum variants become `Pat::Path`
                let pat_id = head.as_id().expect("we know this isn't a wild");
                let variant_id: VariantId = match constructor {
                    &Constructor::Enum(e) => e.into(),
                    &Constructor::Struct(s) => s.into(),
                    _ => return Err(MatchCheckErr::NotImplemented),
                };
                if Some(variant_id) != cx.infer.variant_resolution_for_pat(pat_id) {
                    None
                } else {
                    Some(self.to_tail())
                }
            }
            (Pat::TupleStruct { args: ref pat_ids, ellipsis, .. }, constructor) => {
                let pat_id = head.as_id().expect("we know this isn't a wild");
                let variant_id: VariantId = match constructor {
                    &Constructor::Enum(e) => e.into(),
                    &Constructor::Struct(s) => s.into(),
                    _ => return Err(MatchCheckErr::MalformedMatchArm),
                };
                if Some(variant_id) != cx.infer.variant_resolution_for_pat(pat_id) {
                    None
                } else {
                    let constructor_arity = constructor.arity(cx)?;
                    if let Some(ellipsis_position) = ellipsis {
                        // If there are ellipsis in the pattern, the ellipsis must take the place
                        // of at least one sub-pattern, so `pat_ids` should be smaller than the
                        // constructor arity.
                        if pat_ids.len() < constructor_arity {
                            let mut new_patterns: Vec<PatIdOrWild> = vec![];

                            for pat_id in &pat_ids[0..ellipsis_position] {
                                new_patterns.push((*pat_id).into());
                            }

                            for _ in 0..(constructor_arity - pat_ids.len()) {
                                new_patterns.push(PatIdOrWild::Wild);
                            }

                            for pat_id in &pat_ids[ellipsis_position..pat_ids.len()] {
                                new_patterns.push((*pat_id).into());
                            }

                            Some(self.replace_head_with(new_patterns.into_iter()))
                        } else {
                            return Err(MatchCheckErr::MalformedMatchArm);
                        }
                    } else {
                        // If there is no ellipsis in the tuple pattern, the number
                        // of patterns must equal the constructor arity.
                        if pat_ids.len() == constructor_arity {
                            Some(self.replace_head_with(pat_ids.into_iter()))
                        } else {
                            return Err(MatchCheckErr::MalformedMatchArm);
                        }
                    }
                }
            }
            (Pat::Record { args: ref arg_patterns, .. }, constructor) => {
                let pat_id = head.as_id().expect("we know this isn't a wild");
                let (variant_id, variant_data) = match constructor {
                    &Constructor::Enum(e) => (
                        e.into(),
                        cx.db.enum_data(e.parent).variants[e.local_id].variant_data.clone(),
                    ),
                    &Constructor::Struct(s) => {
                        (s.into(), cx.db.struct_data(s).variant_data.clone())
                    }
                    _ => return Err(MatchCheckErr::MalformedMatchArm),
                };
                if Some(variant_id) != cx.infer.variant_resolution_for_pat(pat_id) {
                    None
                } else {
                    match variant_data.as_ref() {
                        VariantData::Record(struct_field_arena) => {
                            // Here we treat any missing fields in the record as the wild pattern, as
                            // if the record has ellipsis. We want to do this here even if the
                            // record does not contain ellipsis, because it allows us to continue
                            // enforcing exhaustiveness for the rest of the match statement.
                            //
                            // Creating the diagnostic for the missing field in the pattern
                            // should be done in a different diagnostic.
                            let patterns = struct_field_arena.iter().map(|(_, struct_field)| {
                                arg_patterns
                                    .iter()
                                    .find(|pat| pat.name == struct_field.name)
                                    .map(|pat| PatIdOrWild::from(pat.pat))
                                    .unwrap_or(PatIdOrWild::Wild)
                            });

                            Some(self.replace_head_with(patterns))
                        }
                        _ => return Err(MatchCheckErr::Unknown),
                    }
                }
            }
            (Pat::Or(_), _) => return Err(MatchCheckErr::NotImplemented),
            (_, _) => return Err(MatchCheckErr::NotImplemented),
        };

        Ok(result)
    }

    /// A special case of `specialize_constructor` where the head of the pattern stack
    /// is a Wild pattern.
    ///
    /// Replaces the Wild pattern at the head of the pattern stack with N Wild patterns
    /// (N >= 0), where N is the arity of the given constructor.
    fn expand_wildcard(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> MatchCheckResult<PatStack> {
        assert_eq!(
            Pat::Wild,
            self.get_head().expect("expand_wildcard called on empty PatStack").as_pat(cx),
            "expand_wildcard must only be called on PatStack with wild at head",
        );

        let mut patterns: PatStackInner = smallvec![];

        for _ in 0..constructor.arity(cx)? {
            patterns.push(PatIdOrWild::Wild);
        }

        for pat in &self.0[1..] {
            patterns.push(*pat);
        }

        Ok(PatStack::from_vec(patterns))
    }
}

/// A collection of PatStack.
///
/// This type is modeled from the struct of the same name in `rustc`.
pub(super) struct Matrix(Vec<PatStack>);

impl Matrix {
    pub(super) fn empty() -> Self {
        Self(vec![])
    }

    pub(super) fn push(&mut self, cx: &MatchCheckCtx, row: PatStack) {
        if let Some(Pat::Or(pat_ids)) = row.get_head().map(|pat_id| pat_id.as_pat(cx)) {
            // Or patterns are expanded here
            for pat_id in pat_ids {
                self.0.push(row.replace_head_with([pat_id].iter()));
            }
        } else {
            self.0.push(row);
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn heads(&self) -> Vec<PatIdOrWild> {
        self.0.iter().flat_map(|p| p.get_head()).collect()
    }

    /// Computes `D(self)` for each contained PatStack.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_wildcard(&self, cx: &MatchCheckCtx) -> Self {
        Self::collect(cx, self.0.iter().filter_map(|r| r.specialize_wildcard(cx)))
    }

    /// Computes `S(constructor, self)` for each contained PatStack.
    ///
    /// See the module docs and the associated documentation in rustc for details.
    fn specialize_constructor(
        &self,
        cx: &MatchCheckCtx,
        constructor: &Constructor,
    ) -> MatchCheckResult<Self> {
        let mut new_matrix = Matrix::empty();
        for pat in &self.0 {
            if let Some(pat) = pat.specialize_constructor(cx, constructor)? {
                new_matrix.push(cx, pat);
            }
        }

        Ok(new_matrix)
    }

    fn collect<T: IntoIterator<Item = PatStack>>(cx: &MatchCheckCtx, iter: T) -> Self {
        let mut matrix = Matrix::empty();

        for pat in iter {
            // using push ensures we expand or-patterns
            matrix.push(cx, pat);
        }

        matrix
    }
}

#[derive(Clone, Debug, PartialEq)]
/// An indication of the usefulness of a given match arm, where
/// usefulness is defined as matching some patterns which were
/// not matched by an prior match arms.
///
/// We may eventually need an `Unknown` variant here.
pub(super) enum Usefulness {
    Useful,
    NotUseful,
}

pub(super) struct MatchCheckCtx<'a> {
    pub(super) match_expr: Idx<Expr>,
    pub(super) body: Arc<Body>,
    pub(super) infer: Arc<InferenceResult>,
    pub(super) db: &'a dyn HirDatabase,
}

/// Given a set of patterns `matrix`, and pattern to consider `v`, determines
/// whether `v` is useful. A pattern is useful if it covers cases which were
/// not previously covered.
///
/// When calling this function externally (that is, not the recursive calls) it
/// expected that you have already type checked the match arms. All patterns in
/// matrix should be the same type as v, as well as they should all be the same
/// type as the match expression.
pub(super) fn is_useful(
    cx: &MatchCheckCtx,
    matrix: &Matrix,
    v: &PatStack,
) -> MatchCheckResult<Usefulness> {
    // Handle two special cases:
    // - enum with no variants
    // - `!` type
    // In those cases, no match arm is useful.
    match cx.infer[cx.match_expr].strip_references().kind(&Interner) {
        TyKind::Adt(AdtId(hir_def::AdtId::EnumId(enum_id)), ..) => {
            if cx.db.enum_data(*enum_id).variants.is_empty() {
                return Ok(Usefulness::NotUseful);
            }
        }
        TyKind::Never => return Ok(Usefulness::NotUseful),
        _ => (),
    }

    let head = match v.get_head() {
        Some(head) => head,
        None => {
            let result = if matrix.is_empty() { Usefulness::Useful } else { Usefulness::NotUseful };

            return Ok(result);
        }
    };

    if let Pat::Or(pat_ids) = head.as_pat(cx) {
        let mut found_unimplemented = false;
        let any_useful = pat_ids.iter().any(|&pat_id| {
            let v = PatStack::from_pattern(pat_id);

            match is_useful(cx, matrix, &v) {
                Ok(Usefulness::Useful) => true,
                Ok(Usefulness::NotUseful) => false,
                _ => {
                    found_unimplemented = true;
                    false
                }
            }
        });

        return if any_useful {
            Ok(Usefulness::Useful)
        } else if found_unimplemented {
            Err(MatchCheckErr::NotImplemented)
        } else {
            Ok(Usefulness::NotUseful)
        };
    }

    if let Some(constructor) = pat_constructor(cx, head)? {
        let matrix = matrix.specialize_constructor(&cx, &constructor)?;
        let v = v
            .specialize_constructor(&cx, &constructor)?
            .expect("we know this can't fail because we get the constructor from `v.head()` above");

        is_useful(&cx, &matrix, &v)
    } else {
        // expanding wildcard
        let mut used_constructors: Vec<Constructor> = vec![];
        for pat in matrix.heads() {
            if let Some(constructor) = pat_constructor(cx, pat)? {
                used_constructors.push(constructor);
            }
        }

        // We assume here that the first constructor is the "correct" type. Since we
        // only care about the "type" of the constructor (i.e. if it is a bool we
        // don't care about the value), this assumption should be valid as long as
        // the match statement is well formed. We currently uphold this invariant by
        // filtering match arms before calling `is_useful`, only passing in match arms
        // whose type matches the type of the match expression.
        match &used_constructors.first() {
            Some(constructor) if all_constructors_covered(&cx, constructor, &used_constructors) => {
                // If all constructors are covered, then we need to consider whether
                // any values are covered by this wildcard.
                //
                // For example, with matrix '[[Some(true)], [None]]', all
                // constructors are covered (`Some`/`None`), so we need
                // to perform specialization to see that our wildcard will cover
                // the `Some(false)` case.
                //
                // Here we create a constructor for each variant and then check
                // usefulness after specializing for that constructor.
                let mut found_unimplemented = false;
                for constructor in constructor.all_constructors(cx) {
                    let matrix = matrix.specialize_constructor(&cx, &constructor)?;
                    let v = v.expand_wildcard(&cx, &constructor)?;

                    match is_useful(&cx, &matrix, &v) {
                        Ok(Usefulness::Useful) => return Ok(Usefulness::Useful),
                        Ok(Usefulness::NotUseful) => continue,
                        _ => found_unimplemented = true,
                    };
                }

                if found_unimplemented {
                    Err(MatchCheckErr::NotImplemented)
                } else {
                    Ok(Usefulness::NotUseful)
                }
            }
            _ => {
                // Either not all constructors are covered, or the only other arms
                // are wildcards. Either way, this pattern is useful if it is useful
                // when compared to those arms with wildcards.
                let matrix = matrix.specialize_wildcard(&cx);
                let v = v.to_tail();

                is_useful(&cx, &matrix, &v)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Similar to TypeCtor, but includes additional information about the specific
/// value being instantiated. For example, TypeCtor::Bool doesn't contain the
/// boolean value.
enum Constructor {
    Bool(bool),
    Tuple { arity: usize },
    Enum(EnumVariantId),
    Struct(StructId),
}

impl Constructor {
    fn arity(&self, cx: &MatchCheckCtx) -> MatchCheckResult<usize> {
        let arity = match self {
            Constructor::Bool(_) => 0,
            Constructor::Tuple { arity } => *arity,
            Constructor::Enum(e) => {
                match cx.db.enum_data(e.parent).variants[e.local_id].variant_data.as_ref() {
                    VariantData::Tuple(struct_field_data) => struct_field_data.len(),
                    VariantData::Record(struct_field_data) => struct_field_data.len(),
                    VariantData::Unit => 0,
                }
            }
            &Constructor::Struct(s) => match cx.db.struct_data(s).variant_data.as_ref() {
                VariantData::Tuple(struct_field_data) => struct_field_data.len(),
                VariantData::Record(struct_field_data) => struct_field_data.len(),
                VariantData::Unit => 0,
            },
        };

        Ok(arity)
    }

    fn all_constructors(&self, cx: &MatchCheckCtx) -> Vec<Constructor> {
        match self {
            Constructor::Bool(_) => vec![Constructor::Bool(true), Constructor::Bool(false)],
            Constructor::Tuple { .. } | Constructor::Struct(_) => vec![*self],
            Constructor::Enum(e) => cx
                .db
                .enum_data(e.parent)
                .variants
                .iter()
                .map(|(local_id, _)| {
                    Constructor::Enum(EnumVariantId { parent: e.parent, local_id })
                })
                .collect(),
        }
    }
}

/// Returns the constructor for the given pattern. Should only return None
/// in the case of a Wild pattern.
fn pat_constructor(cx: &MatchCheckCtx, pat: PatIdOrWild) -> MatchCheckResult<Option<Constructor>> {
    let res = match pat.as_pat(cx) {
        Pat::Wild => None,
        Pat::Tuple { .. } => {
            let pat_id = pat.as_id().expect("we already know this pattern is not a wild");
            Some(Constructor::Tuple {
                arity: cx.infer.type_of_pat[pat_id]
                    .as_tuple()
                    .ok_or(MatchCheckErr::Unknown)?
                    .len(&Interner),
            })
        }
        Pat::Lit(lit_expr) => match cx.body.exprs[lit_expr] {
            Expr::Literal(Literal::Bool(val)) => Some(Constructor::Bool(val)),
            _ => return Err(MatchCheckErr::NotImplemented),
        },
        Pat::TupleStruct { .. } | Pat::Path(_) | Pat::Record { .. } => {
            let pat_id = pat.as_id().expect("we already know this pattern is not a wild");
            let variant_id =
                cx.infer.variant_resolution_for_pat(pat_id).ok_or(MatchCheckErr::Unknown)?;
            match variant_id {
                VariantId::EnumVariantId(enum_variant_id) => {
                    Some(Constructor::Enum(enum_variant_id))
                }
                VariantId::StructId(struct_id) => Some(Constructor::Struct(struct_id)),
                _ => return Err(MatchCheckErr::NotImplemented),
            }
        }
        _ => return Err(MatchCheckErr::NotImplemented),
    };

    Ok(res)
}

fn all_constructors_covered(
    cx: &MatchCheckCtx,
    constructor: &Constructor,
    used_constructors: &[Constructor],
) -> bool {
    match constructor {
        Constructor::Tuple { arity } => {
            used_constructors.iter().any(|constructor| match constructor {
                Constructor::Tuple { arity: used_arity } => arity == used_arity,
                _ => false,
            })
        }
        Constructor::Bool(_) => {
            if used_constructors.is_empty() {
                return false;
            }

            let covers_true =
                used_constructors.iter().any(|c| matches!(c, Constructor::Bool(true)));
            let covers_false =
                used_constructors.iter().any(|c| matches!(c, Constructor::Bool(false)));

            covers_true && covers_false
        }
        Constructor::Enum(e) => cx.db.enum_data(e.parent).variants.iter().all(|(id, _)| {
            for constructor in used_constructors {
                if let Constructor::Enum(e) = constructor {
                    if id == e.local_id {
                        return true;
                    }
                }
            }

            false
        }),
        &Constructor::Struct(s) => used_constructors.iter().any(|constructor| match constructor {
            &Constructor::Struct(sid) => sid == s,
            _ => false,
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn empty_tuple() {
        check_diagnostics(
            r#"
fn main() {
    match () { }
        //^^ Missing match arm
    match (()) { }
        //^^^^ Missing match arm

    match () { _ => (), }
    match () { () => (), }
    match (()) { (()) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_two_empty_tuple() {
        check_diagnostics(
            r#"
fn main() {
    match ((), ()) { }
        //^^^^^^^^ Missing match arm

    match ((), ()) { ((), ()) => (), }
}
"#,
        );
    }

    #[test]
    fn boolean() {
        check_diagnostics(
            r#"
fn test_main() {
    match false { }
        //^^^^^ Missing match arm
    match false { true => (), }
        //^^^^^ Missing match arm
    match (false, true) {}
        //^^^^^^^^^^^^^ Missing match arm
    match (false, true) { (true, true) => (), }
        //^^^^^^^^^^^^^ Missing match arm
    match (false, true) {
        //^^^^^^^^^^^^^ Missing match arm
        (false, true) => (),
        (false, false) => (),
        (true, false) => (),
    }
    match (false, true) { (true, _x) => (), }
        //^^^^^^^^^^^^^ Missing match arm

    match false { true => (), false => (), }
    match (false, true) {
        (false, _) => (),
        (true, false) => (),
        (_, true) => (),
    }
    match (false, true) {
        (true, true) => (),
        (true, false) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true) {
        (true, _x) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true, false) {
        (false, ..) => (),
        (true, ..) => (),
    }
    match (false, true, false) {
        (.., false) => (),
        (.., true) => (),
    }
    match (false, true, false) { (..) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_tuple_and_bools() {
        check_diagnostics(
            r#"
fn main() {
    match (false, ((), false)) {}
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
    match (false, ((), false)) { (true, ((), true)) => (), }
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
    match (false, ((), false)) { (true, _) => (), }
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm

    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, ((), true)) => (),
        (false, ((), false)) => (),
    }
    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn enums() {
        check_diagnostics(
            r#"
enum Either { A, B, }

fn main() {
    match Either::A { }
        //^^^^^^^^^ Missing match arm
    match Either::B { Either::A => (), }
        //^^^^^^^^^ Missing match arm

    match &Either::B {
        //^^^^^^^^^^ Missing match arm
        Either::A => (),
    }

    match Either::B {
        Either::A => (), Either::B => (),
    }
    match &Either::B {
        Either::A => (), Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_containing_bool() {
        check_diagnostics(
            r#"
enum Either { A(bool), B }

fn main() {
    match Either::B { }
        //^^^^^^^^^ Missing match arm
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true) => (), Either::B => ()
    }

    match Either::B {
        Either::A(true) => (),
        Either::A(false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::B => (),
        _ => (),
    }
    match Either::B {
        Either::A(_) => (),
        Either::B => (),
    }

}
        "#,
        );
    }

    #[test]
    fn enum_different_sizes() {
        check_diagnostics(
            r#"
enum Either { A(bool), B(bool, bool) }

fn main() {
    match Either::A(false) {
        //^^^^^^^^^^^^^^^^ Missing match arm
        Either::A(_) => (),
        Either::B(false, _) => (),
    }

    match Either::A(false) {
        Either::A(_) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
    match Either::A(false) {
        Either::A(true) | Either::A(false) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_enum_no_diagnostic() {
        check_diagnostics(
            r#"
enum Either { A(bool), B(bool, bool) }
enum Either2 { C, D }

fn main() {
    match (Either::A(false), Either2::C) {
        (Either::A(true), _) | (Either::A(false), _) => (),
        (Either::B(true, _), Either2::C) => (),
        (Either::B(false, _), Either2::C) => (),
        (Either::B(_, _), Either2::D) => (),
    }
}
"#,
        );
    }

    #[test]
    fn or_pattern_no_diagnostic() {
        check_diagnostics(
            r#"
enum Either {A, B}

fn main() {
    match (Either::A, Either::B) {
        (Either::A | Either::B, _) => (),
    }
}"#,
        )
    }

    #[test]
    fn mismatched_types() {
        // Match statements with arms that don't match the
        // expression pattern do not fire this diagnostic.
        check_diagnostics(
            r#"
enum Either { A, B }
enum Either2 { C, D }

fn main() {
    match Either::A {
        Either2::C => (),
        Either2::D => (),
    }
    match (true, false) {
        (true, false, true) => (),
        (true) => (),
    }
    match (0) { () => () }
    match Unresolved::Bar { Unresolved::Baz => () }
}
        "#,
        );
    }

    #[test]
    fn malformed_match_arm_tuple_enum_missing_pattern() {
        // We are testing to be sure we don't panic here when the match
        // arm `Either::B` is missing its pattern.
        check_diagnostics(
            r#"
enum Either { A, B(u32) }

fn main() {
    match Either::A {
        Either::A => (),
        Either::B() => (),
    }
}
"#,
        );
    }

    #[test]
    fn expr_diverges() {
        check_diagnostics(
            r#"
enum Either { A, B }

fn main() {
    match loop {} {
        Either::A => (),
        Either::B => (),
    }
    match loop {} {
        Either::A => (),
    }
    match loop { break Foo::A } {
        //^^^^^^^^^^^^^^^^^^^^^ Missing match arm
        Either::A => (),
    }
    match loop { break Foo::A } {
        Either::A => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn expr_partially_diverges() {
        check_diagnostics(
            r#"
enum Either<T> { A(T), B }

fn foo() -> Either<!> { Either::B }
fn main() -> u32 {
    match foo() {
        Either::A(val) => val,
        Either::B => 0,
    }
}
"#,
        );
    }

    #[test]
    fn enum_record() {
        check_diagnostics(
            r#"
enum Either { A { foo: bool }, B }

fn main() {
    let a = Either::A { foo: true };
    match a { }
        //^ Missing match arm
    match a { Either::A { foo: true } => () }
        //^ Missing match arm
    match a {
        Either::A { } => (),
      //^^^^^^^^^ Missing structure fields:
      //        | - foo
        Either::B => (),
    }
    match a {
        //^ Missing match arm
        Either::A { } => (),
    } //^^^^^^^^^ Missing structure fields:
      //        | - foo

    match a {
        Either::A { foo: true } => (),
        Either::A { foo: false } => (),
        Either::B => (),
    }
    match a {
        Either::A { foo: _ } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_fields_out_of_order() {
        check_diagnostics(
            r#"
enum Either {
    A { foo: bool, bar: () },
    B,
}

fn main() {
    let a = Either::A { foo: true, bar: () };
    match a {
        //^ Missing match arm
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
    }

    match a {
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_ellipsis() {
        check_diagnostics(
            r#"
enum Either {
    A { foo: bool, bar: bool },
    B,
}

fn main() {
    let a = Either::B;
    match a {
        //^ Missing match arm
        Either::A { foo: true, .. } => (),
        Either::B => (),
    }
    match a {
        //^ Missing match arm
        Either::A { .. } => (),
    }

    match a {
        Either::A { foo: true, .. } => (),
        Either::A { foo: false, .. } => (),
        Either::B => (),
    }

    match a {
        Either::A { .. } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_tuple_partial_ellipsis() {
        check_diagnostics(
            r#"
enum Either {
    A(bool, bool, bool, bool),
    B,
}

fn main() {
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::B => (),
    }

    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., true) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::A(.., false) => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn never() {
        check_diagnostics(
            r#"
enum Never {}

fn enum_(never: Never) {
    match never {}
}
fn enum_ref(never: &Never) {
    match never {}
}
fn bang(never: !) {
    match never {}
}
"#,
        );
    }

    #[test]
    fn unknown_type() {
        check_diagnostics(
            r#"
enum Option<T> { Some(T), None }

fn main() {
    // `Never` is deliberately not defined so that it's an uninferred type.
    match Option::<Never>::None {
        None => (),
        Some(never) => match never {},
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_end_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (false, ..) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_beginning_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (.., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_in_middle_missing_arm() {
        check_diagnostics(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ Missing match arm
        (true, .., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn record_struct() {
        check_diagnostics(
            r#"struct Foo { a: bool }
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo { a: true } => () }
        //^ Missing match arm
    match &f { Foo { a: true } => () }
        //^^ Missing match arm
    match f { Foo { a: _ } => () }
    match f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
    match &f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_struct() {
        check_diagnostics(
            r#"struct Foo(bool);
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo(true) => () }
        //^ Missing match arm
    match f {
        Foo(true) => (),
        Foo(false) => (),
    }
}
"#,
        );
    }

    #[test]
    fn unit_struct() {
        check_diagnostics(
            r#"struct Foo;
fn main(f: Foo) {
    match f {}
        //^ Missing match arm
    match f { Foo => () }
}
"#,
        );
    }

    #[test]
    fn record_struct_ellipsis() {
        check_diagnostics(
            r#"struct Foo { foo: bool, bar: bool }
fn main(f: Foo) {
    match f { Foo { foo: true, .. } => () }
        //^ Missing match arm
    match f {
        //^ Missing match arm
        Foo { foo: true, .. } => (),
        Foo { bar: false, .. } => ()
    }
    match f { Foo { .. } => () }
    match f {
        Foo { foo: true, .. } => (),
        Foo { foo: false, .. } => ()
    }
}
"#,
        );
    }

    #[test]
    fn internal_or() {
        check_diagnostics(
            r#"
fn main() {
    enum Either { A(bool), B }
    match Either::B {
        //^^^^^^^^^ Missing match arm
        Either::A(true | false) => (),
    }
}
"#,
        );
    }
    mod false_negatives {
        //! The implementation of match checking here is a work in progress. As we roll this out, we
        //! prefer false negatives to false positives (ideally there would be no false positives). This
        //! test module should document known false negatives. Eventually we will have a complete
        //! implementation of match checking and this module will be empty.
        //!
        //! The reasons for documenting known false negatives:
        //!
        //!   1. It acts as a backlog of work that can be done to improve the behavior of the system.
        //!   2. It ensures the code doesn't panic when handling these cases.
        use super::*;

        #[test]
        fn integers() {
            // We don't currently check integer exhaustiveness.
            check_diagnostics(
                r#"
fn main() {
    match 5 {
        10 => (),
        11..20 => (),
    }
}
"#,
            );
        }
    }
}
