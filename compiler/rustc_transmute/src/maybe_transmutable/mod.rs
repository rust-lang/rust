pub(crate) mod query_context;
#[cfg(test)]
mod tests;

use crate::{
    layout::{self, dfa, Byte, Dfa, Nfa, Ref, Tree, Uninhabited},
    maybe_transmutable::query_context::QueryContext,
    Answer, Condition, Map, Reason,
};

pub(crate) struct MaybeTransmutableQuery<L, C>
where
    C: QueryContext,
{
    src: L,
    dst: L,
    scope: <C as QueryContext>::Scope,
    assume: crate::Assume,
    context: C,
}

impl<L, C> MaybeTransmutableQuery<L, C>
where
    C: QueryContext,
{
    pub(crate) fn new(
        src: L,
        dst: L,
        scope: <C as QueryContext>::Scope,
        assume: crate::Assume,
        context: C,
    ) -> Self {
        Self { src, dst, scope, assume, context }
    }

    pub(crate) fn map_layouts<F, M>(
        self,
        f: F,
    ) -> Result<MaybeTransmutableQuery<M, C>, Answer<<C as QueryContext>::Ref>>
    where
        F: FnOnce(
            L,
            L,
            <C as QueryContext>::Scope,
            &C,
        ) -> Result<(M, M), Answer<<C as QueryContext>::Ref>>,
    {
        let Self { src, dst, scope, assume, context } = self;

        let (src, dst) = f(src, dst, scope, &context)?;

        Ok(MaybeTransmutableQuery { src, dst, scope, assume, context })
    }
}

// FIXME: Nix this cfg, so we can write unit tests independently of rustc
#[cfg(feature = "rustc")]
mod rustc {
    use super::*;
    use crate::layout::tree::rustc::Err;

    use rustc_middle::ty::Ty;
    use rustc_middle::ty::TyCtxt;

    impl<'tcx> MaybeTransmutableQuery<Ty<'tcx>, TyCtxt<'tcx>> {
        /// This method begins by converting `src` and `dst` from `Ty`s to `Tree`s,
        /// then computes an answer using those trees.
        #[instrument(level = "debug", skip(self), fields(src = ?self.src, dst = ?self.dst))]
        pub fn answer(self) -> Answer<<TyCtxt<'tcx> as QueryContext>::Ref> {
            let query_or_answer = self.map_layouts(|src, dst, scope, &context| {
                // Convert `src` and `dst` from their rustc representations, to `Tree`-based
                // representations. If these conversions fail, conclude that the transmutation is
                // unacceptable; the layouts of both the source and destination types must be
                // well-defined.
                let src = Tree::from_ty(src, context);
                let dst = Tree::from_ty(dst, context);

                match (src, dst) {
                    // Answer `Ok(None)` here, because 'unknown layout' and type errors will already
                    // be reported by rustc. No need to spam the user with more errors.
                    (Err(Err::TypeError(_)), _)
                    | (_, Err(Err::TypeError(_)))
                    | (Err(Err::Unknown), _)
                    | (_, Err(Err::Unknown)) => Err(Ok(None)),
                    (Err(Err::Unspecified), _) => Err(Err(Reason::SrcIsUnspecified)),
                    (_, Err(Err::Unspecified)) => Err(Err(Reason::DstIsUnspecified)),
                    (Ok(src), Ok(dst)) => Ok((src, dst)),
                }
            });

            match query_or_answer {
                Ok(query) => query.answer(),
                Err(answer) => answer,
            }
        }
    }
}

impl<C> MaybeTransmutableQuery<Tree<<C as QueryContext>::Def, <C as QueryContext>::Ref>, C>
where
    C: QueryContext,
{
    /// Answers whether a `Tree` is transmutable into another `Tree`.
    ///
    /// This method begins by de-def'ing `src` and `dst`, and prunes private paths from `dst`,
    /// then converts `src` and `dst` to `Nfa`s, and computes an answer using those NFAs.
    #[inline(always)]
    #[instrument(level = "debug", skip(self), fields(src = ?self.src, dst = ?self.dst))]
    pub(crate) fn answer(self) -> Answer<<C as QueryContext>::Ref> {
        let assume_visibility = self.assume.safety;
        let query_or_answer = self.map_layouts(|src, dst, scope, context| {
            // Remove all `Def` nodes from `src`, without checking their visibility.
            let src = src.prune(&|def| true);

            trace!(?src, "pruned src");

            // Remove all `Def` nodes from `dst`, additionally...
            let dst = if assume_visibility {
                // ...if visibility is assumed, don't check their visibility.
                dst.prune(&|def| true)
            } else {
                // ...otherwise, prune away all unreachable paths through the `Dst` layout.
                dst.prune(&|def| context.is_accessible_from(def, scope))
            };

            trace!(?dst, "pruned dst");

            // Convert `src` from a tree-based representation to an NFA-based representation.
            // If the conversion fails because `src` is uninhabited, conclude that the transmutation
            // is acceptable, because instances of the `src` type do not exist.
            let src = Nfa::from_tree(src).map_err(|Uninhabited| Ok(None))?;

            // Convert `dst` from a tree-based representation to an NFA-based representation.
            // If the conversion fails because `src` is uninhabited, conclude that the transmutation
            // is unacceptable, because instances of the `dst` type do not exist.
            let dst = Nfa::from_tree(dst).map_err(|Uninhabited| Err(Reason::DstIsPrivate))?;

            Ok((src, dst))
        });

        match query_or_answer {
            Ok(query) => query.answer(),
            Err(answer) => answer,
        }
    }
}

impl<C> MaybeTransmutableQuery<Nfa<<C as QueryContext>::Ref>, C>
where
    C: QueryContext,
{
    /// Answers whether a `Nfa` is transmutable into another `Nfa`.
    ///
    /// This method converts `src` and `dst` to DFAs, then computes an answer using those DFAs.
    #[inline(always)]
    #[instrument(level = "debug", skip(self), fields(src = ?self.src, dst = ?self.dst))]
    pub(crate) fn answer(self) -> Answer<<C as QueryContext>::Ref> {
        let query_or_answer = self
            .map_layouts(|src, dst, scope, context| Ok((Dfa::from_nfa(src), Dfa::from_nfa(dst))));

        match query_or_answer {
            Ok(query) => query.answer(),
            Err(answer) => answer,
        }
    }
}

impl<C> MaybeTransmutableQuery<Dfa<<C as QueryContext>::Ref>, C>
where
    C: QueryContext,
{
    /// Answers whether a `Nfa` is transmutable into another `Nfa`.
    ///
    /// This method converts `src` and `dst` to DFAs, then computes an answer using those DFAs.
    pub(crate) fn answer(self) -> Answer<<C as QueryContext>::Ref> {
        MaybeTransmutableQuery {
            src: &self.src,
            dst: &self.dst,
            scope: self.scope,
            assume: self.assume,
            context: self.context,
        }
        .answer()
    }
}

impl<'l, C> MaybeTransmutableQuery<&'l Dfa<<C as QueryContext>::Ref>, C>
where
    C: QueryContext,
{
    pub(crate) fn answer(&mut self) -> Answer<<C as QueryContext>::Ref> {
        self.answer_memo(&mut Map::default(), self.src.start, self.dst.start)
    }

    #[inline(always)]
    #[instrument(level = "debug", skip(self))]
    fn answer_memo(
        &self,
        cache: &mut Map<(dfa::State, dfa::State), Answer<<C as QueryContext>::Ref>>,
        src_state: dfa::State,
        dst_state: dfa::State,
    ) -> Answer<<C as QueryContext>::Ref> {
        if let Some(answer) = cache.get(&(src_state, dst_state)) {
            answer.clone()
        } else {
            let answer = if dst_state == self.dst.accepting {
                // truncation: `size_of(Src) >= size_of(Dst)`
                Ok(None)
            } else if src_state == self.src.accepting {
                // extension: `size_of(Src) >= size_of(Dst)`
                if let Some(dst_state_prime) = self.dst.byte_from(dst_state, Byte::Uninit) {
                    self.answer_memo(cache, src_state, dst_state_prime)
                } else {
                    Err(Reason::DstIsTooBig)
                }
            } else {
                let src_quantifier = if self.assume.validity {
                    // if the compiler may assume that the programmer is doing additional validity checks,
                    // (e.g.: that `src != 3u8` when the destination type is `bool`)
                    // then there must exist at least one transition out of `src_state` such that the transmute is viable...
                    Quantifier::ThereExists
                } else {
                    // if the compiler cannot assume that the programmer is doing additional validity checks,
                    // then for all transitions out of `src_state`, such that the transmute is viable...
                    // then there must exist at least one transition out of `dst_state` such that the transmute is viable...
                    Quantifier::ForAll
                };

                let bytes_answer = src_quantifier.apply(
                    // for each of the byte transitions out of the `src_state`...
                    self.src.bytes_from(src_state).unwrap_or(&Map::default()).into_iter().map(
                        |(&src_validity, &src_state_prime)| {
                            // ...try to find a matching transition out of `dst_state`.
                            if let Some(dst_state_prime) =
                                self.dst.byte_from(dst_state, src_validity)
                            {
                                self.answer_memo(cache, src_state_prime, dst_state_prime)
                            } else if let Some(dst_state_prime) =
                                // otherwise, see if `dst_state` has any outgoing `Uninit` transitions
                                // (any init byte is a valid uninit byte)
                                self.dst.byte_from(dst_state, Byte::Uninit)
                            {
                                self.answer_memo(cache, src_state_prime, dst_state_prime)
                            } else {
                                // otherwise, we've exhausted our options.
                                // the DFAs, from this point onwards, are bit-incompatible.
                                Err(Reason::DstIsBitIncompatible)
                            }
                        },
                    ),
                );

                // The below early returns reflect how this code would behave:
                //   if self.assume.validity {
                //       or(bytes_answer, refs_answer)
                //   } else {
                //       and(bytes_answer, refs_answer)
                //   }
                // ...if `refs_answer` was computed lazily. The below early
                // returns can be deleted without impacting the correctness of
                // the algoritm; only its performance.
                match bytes_answer {
                    Err(_) if !self.assume.validity => return bytes_answer,
                    Ok(None) if self.assume.validity => return bytes_answer,
                    _ => {}
                };

                let refs_answer = src_quantifier.apply(
                    // for each reference transition out of `src_state`...
                    self.src.refs_from(src_state).unwrap_or(&Map::default()).into_iter().map(
                        |(&src_ref, &src_state_prime)| {
                            // ...there exists a reference transition out of `dst_state`...
                            Quantifier::ThereExists.apply(
                                self.dst
                                    .refs_from(dst_state)
                                    .unwrap_or(&Map::default())
                                    .into_iter()
                                    .map(|(&dst_ref, &dst_state_prime)| {
                                        if !src_ref.is_mutable() && dst_ref.is_mutable() {
                                            Err(Reason::DstIsMoreUnique)
                                        } else if !self.assume.alignment
                                            && src_ref.min_align() < dst_ref.min_align()
                                        {
                                            Err(Reason::DstHasStricterAlignment)
                                        } else {
                                            // ...such that `src` is transmutable into `dst`, if
                                            // `src_ref` is transmutability into `dst_ref`.
                                            and(
                                                Ok(Some(Condition::IfTransmutable {
                                                    src: src_ref,
                                                    dst: dst_ref,
                                                })),
                                                self.answer_memo(
                                                    cache,
                                                    src_state_prime,
                                                    dst_state_prime,
                                                ),
                                            )
                                        }
                                    }),
                            )
                        },
                    ),
                );

                if self.assume.validity {
                    or(bytes_answer, refs_answer)
                } else {
                    and(bytes_answer, refs_answer)
                }
            };
            if let Some(..) = cache.insert((src_state, dst_state), answer.clone()) {
                panic!("failed to correctly cache transmutability")
            }
            answer
        }
    }
}

fn and<R>(lhs: Answer<R>, rhs: Answer<R>) -> Answer<R>
where
    R: PartialEq,
{
    // If both are errors, then we should return the more specific one
    if lhs.is_err() && rhs.is_err() {
        if lhs == Err(Reason::DstIsBitIncompatible) {
            return rhs;
        } else {
            return lhs;
        }
    }
    Ok(match (lhs?, rhs?) {
        // If only one side has a condition, pass it along
        (None, other) | (other, None) => other,
        // If both sides have IfAll conditions, merge them
        (Some(Condition::IfAll(mut lhs)), Some(Condition::IfAll(ref mut rhs))) => {
            lhs.append(rhs);
            Some(Condition::IfAll(lhs))
        }
        // If only one side is an IfAll, add the other Condition to it
        (constraint, Some(Condition::IfAll(mut constraints)))
        | (Some(Condition::IfAll(mut constraints)), constraint) => {
            constraints.push(Ok(constraint));
            Some(Condition::IfAll(constraints))
        }
        // Otherwise, both lhs and rhs conditions can be combined in a parent IfAll
        (lhs, rhs) => Some(Condition::IfAll(vec![Ok(lhs), Ok(rhs)])),
    })
}

fn or<R>(lhs: Answer<R>, rhs: Answer<R>) -> Answer<R>
where
    R: PartialEq,
{
    // If both are errors, then we should return the more specific one
    if lhs.is_err() && rhs.is_err() {
        if lhs == Err(Reason::DstIsBitIncompatible) {
            return rhs;
        } else {
            return lhs;
        }
    }
    // Otherwise, errors can be ignored for the rest of the pattern matching
    let lhs = lhs.unwrap_or(None);
    let rhs = rhs.unwrap_or(None);
    Ok(match (lhs, rhs) {
        // If only one side has a condition, pass it along
        (None, other) | (other, None) => other,
        // If both sides have IfAny conditions, merge them
        (Some(Condition::IfAny(mut lhs)), Some(Condition::IfAny(ref mut rhs))) => {
            lhs.append(rhs);
            Some(Condition::IfAny(lhs))
        }
        // If only one side is an IfAny, add the other Condition to it
        (constraint, Some(Condition::IfAny(mut constraints)))
        | (Some(Condition::IfAny(mut constraints)), constraint) => {
            constraints.push(Ok(constraint));
            Some(Condition::IfAny(constraints))
        }
        // Otherwise, both lhs and rhs conditions can be combined in a parent IfAny
        (lhs, rhs) => Some(Condition::IfAny(vec![Ok(lhs), Ok(rhs)])),
    })
}

pub enum Quantifier {
    ThereExists,
    ForAll,
}

impl Quantifier {
    pub fn apply<R, I>(&self, iter: I) -> Answer<R>
    where
        R: layout::Ref,
        I: IntoIterator<Item = Answer<R>>,
    {
        use std::ops::ControlFlow::{Break, Continue};

        let (init, try_fold_f): (_, fn(_, _) -> _) = match self {
            Self::ThereExists => (Err(Reason::DstIsBitIncompatible), |accum: Answer<R>, next| {
                match or(accum, next) {
                    Ok(None) => Break(Ok(None)),
                    maybe => Continue(maybe),
                }
            }),
            Self::ForAll => (Ok(None), |accum: Answer<R>, next| match and(accum, next) {
                Err(reason) => Break(Err(reason)),
                maybe => Continue(maybe),
            }),
        };

        let (Continue(result) | Break(result)) = iter.into_iter().try_fold(init, try_fold_f);
        result
    }
}
