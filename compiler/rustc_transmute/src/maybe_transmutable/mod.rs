use rustc_data_structures::stack::ensure_sufficient_stack;
use tracing::{debug, instrument, trace};

pub(crate) mod query_context;
#[cfg(test)]
mod tests;

use crate::layout::{self, Def, Dfa, Reference, Tree, dfa, union};
use crate::maybe_transmutable::query_context::QueryContext;
use crate::{Answer, Condition, Map, Reason};

pub(crate) struct MaybeTransmutableQuery<L, C>
where
    C: QueryContext,
{
    src: L,
    dst: L,
    assume: crate::Assume,
    context: C,
}

impl<L, C> MaybeTransmutableQuery<L, C>
where
    C: QueryContext,
{
    pub(crate) fn new(src: L, dst: L, assume: crate::Assume, context: C) -> Self {
        Self { src, dst, assume, context }
    }
}

// FIXME: Nix this cfg, so we can write unit tests independently of rustc
#[cfg(feature = "rustc")]
mod rustc {
    use rustc_middle::ty::layout::LayoutCx;
    use rustc_middle::ty::{Ty, TyCtxt, TypingEnv};

    use super::*;
    use crate::layout::tree::rustc::Err;

    impl<'tcx> MaybeTransmutableQuery<Ty<'tcx>, TyCtxt<'tcx>> {
        /// This method begins by converting `src` and `dst` from `Ty`s to `Tree`s,
        /// then computes an answer using those trees.
        #[instrument(level = "debug", skip(self), fields(src = ?self.src, dst = ?self.dst))]
        pub(crate) fn answer(
            self,
        ) -> Answer<<TyCtxt<'tcx> as QueryContext>::Region, <TyCtxt<'tcx> as QueryContext>::Type>
        {
            let Self { src, dst, assume, context } = self;

            let layout_cx = LayoutCx::new(context, TypingEnv::fully_monomorphized());

            // Convert `src` and `dst` from their rustc representations, to `Tree`-based
            // representations.
            let src = Tree::from_ty(src, layout_cx);
            let dst = Tree::from_ty(dst, layout_cx);

            match (src, dst) {
                (Err(Err::TypeError(_)), _) | (_, Err(Err::TypeError(_))) => {
                    Answer::No(Reason::TypeError)
                }
                (Err(Err::UnknownLayout), _) => Answer::No(Reason::SrcLayoutUnknown),
                (_, Err(Err::UnknownLayout)) => Answer::No(Reason::DstLayoutUnknown),
                (Err(Err::NotYetSupported), _) => Answer::No(Reason::SrcIsNotYetSupported),
                (_, Err(Err::NotYetSupported)) => Answer::No(Reason::DstIsNotYetSupported),
                (Err(Err::SizeOverflow), _) => Answer::No(Reason::SrcSizeOverflow),
                (_, Err(Err::SizeOverflow)) => Answer::No(Reason::DstSizeOverflow),
                (Ok(src), Ok(dst)) => MaybeTransmutableQuery { src, dst, assume, context }.answer(),
            }
        }
    }
}

impl<C>
    MaybeTransmutableQuery<
        Tree<<C as QueryContext>::Def, <C as QueryContext>::Region, <C as QueryContext>::Type>,
        C,
    >
where
    C: QueryContext,
{
    /// Answers whether a `Tree` is transmutable into another `Tree`.
    ///
    /// This method begins by de-def'ing `src` and `dst`, and prunes private paths from `dst`,
    /// then converts `src` and `dst` to `Dfa`s, and computes an answer using those DFAs.
    #[inline(always)]
    #[instrument(level = "debug", skip(self), fields(src = ?self.src, dst = ?self.dst))]
    pub(crate) fn answer(self) -> Answer<<C as QueryContext>::Region, <C as QueryContext>::Type> {
        let Self { src, dst, assume, context } = self;

        // Unconditionally remove all `Def` nodes from `src`, without pruning away the
        // branches they appear in. This is valid to do for value-to-value
        // transmutations, but not for `&mut T` to `&mut U`; we will need to be
        // more sophisticated to handle transmutations between mutable
        // references.
        let src = src.prune(&|_def| false);

        if src.is_inhabited() && !dst.is_inhabited() {
            return Answer::No(Reason::DstUninhabited);
        }

        trace!(?src, "pruned src");

        // Remove all `Def` nodes from `dst`, additionally...
        let dst = if assume.safety {
            // ...if safety is assumed, don't check if they carry safety
            // invariants; retain all paths.
            dst.prune(&|_def| false)
        } else {
            // ...otherwise, prune away all paths with safety invariants from
            // the `Dst` layout.
            dst.prune(&|def| def.has_safety_invariants())
        };

        trace!(?dst, "pruned dst");

        // Convert `src` from a tree-based representation to an DFA-based
        // representation. If the conversion fails because `src` is uninhabited,
        // conclude that the transmutation is acceptable, because instances of
        // the `src` type do not exist.
        let src = match Dfa::from_tree(src) {
            Ok(src) => src,
            Err(layout::Uninhabited) => return Answer::Yes,
        };

        // Convert `dst` from a tree-based representation to an DFA-based
        // representation. If the conversion fails because `src` is uninhabited,
        // conclude that the transmutation is unacceptable. Valid instances of
        // the `dst` type do not exist, either because it's genuinely
        // uninhabited, or because there are no branches of the tree that are
        // free of safety invariants.
        let dst = match Dfa::from_tree(dst) {
            Ok(dst) => dst,
            Err(layout::Uninhabited) => return Answer::No(Reason::DstMayHaveSafetyInvariants),
        };

        MaybeTransmutableQuery { src, dst, assume, context }.answer()
    }
}

impl<C> MaybeTransmutableQuery<Dfa<<C as QueryContext>::Region, <C as QueryContext>::Type>, C>
where
    C: QueryContext,
{
    /// Answers whether a `Dfa` is transmutable into another `Dfa`.
    pub(crate) fn answer(self) -> Answer<<C as QueryContext>::Region, <C as QueryContext>::Type> {
        self.answer_memo(&mut Map::default(), self.src.start, self.dst.start)
    }

    #[inline(always)]
    #[instrument(level = "debug", skip(self))]
    fn answer_memo(
        &self,
        cache: &mut Map<
            (dfa::State, dfa::State),
            Answer<<C as QueryContext>::Region, <C as QueryContext>::Type>,
        >,
        src_state: dfa::State,
        dst_state: dfa::State,
    ) -> Answer<<C as QueryContext>::Region, <C as QueryContext>::Type> {
        if let Some(answer) = cache.get(&(src_state, dst_state)) {
            answer.clone()
        } else {
            let answer = ensure_sufficient_stack(|| self.answer_impl(cache, src_state, dst_state));
            if let Some(..) = cache.insert((src_state, dst_state), answer.clone()) {
                panic!("failed to correctly cache transmutability")
            }
            answer
        }
    }

    fn answer_impl(
        &self,
        cache: &mut Map<
            (dfa::State, dfa::State),
            Answer<<C as QueryContext>::Region, <C as QueryContext>::Type>,
        >,
        src_state: dfa::State,
        dst_state: dfa::State,
    ) -> Answer<<C as QueryContext>::Region, <C as QueryContext>::Type> {
        debug!(?src_state, ?dst_state);
        debug!(src = ?self.src);
        debug!(dst = ?self.dst);
        debug!(
            src_transitions_len = self.src.transitions.len(),
            dst_transitions_len = self.dst.transitions.len()
        );
        if dst_state == self.dst.accept {
            // truncation: `size_of(Src) >= size_of(Dst)`
            //
            // Why is truncation OK to do? Because even though the Src is bigger, all we care about
            // is whether we have enough data for the Dst to be valid in accordance with what its
            // type dictates.
            // For example, in a u8 to `()` transmutation, we have enough data available from the u8
            // to transmute it to a `()` (though in this case does `()` really need any data to
            // begin with? It doesn't). Same thing with u8 to fieldless struct.
            // Now then, why is something like u8 to bool not allowed? That is not because the bool
            // is smaller in size, but rather because those 2 bits that we are re-interpreting from
            // the u8 could introduce invalid states for the bool type.
            //
            // So, if it's possible to transmute to a smaller Dst by truncating, and we can guarantee
            // that none of the actually-used data can introduce an invalid state for Dst's type, we
            // are able to safely transmute, even with truncation.
            Answer::Yes
        } else if src_state == self.src.accept {
            // extension: `size_of(Src) <= size_of(Dst)`
            if let Some(dst_state_prime) = self.dst.get_uninit_edge_dst(dst_state) {
                self.answer_memo(cache, src_state, dst_state_prime)
            } else {
                Answer::No(Reason::DstIsTooBig)
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
                union(self.src.bytes_from(src_state), self.dst.bytes_from(dst_state)).filter_map(
                    |(_range, (src_state_prime, dst_state_prime))| {
                        match (src_state_prime, dst_state_prime) {
                            // No matching transitions in `src`. Skip.
                            (None, _) => None,
                            // No matching transitions in `dst`. Fail.
                            (Some(_), None) => Some(Answer::No(Reason::DstIsBitIncompatible)),
                            // Matching transitions. Continue with successor states.
                            (Some(src_state_prime), Some(dst_state_prime)) => {
                                Some(self.answer_memo(cache, src_state_prime, dst_state_prime))
                            }
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
            // the algorithm; only its performance.
            debug!(?bytes_answer);
            match bytes_answer {
                Answer::No(_) if !self.assume.validity => return bytes_answer,
                Answer::Yes if self.assume.validity => return bytes_answer,
                _ => {}
            };

            let refs_answer = src_quantifier.apply(
                // for each reference transition out of `src_state`...
                self.src.refs_from(src_state).map(|(src_ref, src_state_prime)| {
                    // ...there exists a reference transition out of `dst_state`...
                    Quantifier::ThereExists.apply(self.dst.refs_from(dst_state).map(
                        |(dst_ref, dst_state_prime)| {
                            if !src_ref.is_mut && dst_ref.is_mut {
                                Answer::No(Reason::DstIsMoreUnique)
                            } else if !self.assume.alignment
                                && src_ref.referent_align < dst_ref.referent_align
                            {
                                Answer::No(Reason::DstHasStricterAlignment {
                                    src_min_align: src_ref.referent_align,
                                    dst_min_align: dst_ref.referent_align,
                                })
                            } else if dst_ref.referent_size > src_ref.referent_size {
                                Answer::No(Reason::DstRefIsTooBig {
                                    src: src_ref.referent,
                                    src_size: src_ref.referent_size,
                                    dst: dst_ref.referent,
                                    dst_size: dst_ref.referent_size,
                                })
                            } else {
                                let mut conditions = Vec::with_capacity(4);
                                let mut is_transmutable =
                                    |src: Reference<_, _>, dst: Reference<_, _>| {
                                        conditions.push(Condition::Transmutable {
                                            src: src.referent,
                                            dst: dst.referent,
                                        });
                                        if !self.assume.lifetimes {
                                            conditions.push(Condition::Outlives {
                                                long: src.region,
                                                short: dst.region,
                                            });
                                        }
                                    };

                                is_transmutable(src_ref, dst_ref);

                                if dst_ref.is_mut {
                                    is_transmutable(dst_ref, src_ref);
                                } else {
                                    conditions.push(Condition::Immutable { ty: dst_ref.referent });
                                }

                                Answer::If(Condition::IfAll(conditions)).and(self.answer_memo(
                                    cache,
                                    src_state_prime,
                                    dst_state_prime,
                                ))
                            }
                        },
                    ))
                }),
            );

            if self.assume.validity {
                bytes_answer.or(refs_answer)
            } else {
                bytes_answer.and(refs_answer)
            }
        }
    }
}

impl<R, T> Answer<R, T> {
    fn and(self, rhs: Answer<R, T>) -> Answer<R, T> {
        let lhs = self;
        match (lhs, rhs) {
            // If both are errors, then we should return the more specific one
            (Answer::No(Reason::DstIsBitIncompatible), Answer::No(reason))
            | (Answer::No(reason), Answer::No(_))
            // If either is an error, return it
            | (Answer::No(reason), _) | (_, Answer::No(reason)) => Answer::No(reason),
            // If only one side has a condition, pass it along
            | (Answer::Yes, other) | (other, Answer::Yes) => other,
            // If both sides have IfAll conditions, merge them
            (Answer::If(Condition::IfAll(mut lhs)), Answer::If(Condition::IfAll(ref mut rhs))) => {
                lhs.append(rhs);
                Answer::If(Condition::IfAll(lhs))
            }
            // If only one side is an IfAll, add the other Condition to it
            (Answer::If(cond), Answer::If(Condition::IfAll(mut conds)))
            | (Answer::If(Condition::IfAll(mut conds)), Answer::If(cond)) => {
                conds.push(cond);
                Answer::If(Condition::IfAll(conds))
            }
            // Otherwise, both lhs and rhs conditions can be combined in a parent IfAll
            (Answer::If(lhs), Answer::If(rhs)) => Answer::If(Condition::IfAll(vec![lhs, rhs])),
        }
    }

    fn or(self, rhs: Answer<R, T>) -> Answer<R, T> {
        let lhs = self;
        match (lhs, rhs) {
            // If both are errors, then we should return the more specific one
            (Answer::No(Reason::DstIsBitIncompatible), Answer::No(reason))
            | (Answer::No(reason), Answer::No(_)) => Answer::No(reason),
            // Otherwise, errors can be ignored for the rest of the pattern matching
            (Answer::No(_), other) | (other, Answer::No(_)) => other.or(Answer::Yes),
            // If only one side has a condition, pass it along
            (Answer::Yes, other) | (other, Answer::Yes) => other,
            // If both sides have IfAny conditions, merge them
            (Answer::If(Condition::IfAny(mut lhs)), Answer::If(Condition::IfAny(ref mut rhs))) => {
                lhs.append(rhs);
                Answer::If(Condition::IfAny(lhs))
            }
            // If only one side is an IfAny, add the other Condition to it
            (Answer::If(cond), Answer::If(Condition::IfAny(mut conds)))
            | (Answer::If(Condition::IfAny(mut conds)), Answer::If(cond)) => {
                conds.push(cond);
                Answer::If(Condition::IfAny(conds))
            }
            // Otherwise, both lhs and rhs conditions can be combined in a parent IfAny
            (Answer::If(lhs), Answer::If(rhs)) => Answer::If(Condition::IfAny(vec![lhs, rhs])),
        }
    }
}

enum Quantifier {
    ThereExists,
    ForAll,
}

impl Quantifier {
    fn apply<R, T, I>(&self, iter: I) -> Answer<R, T>
    where
        R: layout::Region,
        T: layout::Type,
        I: IntoIterator<Item = Answer<R, T>>,
    {
        use std::ops::ControlFlow::{Break, Continue};

        let (init, try_fold_f): (_, fn(_, _) -> _) = match self {
            Self::ThereExists => {
                (Answer::No(Reason::DstIsBitIncompatible), |accum: Answer<R, T>, next| match accum
                    .or(next)
                {
                    Answer::Yes => Break(Answer::Yes),
                    maybe => Continue(maybe),
                })
            }
            Self::ForAll => (Answer::Yes, |accum: Answer<R, T>, next| {
                let answer = accum.and(next);
                match answer {
                    Answer::No(_) => Break(answer),
                    maybe => Continue(maybe),
                }
            }),
        };

        let (Continue(result) | Break(result)) = iter.into_iter().try_fold(init, try_fold_f);
        result
    }
}
