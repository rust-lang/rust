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
    /// Removes definition markers from both trees and, unless safety is assumed,
    /// prunes destination paths that may carry safety invariants. It then converts
    /// the remaining layouts to `Dfa`s and compares them.
    #[inline(always)]
    #[instrument(level = "debug", skip(self), fields(src = ?self.src, dst = ?self.dst))]
    pub(crate) fn answer(self) -> Answer<<C as QueryContext>::Region, <C as QueryContext>::Type> {
        let Self { src, dst, assume, context } = self;

        // Keep every source representation while removing its definition markers.
        // Reference nodes remain intact; mutable destination references also generate
        // a reverse transmutability obligation for their referents.
        let src = src.prune(&|_def| false);

        if src.is_inhabited() && !dst.is_inhabited() {
            return Answer::No(Reason::DstUninhabited);
        }

        trace!(?src, "pruned src");

        // Remove destination definition markers. Unless the caller assumes safety,
        // prune paths whose definitions may carry safety invariants.
        let dst = if assume.safety {
            dst.prune(&|_def| false)
        } else {
            dst.prune(&|def| def.has_safety_invariants())
        };

        trace!(?dst, "pruned dst");

        // Convert `src` from a tree-based representation to a DFA-based
        // representation. If the conversion fails because `src` is uninhabited,
        // conclude that the transmutation is acceptable, because instances of
        // the `src` type do not exist.
        let src = match Dfa::from_tree(src) {
            Ok(src) => src,
            Err(layout::Uninhabited) => return Answer::Yes,
        };

        // An inhabited source and an originally uninhabited destination were
        // rejected above. If the pruned destination is now uninhabited, no path
        // remains whose definitions are known to be free of safety invariants.
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
        debug!(src = ?self.src);
        debug!(dst = ?self.dst);
        debug!(
            src_transitions_len = self.src.transitions.len(),
            dst_transitions_len = self.dst.transitions.len()
        );
        self.answer_memo(&mut Map::default(), self.src.start, self.dst.start)
    }

    #[inline(always)]
    #[instrument(level = "debug", skip(self, cache))]
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
            let answer = self.answer_impl(cache, src_state, dst_state);
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
        if dst_state == self.dst.accept {
            // The destination needs no more input. Union transmutation permits
            // truncating the remaining source bytes: for example, `u8` to `()`.
            // Compatibility of the consumed prefix is checked by the preceding
            // transitions, including any conditions they generate for references.
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
                // The caller checks validity (for example, `src <= 1u8` for a
                // `u8`-to-`bool` transmutation), so at least one source transition
                // must admit a compatible continuation.
                Quantifier::ThereExists
            } else {
                // Every source transition must admit a compatible continuation
                // in the destination when the caller does not assume validity.
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
    /// Requires both answers, combining their conditions into a conjunction.
    fn and(self, rhs: Answer<R, T>) -> Answer<R, T> {
        let lhs = self;
        match (lhs, rhs) {
            // Prefer a specific reason over generic bit incompatibility;
            // otherwise, retain the left-hand reason.
            (Answer::No(Reason::DstIsBitIncompatible), Answer::No(reason))
            | (Answer::No(reason), Answer::No(_))
            // If either is an error, return it
            | (Answer::No(reason), _) | (_, Answer::No(reason)) => Answer::No(reason),
            // If only one side has a condition, pass it along
            (Answer::Yes, other) | (other, Answer::Yes) => other,
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

    /// Combines alternative answers and collects their conditions in an `IfAny`.
    ///
    /// Currently, combining `Yes` with `If` retains the condition. This differs
    /// from Boolean disjunction, where unconditional success would suffice.
    fn or(self, rhs: Answer<R, T>) -> Answer<R, T> {
        let lhs = self;
        match (lhs, rhs) {
            // Prefer a specific reason over generic bit incompatibility;
            // otherwise, retain the left-hand reason.
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
    /// Folds answers with `or` or `and`, stopping on `Yes` or `No`, respectively.
    /// An empty iterator yields bit incompatibility for `ThereExists` and `Yes`
    /// for `ForAll`.
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
