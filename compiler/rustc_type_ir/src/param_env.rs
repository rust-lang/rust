use std::iter;
use std::marker::PhantomData;

pub use clause_filter::NoClauses;
pub use iterator::CallerBoundsIterator;
use rustc_data_structures::either::Either;

use clause_filter::*;
use output_as::*;
use polarity_filter::*;
use self_ty_filter::*;
use trait_filter::*;

use crate::inherent::*;
use crate::lang_items::SolverTraitLangItem;
use crate::{
    Binder, ClauseKind, ClausePolarity, CompressedTraitRefs, Interner, TraitClause, Upcast,
};

#[must_use]
pub struct CallerBoundsAccessor<I, Iter, CF>
where
    I: Interner,
    Iter: Iterator<Item = I::Clause>,
    CF: ClauseFilter,
{
    clauses: Iter,
    clause_filter: CF,
    cx: PhantomData<I>,
}

impl<I, Iter> CallerBoundsAccessor<I, Iter, NoClauses>
where
    I: Interner,
    Iter: Iterator<Item = I::Clause>,
{
    #[inline]
    pub fn new(clauses: Iter) -> Self {
        Self { clauses, clause_filter: NoClauses, cx: PhantomData }
    }
}

mod iterator {
    use super::*;

    pub trait CallerBoundsIterator<I, Iter>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
    {
        type Item;
        /// Note: likely less efficient than staying in `CallerBoundsIterator` land.
        fn iter(self) -> impl Iterator<Item = Self::Item>;
    }
    impl<I, Iter> CallerBoundsIterator<I, Iter> for CallerBoundsAccessor<I, Iter, NonTraitClauses>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
    {
        type Item = I::Clause;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            self.clauses.filter(|&i| {
                !matches!(
                    i.kind().skip_binder(),
                    ClauseKind::Trait(_) | ClauseKind::CompressedTrait(_)
                )
            })
        }
    }

    impl<I, Iter> CallerBoundsIterator<I, Iter> for CallerBoundsAccessor<I, Iter, AllClauses<I>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
    {
        type Item = I::Clause;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            let cx = self.clause_filter.0;
            self.clauses.flat_map(move |clause| match clause.kind().skip_binder() {
                // Reconstruct clauses from compressed trait clauses.
                ClauseKind::CompressedTrait(ct) => Either::Left(
                    ct.compressed_trait_refs
                        .iter()
                        .map(move |c| c.trait_clause_from_single_bit(cx, ct.self_ty).upcast(cx)),
                ),
                _ => Either::Right(iter::once(clause)),
            })
        }
    }

    #[inline]
    fn filter_trait_clause_helper<I, P, S, T, O>(
        filter: &TraitClauses<I, P, S, T, O>,
        clause: TraitClause<I>,
    ) -> bool
    where
        I: Interner,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        O: OutputAs<I>,
    {
        let TraitClauses { cx: _, polarity_filter, self_type_filter, trait_filter, output: _ } =
            filter;

        polarity_filter.allow_polarity(clause.polarity)
            && self_type_filter.allow_self_ty(clause.self_ty())
            && trait_filter.allow_clause(clause)
    }

    #[inline]
    fn filter_compressed_trait_clause_helper<I, P, S, T, O>(
        filter: &TraitClauses<I, P, S, T, O>,
        self_ty: I::Ty,
        trait_ref_single_bit: CompressedTraitRefs,
    ) -> Option<Binder<I, TraitClause<I>>>
    where
        I: Interner,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        O: OutputAs<I>,
    {
        let &TraitClauses {
            cx,
            ref polarity_filter,
            ref self_type_filter,
            ref trait_filter,
            output: _,
        } = filter;
        // All compressed clauses are positive clauses, so filter negatives out.
        if polarity_filter.allow_polarity(ClausePolarity::Positive)
            // Match the self ty, of course.
            && self_type_filter.allow_self_ty(self_ty)
            && let Some(def_id) = trait_filter.allow_compressed_trait_ref_single_bit(cx, trait_ref_single_bit)
        {
            // Only if all filtering allows this compressed clause do we reconstruct the trait clause,
            // but we don't intern it yet, since that depends on the output transformation. If we can
            // avoid re-interning, that's worth a *lot*.
            Some(CompressedTraitRefs::trait_clause_from_def_id(cx, self_ty, def_id))
        } else {
            None
        }
    }

    impl<I, Iter, P, S, T, O> CallerBoundsIterator<I, Iter>
        for CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, O>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        O: OutputAs<I>,
    {
        type Item = O::Output;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            let Self { clauses, clause_filter: filter, cx: _ } = self;
            let cx = filter.cx;

            clauses.flat_map(move |clause| {
                let kind = clause.kind();
                match kind.skip_binder() {
                    ClauseKind::Trait(trait_clause)
                        if filter_trait_clause_helper(&filter, trait_clause) =>
                    {
                        // Use either to allow returning two kinds of iterators.
                        Either::Left(Some(kind.rebind(trait_clause)).into_iter())
                    }
                    ClauseKind::CompressedTrait(ct) => Either::Right(
                        ct.compressed_trait_refs
                            .iter()
                            .filter_map(|i| {
                                filter_compressed_trait_clause_helper(&filter, ct.self_ty, i)
                            })
                            .collect::<Vec<_>>()
                            .into_iter(),
                    ),
                    _ => Either::Left(None.into_iter()),
                }
            }).map(move |i| O::construct_from_trait_clause(i, cx))
        }
    }
}

mod clause_filter {
    use super::*;

    pub trait ClauseFilter {}

    pub struct NoClauses;
    impl ClauseFilter for NoClauses {}

    pub struct NonTraitClauses;
    impl ClauseFilter for NonTraitClauses {}

    #[derive(Clone, Copy)]
    pub struct TraitClauses<I, P, S, T, O> {
        pub cx: I,
        pub polarity_filter: P,
        pub self_type_filter: S,
        pub trait_filter: T,
        pub output: O,
    }

    impl<I, P, S, T, O> ClauseFilter for TraitClauses<I, P, S, T, O> {}

    pub struct AllClauses<I>(pub I)
    where
        I: Interner;
    impl<I> ClauseFilter for AllClauses<I> where I: Interner {}

    impl<I, Iter> CallerBoundsAccessor<I, Iter, NoClauses>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
    {
        #[inline]
        pub fn non_trait_clauses(self) -> CallerBoundsAccessor<I, Iter, NonTraitClauses> {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: NonTraitClauses,
                cx: PhantomData,
            }
        }

        #[inline]
        pub fn trait_clauses(
            self,
            cx: I,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, (), (), (), OutputAsTraitClause>>
        {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: TraitClauses {
                    polarity_filter: (),
                    self_type_filter: (),
                    trait_filter: (),
                    output: OutputAsTraitClause,
                    cx,
                },
                cx: PhantomData,
            }
        }

        /// Warning: may be a performance hazzard!
        #[inline]
        pub fn all_clauses(self, cx: I) -> CallerBoundsAccessor<I, Iter, AllClauses<I>> {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: AllClauses(cx),
                cx: PhantomData,
            }
        }
    }
}

mod polarity_filter {
    use super::*;

    pub trait PolarityFilter {
        fn allow_polarity(&self, p: ClausePolarity) -> bool;
    }
    impl PolarityFilter for () {
        #[inline]
        fn allow_polarity(&self, _p: ClausePolarity) -> bool {
            true
        }
    }
    impl PolarityFilter for ClausePolarity {
        #[inline]
        fn allow_polarity(&self, p: ClausePolarity) -> bool {
            *self == p
        }
    }

    impl<I, Iter, P, S, T, O> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, O>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        O: OutputAs<I>,
    {
        #[inline]
        pub fn with_polarity(
            self,
            polarity: ClausePolarity,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, ClausePolarity, S, T, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses { cx, polarity_filter: _, self_type_filter, trait_filter, output },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    cx,
                    polarity_filter: polarity,
                    self_type_filter,
                    trait_filter,
                    output,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn validate_self_ty<F: Fn(I::Ty) -> bool>(
            self,
            validate_fn: F,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, FilterFn<F>, T, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses { cx, polarity_filter, self_type_filter: _, trait_filter, output },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    cx,
                    polarity_filter,
                    self_type_filter: FilterFn { f: validate_fn },
                    trait_filter,
                    output,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn with_self_ty(
            self,
            desired_self_ty: I::Ty,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, FilterFn<impl Fn(I::Ty) -> bool>, T, O>>
        {
            self.validate_self_ty(move |actual_self_ty| actual_self_ty == desired_self_ty)
        }

        #[inline]
        fn for_target_trait(
            self,
            trait_filter: TargetTrait<I>,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TargetTrait<I>, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        cx,
                        polarity_filter,
                        self_type_filter,
                        trait_filter: _,
                        output,
                    },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    cx,
                    polarity_filter,
                    self_type_filter,
                    trait_filter,
                    output,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn for_trait(
            self,
            def_id: I::TraitId,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TargetTrait<I>, O>> {
            let cx = self.clause_filter.cx;

            self.for_target_trait(TargetTrait {
                def_id,
                lang_item_compressed: cx
                    .as_trait_lang_item(def_id)
                    .and_then(|lang_item| CompressedTraitRefs::from_lang_item(lang_item)),
            })
        }
        #[inline]
        pub fn for_trait_lang_item(
            self,
            lang_item: SolverTraitLangItem,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TargetTrait<I>, O>> {
            let cx = self.clause_filter.cx;
            self.for_target_trait(TargetTrait {
                def_id: cx.require_trait_lang_item(lang_item),
                lang_item_compressed: CompressedTraitRefs::from_lang_item(lang_item),
            })
        }

        #[inline]
        pub fn interned(
            self,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, OutputAsInternedClause>>
        {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses { cx, polarity_filter, self_type_filter, trait_filter, output: _ },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    cx,
                    polarity_filter,
                    self_type_filter,
                    trait_filter,
                    output: OutputAsInternedClause,
                },
                cx: PhantomData,
            }
        }
    }
}

mod self_ty_filter {
    use super::*;

    pub trait SelfTyFilter<I>
    where
        I: Interner,
    {
        fn allow_self_ty(&self, ty: I::Ty) -> bool;
    }

    impl<I> SelfTyFilter<I> for ()
    where
        I: Interner,
    {
        #[inline]
        fn allow_self_ty(&self, _ty: I::Ty) -> bool {
            true
        }
    }

    pub struct FilterFn<F> {
        pub f: F,
    }
    impl<I, F> SelfTyFilter<I> for FilterFn<F>
    where
        I: Interner,
        F: Fn(I::Ty) -> bool,
    {
        #[inline]
        fn allow_self_ty(&self, ty: I::Ty) -> bool {
            (self.f)(ty)
        }
    }
}

mod trait_filter {
    use super::*;
    use crate::CompressedTraitRefs;

    pub trait TraitFilter<I>
    where
        I: Interner,
    {
        fn allow_clause(&self, clause: TraitClause<I>) -> bool;
        fn allow_compressed_trait_ref_single_bit(
            &self,
            cx: I,
            trait_ref: CompressedTraitRefs,
        ) -> Option<I::TraitId>;
    }

    impl<I> TraitFilter<I> for ()
    where
        I: Interner,
    {
        #[inline]
        fn allow_clause(&self, _clause: TraitClause<I>) -> bool {
            true
        }

        #[inline]
        fn allow_compressed_trait_ref_single_bit(
            &self,
            cx: I,
            trait_ref: CompressedTraitRefs,
        ) -> Option<I::TraitId> {
            Some(cx.require_trait_lang_item(trait_ref.lang_item_from_single_bit()))
        }
    }

    // TODO: use bit operations to test an entire compressed clause against a single filter.
    pub struct TargetTrait<I>
    where
        I: Interner,
    {
        pub def_id: I::TraitId,
        /// When scanning specifically for trait refs that can be compressed,
        /// this value has a single bit set, representing the compressed form
        /// of the trait we're looking for.
        pub lang_item_compressed: Option<CompressedTraitRefs>,
    }
    impl<I> TraitFilter<I> for TargetTrait<I>
    where
        I: Interner,
    {
        #[inline]
        fn allow_clause(&self, clause: TraitClause<I>) -> bool {
            // When looking for a compressed trait, any normal clause can never match so we
            // return false unconditionally.
            if self.lang_item_compressed.is_some() {
                return false;
            }

            clause.def_id() == self.def_id
        }

        #[inline]
        fn allow_compressed_trait_ref_single_bit(
            &self,
            _cx: I,
            trait_ref: CompressedTraitRefs,
        ) -> Option<I::TraitId> {
            if let Some(target) = self.lang_item_compressed {
                (trait_ref == target).then_some(self.def_id)
            } else {
                // When looking for a normal trait, any compressed clause can never match so we
                // return false unconditionally.
                None
            }
        }
    }
}

mod output_as {
    use super::*;

    pub trait OutputAs<I: Interner> {
        type Output;
        fn construct_from_trait_clause(tc: Binder<I, TraitClause<I>>, cx: I) -> Self::Output;
    }

    pub struct OutputAsInternedClause;
    impl<I: Interner> OutputAs<I> for OutputAsInternedClause {
        type Output = I::Clause;
        fn construct_from_trait_clause(tc: Binder<I, TraitClause<I>>, cx: I) -> I::Clause {
            tc.upcast(cx)
        }
    }

    pub struct OutputAsTraitClause;
    impl<I: Interner> OutputAs<I> for OutputAsTraitClause {
        type Output = Binder<I, TraitClause<I>>;
        fn construct_from_trait_clause(
            tc: Binder<I, TraitClause<I>>,
            _cx: I,
        ) -> Binder<I, TraitClause<I>> {
            tc
        }
    }
}
