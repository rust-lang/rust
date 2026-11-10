use std::iter;
use std::marker::PhantomData;

pub use clause_filter::NoClauses;
use clause_filter::*;
pub use iterator::CallerBoundsIterator;
use output_as::*;
use output_filter::*;
use polarity_filter::*;
use rustc_data_structures::either::Either;
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
        type Item = Binder<I, ClauseKind<I>>;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            let cx = self.clause_filter.0;
            self.clauses.flat_map(move |clause| match clause.kind().skip_binder() {
                // Reconstruct clauses from compressed trait clauses.
                ClauseKind::CompressedTrait(ct) => {
                    Either::Left(ct.compressed_trait_refs.iter().map(move |c| {
                        c.trait_clause_from_single_bit(cx, ct.self_ty).map_bound(ClauseKind::Trait)
                    }))
                }
                _ => Either::Right(iter::once(clause.kind())),
            })
        }
    }

    impl<I, Iter> CallerBoundsIterator<I, Iter> for CallerBoundsAccessor<I, Iter, AllClausesInterned<I>>
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
    fn filter_trait_clause_helper<I, P, S, T, F, O, X>(
        filter: &TraitClauses<I, P, S, T, F, O>,
        clause: TraitClause<I>,
        bound: Binder<I, X>,
    ) -> bool
    where
        I: Interner,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        F: OutputFilter<I>,
        O: OutputAs<I>,
    {
        let TraitClauses {
            cx: _,
            polarity_filter,
            self_type_filter,
            trait_filter,
            output_filter,
            output: _,
        } = filter;
        polarity_filter.allow_polarity(clause.polarity)
            && self_type_filter.allow_self_ty(clause.self_ty())
            && trait_filter.allow_clause(clause)
            && output_filter.allow_trait_clause(bound.rebind(clause))
    }

    #[inline]
    fn filter_compressed_trait_helper<I, T, F>(
        trait_filter: &T,
        output_filter: &F,
        cx: I,
        self_ty: I::GenericArgs,
        trait_ref_single_bit: CompressedTraitRefs,
    ) -> Option<Binder<I, TraitClause<I>>>
    where
        I: Interner,
        T: TraitFilter<I>,
        F: OutputFilter<I>,
    {
        if let Some(def_id) =
            trait_filter.allow_compressed_trait_ref_single_bit(cx, trait_ref_single_bit)
        {
            // Only if all filtering allows this compressed clause do we reconstruct the trait clause,
            // but we don't intern it yet, since that depends on the output transformation. If we can
            // avoid re-interning, that's worth a *lot*.
            Some(CompressedTraitRefs::trait_clause_from_def_id(cx, self_ty, def_id))
                .filter(|i| output_filter.allow_trait_clause(*i))
        } else {
            None
        }
    }

    struct CallerBoundsIteratorImpl<I, Iter, CF>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        CF: ClauseFilter,
    {
        accessor: CallerBoundsAccessor<I, Iter, CF>,
        decompression_in_progress: CompressedTraitRefs,
        // self ty of the currently decompression-in-progress trait ref,
        // if applicable.
        self_ty: Option<I::GenericArgs>,
    }

    impl<I, Iter, P, S, T, F, O> Iterator
        for CallerBoundsIteratorImpl<I, Iter, TraitClauses<I, P, S, T, F, O>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        F: OutputFilter<I>,
        O: OutputAs<I>,
    {
        type Item = O::Output;

        fn next(&mut self) -> Option<Self::Item> {
            let CallerBoundsAccessor { clauses, clause_filter: filter, cx: _ } = &mut self.accessor;
            let cx = filter.cx;

            // This loop finds the next item in `clauses` that can be returned.
            // If `clauses` is empty, we bail out of the function inside the loop,
            // otherwise the next non-filtered element is returned using `break`,
            // expanding compressed trait refs.
            let res = loop {
                // First check whether we're still expanding some compressed trait,
                // and make one step of progress towards it if we can.
                // `self.compressed_elem` is `0` when we're not expanding a compressed trait,
                // and in that case this function returns `None`
                if let Some(trait_ref_single_bit) =
                    self.decompression_in_progress.get_and_mask_least_significant_bit()
                {
                    match filter_compressed_trait_helper(
                        &filter.trait_filter,
                        &filter.output_filter,
                        cx,
                        self.self_ty.unwrap(),
                        trait_ref_single_bit,
                    ) {
                        Some(tc) => break tc,
                        None => continue,
                    }
                } else {
                    // Only if we indeed are not, do we check the next item in the iterator.
                    // If exhausted, we also return None.
                    let clause = clauses.next()?;
                    let kind = clause.kind();
                    match kind.skip_binder() {
                        // If we find a trait clause, and it matches, return it
                        ClauseKind::Trait(trait_clause)
                            if filter_trait_clause_helper(&filter, trait_clause, kind) =>
                        {
                            break kind.rebind(trait_clause);
                        }
                        // If we find a compressed trait clause, and it matches, store it in `self.compressed_elem`
                        ClauseKind::CompressedTrait(ct)
                            if filter.self_type_filter.allow_self_ty(ct.self_ty.type_at(0))
                                && filter
                                    .polarity_filter
                                    .allow_polarity(ClausePolarity::Positive)
                                && filter
                                    .trait_filter
                                    .allow_any_compressed_trait_ref(ct.compressed_trait_refs) =>
                        {
                            self.decompression_in_progress = ct.compressed_trait_refs;
                            self.self_ty = Some(ct.self_ty);
                        }
                        _ => continue,
                    }
                }
            };

            Some(O::construct_from_trait_clause(res, cx))
        }
    }

    impl<I, Iter, P, S, T, F, O> CallerBoundsIterator<I, Iter>
        for CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, F, O>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        F: OutputFilter<I>,
        O: OutputAs<I>,
    {
        type Item = O::Output;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            CallerBoundsIteratorImpl {
                accessor: self,
                decompression_in_progress: CompressedTraitRefs::empty(),
                self_ty: None,
            }
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
    pub struct TraitClauses<I, P, S, T, F, O> {
        pub cx: I,
        pub polarity_filter: P,
        pub self_type_filter: S,
        pub trait_filter: T,
        pub output_filter: F,
        pub output: O,
    }

    impl<I, P, S, T, F, O> ClauseFilter for TraitClauses<I, P, S, T, F, O> {}

    pub struct AllClauses<I>(pub I)
    where
        I: Interner;
    impl<I> ClauseFilter for AllClauses<I> where I: Interner {}

    pub struct AllClausesInterned<I>(pub I)
    where
        I: Interner;
    impl<I> ClauseFilter for AllClausesInterned<I> where I: Interner {}

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
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, (), (), (), (), OutputAsTraitClause>>
        {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: TraitClauses {
                    polarity_filter: (),
                    self_type_filter: (),
                    trait_filter: (),
                    output_filter: (),
                    output: OutputAsTraitClause,
                    cx,
                },
                cx: PhantomData,
            }
        }

        /// Warning: may be a performance hazzard!
        #[inline]
        pub fn all_clauses_interned(
            self,
            cx: I,
        ) -> CallerBoundsAccessor<I, Iter, AllClausesInterned<I>> {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: AllClausesInterned(cx),
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

    impl<I, Iter, P, S, T, F, O> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, F, O>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        F: OutputFilter<I>,
        O: OutputAs<I>,
    {
        #[inline]
        pub fn with_polarity(
            self,
            polarity: ClausePolarity,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, ClausePolarity, S, T, F, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        cx,
                        polarity_filter: _,
                        self_type_filter,
                        trait_filter,
                        output_filter,
                        output,
                    },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    cx,
                    polarity_filter: polarity,
                    self_type_filter,
                    trait_filter,
                    output_filter,
                    output,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn validate_self_ty<SF: Fn(I::Ty) -> bool>(
            self,
            validate_fn: SF,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, SelfTyFilterFn<SF>, T, F, O>>
        {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        cx,
                        polarity_filter,
                        self_type_filter: _,
                        trait_filter,
                        output_filter,
                        output,
                    },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    cx,
                    polarity_filter,
                    self_type_filter: SelfTyFilterFn { f: validate_fn },
                    trait_filter,
                    output_filter,
                    output,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn with_self_ty(
            self,
            desired_self_ty: I::Ty,
        ) -> CallerBoundsAccessor<
            I,
            Iter,
            TraitClauses<I, P, SelfTyFilterFn<impl Fn(I::Ty) -> bool>, T, F, O>,
        > {
            self.validate_self_ty(move |actual_self_ty| actual_self_ty == desired_self_ty)
        }

        #[inline]
        fn for_target_trait(
            self,
            trait_filter: TargetTrait<I>,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TargetTrait<I>, F, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        cx,
                        polarity_filter,
                        self_type_filter,
                        trait_filter: _,
                        output_filter,
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
                    output_filter,
                    output,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn for_trait(
            self,
            def_id: I::TraitId,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TargetTrait<I>, F, O>> {
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
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TargetTrait<I>, F, O>> {
            let cx = self.clause_filter.cx;
            self.for_target_trait(TargetTrait {
                def_id: cx.require_trait_lang_item(lang_item),
                lang_item_compressed: CompressedTraitRefs::from_lang_item(lang_item),
            })
        }

        /// Filter the final [`TraitClause`] before interning.
        #[inline]
        pub fn filter<OF>(
            self,
            f: OF,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, OutputFilterFn<OF>, O>>
        where
            OF: Fn(Binder<I, TraitClause<I>>) -> bool,
        {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        cx,
                        polarity_filter,
                        self_type_filter,
                        trait_filter,
                        output_filter: _,
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
                    output_filter: OutputFilterFn { f },
                    output,
                },
                cx: PhantomData,
            }
        }

        #[inline]
        pub fn interned(
            self,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, T, F, OutputAsInternedClause>>
        {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        cx,
                        polarity_filter,
                        self_type_filter,
                        trait_filter,
                        output_filter,
                        output: _,
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
                    output_filter,
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

    pub struct SelfTyFilterFn<F> {
        pub f: F,
    }
    impl<I, F> SelfTyFilter<I> for SelfTyFilterFn<F>
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
        /// Returns the defid of the trait if this filter allows the given `trait_ref`.
        /// Note that `trait_ref` should have only a single bit set.
        fn allow_compressed_trait_ref_single_bit(
            &self,
            cx: I,
            trait_ref: CompressedTraitRefs,
        ) -> Option<I::TraitId>;
        /// returns true if this trait filter *might* allow any
        /// of the traits in `trait_ref` to match.
        fn allow_any_compressed_trait_ref(&self, trait_ref: CompressedTraitRefs) -> bool;
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

        fn allow_any_compressed_trait_ref(&self, _trait_ref: CompressedTraitRefs) -> bool {
            true
        }
    }

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
            // When looking for a compressed trait, you'd think no normal clause
            // can ever match so we could return false unconditionally when
            // self.lang_item_compressed is `None`. However, bound clauses are
            // never compressed, so in cases where we have `for<'a> T: Send`,
            // a `Send` bound which would otherwise be compressed is actually not,
            // and we should return it.
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

        fn allow_any_compressed_trait_ref(&self, trait_ref: CompressedTraitRefs) -> bool {
            if let Some(target) = self.lang_item_compressed {
                // only if the given `trait_ref` has the bit set that corresponds to `target`, do we
                // return true. Otherwise we can skip this compressed trait instantly.
                trait_ref.contains(target)
            } else {
                // When looking for a normal trait, any compressed clause can never match so we
                // return false unconditionally.
                false
            }
        }
    }
}

mod output_filter {
    use super::*;

    pub trait OutputFilter<I>
    where
        I: Interner,
    {
        fn allow_trait_clause(&self, clause: Binder<I, TraitClause<I>>) -> bool;
    }

    impl<I> OutputFilter<I> for ()
    where
        I: Interner,
    {
        #[inline]
        fn allow_trait_clause(&self, _clause: Binder<I, TraitClause<I>>) -> bool {
            true
        }
    }

    pub struct OutputFilterFn<F> {
        pub f: F,
    }
    impl<I, F> OutputFilter<I> for OutputFilterFn<F>
    where
        I: Interner,
        F: Fn(Binder<I, TraitClause<I>>) -> bool,
    {
        #[inline]
        fn allow_trait_clause(&self, clause: Binder<I, TraitClause<I>>) -> bool {
            (self.f)(clause)
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
