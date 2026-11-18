use std::marker::PhantomData;

pub use clause_filter::NoClauses;
use clause_filter::*;
pub use iterator::CallerBoundsIterator;
use output_as::*;
use polarity_filter::*;
use self_ty_filter::*;
use trait_filter::*;

use crate::inherent::*;
use crate::lang_items::SolverTraitLangItem;
use crate::{Binder, ClauseKind, ClausePolarity, Interner, TraitClause, Upcast};

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
    use crate::param_env::output_as::OutputAs;

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
            self.clauses.filter(|&i| !matches!(i.kind().skip_binder(), ClauseKind::Trait(_)))
        }
    }

    impl<I, Iter> CallerBoundsIterator<I, Iter> for CallerBoundsAccessor<I, Iter, AllClauses>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
    {
        type Item = I::Clause;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            self.clauses
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
            self.clauses.filter_map(move |clause| {
                let kind = clause.kind();
                match kind.skip_binder() {
                    ClauseKind::Trait(trait_clause)
                        if filter_trait_clause_helper(&self.clause_filter, trait_clause) =>
                    {
                        Some(O::construct_from_trait_clause(
                            kind.rebind(trait_clause),
                            self.clause_filter.cx,
                        ))
                    }
                    _ => None,
                }
            })
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

    pub struct TraitClauses<I, P, S, T, O> {
        pub cx: I,
        pub polarity_filter: P,
        pub self_type_filter: S,
        pub trait_filter: T,
        pub output: O,
    }

    impl<I, P, S, T, O> ClauseFilter for TraitClauses<I, P, S, T, O> {}

    pub struct AllClauses;
    impl ClauseFilter for AllClauses {}

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
        // FIXME: asking for _cx here is useful down the line.
        pub fn all_clauses(self, _cx: I) -> CallerBoundsAccessor<I, Iter, AllClauses> {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: AllClauses,
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
        pub fn for_trait(
            self,
            def_id: I::TraitId,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TraitWithId<I>, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses { polarity_filter, self_type_filter, trait_filter: _, output, cx },
                cx: PhantomData,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    polarity_filter,
                    self_type_filter,
                    trait_filter: TraitWithId(def_id),
                    output,
                    cx,
                },
                cx: PhantomData,
            }
        }
        #[inline]
        pub fn for_trait_lang_item(
            self,
            lang_item: SolverTraitLangItem,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<I, P, S, TraitWithId<I>, O>> {
            let cx = self.clause_filter.cx;
            self.for_trait(cx.require_trait_lang_item(lang_item))
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

    pub trait TraitFilter<I>
    where
        I: Interner,
    {
        fn allow_clause(&self, clause: TraitClause<I>) -> bool;
    }

    impl<I> TraitFilter<I> for ()
    where
        I: Interner,
    {
        #[inline]
        fn allow_clause(&self, _clause: TraitClause<I>) -> bool {
            true
        }
    }

    pub struct TraitWithId<I>(pub I::TraitId)
    where
        I: Interner;
    impl<I> TraitFilter<I> for TraitWithId<I>
    where
        I: Interner,
    {
        #[inline]
        fn allow_clause(&self, clause: TraitClause<I>) -> bool {
            clause.def_id() == self.0
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
