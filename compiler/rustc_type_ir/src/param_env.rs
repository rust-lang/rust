use std::marker::PhantomData;

pub use clause_filter::NoClauses;
pub use iterator::CallerBoundsIterator;

use crate::inherent::*;
use crate::lang_items::SolverTraitLangItem;
use crate::param_env::clause_filter::*;
use crate::param_env::output_transformer::*;
use crate::param_env::polarity_filter::*;
use crate::param_env::self_ty_filter::*;
use crate::param_env::trait_filter::*;
use crate::{Binder, ClauseKind, ClausePolarity, Interner, TraitClause};

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
        filter: &TraitClauses<P, S, T, O>,
        clause: TraitClause<I>,
    ) -> bool
    where
        I: Interner,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        O: OutputTransformer,
    {
        let TraitClauses { polarity_filter, self_type_filter, trait_filter, output_transformer: _ } =
            filter;
        polarity_filter.allow_polarity(clause.polarity)
            && self_type_filter.allow_self_ty(clause.self_ty())
            && trait_filter.allow_clause(clause)
    }

    impl<I, Iter, P, S, T> CallerBoundsIterator<I, Iter>
        for CallerBoundsAccessor<I, Iter, TraitClauses<P, S, T, OutputAsInternedClause>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
    {
        type Item = I::Clause;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            self.clauses.filter_map(move |clause| match clause.kind().skip_binder() {
                ClauseKind::Trait(trait_clause)
                    if filter_trait_clause_helper(&self.clause_filter, trait_clause) =>
                {
                    Some(clause)
                }
                _ => None,
            })
        }
    }

    impl<I, Iter, P, S, T> CallerBoundsIterator<I, Iter>
        for CallerBoundsAccessor<I, Iter, TraitClauses<P, S, T, OutputAsTraitClause>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
    {
        type Item = Binder<I, TraitClause<I>>;

        #[inline]
        fn iter(self) -> impl Iterator<Item = Self::Item> {
            self.clauses.filter_map(move |clause| {
                let kind = clause.kind();
                match kind.skip_binder() {
                    ClauseKind::Trait(trait_clause)
                        if filter_trait_clause_helper(&self.clause_filter, trait_clause) =>
                    {
                        Some(kind.rebind(trait_clause))
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

    pub struct TraitClauses<P, S, T, O> {
        pub polarity_filter: P,
        pub self_type_filter: S,
        pub trait_filter: T,
        pub output_transformer: O,
    }

    impl<P, S, T, O> ClauseFilter for TraitClauses<P, S, T, O> {}

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
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<(), (), (), OutputAsTraitClause>> {
            CallerBoundsAccessor {
                clauses: self.clauses,
                clause_filter: TraitClauses {
                    polarity_filter: (),
                    self_type_filter: (),
                    trait_filter: (),
                    output_transformer: OutputAsTraitClause,
                },
                cx: PhantomData,
            }
        }

        /// Warning: may be a performance hazzard!
        #[inline]
        pub fn all_clauses(self) -> CallerBoundsAccessor<I, Iter, AllClauses> {
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

    impl<I, Iter, P, S, T, O> CallerBoundsAccessor<I, Iter, TraitClauses<P, S, T, O>>
    where
        I: Interner,
        Iter: Iterator<Item = I::Clause>,
        P: PolarityFilter,
        S: SelfTyFilter<I>,
        T: TraitFilter<I>,
        O: OutputTransformer,
    {
        #[inline]
        pub fn with_polarity(
            self,
            polarity: ClausePolarity,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<ClausePolarity, S, T, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        polarity_filter: _,
                        self_type_filter,
                        trait_filter,
                        output_transformer,
                    },
                cx,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    polarity_filter: polarity,
                    self_type_filter,
                    trait_filter,
                    output_transformer,
                },
                cx,
            }
        }
        #[inline]
        pub fn validate_self_ty<F: Fn(I::Ty) -> bool>(
            self,
            validate_fn: F,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<P, FilterFn<F>, T, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        polarity_filter,
                        self_type_filter: _,
                        trait_filter,
                        output_transformer,
                    },
                cx,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    polarity_filter,
                    self_type_filter: FilterFn { f: validate_fn },
                    trait_filter,
                    output_transformer,
                },
                cx,
            }
        }
        #[inline]
        pub fn with_self_ty(
            self,
            desired_self_ty: I::Ty,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<P, FilterFn<impl Fn(I::Ty) -> bool>, T, O>>
        {
            self.validate_self_ty(move |actual_self_ty| actual_self_ty == desired_self_ty)
        }
        #[inline]
        pub fn for_trait(
            self,
            def_id: I::TraitId,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<P, S, TraitWithId<I>, O>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        polarity_filter,
                        self_type_filter,
                        trait_filter: _,
                        output_transformer,
                    },
                cx,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    polarity_filter,
                    self_type_filter,
                    trait_filter: TraitWithId(def_id),
                    output_transformer,
                },
                cx,
            }
        }
        #[inline]
        pub fn for_trait_lang_item(
            self,
            cx: I,
            lang_item: SolverTraitLangItem,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<P, S, TraitWithId<I>, O>> {
            self.for_trait(cx.require_trait_lang_item(lang_item))
        }

        #[inline]
        pub fn interned(
            self,
        ) -> CallerBoundsAccessor<I, Iter, TraitClauses<P, S, T, OutputAsInternedClause>> {
            let CallerBoundsAccessor {
                clauses: iter,
                clause_filter:
                    TraitClauses {
                        polarity_filter,
                        self_type_filter,
                        trait_filter,
                        output_transformer: _,
                    },
                cx,
            } = self;
            CallerBoundsAccessor {
                clauses: iter,
                clause_filter: TraitClauses {
                    polarity_filter,
                    self_type_filter,
                    trait_filter,
                    output_transformer: OutputAsInternedClause,
                },
                cx,
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

mod output_transformer {
    pub trait OutputTransformer {}

    pub struct OutputAsInternedClause;
    impl OutputTransformer for OutputAsInternedClause {}

    pub struct OutputAsTraitClause;
    impl OutputTransformer for OutputAsTraitClause {}
}
