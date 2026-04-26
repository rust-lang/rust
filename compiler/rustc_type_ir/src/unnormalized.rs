use std::marker::PhantomData;

use derive_where::derive_where;

use crate::inherent::*;
use crate::upcast::Upcast;
use crate::{
    Binder, BoundConstness, ClauseKind, HostEffectPredicate, Interner, PredicatePolarity,
    TraitPredicate, TraitRef,
};

/// A wrapper for values that need normalization.
///
/// FIXME(#155345): This is very WIP. The plan is to replace the `skip_norm_wip`
/// spread throughout the codebase with proper normalization. This is the first
/// step toward switching to eager normalization with the next solver. See the
/// normalization refactor plan [here].
///
/// We're in a weird intermediate state as the change is too big to land in a
/// single PR. While this work is in progress, just use `Unnormalized::new_wip`
/// and `Unnormalized::skip_norm_wip` as needed.
///
/// The interner type parameter exists to constraint generic for certain impl,
/// e.g., `Unnormalized<I, I::Clause>`.
///
/// [here]: https://rust-lang.zulipchat.com/#narrow/channel/364551-t-types.2Ftrait-system-refactor/topic/Eager.20normalization.2C.20ahoy.21/with/582996293
#[derive_where(Clone, Copy, PartialOrd, PartialEq, Debug; T)]
pub struct Unnormalized<I: Interner, T> {
    value: T,
    #[derive_where(skip(Debug))]
    _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner, T> Unnormalized<I, T> {
    /// Should only be used in limited situations where you produce an potentially
    /// unnormalized value, like in (Early)Binder/GenericPredicates instantiation.
    pub fn new(value: T) -> Unnormalized<I, T> {
        Unnormalized { value, _tcx: PhantomData }
    }

    /// FIXME: This is going to be eventually removed once we migrate the relevant
    /// APIs to return `Unnormalized`.
    pub fn new_wip(value: T) -> Unnormalized<I, T> {
        Unnormalized { value, _tcx: PhantomData }
    }

    /// Intentionally skip normalization.
    /// You probably should perform normalization in most cases.
    pub fn skip_normalization(self) -> T {
        self.value
    }

    /// FIXME: This is going to be eventually removed.
    /// If you meet this in codebase, try using one of the normalization routines
    /// to consume the `Unnormalized` wrapper. Or use `skip_normalization` when normalization
    /// is really unnecessary.
    pub fn skip_norm_wip(self) -> T {
        self.value
    }

    pub fn map<F, U>(self, f: F) -> Unnormalized<I, U>
    where
        F: FnOnce(T) -> U,
    {
        Unnormalized { value: f(self.value), _tcx: PhantomData }
    }

    pub fn as_ref(&self) -> Unnormalized<I, &T> {
        Unnormalized { value: &self.value, _tcx: PhantomData }
    }

    pub fn map_ref<U, F>(&self, f: F) -> Unnormalized<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        Unnormalized { value: f(&self.value), _tcx: PhantomData }
    }
}

impl<I: Interner, T, U> Unnormalized<I, (T, U)> {
    pub fn unzip(self) -> (Unnormalized<I, T>, Unnormalized<I, U>) {
        (Unnormalized::new(self.value.0), Unnormalized::new(self.value.1))
    }
}

impl<I: Interner, T> Unnormalized<I, Binder<I, T>> {
    pub fn skip_binder(self) -> T {
        self.value.skip_binder()
    }
}

impl<I: Interner> Unnormalized<I, I::Clause> {
    pub fn as_trait_clause(self) -> Option<Unnormalized<I, Binder<I, TraitPredicate<I>>>> {
        self.value.as_trait_clause().map(|v| Unnormalized::new(v))
    }

    pub fn kind(self) -> Unnormalized<I, Binder<I, ClauseKind<I>>> {
        self.map(|v| v.kind())
    }
}

impl<I: Interner> Unnormalized<I, Binder<I, TraitPredicate<I>>> {
    pub fn self_ty(self) -> Unnormalized<I, Binder<I, I::Ty>> {
        self.map(|pred| pred.self_ty())
    }

    pub fn def_id(self) -> I::TraitId {
        self.value.skip_binder().def_id()
    }

    #[inline]
    pub fn polarity(self) -> PredicatePolarity {
        self.value.skip_binder().polarity
    }
}

impl<I: Interner> Unnormalized<I, Binder<I, TraitRef<I>>> {
    pub fn self_ty(&self) -> Unnormalized<I, Binder<I, I::Ty>> {
        self.map_ref(|tr| tr.self_ty())
    }

    pub fn def_id(&self) -> I::TraitId {
        self.value.skip_binder().def_id
    }

    pub fn to_host_effect_clause(
        self,
        cx: I,
        constness: BoundConstness,
    ) -> Unnormalized<I, I::Clause> {
        let inner = self
            .value
            .map_bound(|trait_ref| {
                ClauseKind::HostEffect(HostEffectPredicate { trait_ref, constness })
            })
            .upcast(cx);
        Unnormalized::new(inner)
    }
}
