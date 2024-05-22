use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{ControlFlow, Deref};

#[cfg(feature = "nightly")]
use rustc_macros::HashStable_NoContext;
use rustc_serialize::Decodable;

use crate::fold::{FallibleTypeFolder, TypeFoldable, TypeSuperFoldable};
use crate::inherent::*;
use crate::lift::Lift;
use crate::visit::{Flags, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use crate::{self as ty, Interner, SsoHashSet};

/// Binder is a binder for higher-ranked lifetimes or types. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef ==
/// Binder<I, TraitRef>`). Note that when we instantiate,
/// erase, or otherwise "discharge" these bound vars, we change the
/// type from `Binder<I, T>` to just `T` (see
/// e.g., `liberate_late_bound_regions`).
///
/// `Decodable` and `Encodable` are implemented for `Binder<T>` using the `impl_binder_encode_decode!` macro.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "T: Clone"),
    Copy(bound = "T: Copy"),
    Hash(bound = "T: Hash"),
    PartialEq(bound = "T: PartialEq"),
    Eq(bound = "T: Eq"),
    Debug(bound = "T: Debug")
)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct Binder<I: Interner, T> {
    value: T,
    bound_vars: I::BoundVarKinds,
}

// FIXME: We manually derive `Lift` because the `derive(Lift_Generic)` doesn't
// understand how to turn `T` to `T::Lifted` in the output `type Lifted`.
impl<I: Interner, U: Interner, T> Lift<U> for Binder<I, T>
where
    T: Lift<U>,
    I::BoundVarKinds: Lift<U, Lifted = U::BoundVarKinds>,
{
    type Lifted = Binder<U, T::Lifted>;

    fn lift_to_tcx(self, tcx: U) -> Option<Self::Lifted> {
        Some(Binder {
            value: self.value.lift_to_tcx(tcx)?,
            bound_vars: self.bound_vars.lift_to_tcx(tcx)?,
        })
    }
}

macro_rules! impl_binder_encode_decode {
    ($($t:ty),+ $(,)?) => {
        $(
            impl<I: Interner, E: crate::TyEncoder<I = I>> rustc_serialize::Encodable<E> for ty::Binder<I, $t>
            where
                $t: rustc_serialize::Encodable<E>,
                I::BoundVarKinds: rustc_serialize::Encodable<E>,
            {
                fn encode(&self, e: &mut E) {
                    self.bound_vars().encode(e);
                    self.as_ref().skip_binder().encode(e);
                }
            }
            impl<I: Interner, D: crate::TyDecoder<I = I>> Decodable<D> for ty::Binder<I, $t>
            where
                $t: TypeVisitable<I> + rustc_serialize::Decodable<D>,
                I::BoundVarKinds: rustc_serialize::Decodable<D>,
            {
                fn decode(decoder: &mut D) -> Self {
                    let bound_vars = Decodable::decode(decoder);
                    ty::Binder::bind_with_vars(<$t>::decode(decoder), bound_vars)
                }
            }
        )*
    }
}

impl_binder_encode_decode! {
    ty::FnSig<I>,
    ty::TraitPredicate<I>,
    ty::ExistentialPredicate<I>,
    ty::TraitRef<I>,
    ty::ExistentialTraitRef<I>,
}

impl<I: Interner, T> Binder<I, T>
where
    T: TypeVisitable<I>,
{
    /// Wraps `value` in a binder, asserting that `value` does not
    /// contain any bound vars that would be bound by the
    /// binder. This is commonly used to 'inject' a value T into a
    /// different binding level.
    #[track_caller]
    pub fn dummy(value: T) -> Binder<I, T> {
        assert!(
            !value.has_escaping_bound_vars(),
            "`{value:?}` has escaping bound vars, so it cannot be wrapped in a dummy binder."
        );
        Binder { value, bound_vars: Default::default() }
    }

    pub fn bind_with_vars(value: T, bound_vars: I::BoundVarKinds) -> Binder<I, T> {
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Binder<I, T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_binder(self)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Binder<I, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_binder(self)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeSuperFoldable<I> for Binder<I, T> {
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.try_map_bound(|ty| ty.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeSuperVisitable<I> for Binder<I, T> {
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        self.as_ref().skip_binder().visit_with(visitor)
    }
}

impl<I: Interner, T> Binder<I, T> {
    /// Skips the binder and returns the "bound" value. This is a
    /// risky thing to do because it's easy to get confused about
    /// De Bruijn indices and the like. It is usually better to
    /// discharge the binder using `no_bound_vars` or
    /// `instantiate_bound_regions` or something like
    /// that. `skip_binder` is only valid when you are either
    /// extracting data that has nothing to do with bound vars, you
    /// are doing some sort of test that does not involve bound
    /// regions, or you are being very careful about your depth
    /// accounting.
    ///
    /// Some examples where `skip_binder` is reasonable:
    ///
    /// - extracting the `DefId` from a PolyTraitRef;
    /// - comparing the self type of a PolyTraitRef to see if it is equal to
    ///   a type parameter `X`, since the type `X` does not reference any regions
    pub fn skip_binder(self) -> T {
        self.value
    }

    pub fn bound_vars(&self) -> I::BoundVarKinds {
        self.bound_vars
    }

    pub fn as_ref(&self) -> Binder<I, &T> {
        Binder { value: &self.value, bound_vars: self.bound_vars }
    }

    pub fn as_deref(&self) -> Binder<I, &T::Target>
    where
        T: Deref,
    {
        Binder { value: &self.value, bound_vars: self.bound_vars }
    }

    pub fn map_bound_ref<F, U: TypeVisitable<I>>(&self, f: F) -> Binder<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U: TypeVisitable<I>>(self, f: F) -> Binder<I, U>
    where
        F: FnOnce(T) -> U,
    {
        let Binder { value, bound_vars } = self;
        let value = f(value);
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }

    pub fn try_map_bound<F, U: TypeVisitable<I>, E>(self, f: F) -> Result<Binder<I, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let Binder { value, bound_vars } = self;
        let value = f(value)?;
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            value.visit_with(&mut validator);
        }
        Ok(Binder { value, bound_vars })
    }

    /// Wraps a `value` in a binder, using the same bound variables as the
    /// current `Binder`. This should not be used if the new value *changes*
    /// the bound variables. Note: the (old or new) value itself does not
    /// necessarily need to *name* all the bound variables.
    ///
    /// This currently doesn't do anything different than `bind`, because we
    /// don't actually track bound vars. However, semantically, it is different
    /// because bound vars aren't allowed to change here, whereas they are
    /// in `bind`. This may be (debug) asserted in the future.
    pub fn rebind<U>(&self, value: U) -> Binder<I, U>
    where
        U: TypeVisitable<I>,
    {
        Binder::bind_with_vars(value, self.bound_vars)
    }

    /// Unwraps and returns the value within, but only if it contains
    /// no bound vars at all. (In other words, if this binder --
    /// and indeed any enclosing binder -- doesn't bind anything at
    /// all.) Otherwise, returns `None`.
    ///
    /// (One could imagine having a method that just unwraps a single
    /// binder, but permits late-bound vars bound by enclosing
    /// binders, but that would require adjusting the debruijn
    /// indices, and given the shallow binding structure we often use,
    /// would not be that useful.)
    pub fn no_bound_vars(self) -> Option<T>
    where
        T: TypeVisitable<I>,
    {
        // `self.value` is equivalent to `self.skip_binder()`
        if self.value.has_escaping_bound_vars() { None } else { Some(self.skip_binder()) }
    }

    /// Splits the contents into two things that share the same binder
    /// level as the original, returning two distinct binders.
    ///
    /// `f` should consider bound regions at depth 1 to be free, and
    /// anything it produces with bound regions at depth 1 will be
    /// bound in the resulting return values.
    pub fn split<U, V, F>(self, f: F) -> (Binder<I, U>, Binder<I, V>)
    where
        F: FnOnce(T) -> (U, V),
    {
        let Binder { value, bound_vars } = self;
        let (u, v) = f(value);
        (Binder { value: u, bound_vars }, Binder { value: v, bound_vars })
    }
}

impl<I: Interner, T> Binder<I, Option<T>> {
    pub fn transpose(self) -> Option<Binder<I, T>> {
        let Binder { value, bound_vars } = self;
        value.map(|value| Binder { value, bound_vars })
    }
}

impl<I: Interner, T: IntoIterator> Binder<I, T> {
    pub fn iter(self) -> impl Iterator<Item = Binder<I, T::Item>> {
        let Binder { value, bound_vars } = self;
        value.into_iter().map(move |value| Binder { value, bound_vars })
    }
}

pub struct ValidateBoundVars<I: Interner> {
    bound_vars: I::BoundVarKinds,
    binder_index: ty::DebruijnIndex,
    // We may encounter the same variable at different levels of binding, so
    // this can't just be `Ty`
    visited: SsoHashSet<(ty::DebruijnIndex, I::Ty)>,
}

impl<I: Interner> ValidateBoundVars<I> {
    pub fn new(bound_vars: I::BoundVarKinds) -> Self {
        ValidateBoundVars {
            bound_vars,
            binder_index: ty::INNERMOST,
            visited: SsoHashSet::default(),
        }
    }
}

impl<I: Interner> TypeVisitor<I> for ValidateBoundVars<I> {
    type Result = ControlFlow<()>;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &Binder<I, T>) -> Self::Result {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: I::Ty) -> Self::Result {
        if t.outer_exclusive_binder() < self.binder_index
            || !self.visited.insert((self.binder_index, t))
        {
            return ControlFlow::Break(());
        }
        match t.kind() {
            ty::Bound(debruijn, bound_ty) if debruijn == self.binder_index => {
                let idx = bound_ty.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", t, self.bound_vars);
                }
                bound_ty.assert_eq(self.bound_vars[idx]);
            }
            _ => {}
        };

        t.super_visit_with(self)
    }

    fn visit_region(&mut self, r: I::Region) -> Self::Result {
        match r.kind() {
            ty::ReBound(index, br) if index == self.binder_index => {
                let idx = br.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", r, self.bound_vars);
                }
                br.assert_eq(self.bound_vars[idx]);
            }

            _ => (),
        };

        ControlFlow::Continue(())
    }
}
