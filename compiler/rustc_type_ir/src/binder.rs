use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use tracing::instrument;

use crate::data_structures::SsoHashSet;
use crate::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
use crate::inherent::*;
use crate::lift::Lift;
use crate::visit::{Flags, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use crate::{self as ty, Interner};

/// Binder is a binder for higher-ranked lifetimes or types. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef ==
/// Binder<I, TraitRef>`). Note that when we instantiate,
/// erase, or otherwise "discharge" these bound vars, we change the
/// type from `Binder<I, T>` to just `T` (see
/// e.g., `liberate_late_bound_regions`).
///
/// `Decodable` and `Encodable` are implemented for `Binder<T>` using the `impl_binder_encode_decode!` macro.
#[derive_where(Clone; I: Interner, T: Clone)]
#[derive_where(Copy; I: Interner, T: Copy)]
#[derive_where(Hash; I: Interner, T: Hash)]
#[derive_where(PartialEq; I: Interner, T: PartialEq)]
#[derive_where(Eq; I: Interner, T: Eq)]
#[derive_where(Debug; I: Interner, T: Debug)]
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

    fn lift_to_interner(self, cx: U) -> Option<Self::Lifted> {
        Some(Binder {
            value: self.value.lift_to_interner(cx)?,
            bound_vars: self.bound_vars.lift_to_interner(cx)?,
        })
    }
}

#[cfg(feature = "nightly")]
macro_rules! impl_binder_encode_decode {
    ($($t:ty),+ $(,)?) => {
        $(
            impl<I: Interner, E: rustc_serialize::Encoder> rustc_serialize::Encodable<E> for ty::Binder<I, $t>
            where
                $t: rustc_serialize::Encodable<E>,
                I::BoundVarKinds: rustc_serialize::Encodable<E>,
            {
                fn encode(&self, e: &mut E) {
                    self.bound_vars().encode(e);
                    self.as_ref().skip_binder().encode(e);
                }
            }
            impl<I: Interner, D: rustc_serialize::Decoder> rustc_serialize::Decodable<D> for ty::Binder<I, $t>
            where
                $t: TypeVisitable<I> + rustc_serialize::Decodable<D>,
                I::BoundVarKinds: rustc_serialize::Decodable<D>,
            {
                fn decode(decoder: &mut D) -> Self {
                    let bound_vars = rustc_serialize::Decodable::decode(decoder);
                    ty::Binder::bind_with_vars(rustc_serialize::Decodable::decode(decoder), bound_vars)
                }
            }
        )*
    }
}

#[cfg(feature = "nightly")]
impl_binder_encode_decode! {
    ty::FnSig<I>,
    ty::FnSigTys<I>,
    ty::TraitPredicate<I>,
    ty::ExistentialPredicate<I>,
    ty::TraitRef<I>,
    ty::ExistentialTraitRef<I>,
    ty::HostEffectPredicate<I>,
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
            let _ = value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Binder<I, T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_binder(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_binder(self)
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
        self.try_map_bound(|t| t.try_fold_with(folder))
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        self.map_bound(|t| t.fold_with(folder))
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
            let _ = value.visit_with(&mut validator);
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
            let _ = value.visit_with(&mut validator);
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
                bound_ty.assert_eq(self.bound_vars.get(idx).unwrap());
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
                br.assert_eq(self.bound_vars.get(idx).unwrap());
            }

            _ => (),
        };

        ControlFlow::Continue(())
    }
}

/// Similar to [`super::Binder`] except that it tracks early bound generics, i.e. `struct Foo<T>(T)`
/// needs `T` instantiated immediately. This type primarily exists to avoid forgetting to call
/// `instantiate`.
///
/// If you don't have anything to `instantiate`, you may be looking for
/// [`instantiate_identity`](EarlyBinder::instantiate_identity) or [`skip_binder`](EarlyBinder::skip_binder).
#[derive_where(Clone; I: Interner, T: Clone)]
#[derive_where(Copy; I: Interner, T: Copy)]
#[derive_where(PartialEq; I: Interner, T: PartialEq)]
#[derive_where(Eq; I: Interner, T: Eq)]
#[derive_where(Ord; I: Interner, T: Ord)]
#[derive_where(PartialOrd; I: Interner, T: Ord)]
#[derive_where(Hash; I: Interner, T: Hash)]
#[derive_where(Debug; I: Interner, T: Debug)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct EarlyBinder<I: Interner, T> {
    value: T,
    #[derive_where(skip(Debug))]
    _tcx: PhantomData<I>,
}

/// For early binders, you should first call `instantiate` before using any visitors.
#[cfg(feature = "nightly")]
impl<I: Interner, T> !TypeFoldable<I> for ty::EarlyBinder<I, T> {}

/// For early binders, you should first call `instantiate` before using any visitors.
#[cfg(feature = "nightly")]
impl<I: Interner, T> !TypeVisitable<I> for ty::EarlyBinder<I, T> {}

impl<I: Interner, T> EarlyBinder<I, T> {
    pub fn bind(value: T) -> EarlyBinder<I, T> {
        EarlyBinder { value, _tcx: PhantomData }
    }

    pub fn as_ref(&self) -> EarlyBinder<I, &T> {
        EarlyBinder { value: &self.value, _tcx: PhantomData }
    }

    pub fn map_bound_ref<F, U>(&self, f: F) -> EarlyBinder<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U>(self, f: F) -> EarlyBinder<I, U>
    where
        F: FnOnce(T) -> U,
    {
        let value = f(self.value);
        EarlyBinder { value, _tcx: PhantomData }
    }

    pub fn try_map_bound<F, U, E>(self, f: F) -> Result<EarlyBinder<I, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let value = f(self.value)?;
        Ok(EarlyBinder { value, _tcx: PhantomData })
    }

    pub fn rebind<U>(&self, value: U) -> EarlyBinder<I, U> {
        EarlyBinder { value, _tcx: PhantomData }
    }

    /// Skips the binder and returns the "bound" value.
    /// This can be used to extract data that does not depend on generic parameters
    /// (e.g., getting the `DefId` of the inner value or getting the number of
    /// arguments of an `FnSig`). Otherwise, consider using
    /// [`instantiate_identity`](EarlyBinder::instantiate_identity).
    ///
    /// To skip the binder on `x: &EarlyBinder<I, T>` to obtain `&T`, leverage
    /// [`EarlyBinder::as_ref`](EarlyBinder::as_ref): `x.as_ref().skip_binder()`.
    ///
    /// See also [`Binder::skip_binder`](super::Binder::skip_binder), which is
    /// the analogous operation on [`super::Binder`].
    pub fn skip_binder(self) -> T {
        self.value
    }
}

impl<I: Interner, T> EarlyBinder<I, Option<T>> {
    pub fn transpose(self) -> Option<EarlyBinder<I, T>> {
        self.value.map(|value| EarlyBinder { value, _tcx: PhantomData })
    }
}

impl<I: Interner, Iter: IntoIterator> EarlyBinder<I, Iter>
where
    Iter::Item: TypeFoldable<I>,
{
    pub fn iter_instantiated<A>(self, cx: I, args: A) -> IterInstantiated<I, Iter, A>
    where
        A: SliceLike<Item = I::GenericArg>,
    {
        IterInstantiated { it: self.value.into_iter(), cx, args }
    }

    /// Similar to [`instantiate_identity`](EarlyBinder::instantiate_identity),
    /// but on an iterator of `TypeFoldable` values.
    pub fn iter_identity(self) -> Iter::IntoIter {
        self.value.into_iter()
    }
}

pub struct IterInstantiated<I: Interner, Iter: IntoIterator, A> {
    it: Iter::IntoIter,
    cx: I,
    args: A,
}

impl<I: Interner, Iter: IntoIterator, A> Iterator for IterInstantiated<I, Iter, A>
where
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
    type Item = Iter::Item;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            EarlyBinder { value: self.it.next()?, _tcx: PhantomData }
                .instantiate(self.cx, self.args),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator, A> DoubleEndedIterator for IterInstantiated<I, Iter, A>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(
            EarlyBinder { value: self.it.next_back()?, _tcx: PhantomData }
                .instantiate(self.cx, self.args),
        )
    }
}

impl<I: Interner, Iter: IntoIterator, A> ExactSizeIterator for IterInstantiated<I, Iter, A>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
}

impl<'s, I: Interner, Iter: IntoIterator> EarlyBinder<I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    pub fn iter_instantiated_copied(
        self,
        cx: I,
        args: &'s [I::GenericArg],
    ) -> IterInstantiatedCopied<'s, I, Iter> {
        IterInstantiatedCopied { it: self.value.into_iter(), cx, args }
    }

    /// Similar to [`instantiate_identity`](EarlyBinder::instantiate_identity),
    /// but on an iterator of values that deref to a `TypeFoldable`.
    pub fn iter_identity_copied(self) -> IterIdentityCopied<Iter> {
        IterIdentityCopied { it: self.value.into_iter() }
    }
}

pub struct IterInstantiatedCopied<'a, I: Interner, Iter: IntoIterator> {
    it: Iter::IntoIter,
    cx: I,
    args: &'a [I::GenericArg],
}

impl<I: Interner, Iter: IntoIterator> Iterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    type Item = <Iter::Item as Deref>::Target;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|value| {
            EarlyBinder { value: *value, _tcx: PhantomData }.instantiate(self.cx, self.args)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator> DoubleEndedIterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|value| {
            EarlyBinder { value: *value, _tcx: PhantomData }.instantiate(self.cx, self.args)
        })
    }
}

impl<I: Interner, Iter: IntoIterator> ExactSizeIterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
}

pub struct IterIdentityCopied<Iter: IntoIterator> {
    it: Iter::IntoIter,
}

impl<Iter: IntoIterator> Iterator for IterIdentityCopied<Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
    type Item = <Iter::Item as Deref>::Target;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|i| *i)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<Iter: IntoIterator> DoubleEndedIterator for IterIdentityCopied<Iter>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|i| *i)
    }
}

impl<Iter: IntoIterator> ExactSizeIterator for IterIdentityCopied<Iter>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
}
pub struct EarlyBinderIter<I, T> {
    t: T,
    _tcx: PhantomData<I>,
}

impl<I: Interner, T: IntoIterator> EarlyBinder<I, T> {
    pub fn transpose_iter(self) -> EarlyBinderIter<I, T::IntoIter> {
        EarlyBinderIter { t: self.value.into_iter(), _tcx: PhantomData }
    }
}

impl<I: Interner, T: Iterator> Iterator for EarlyBinderIter<I, T> {
    type Item = EarlyBinder<I, T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.t.next().map(|value| EarlyBinder { value, _tcx: PhantomData })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.t.size_hint()
    }
}

impl<I: Interner, T: TypeFoldable<I>> ty::EarlyBinder<I, T> {
    pub fn instantiate<A>(self, cx: I, args: A) -> T
    where
        A: SliceLike<Item = I::GenericArg>,
    {
        let mut folder = ArgFolder { cx, args: args.as_slice(), binders_passed: 0 };
        self.value.fold_with(&mut folder)
    }

    /// Makes the identity replacement `T0 => T0, ..., TN => TN`.
    /// Conceptually, this converts universally bound variables into placeholders
    /// when inside of a given item.
    ///
    /// For example, consider `for<T> fn foo<T>(){ .. }`:
    /// - Outside of `foo`, `T` is bound (represented by the presence of `EarlyBinder`).
    /// - Inside of the body of `foo`, we treat `T` as a placeholder by calling
    /// `instantiate_identity` to discharge the `EarlyBinder`.
    pub fn instantiate_identity(self) -> T {
        self.value
    }

    /// Returns the inner value, but only if it contains no bound vars.
    pub fn no_bound_vars(self) -> Option<T> {
        if !self.value.has_param() { Some(self.value) } else { None }
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual instantiation engine itself is a type folder.

struct ArgFolder<'a, I: Interner> {
    cx: I,
    args: &'a [I::GenericArg],

    /// Number of region binders we have passed through while doing the instantiation
    binders_passed: u32,
}

impl<'a, I: Interner> TypeFolder<I> for ArgFolder<'a, I> {
    #[inline]
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_binder<T: TypeFoldable<I>>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T> {
        self.binders_passed += 1;
        let t = t.super_fold_with(self);
        self.binders_passed -= 1;
        t
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region instantiation of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match r.kind() {
            ty::ReEarlyParam(data) => {
                let rk = self.args.get(data.index() as usize).map(|arg| arg.kind());
                match rk {
                    Some(ty::GenericArgKind::Lifetime(lt)) => self.shift_region_through_binders(lt),
                    Some(other) => self.region_param_expected(data, r, other),
                    None => self.region_param_out_of_range(data, r),
                }
            }
            ty::ReBound(..)
            | ty::ReLateParam(_)
            | ty::ReStatic
            | ty::RePlaceholder(_)
            | ty::ReErased
            | ty::ReError(_) => r,
            ty::ReVar(_) => panic!("unexpected region: {r:?}"),
        }
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        if !t.has_param() {
            return t;
        }

        match t.kind() {
            ty::Param(p) => self.ty_for_param(p, t),
            _ => t.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        if let ty::ConstKind::Param(p) = c.kind() {
            self.const_for_param(p, c)
        } else {
            c.super_fold_with(self)
        }
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if p.has_param() { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: I::Clauses) -> I::Clauses {
        if c.has_param() { c.super_fold_with(self) } else { c }
    }
}

impl<'a, I: Interner> ArgFolder<'a, I> {
    fn ty_for_param(&self, p: I::ParamTy, source_ty: I::Ty) -> I::Ty {
        // Look up the type in the args. It really should be in there.
        let opt_ty = self.args.get(p.index() as usize).map(|arg| arg.kind());
        let ty = match opt_ty {
            Some(ty::GenericArgKind::Type(ty)) => ty,
            Some(kind) => self.type_param_expected(p, source_ty, kind),
            None => self.type_param_out_of_range(p, source_ty),
        };

        self.shift_vars_through_binders(ty)
    }

    #[cold]
    #[inline(never)]
    fn type_param_expected(&self, p: I::ParamTy, ty: I::Ty, kind: ty::GenericArgKind<I>) -> ! {
        panic!(
            "expected type for `{:?}` ({:?}/{}) but found {:?} when instantiating, args={:?}",
            p,
            ty,
            p.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn type_param_out_of_range(&self, p: I::ParamTy, ty: I::Ty) -> ! {
        panic!(
            "type parameter `{:?}` ({:?}/{}) out of range when instantiating, args={:?}",
            p,
            ty,
            p.index(),
            self.args,
        )
    }

    fn const_for_param(&self, p: I::ParamConst, source_ct: I::Const) -> I::Const {
        // Look up the const in the args. It really should be in there.
        let opt_ct = self.args.get(p.index() as usize).map(|arg| arg.kind());
        let ct = match opt_ct {
            Some(ty::GenericArgKind::Const(ct)) => ct,
            Some(kind) => self.const_param_expected(p, source_ct, kind),
            None => self.const_param_out_of_range(p, source_ct),
        };

        self.shift_vars_through_binders(ct)
    }

    #[cold]
    #[inline(never)]
    fn const_param_expected(
        &self,
        p: I::ParamConst,
        ct: I::Const,
        kind: ty::GenericArgKind<I>,
    ) -> ! {
        panic!(
            "expected const for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}",
            p,
            ct,
            p.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn const_param_out_of_range(&self, p: I::ParamConst, ct: I::Const) -> ! {
        panic!(
            "const parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}",
            p,
            ct,
            p.index(),
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn region_param_expected(
        &self,
        ebr: I::EarlyParamRegion,
        r: I::Region,
        kind: ty::GenericArgKind<I>,
    ) -> ! {
        panic!(
            "expected region for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}",
            ebr,
            r,
            ebr.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn region_param_out_of_range(&self, ebr: I::EarlyParamRegion, r: I::Region) -> ! {
        panic!(
            "region parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}",
            ebr,
            r,
            ebr.index(),
            self.args,
        )
    }

    /// It is sometimes necessary to adjust the De Bruijn indices during instantiation. This occurs
    /// when we are instantiating a type with escaping bound vars into a context where we have
    /// passed through binders. That's quite a mouthful. Let's see an example:
    ///
    /// ```
    /// type Func<A> = fn(A);
    /// type MetaFunc = for<'a> fn(Func<&'a i32>);
    /// ```
    ///
    /// The type `MetaFunc`, when fully expanded, will be
    /// ```ignore (illustrative)
    /// for<'a> fn(fn(&'a i32))
    /// //      ^~ ^~ ^~~
    /// //      |  |  |
    /// //      |  |  DebruijnIndex of 2
    /// //      Binders
    /// ```
    /// Here the `'a` lifetime is bound in the outer function, but appears as an argument of the
    /// inner one. Therefore, that appearance will have a DebruijnIndex of 2, because we must skip
    /// over the inner binder (remember that we count De Bruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a i32` will have a
    /// De Bruijn index of 1. It's only during the instantiation that we can see we must increase the
    /// depth by 1 to account for the binder that we passed through.
    ///
    /// As a second example, consider this twist:
    ///
    /// ```
    /// type FuncTuple<A> = (A,fn(A));
    /// type MetaFuncTuple = for<'a> fn(FuncTuple<&'a i32>);
    /// ```
    ///
    /// Here the final type will be:
    /// ```ignore (illustrative)
    /// for<'a> fn((&'a i32, fn(&'a i32)))
    /// //          ^~~         ^~~
    /// //          |           |
    /// //   DebruijnIndex of 1 |
    /// //               DebruijnIndex of 2
    /// ```
    /// As indicated in the diagram, here the same type `&'a i32` is instantiated once, but in the
    /// first case we do not increase the De Bruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    #[instrument(level = "trace", skip(self), fields(binders_passed = self.binders_passed), ret)]
    fn shift_vars_through_binders<T: TypeFoldable<I>>(&self, val: T) -> T {
        if self.binders_passed == 0 || !val.has_escaping_bound_vars() {
            val
        } else {
            ty::shift_vars(self.cx, val, self.binders_passed)
        }
    }

    fn shift_region_through_binders(&self, region: I::Region) -> I::Region {
        if self.binders_passed == 0 || !region.has_escaping_bound_vars() {
            region
        } else {
            ty::shift_region(self.cx, region, self.binders_passed)
        }
    }
}
