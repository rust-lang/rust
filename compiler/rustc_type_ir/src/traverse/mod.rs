//! A visiting traversal mechanism for complex data structures that contain type
//! information. See the documentation of the [visit] and [fold] modules for more
//! details.

#[macro_use]
pub mod visit;
pub mod fold;

use std::fmt;

use fold::{FallibleTypeFolder, TypeFoldable};
use rustc_ast_ir::visit::VisitorResult;
use rustc_type_ir_macros::{TypeFoldable_Generic, TypeVisitable_Generic};
use visit::{TypeVisitable, TypeVisitor};

use crate::Interner;

#[derive(Debug, Clone, TypeVisitable_Generic, TypeFoldable_Generic)]
pub struct AlwaysTraversable<T>(pub T);

/// A trait which allows the compiler to reason about the disjointness
/// of `TypeVisitable` and `NoopTypeTraversable`.
///
/// This trait has a blanket impls for everything that implements `TypeVisitable`
/// while requiring a manual impl for all types whose traversal is a noop.
pub trait TypeTraversable<I: Interner>: fmt::Debug + Clone {
    type Kind;

    #[inline(always)]
    fn noop_visit_with<V: TypeVisitor<I>>(&self, _: &mut V) -> V::Result
    where
        Self: TypeTraversable<I, Kind = NoopTypeTraversal>,
    {
        V::Result::output()
    }

    #[inline(always)]
    fn noop_try_fold_with<F: FallibleTypeFolder<I>>(self, _: &mut F) -> Result<Self, F::Error>
    where
        Self: TypeTraversable<I, Kind = NoopTypeTraversal>,
    {
        Ok(self)
    }
}
pub struct ImportantTypeTraversal;
pub struct NoopTypeTraversal;

pub trait OptVisitWith<I: Interner>: TypeTraversable<I> {
    fn mk_visit_with<V: TypeVisitor<I>>() -> fn(&Self, &mut V) -> V::Result;
}

impl<I, T> OptVisitWith<I> for T
where
    I: Interner,
    T: TypeTraversable<I> + Clone + OptVisitWithHelper<I, T::Kind>,
{
    #[inline(always)]
    fn mk_visit_with<V: TypeVisitor<I>>() -> fn(&Self, &mut V) -> V::Result {
        Self::mk_visit_with_helper()
    }
}

trait OptVisitWithHelper<I: Interner, KIND> {
    fn mk_visit_with_helper<V: TypeVisitor<I>>() -> fn(&Self, &mut V) -> V::Result;
}

impl<I, T> OptVisitWithHelper<I, ImportantTypeTraversal> for T
where
    I: Interner,
    T: TypeVisitable<I>,
{
    #[inline(always)]
    fn mk_visit_with_helper<V: TypeVisitor<I>>() -> fn(&Self, &mut V) -> V::Result {
        Self::visit_with
    }
}

/// While this is implemented for all `T`, it is only useable via `OptVisitWith` if
/// `T` implements `TypeTraversable<I, Kind = NoopTypeTraversal>`.
impl<I, T> OptVisitWithHelper<I, NoopTypeTraversal> for T
where
    I: Interner,
{
    #[inline(always)]
    fn mk_visit_with_helper<V: TypeVisitor<I>>() -> fn(&Self, &mut V) -> V::Result {
        |_, _| V::Result::output()
    }
}

pub trait OptTryFoldWith<I: Interner>: OptVisitWith<I> + Sized {
    fn mk_try_fold_with<F: FallibleTypeFolder<I>>() -> fn(Self, &mut F) -> Result<Self, F::Error>;
}

impl<I, T> OptTryFoldWith<I> for T
where
    I: Interner,
    T: OptVisitWith<I> + OptTryFoldWithHelper<I, T::Kind>,
{
    #[inline(always)]
    fn mk_try_fold_with<F: FallibleTypeFolder<I>>() -> fn(Self, &mut F) -> Result<Self, F::Error> {
        Self::mk_try_fold_with_helper()
    }
}

pub trait OptTryFoldWithHelper<I: Interner, KIND>: Sized {
    fn mk_try_fold_with_helper<F: FallibleTypeFolder<I>>()
    -> fn(Self, &mut F) -> Result<Self, F::Error>;
}

impl<I, T> OptTryFoldWithHelper<I, ImportantTypeTraversal> for T
where
    I: Interner,
    T: TypeFoldable<I>,
{
    #[inline(always)]
    fn mk_try_fold_with_helper<F: FallibleTypeFolder<I>>()
    -> fn(Self, &mut F) -> Result<Self, F::Error> {
        Self::try_fold_with
    }
}

/// While this is implemented for all `T`, it is only useable via `OptTryFoldWith` if
/// `T` implements `TypeTraversable<I, Kind = NoopTypeTraversal>`.
impl<I, T> OptTryFoldWithHelper<I, NoopTypeTraversal> for T
where
    I: Interner,
{
    #[inline(always)]
    fn mk_try_fold_with_helper<F: FallibleTypeFolder<I>>()
    -> fn(Self, &mut F) -> Result<Self, F::Error> {
        |this, _| Ok(this)
    }
}
