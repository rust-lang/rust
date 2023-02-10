//! This module contains implementations of the `TypeFoldable` and `TypeVisitable`
//! traits for various types in the Rust compiler. Most are written by hand, though
//! we've recently added some macros and proc-macros to help with the tedium.

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::Interner;
use rustc_data_structures::functor::IdFunctor;
use rustc_index::vec::{Idx, IndexVec};

use std::mem::ManuallyDrop;
use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeTraversalImpls! {
    (),
    bool,
    usize,
    u16,
    u32,
    u64,
    String,
    crate::DebruijnIndex,
}

///////////////////////////////////////////////////////////////////////////
// Traversal implementations.

impl<I: Interner, T: TypeFoldable<I>, U: TypeFoldable<I>> TypeFoldable<I> for (T, U) {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<(T, U), F::Error> {
        Ok((self.0.try_fold_with(folder)?, self.1.try_fold_with(folder)?))
    }
}

impl<I: Interner, T: TypeVisitable<I>, U: TypeVisitable<I>> TypeVisitable<I> for (T, U) {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)
    }
}

impl<I: Interner, A: TypeFoldable<I>, B: TypeFoldable<I>, C: TypeFoldable<I>> TypeFoldable<I>
    for (A, B, C)
{
    fn try_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<(A, B, C), F::Error> {
        Ok((
            self.0.try_fold_with(folder)?,
            self.1.try_fold_with(folder)?,
            self.2.try_fold_with(folder)?,
        ))
    }
}

impl<I: Interner, A: TypeVisitable<I>, B: TypeVisitable<I>, C: TypeVisitable<I>> TypeVisitable<I>
    for (A, B, C)
{
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)?;
        self.2.visit_with(visitor)
    }
}

EnumTypeTraversalImpl! {
    impl<I, T> TypeFoldable<I> for Option<T> {
        (Some)(a),
        (None),
    } where I: Interner, T: TypeFoldable<I>
}
EnumTypeTraversalImpl! {
    impl<I, T> TypeVisitable<I> for Option<T> {
        (Some)(a),
        (None),
    } where I: Interner, T: TypeVisitable<I>
}

EnumTypeTraversalImpl! {
    impl<I, T, E> TypeFoldable<I> for Result<T, E> {
        (Ok)(a),
        (Err)(a),
    } where I: Interner, T: TypeFoldable<I>, E: TypeFoldable<I>,
}
EnumTypeTraversalImpl! {
    impl<I, T, E> TypeVisitable<I> for Result<T, E> {
        (Ok)(a),
        (Err)(a),
    } where I: Interner, T: TypeVisitable<I>, E: TypeVisitable<I>,
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Rc<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(mut self, folder: &mut F) -> Result<Self, F::Error> {
        // We merely want to replace the contained `T`, if at all possible,
        // so that we don't needlessly allocate a new `Rc` or indeed clone
        // the contained type.
        unsafe {
            // First step is to ensure that we have a unique reference to
            // the contained type, which `Rc::make_mut` will accomplish (by
            // allocating a new `Rc` and cloning the `T` only if required).
            // This is done *before* casting to `Rc<ManuallyDrop<T>>` so that
            // panicking during `make_mut` does not leak the `T`.
            Rc::make_mut(&mut self);

            // Casting to `Rc<ManuallyDrop<T>>` is safe because `ManuallyDrop`
            // is `repr(transparent)`.
            let ptr = Rc::into_raw(self).cast::<ManuallyDrop<T>>();
            let mut unique = Rc::from_raw(ptr);

            // Call to `Rc::make_mut` above guarantees that `unique` is the
            // sole reference to the contained value, so we can avoid doing
            // a checked `get_mut` here.
            let slot = Rc::get_mut_unchecked(&mut unique);

            // Semantically move the contained type out from `unique`, fold
            // it, then move the folded value back into `unique`. Should
            // folding fail, `ManuallyDrop` ensures that the "moved-out"
            // value is not re-dropped.
            let owned = ManuallyDrop::take(slot);
            let folded = owned.try_fold_with(folder)?;
            *slot = ManuallyDrop::new(folded);

            // Cast back to `Rc<T>`.
            Ok(Rc::from_raw(Rc::into_raw(unique).cast()))
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Rc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Arc<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(mut self, folder: &mut F) -> Result<Self, F::Error> {
        // We merely want to replace the contained `T`, if at all possible,
        // so that we don't needlessly allocate a new `Arc` or indeed clone
        // the contained type.
        unsafe {
            // First step is to ensure that we have a unique reference to
            // the contained type, which `Arc::make_mut` will accomplish (by
            // allocating a new `Arc` and cloning the `T` only if required).
            // This is done *before* casting to `Arc<ManuallyDrop<T>>` so that
            // panicking during `make_mut` does not leak the `T`.
            Arc::make_mut(&mut self);

            // Casting to `Arc<ManuallyDrop<T>>` is safe because `ManuallyDrop`
            // is `repr(transparent)`.
            let ptr = Arc::into_raw(self).cast::<ManuallyDrop<T>>();
            let mut unique = Arc::from_raw(ptr);

            // Call to `Arc::make_mut` above guarantees that `unique` is the
            // sole reference to the contained value, so we can avoid doing
            // a checked `get_mut` here.
            let slot = Arc::get_mut_unchecked(&mut unique);

            // Semantically move the contained type out from `unique`, fold
            // it, then move the folded value back into `unique`. Should
            // folding fail, `ManuallyDrop` ensures that the "moved-out"
            // value is not re-dropped.
            let owned = ManuallyDrop::take(slot);
            let folded = owned.try_fold_with(folder)?;
            *slot = ManuallyDrop::new(folded);

            // Cast back to `Arc<T>`.
            Ok(Arc::from_raw(Arc::into_raw(unique).cast()))
        }
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Arc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Box<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|value| value.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Vec<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|t| t.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Vec<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for &[T] {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Box<[T]> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|t| t.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<[T]> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<I: Interner, T: TypeFoldable<I>, Ix: Idx> TypeFoldable<I> for IndexVec<Ix, T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|x| x.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>, Ix: Idx> TypeVisitable<I> for IndexVec<Ix, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}
