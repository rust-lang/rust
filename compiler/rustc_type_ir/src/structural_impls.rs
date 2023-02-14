//! This module contains implementations of the `TypeFoldable` and `TypeVisitable`
//! traits for various types in the Rust compiler. Most are written by hand, though
//! we've recently added some macros and proc-macros to help with the tedium.

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::Interner;
use rustc_data_structures::functor::IdFunctor;
use rustc_index::vec::{Idx, IndexVec};

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
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|value| value.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Rc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Arc<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|value| value.try_fold_with(folder))
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
