//! This module contains implementations of the `TypeVisitable`
//! trait for various types in the Rust compiler. Most are written by
//! hand, though we've recently added some macros and proc-macros to help with the tedium.

use crate::{
    visit::{TypeVisitable, TypeVisitor},
    DebruijnIndex, Interner,
};
use rustc_index::vec::{Idx, IndexVec};

use std::ops::ControlFlow;
use std::rc::Rc;
use std::sync::Arc;

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeVisitableImpls! {
    (),
    bool,
    usize,
    u16,
    u32,
    u64,
    String,
    DebruijnIndex,
}

///////////////////////////////////////////////////////////////////////////
// TypeVisitable implementations.

impl<I: Interner, T: TypeVisitable<I>, U: TypeVisitable<I>> TypeVisitable<I> for (T, U) {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.0.visit_with(visitor)?;
        self.1.visit_with(visitor)
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

EnumTypeVisitableImpl! {
    impl<I, T> TypeVisitable<I> for Option<T> {
        (Some)(a),
        (None),
    } where I: Interner, T: TypeVisitable<I>
}

EnumTypeVisitableImpl! {
    impl<I, T, E> TypeVisitable<I> for Result<T, E> {
        (Ok)(a),
        (Err)(a),
    } where I: Interner, T: TypeVisitable<I>, E: TypeVisitable<I>,
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Rc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Arc<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        (**self).visit_with(visitor)
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

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Box<[T]> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}

impl<I: Interner, T: TypeVisitable<I>, Ix: Idx> TypeVisitable<I> for IndexVec<Ix, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.iter().try_for_each(|t| t.visit_with(visitor))
    }
}
