//! This module contains implementations of the `TypeFoldable` and `TypeVisitable`
//! traits for various types in the Rust compiler. Most are written by hand, though
//! we've recently added some macros and proc-macros to help with the tedium.

use crate::fold::{FallibleTypeFolder, TypeFoldable};
use crate::visit::{TypeVisitable, TypeVisitor};
use crate::{ConstKind, FloatTy, InferTy, IntTy, Interner, UintTy, UniverseIndex};
use rustc_data_structures::functor::IdFunctor;
use rustc_data_structures::sync::Lrc;
use rustc_index::{Idx, IndexVec};

use core::fmt;
use std::marker::PhantomData;
use std::ops::ControlFlow;

///////////////////////////////////////////////////////////////////////////
// Atomic structs
//
// For things that don't carry any arena-allocated data (and are
// copy...), just add them to this list.

TrivialTypeTraversalImpls! {
    (),
    bool,
    usize,
    u8,
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

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Option<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            Some(v) => Some(v.try_fold_with(folder)?),
            None => None,
        })
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Option<T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self {
            Some(v) => v.visit_with(visitor),
            None => ControlFlow::Continue(()),
        }
    }
}

impl<I: Interner, T: TypeFoldable<I>, E: TypeFoldable<I>> TypeFoldable<I> for Result<T, E> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        Ok(match self {
            Ok(v) => Ok(v.try_fold_with(folder)?),
            Err(e) => Err(e.try_fold_with(folder)?),
        })
    }
}

impl<I: Interner, T: TypeVisitable<I>, E: TypeVisitable<I>> TypeVisitable<I> for Result<T, E> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        match self {
            Ok(v) => v.visit_with(visitor),
            Err(e) => e.visit_with(visitor),
        }
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Lrc<T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        self.try_map_id(|value| value.try_fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Lrc<T> {
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

// `TypeFoldable` isn't impl'd for `&[T]`. It doesn't make sense in the general
// case, because we can't return a new slice. But note that there are a couple
// of trivial impls of `TypeFoldable` for specific slice types elsewhere.

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for &[T] {
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

///////////////////////////////////////////////////
//  Debug impls

pub trait InferCtxtLike<I: Interner> {
    fn universe_of_ty(&self, ty: I::InferTy) -> Option<UniverseIndex>;
    fn universe_of_lt(&self, lt: I::RegionVid) -> Option<UniverseIndex>;
    fn universe_of_ct(&self, ct: I::InferConst) -> Option<UniverseIndex>;
}

impl<I: Interner> InferCtxtLike<I> for core::convert::Infallible {
    fn universe_of_ty(&self, _ty: <I as Interner>::InferTy) -> Option<UniverseIndex> {
        match *self {}
    }
    fn universe_of_ct(&self, _ct: <I as Interner>::InferConst) -> Option<UniverseIndex> {
        match *self {}
    }
    fn universe_of_lt(&self, _lt: <I as Interner>::RegionVid) -> Option<UniverseIndex> {
        match *self {}
    }
}

pub trait DebugWithInfcx<I: Interner>: fmt::Debug {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
}

impl<I: Interner, T: DebugWithInfcx<I> + ?Sized> DebugWithInfcx<I> for &'_ T {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        <T as DebugWithInfcx<I>>::fmt(this.map(|&data| data), f)
    }
}
impl<I: Interner, T: DebugWithInfcx<I>> DebugWithInfcx<I> for [T] {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match f.alternate() {
            true => {
                write!(f, "[\n")?;
                for element in this.data.iter() {
                    write!(f, "{:?},\n", &this.wrap(element))?;
                }
                write!(f, "]")
            }
            false => {
                write!(f, "[")?;
                if this.data.len() > 0 {
                    for element in &this.data[..(this.data.len() - 1)] {
                        write!(f, "{:?}, ", &this.wrap(element))?;
                    }
                    if let Some(element) = this.data.last() {
                        write!(f, "{:?}", &this.wrap(element))?;
                    }
                }
                write!(f, "]")
            }
        }
    }
}

pub struct OptWithInfcx<'a, I: Interner, InfCtx: InferCtxtLike<I>, T> {
    pub data: T,
    pub infcx: Option<&'a InfCtx>,
    _interner: PhantomData<I>,
}

impl<I: Interner, InfCtx: InferCtxtLike<I>, T: Copy> Copy for OptWithInfcx<'_, I, InfCtx, T> {}
impl<I: Interner, InfCtx: InferCtxtLike<I>, T: Clone> Clone for OptWithInfcx<'_, I, InfCtx, T> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), infcx: self.infcx, _interner: self._interner }
    }
}

impl<'a, I: Interner, T> OptWithInfcx<'a, I, core::convert::Infallible, T> {
    pub fn new_no_ctx(data: T) -> Self {
        Self { data, infcx: None, _interner: PhantomData }
    }
}

impl<'a, I: Interner, InfCtx: InferCtxtLike<I>, T> OptWithInfcx<'a, I, InfCtx, T> {
    pub fn new(data: T, infcx: &'a InfCtx) -> Self {
        Self { data, infcx: Some(infcx), _interner: PhantomData }
    }

    pub fn wrap<U>(self, u: U) -> OptWithInfcx<'a, I, InfCtx, U> {
        OptWithInfcx { data: u, infcx: self.infcx, _interner: PhantomData }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> OptWithInfcx<'a, I, InfCtx, U> {
        OptWithInfcx { data: f(self.data), infcx: self.infcx, _interner: PhantomData }
    }

    pub fn as_ref(&self) -> OptWithInfcx<'a, I, InfCtx, &T> {
        OptWithInfcx { data: &self.data, infcx: self.infcx, _interner: PhantomData }
    }
}

impl<I: Interner, InfCtx: InferCtxtLike<I>, T: DebugWithInfcx<I>> fmt::Debug
    for OptWithInfcx<'_, I, InfCtx, T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DebugWithInfcx::fmt(self.as_ref(), f)
    }
}

impl fmt::Debug for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl fmt::Debug for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl fmt::Debug for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl fmt::Debug for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InferTy::*;
        match *self {
            TyVar(ref v) => v.fmt(f),
            IntVar(ref v) => v.fmt(f),
            FloatVar(ref v) => v.fmt(f),
            FreshTy(v) => write!(f, "FreshTy({v:?})"),
            FreshIntTy(v) => write!(f, "FreshIntTy({v:?})"),
            FreshFloatTy(v) => write!(f, "FreshFloatTy({v:?})"),
        }
    }
}
impl<I: Interner<InferTy = InferTy>> DebugWithInfcx<I> for InferTy {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        use InferTy::*;
        match this.infcx.and_then(|infcx| infcx.universe_of_ty(*this.data)) {
            None => write!(f, "{:?}", this.data),
            Some(universe) => match *this.data {
                TyVar(ty_vid) => write!(f, "?{}_{}t", ty_vid.index(), universe.index()),
                IntVar(_) | FloatVar(_) | FreshTy(_) | FreshIntTy(_) | FreshFloatTy(_) => {
                    unreachable!()
                }
            },
        }
    }
}

impl<I: Interner> fmt::Debug for ConstKind<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}
impl<I: Interner> DebugWithInfcx<I> for ConstKind<I> {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        use ConstKind::*;

        match this.data {
            Param(param) => write!(f, "{param:?}"),
            Infer(var) => write!(f, "{:?}", &this.wrap(var)),
            Bound(debruijn, var) => crate::debug_bound_var(f, *debruijn, var.clone()),
            Placeholder(placeholder) => write!(f, "{placeholder:?}"),
            Unevaluated(uv) => {
                write!(f, "{:?}", &this.wrap(uv))
            }
            Value(valtree) => write!(f, "{valtree:?}"),
            Error(_) => write!(f, "{{const error}}"),
            Expr(expr) => write!(f, "{:?}", &this.wrap(expr)),
        }
    }
}
