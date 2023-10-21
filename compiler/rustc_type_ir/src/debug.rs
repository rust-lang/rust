use crate::{Interner, UniverseIndex};

use core::fmt;
use std::marker::PhantomData;

pub trait InferCtxtLike<I: Interner> {
    fn universe_of_ty(&self, ty: I::InferTy) -> Option<UniverseIndex>;

    fn universe_of_lt(&self, lt: I::InferRegion) -> Option<UniverseIndex>;

    fn universe_of_ct(&self, ct: I::InferConst) -> Option<UniverseIndex>;
}

impl<I: Interner> InferCtxtLike<I> for core::convert::Infallible {
    fn universe_of_ty(&self, _ty: <I as Interner>::InferTy) -> Option<UniverseIndex> {
        match *self {}
    }

    fn universe_of_ct(&self, _ct: <I as Interner>::InferConst) -> Option<UniverseIndex> {
        match *self {}
    }

    fn universe_of_lt(&self, _lt: <I as Interner>::InferRegion) -> Option<UniverseIndex> {
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
