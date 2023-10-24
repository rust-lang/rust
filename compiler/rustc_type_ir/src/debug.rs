use crate::{Interner, UniverseIndex};

use core::fmt;
use std::marker::PhantomData;

pub trait InferCtxtLike {
    type Interner: Interner;

    fn universe_of_ty(&self, ty: <Self::Interner as Interner>::InferTy) -> Option<UniverseIndex>;

    fn universe_of_lt(
        &self,
        lt: <Self::Interner as Interner>::InferRegion,
    ) -> Option<UniverseIndex>;

    fn universe_of_ct(&self, ct: <Self::Interner as Interner>::InferConst)
    -> Option<UniverseIndex>;
}

pub struct NoInfcx<I>(PhantomData<I>);

impl<I: Interner> InferCtxtLike for NoInfcx<I> {
    type Interner = I;

    fn universe_of_ty(&self, _ty: <I as Interner>::InferTy) -> Option<UniverseIndex> {
        None
    }

    fn universe_of_ct(&self, _ct: <I as Interner>::InferConst) -> Option<UniverseIndex> {
        None
    }

    fn universe_of_lt(&self, _lt: <I as Interner>::InferRegion) -> Option<UniverseIndex> {
        None
    }
}

pub trait DebugWithInfcx<I: Interner>: fmt::Debug {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
}

impl<I: Interner, T: DebugWithInfcx<I> + ?Sized> DebugWithInfcx<I> for &'_ T {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        <T as DebugWithInfcx<I>>::fmt(this.map(|&data| data), f)
    }
}

impl<I: Interner, T: DebugWithInfcx<I>> DebugWithInfcx<I> for [T] {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
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

pub struct WithInfcx<'a, Infcx: InferCtxtLike, T> {
    pub data: T,
    pub infcx: &'a Infcx,
}

impl<Infcx: InferCtxtLike, T: Copy> Copy for WithInfcx<'_, Infcx, T> {}

impl<Infcx: InferCtxtLike, T: Clone> Clone for WithInfcx<'_, Infcx, T> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), infcx: self.infcx }
    }
}

impl<'a, I: Interner, T> WithInfcx<'a, NoInfcx<I>, T> {
    pub fn with_no_infcx(data: T) -> Self {
        Self { data, infcx: &NoInfcx(PhantomData) }
    }
}

impl<'a, Infcx: InferCtxtLike, T> WithInfcx<'a, Infcx, T> {
    pub fn new(data: T, infcx: &'a Infcx) -> Self {
        Self { data, infcx }
    }

    pub fn wrap<U>(self, u: U) -> WithInfcx<'a, Infcx, U> {
        WithInfcx { data: u, infcx: self.infcx }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> WithInfcx<'a, Infcx, U> {
        WithInfcx { data: f(self.data), infcx: self.infcx }
    }

    pub fn as_ref(&self) -> WithInfcx<'a, Infcx, &T> {
        WithInfcx { data: &self.data, infcx: self.infcx }
    }
}

impl<Infcx: InferCtxtLike, T: DebugWithInfcx<Infcx::Interner>> fmt::Debug
    for WithInfcx<'_, Infcx, T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DebugWithInfcx::fmt(self.as_ref(), f)
    }
}
