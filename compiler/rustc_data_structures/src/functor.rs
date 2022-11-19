use rustc_index::vec::{Idx, IndexVec};
use std::mem;

pub trait IdFunctor: Sized {
    type Inner;

    fn try_map_id<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>;
}

impl<T> IdFunctor for Box<T> {
    type Inner = T;

    #[inline]
    fn try_map_id<F, E>(self, mut f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        let raw = Box::into_raw(self);
        Ok(unsafe {
            // SAFETY: The raw pointer points to a valid value of type `T`.
            let value = raw.read();
            // SAFETY: Converts `Box<T>` to `Box<MaybeUninit<T>>` which is the
            // inverse of `Box::assume_init()` and should be safe.
            let raw: Box<mem::MaybeUninit<T>> = Box::from_raw(raw.cast());
            // SAFETY: Write the mapped value back into the `Box`.
            Box::write(raw, f(value)?)
        })
    }
}

impl<T> IdFunctor for Vec<T> {
    type Inner = T;

    #[inline]
    fn try_map_id<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        self.into_iter().map(f).collect()
    }
}

impl<T> IdFunctor for Box<[T]> {
    type Inner = T;

    #[inline]
    fn try_map_id<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        Vec::from(self).try_map_id(f).map(Into::into)
    }
}

impl<I: Idx, T> IdFunctor for IndexVec<I, T> {
    type Inner = T;

    #[inline]
    fn try_map_id<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        self.raw.try_map_id(f).map(IndexVec::from_raw)
    }
}
