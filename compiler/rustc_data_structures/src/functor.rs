use rustc_index::vec::{Idx, IndexVec};
use std::mem;
use std::ptr;

pub trait IdFunctor: Sized {
    type Inner;

    fn map_id<F>(self, f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner;

    fn try_map_id<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>;
}

impl<T> IdFunctor for Box<T> {
    type Inner = T;

    #[inline]
    fn map_id<F>(self, mut f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner,
    {
        let raw = Box::into_raw(self);
        unsafe {
            // SAFETY: The raw pointer points to a valid value of type `T`.
            let value = ptr::read(raw);
            // SAFETY: Converts `Box<T>` to `Box<MaybeUninit<T>>` which is the
            // inverse of `Box::assume_init()` and should be safe.
            let mut raw: Box<mem::MaybeUninit<T>> = Box::from_raw(raw.cast());
            // SAFETY: Write the mapped value back into the `Box`.
            ptr::write(raw.as_mut_ptr(), f(value));
            // SAFETY: We just initialized `raw`.
            raw.assume_init()
        }
    }

    #[inline]
    fn try_map_id<F, E>(self, mut f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        let raw = Box::into_raw(self);
        Ok(unsafe {
            // SAFETY: The raw pointer points to a valid value of type `T`.
            let value = ptr::read(raw);
            // SAFETY: Converts `Box<T>` to `Box<MaybeUninit<T>>` which is the
            // inverse of `Box::assume_init()` and should be safe.
            let mut raw: Box<mem::MaybeUninit<T>> = Box::from_raw(raw.cast());
            // SAFETY: Write the mapped value back into the `Box`.
            ptr::write(raw.as_mut_ptr(), f(value)?);
            // SAFETY: We just initialized `raw`.
            raw.assume_init()
        })
    }
}

impl<T> IdFunctor for Vec<T> {
    type Inner = T;

    #[inline]
    fn map_id<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner,
    {
        // FIXME: We don't really care about panics here and leak
        // far more than we should, but that should be fine for now.
        let len = self.len();
        unsafe {
            self.set_len(0);
            let start = self.as_mut_ptr();
            for i in 0..len {
                let p = start.add(i);
                ptr::write(p, f(ptr::read(p)));
            }
            self.set_len(len);
        }
        self
    }

    #[inline]
    fn try_map_id<F, E>(mut self, mut f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        // FIXME: We don't really care about panics here and leak
        // far more than we should, but that should be fine for now.
        let len = self.len();
        let mut error = Ok(());
        unsafe {
            self.set_len(0);
            let start = self.as_mut_ptr();
            for i in 0..len {
                let p = start.add(i);
                match f(ptr::read(p)) {
                    Ok(value) => ptr::write(p, value),
                    Err(err) => {
                        error = Err(err);
                        break;
                    }
                }
            }
            // Even if we encountered an error, set the len back
            // so we don't leak memory.
            self.set_len(len);
        }
        error.map(|()| self)
    }
}

impl<T> IdFunctor for Box<[T]> {
    type Inner = T;

    #[inline]
    fn map_id<F>(self, f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner,
    {
        Vec::from(self).map_id(f).into()
    }

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
    fn map_id<F>(self, f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner,
    {
        IndexVec::from_raw(self.raw.map_id(f))
    }

    #[inline]
    fn try_map_id<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        self.raw.try_map_id(f).map(IndexVec::from_raw)
    }
}
