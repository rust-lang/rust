use rustc_index::vec::{Idx, IndexVec};
use std::mem;
use std::ptr;

pub trait IdFunctor {
    type Inner;

    fn map_id<F>(self, f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner;
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
}
