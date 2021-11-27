use rustc_index::vec::{Idx, IndexVec};
use std::mem;

pub trait IdFunctor: Sized {
    type Inner;

    #[inline]
    fn map_id<F>(self, mut f: F) -> Self
    where
        F: FnMut(Self::Inner) -> Self::Inner,
    {
        self.try_map_id::<_, !>(|value| Ok(f(value))).into_ok()
    }

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
            let mut raw: Box<mem::MaybeUninit<T>> = Box::from_raw(raw.cast());
            // SAFETY: Write the mapped value back into the `Box`.
            raw.write(f(value)?);
            // SAFETY: We just initialized `raw`.
            raw.assume_init()
        })
    }
}

impl<T> IdFunctor for Vec<T> {
    type Inner = T;

    #[inline]
    fn try_map_id<F, E>(mut self, mut f: F) -> Result<Self, E>
    where
        F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
    {
        // FIXME: We don't really care about panics here and leak
        // far more than we should, but that should be fine for now.
        let len = self.len();
        unsafe {
            self.set_len(0);
            let start = self.as_mut_ptr();
            for i in 0..len {
                let p = start.add(i);
                match f(p.read()) {
                    Ok(val) => p.write(val),
                    Err(err) => {
                        // drop all other elements in self
                        // (current element was "moved" into the call to f)
                        for j in (0..i).chain(i + 1..len) {
                            start.add(j).drop_in_place();
                        }

                        // returning will drop self, releasing the allocation
                        // (len is 0 so elements will not be re-dropped)
                        return Err(err);
                    }
                }
            }
            // Even if we encountered an error, set the len back
            // so we don't leak memory.
            self.set_len(len);
        }
        Ok(self)
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
