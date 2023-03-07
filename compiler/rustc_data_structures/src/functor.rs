use rustc_index::vec::{Idx, IndexVec};
use std::{mem, rc::Rc, sync::Arc};

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

macro_rules! rc {
    ($($rc:ident),+) => {$(
        impl<T: Clone> IdFunctor for $rc<T> {
            type Inner = T;

            #[inline]
            fn try_map_id<F, E>(mut self, mut f: F) -> Result<Self, E>
            where
                F: FnMut(Self::Inner) -> Result<Self::Inner, E>,
            {
                // We merely want to replace the contained `T`, if at all possible,
                // so that we don't needlessly allocate a new `$rc` or indeed clone
                // the contained type.
                unsafe {
                    // First step is to ensure that we have a unique reference to
                    // the contained type, which `$rc::make_mut` will accomplish (by
                    // allocating a new `$rc` and cloning the `T` only if required).
                    // This is done *before* casting to `$rc<ManuallyDrop<T>>` so that
                    // panicking during `make_mut` does not leak the `T`.
                    $rc::make_mut(&mut self);

                    // Casting to `$rc<ManuallyDrop<T>>` is safe because `ManuallyDrop`
                    // is `repr(transparent)`.
                    let ptr = $rc::into_raw(self).cast::<mem::ManuallyDrop<T>>();
                    let mut unique = $rc::from_raw(ptr);

                    // Call to `$rc::make_mut` above guarantees that `unique` is the
                    // sole reference to the contained value, so we can avoid doing
                    // a checked `get_mut` here.
                    let slot = $rc::get_mut_unchecked(&mut unique);

                    // Semantically move the contained type out from `unique`, fold
                    // it, then move the folded value back into `unique`. Should
                    // folding fail, `ManuallyDrop` ensures that the "moved-out"
                    // value is not re-dropped.
                    let owned = mem::ManuallyDrop::take(slot);
                    let folded = f(owned)?;
                    *slot = mem::ManuallyDrop::new(folded);

                    // Cast back to `$rc<T>`.
                    Ok($rc::from_raw($rc::into_raw(unique).cast()))
                }
            }
        }
    )+};
}

rc! { Rc, Arc }
