use crate::raw::Bucket;
use crate::raw::{RawIterRange, RawTable};
use crate::scopeguard::guard;
use alloc::alloc::dealloc;
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;
use rayon::iter::{
    plumbing::{self, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};

/// Parallel iterator which returns a raw pointer to every full bucket in the table.
pub struct RawParIter<T> {
    iter: RawIterRange<T>,
}

impl<T> ParallelIterator for RawParIter<T> {
    type Item = Bucket<T>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let producer = ParIterProducer { iter: self.iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

/// Producer which returns a `Bucket<T>` for every element.
struct ParIterProducer<T> {
    iter: RawIterRange<T>,
}

impl<T> UnindexedProducer for ParIterProducer<T> {
    type Item = Bucket<T>;

    #[cfg_attr(feature = "inline-more", inline)]
    fn split(self) -> (Self, Option<Self>) {
        let (left, right) = self.iter.split();
        let left = ParIterProducer { iter: left };
        let right = right.map(|right| ParIterProducer { iter: right });
        (left, right)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        folder.consume_iter(self.iter)
    }
}

/// Parallel iterator which consumes a table and returns elements.
pub struct RawIntoParIter<T> {
    table: RawTable<T>,
}

impl<T: Send> ParallelIterator for RawIntoParIter<T> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let iter = unsafe { self.table.iter().iter };
        let _guard = guard(self.table.into_alloc(), |alloc| {
            if let Some((ptr, layout)) = *alloc {
                unsafe {
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        });
        let producer = ParDrainProducer { iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

/// Parallel iterator which consumes elements without freeing the table storage.
pub struct RawParDrain<'a, T> {
    // We don't use a &'a mut RawTable<T> because we want RawParDrain to be
    // covariant over T.
    table: NonNull<RawTable<T>>,
    marker: PhantomData<&'a RawTable<T>>,
}

unsafe impl<T> Send for RawParDrain<'_, T> {}

impl<T: Send> ParallelIterator for RawParDrain<'_, T> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let _guard = guard(self.table, |table| unsafe {
            table.as_mut().clear_no_drop()
        });
        let iter = unsafe { self.table.as_ref().iter().iter };
        mem::forget(self);
        let producer = ParDrainProducer { iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<T> Drop for RawParDrain<'_, T> {
    fn drop(&mut self) {
        // If drive_unindexed is not called then simply clear the table.
        unsafe { self.table.as_mut().clear() }
    }
}

/// Producer which will consume all elements in the range, even if it is dropped
/// halfway through.
struct ParDrainProducer<T> {
    iter: RawIterRange<T>,
}

impl<T: Send> UnindexedProducer for ParDrainProducer<T> {
    type Item = T;

    #[cfg_attr(feature = "inline-more", inline)]
    fn split(self) -> (Self, Option<Self>) {
        let (left, right) = self.iter.clone().split();
        mem::forget(self);
        let left = ParDrainProducer { iter: left };
        let right = right.map(|right| ParDrainProducer { iter: right });
        (left, right)
    }

    #[cfg_attr(feature = "inline-more", inline)]
    fn fold_with<F>(mut self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        // Make sure to modify the iterator in-place so that any remaining
        // elements are processed in our Drop impl.
        while let Some(item) = self.iter.next() {
            folder = folder.consume(unsafe { item.read() });
            if folder.full() {
                return folder;
            }
        }

        // If we processed all elements then we don't need to run the drop.
        mem::forget(self);
        folder
    }
}

impl<T> Drop for ParDrainProducer<T> {
    #[cfg_attr(feature = "inline-more", inline)]
    fn drop(&mut self) {
        // Drop all remaining elements
        if mem::needs_drop::<T>() {
            while let Some(item) = self.iter.next() {
                unsafe {
                    item.drop();
                }
            }
        }
    }
}

impl<T> RawTable<T> {
    /// Returns a parallel iterator over the elements in a `RawTable`.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn par_iter(&self) -> RawParIter<T> {
        RawParIter {
            iter: unsafe { self.iter().iter },
        }
    }

    /// Returns a parallel iterator over the elements in a `RawTable`.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn into_par_iter(self) -> RawIntoParIter<T> {
        RawIntoParIter { table: self }
    }

    /// Returns a parallel iterator which consumes all elements of a `RawTable`
    /// without freeing its memory allocation.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn par_drain(&mut self) -> RawParDrain<'_, T> {
        RawParDrain {
            table: NonNull::from(self),
            marker: PhantomData,
        }
    }
}
