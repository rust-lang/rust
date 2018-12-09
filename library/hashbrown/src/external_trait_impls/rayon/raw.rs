use alloc::alloc::{dealloc, Layout};
use core::mem;
use core::ptr::NonNull;
use raw::Bucket;
use raw::{RawIterRange, RawTable};
use rayon::iter::{
    plumbing::{self, Folder, UnindexedConsumer, UnindexedProducer},
    ParallelIterator,
};
use scopeguard::guard;

/// Parallel iterator which returns a raw pointer to every full bucket in the table.
pub struct RawParIter<T> {
    iter: RawIterRange<T>,
}

unsafe impl<T> Send for RawParIter<T> {}

impl<T> ParallelIterator for RawParIter<T> {
    type Item = Bucket<T>;

    #[inline]
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

unsafe impl<T> Send for ParIterProducer<T> {}

impl<T> UnindexedProducer for ParIterProducer<T> {
    type Item = Bucket<T>;

    #[inline]
    fn split(self) -> (Self, Option<Self>) {
        let (left, right) = self.iter.split();
        let left = ParIterProducer { iter: left };
        let right = right.map(|right| ParIterProducer { iter: right });
        (left, right)
    }

    #[inline]
    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        folder.consume_iter(self.iter)
    }
}

/// Parallel iterator which consumes a table and returns elements.
pub struct RawIntoParIter<T> {
    iter: RawIterRange<T>,
    alloc: Option<(NonNull<u8>, Layout)>,
}

unsafe impl<T> Send for RawIntoParIter<T> {}

impl<T: Send> ParallelIterator for RawIntoParIter<T> {
    type Item = T;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let _guard = guard(self.alloc, |alloc| {
            if let Some((ptr, layout)) = *alloc {
                unsafe {
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        });
        let producer = ParDrainProducer { iter: self.iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

/// Parallel iterator which consumes elements without freeing the table storage.
pub struct RawParDrain<'a, T> {
    table: &'a mut RawTable<T>,
}

unsafe impl<'a, T> Send for RawParDrain<'a, T> {}

impl<'a, T: Send> ParallelIterator for RawParDrain<'a, T> {
    type Item = T;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let iter = unsafe { self.table.iter().iter };
        let _guard = guard(self.table, |table| table.clear_no_drop());
        let producer = ParDrainProducer { iter };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

/// Producer which will consume all elements in the range, even if it is dropped
/// halfway through.
struct ParDrainProducer<T> {
    iter: RawIterRange<T>,
}

unsafe impl<T: Send> Send for ParDrainProducer<T> {}

impl<T: Send> UnindexedProducer for ParDrainProducer<T> {
    type Item = T;

    #[inline]
    fn split(self) -> (Self, Option<Self>) {
        let (left, right) = self.iter.clone().split();
        mem::forget(self);
        let left = ParDrainProducer { iter: left };
        let right = right.map(|right| ParDrainProducer { iter: right });
        (left, right)
    }

    #[inline]
    fn fold_with<F>(mut self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        // Make sure to modify the iterator in-place so that any remaining
        // elements are processed in our Drop impl.
        while let Some(item) = self.iter.next() {
            folder = folder.consume(unsafe { item.read() });
            if folder.full() {
                break;
            }
        }
        folder
    }
}

impl<T> Drop for ParDrainProducer<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements
            if mem::needs_drop::<T>() {
                while let Some(item) = self.iter.next() {
                    item.drop();
                }
            }
        }
    }
}

impl<T> RawTable<T> {
    /// Returns a parallel iterator over the elements in a `RawTable`.
    #[inline]
    pub fn par_iter(&self) -> RawParIter<T> {
        RawParIter {
            iter: unsafe { self.iter().iter },
        }
    }

    /// Returns a parallel iterator over the elements in a `RawTable`.
    #[inline]
    pub fn into_par_iter(self) -> RawIntoParIter<T> {
        RawIntoParIter {
            iter: unsafe { self.iter().iter },
            alloc: self.into_alloc(),
        }
    }

    /// Returns a parallel iterator which consumes all elements of a `RawTable`
    /// without freeing its memory allocation.
    #[inline]
    pub fn par_drain(&mut self) -> RawParDrain<T> {
        RawParDrain { table: self }
    }
}
