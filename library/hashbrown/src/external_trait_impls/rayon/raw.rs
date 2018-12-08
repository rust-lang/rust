use alloc::alloc::dealloc;
use core::alloc::Layout;
use core::fmt;
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

use rayon::iter::{
    plumbing::{self, Folder, UnindexedConsumer, UnindexedProducer},
    IntoParallelIterator, ParallelIterator,
};

use raw::{RawIterRange, RawTable};

#[inline]
unsafe fn raw_iter_range_split_and_convert<T, P>(iter_range: RawIterRange<T>) -> (P, Option<P>)
where
    P: From<RawIterRange<T>>,
{
    let (left, right) = iter_range.split();
    (P::from(left), right.map(P::from))
}

/// Parallel iterator over shared references to entries in a map.
///
/// This iterator is created by the [`par_iter`] method on [`HashMap`]
/// (provided by the [`IntoParallelRefIterator`] trait).
/// See its documentation for more.
///
/// [`par_iter`]: /hashbrown/struct.HashMap.html#method.par_iter
/// [`HashMap`]: /hashbrown/struct.HashMap.html
/// [`IntoParallelRefIterator`]: https://docs.rs/rayon/1.0/rayon/iter/trait.IntoParallelRefIterator.html
pub struct ParIter<'a, K: 'a, V: 'a> {
    table: &'a RawTable<(K, V)>,
}

impl<'a, K: Sync, V: Sync> IntoParallelIterator for &'a RawTable<(K, V)> {
    type Item = (&'a K, &'a V);
    type Iter = ParIter<'a, K, V>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ParIter { table: self }
    }
}

unsafe impl<'a, K: Sync, V: Sync> Send for ParIter<'a, K, V> {}

impl<'a, K: Sync, V: Sync> ParallelIterator for ParIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let buckets = unsafe { self.table.iter().iter };
        let producer = ParIterProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V> Clone for ParIter<'a, K, V> {
    fn clone(&self) -> Self {
        ParIter {
            table: &*self.table,
        }
    }
}

impl<'a, K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIter<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = unsafe { self.table.iter() };
        let entries = iter.map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParIterProducer<'a, K: 'a, V: 'a> {
    iter: RawIterRange<(K, V)>,
    marker: PhantomData<&'a ()>,
}

impl<'a, K, V> From<RawIterRange<(K, V)>> for ParIterProducer<'a, K, V> {
    #[inline]
    fn from(iter: RawIterRange<(K, V)>) -> Self {
        Self {
            iter,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, K: Sync, V: Sync> Send for ParIterProducer<'a, K, V> {}

impl<'a, K: Sync, V: Sync> UnindexedProducer for ParIterProducer<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn split(self) -> (Self, Option<Self>) {
        unsafe { raw_iter_range_split_and_convert(self.iter) }
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });
        folder.consume_iter(iter)
    }
}

/// Parallel iterator over shared references to keys in a map.
///
/// This iterator is created by the [`par_keys`] method on [`HashMap`].
/// See its documentation for more.
///
/// [`par_keys`]: /hashbrown/struct.HashMap.html#method.par_keys
/// [`HashMap`]: /hashbrown/struct.HashMap.html
pub struct ParKeys<'a, K: 'a, V: 'a> {
    table: &'a RawTable<(K, V)>,
}

impl<K: Sync, V> RawTable<(K, V)> {
    #[inline]
    pub fn par_keys(&self) -> ParKeys<K, V> {
        ParKeys { table: self }
    }
}

unsafe impl<'a, K: Sync, V> Send for ParKeys<'a, K, V> {}

impl<'a, K: Sync, V> ParallelIterator for ParKeys<'a, K, V> {
    type Item = &'a K;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let buckets = unsafe { self.table.iter().iter };
        let producer = ParKeysProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V> Clone for ParKeys<'a, K, V> {
    fn clone(&self) -> Self {
        ParKeys {
            table: &*self.table,
        }
    }
}

impl<'a, K: fmt::Debug, V> fmt::Debug for ParKeys<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = unsafe { self.table.iter() };
        let entries = iter.map(|bucket| unsafe { &bucket.as_ref().0 });

        f.debug_list().entries(entries).finish()
    }
}

struct ParKeysProducer<'a, K: 'a, V: 'a> {
    iter: RawIterRange<(K, V)>,
    marker: PhantomData<&'a ()>,
}

impl<'a, K, V> From<RawIterRange<(K, V)>> for ParKeysProducer<'a, K, V> {
    #[inline]
    fn from(iter: RawIterRange<(K, V)>) -> Self {
        Self {
            iter,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, K: Sync, V> Send for ParKeysProducer<'a, K, V> {}

impl<'a, K: Sync, V> UnindexedProducer for ParKeysProducer<'a, K, V> {
    type Item = &'a K;

    fn split(self) -> (Self, Option<Self>) {
        unsafe { raw_iter_range_split_and_convert(self.iter) }
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe { &bucket.as_ref().0 });
        folder.consume_iter(iter)
    }
}

/// Parallel iterator over shared references to values in a map.
///
/// This iterator is created by the [`par_values`] method on [`HashMap`].
/// See its documentation for more.
///
/// [`par_values`]: /hashbrown/struct.HashMap.html#method.par_values
/// [`HashMap`]: /hashbrown/struct.HashMap.html
pub struct ParValues<'a, K: 'a, V: 'a> {
    table: &'a RawTable<(K, V)>,
}

impl<K, V: Sync> RawTable<(K, V)> {
    #[inline]
    pub fn par_values(&self) -> ParValues<K, V> {
        ParValues { table: self }
    }
}

unsafe impl<'a, K, V: Sync> Send for ParValues<'a, K, V> {}

impl<'a, K, V: Sync> ParallelIterator for ParValues<'a, K, V> {
    type Item = &'a V;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let buckets = unsafe { self.table.iter().iter };
        let producer = ParValuesProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V> Clone for ParValues<'a, K, V> {
    fn clone(&self) -> Self {
        ParValues {
            table: &*self.table,
        }
    }
}

impl<'a, K, V: fmt::Debug> fmt::Debug for ParValues<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = unsafe { self.table.iter() };
        let entries = iter.map(|bucket| unsafe { &bucket.as_ref().1 });

        f.debug_list().entries(entries).finish()
    }
}

struct ParValuesProducer<'a, K: 'a, V: 'a> {
    iter: RawIterRange<(K, V)>,
    marker: PhantomData<&'a ()>,
}

impl<'a, K, V> From<RawIterRange<(K, V)>> for ParValuesProducer<'a, K, V> {
    #[inline]
    fn from(iter: RawIterRange<(K, V)>) -> Self {
        Self {
            iter,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, K, V: Sync> Send for ParValuesProducer<'a, K, V> {}

impl<'a, K, V: Sync> UnindexedProducer for ParValuesProducer<'a, K, V> {
    type Item = &'a V;

    fn split(self) -> (Self, Option<Self>) {
        unsafe { raw_iter_range_split_and_convert(self.iter) }
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe { &bucket.as_ref().1 });
        folder.consume_iter(iter)
    }
}

/// Parallel iterator over mutable references to entries in a map.
///
/// This iterator is created by the [`par_iter_mut`] method on [`HashMap`]
/// (provided by the [`IntoParallelRefMutIterator`] trait).
/// See its documentation for more.
///
/// [`par_iter_mut`]: /hashbrown/struct.HashMap.html#method.par_iter_mut
/// [`HashMap`]: /hashbrown/struct.HashMap.html
/// [`IntoParallelRefMutIterator`]: https://docs.rs/rayon/1.0/rayon/iter/trait.IntoParallelRefMutIterator.html
pub struct ParIterMut<'a, K: 'a, V: 'a> {
    table: &'a mut RawTable<(K, V)>,
}

impl<'a, K: Sync, V: Send> IntoParallelIterator for &'a mut RawTable<(K, V)> {
    type Item = (&'a K, &'a mut V);
    type Iter = ParIterMut<'a, K, V>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        ParIterMut { table: self }
    }
}

unsafe impl<'a, K: Sync, V: Send> Send for ParIterMut<'a, K, V> {}

impl<'a, K: Sync, V: Send> ParallelIterator for ParIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let buckets = unsafe { self.table.iter().iter };
        let producer = ParIterMutProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIterMut<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = unsafe { self.table.iter() };
        let entries = iter.map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParIterMutProducer<'a, K: 'a, V: 'a> {
    iter: RawIterRange<(K, V)>,
    // To ensure invariance with respect to V
    marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> From<RawIterRange<(K, V)>> for ParIterMutProducer<'a, K, V> {
    #[inline]
    fn from(iter: RawIterRange<(K, V)>) -> Self {
        Self {
            iter,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, K: Sync, V: Send> Send for ParIterMutProducer<'a, K, V> {}

impl<'a, K: Sync, V: Send> UnindexedProducer for ParIterMutProducer<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn split(self) -> (Self, Option<Self>) {
        unsafe { raw_iter_range_split_and_convert(self.iter) }
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe {
            let (ref k, ref mut v) = *bucket.as_mut();
            (k, v)
        });
        folder.consume_iter(iter)
    }
}

/// Parallel iterator over mutable references to values in a map.
///
/// This iterator is created by the [`par_values_mut`] method on [`HashMap`].
/// See its documentation for more.
///
/// [`par_values_mut`]: /hashbrown/struct.HashMap.html#method.par_values_mut
/// [`HashMap`]: /hashbrown/struct.HashMap.html
pub struct ParValuesMut<'a, K: 'a, V: 'a> {
    table: &'a mut RawTable<(K, V)>,
}

impl<K, V: Send> RawTable<(K, V)> {
    #[inline]
    pub fn par_values_mut(&mut self) -> ParValuesMut<K, V> {
        ParValuesMut { table: self }
    }
}

unsafe impl<'a, K, V: Send> Send for ParValuesMut<'a, K, V> {}

impl<'a, K, V: Send> ParallelIterator for ParValuesMut<'a, K, V> {
    type Item = &'a mut V;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let buckets = unsafe { self.table.iter().iter };
        let producer = ParValuesMutProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V: fmt::Debug> fmt::Debug for ParValuesMut<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = unsafe { self.table.iter() };
        let entries = iter.map(|bucket| unsafe { &bucket.as_ref().1 });

        f.debug_list().entries(entries).finish()
    }
}

struct ParValuesMutProducer<'a, K: 'a, V: 'a> {
    iter: RawIterRange<(K, V)>,
    // To ensure invariance with respect to V
    marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> From<RawIterRange<(K, V)>> for ParValuesMutProducer<'a, K, V> {
    #[inline]
    fn from(iter: RawIterRange<(K, V)>) -> Self {
        Self {
            iter,
            marker: PhantomData,
        }
    }
}

unsafe impl<'a, K, V: Send> Send for ParValuesMutProducer<'a, K, V> {}

impl<'a, K, V: Send> UnindexedProducer for ParValuesMutProducer<'a, K, V> {
    type Item = &'a mut V;

    fn split(self) -> (Self, Option<Self>) {
        unsafe { raw_iter_range_split_and_convert(self.iter) }
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe { &mut bucket.as_mut().1 });
        folder.consume_iter(iter)
    }
}

/// Parallel iterator over entries of a consumed map.
///
/// This iterator is created by the [`into_par_iter`] method on [`HashMap`]
/// (provided by the [`IntoParallelIterator`] trait).
/// See its documentation for more.
///
/// [`into_par_iter`]: /hashbrown/struct.HashMap.html#method.into_par_iter
/// [`HashMap`]: /hashbrown/struct.HashMap.html
/// [`IntoParallelIterator`]: https://docs.rs/rayon/1.0/rayon/iter/trait.IntoParallelIterator.html
pub struct IntoParIter<K, V> {
    table: RawTable<(K, V)>,
}

impl<K: Send, V: Send> IntoParallelIterator for RawTable<(K, V)> {
    type Item = (K, V);
    type Iter = IntoParIter<K, V>;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        IntoParIter { table: self }
    }
}

impl<K: Send, V: Send> ParallelIterator for IntoParIter<K, V> {
    type Item = (K, V);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let iter = unsafe { self.table.iter().iter };
        let alloc = self.table.into_alloc();
        let producer = IntoParIterProducer { iter, alloc };
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K: fmt::Debug, V: fmt::Debug> fmt::Debug for IntoParIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iter = unsafe { self.table.iter() };
        let entries = iter.map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });

        f.debug_list().entries(entries).finish()
    }
}

struct IntoParIterProducer<K, V> {
    iter: RawIterRange<(K, V)>,
    alloc: Option<(NonNull<u8>, Layout)>,
}

unsafe impl<K: Send, V: Send> Send for IntoParIterProducer<K, V> {}

impl<K: Send, V: Send> UnindexedProducer for IntoParIterProducer<K, V> {
    type Item = (K, V);

    fn split(mut self) -> (Self, Option<Self>) {
        let (left_iter, right_iter) = unsafe { self.iter.clone().split() };

        self.iter = left_iter;
        let right = right_iter.map(|iter| IntoParIterProducer { iter, alloc: None });

        (self, right)
    }

    fn fold_with<F>(mut self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.by_ref().map(|bucket| unsafe { bucket.read() });
        folder.consume_iter(iter)
    }
}

impl<K, V> Drop for IntoParIterProducer<K, V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements
            if mem::needs_drop::<(K, V)>() {
                while let Some(item) = self.iter.next() {
                    item.drop();
                }
            }

            // Free the table
            if let Some((ptr, layout)) = self.alloc {
                dealloc(ptr.as_ptr(), layout);
            }
        }
    }
}
