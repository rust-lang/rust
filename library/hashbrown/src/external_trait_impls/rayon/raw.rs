use core::fmt;
use core::marker::PhantomData;

use rayon::iter::{
    plumbing::{self, Folder, UnindexedConsumer, UnindexedProducer},
    IntoParallelIterator, ParallelIterator,
};

use raw::{Bucket, RawTable};


struct SplitBuckets<'a, K: 'a, V: 'a> {
    table: &'a RawTable<(K, V)>,
    pos: usize,
    end: usize,
}

impl<'a, K, V> SplitBuckets<'a, K, V> {
    fn new(table: &'a RawTable<(K, V)>) -> Self {
        let mut begin = 0;
        let mut end = table.buckets();
        trim_empty_buckets(table, &mut begin, &mut end);

        Self { table, pos: begin, end }
    }

    /// Split in halves, if possible. Halves get automatically trimmed.
    #[inline]
    fn split<P: From<Self>>(&self) -> (P, Option<P>) {
        let Self { table, pos, end } = *self;
        let len = end - pos;

        if len > 1 {
            let mid = pos + len / 2;

            let mut left = Self { table, pos, end: mid };
            trim_empty_buckets(left.table, &mut left.pos, &mut left.end);

            let mut right = Self { table, pos: mid, end };
            trim_empty_buckets(right.table, &mut right.pos, &mut right.end);

            (P::from(left), Some(P::from(right)))
        } else {
            (P::from(Self { table, pos, end }), None)
        }
    }
}

/// Trims the (half-open) range `begin..end` of buckets from both sides.
/// As a result, either of these outcomes occur:
///
/// * `begin` becomes equal to `end`;
/// * `begin` becomes strictly less than `end` AND both `begin` and `end - 1`
///   point to non-empty buckets.
#[inline]
fn trim_empty_buckets<T>(table: &RawTable<T>, begin: &mut usize, end: &mut usize) {
    // Trim empty buckets from beginning
    while *begin < *end && table.is_empty_bucket_at(*begin) {
        *begin += 1;
    }

    // Trim empty buckets from end
    while *begin < *end && table.is_empty_bucket_at(*end - 1) {
        *end -= 1;
    }
}

impl<'a, K, V> Iterator for SplitBuckets<'a, K, V> {
    type Item = Bucket<(K, V)>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.end {
            let item = unsafe { self.table.bucket(self.pos) };
            let old_pos = self.pos;
            self.pos += 1;

            if !self.table.is_empty_bucket_at(old_pos) {
                return Some(item);
            }
        }

        None
    }
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
        where C: UnindexedConsumer<Self::Item>
    {
        let buckets = SplitBuckets::new(self.table);
        let producer = ParIterProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V> Clone for ParIter<'a, K, V> {
    fn clone(&self) -> Self {
        ParIter { table: &*self.table }
    }
}

impl<'a, K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIter<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entries = SplitBuckets::new(self.table).map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParIterProducer<'a, K: 'a, V: 'a> {
    iter: SplitBuckets<'a, K, V>,
}

impl<'a, K, V> From<SplitBuckets<'a, K, V>> for ParIterProducer<'a, K, V> {
    #[inline]
    fn from(iter: SplitBuckets<'a, K, V>) -> Self {
        Self { iter }
    }
}

unsafe impl<'a, K: Sync, V: Sync> Send for ParIterProducer<'a, K, V> {}

impl<'a, K: Sync, V: Sync> UnindexedProducer for ParIterProducer<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn split(self) -> (Self, Option<Self>) {
        self.iter.split()
    }

    fn fold_with<F>(self, folder: F) -> F
        where F: Folder<Self::Item>
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
        let buckets = SplitBuckets::new(self.table);
        let producer = ParKeysProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V> Clone for ParKeys<'a, K, V> {
    fn clone(&self) -> Self {
        ParKeys { table: &*self.table }
    }
}

impl<'a, K: fmt::Debug, V> fmt::Debug for ParKeys<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entries = SplitBuckets::new(self.table).map(|bucket| unsafe {
            &bucket.as_ref().0
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParKeysProducer<'a, K: 'a, V: 'a> {
    iter: SplitBuckets<'a, K, V>,
}

impl<'a, K, V> From<SplitBuckets<'a, K, V>> for ParKeysProducer<'a, K, V> {
    #[inline]
    fn from(iter: SplitBuckets<'a, K, V>) -> Self {
        Self { iter }
    }
}

unsafe impl<'a, K: Sync, V> Send for ParKeysProducer<'a, K, V> {}

impl<'a, K: Sync, V> UnindexedProducer for ParKeysProducer<'a, K, V> {
    type Item = &'a K;

    fn split(self) -> (Self, Option<Self>) {
        self.iter.split()
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe {
            &bucket.as_ref().0
        });
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
        let buckets = SplitBuckets::new(self.table);
        let producer = ParValuesProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V> Clone for ParValues<'a, K, V> {
    fn clone(&self) -> Self {
        ParValues { table: &*self.table }
    }
}

impl<'a, K, V: fmt::Debug> fmt::Debug for ParValues<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entries = SplitBuckets::new(self.table).map(|bucket| unsafe {
            &bucket.as_ref().1
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParValuesProducer<'a, K: 'a, V: 'a> {
    iter: SplitBuckets<'a, K, V>,
}

impl<'a, K, V> From<SplitBuckets<'a, K, V>> for ParValuesProducer<'a, K, V> {
    #[inline]
    fn from(iter: SplitBuckets<'a, K, V>) -> Self {
        Self { iter }
    }
}

unsafe impl<'a, K, V: Sync> Send for ParValuesProducer<'a, K, V> {}

impl<'a, K, V: Sync> UnindexedProducer for ParValuesProducer<'a, K, V> {
    type Item = &'a V;

    fn split(self) -> (Self, Option<Self>) {
        self.iter.split()
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe {
            &bucket.as_ref().1
        });
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
        let buckets = SplitBuckets::new(self.table);
        let producer = ParIterMutProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K: fmt::Debug, V: fmt::Debug> fmt::Debug for ParIterMut<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entries = SplitBuckets::new(self.table).map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParIterMutProducer<'a, K: 'a, V: 'a> {
    iter: SplitBuckets<'a, K, V>,
    // To ensure invariance with respect to V
    marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> From<SplitBuckets<'a, K, V>> for ParIterMutProducer<'a, K, V> {
    #[inline]
    fn from(iter: SplitBuckets<'a, K, V>) -> Self {
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
        self.iter.split()
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
        let buckets = SplitBuckets::new(self.table);
        let producer = ParValuesMutProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K, V: fmt::Debug> fmt::Debug for ParValuesMut<'a, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entries = SplitBuckets::new(self.table).map(|bucket| unsafe {
            &bucket.as_ref().1
        });

        f.debug_list().entries(entries).finish()
    }
}

struct ParValuesMutProducer<'a, K: 'a, V: 'a> {
    iter: SplitBuckets<'a, K, V>,
    // To ensure invariance with respect to V
    marker: PhantomData<&'a mut V>,
}

impl<'a, K, V> From<SplitBuckets<'a, K, V>> for ParValuesMutProducer<'a, K, V> {
    #[inline]
    fn from(iter: SplitBuckets<'a, K, V>) -> Self {
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
        self.iter.split()
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        let iter = self.iter.map(|bucket| unsafe {
            &mut bucket.as_mut().1
        });
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
        // Pre-set the map size to zero, indicating all items drained.
        let mut table = self.table;
        unsafe {
            table.set_len(0);
        }

        let buckets = SplitBuckets::new(&table);
        let producer = IntoParIterProducer::from(buckets);
        plumbing::bridge_unindexed(producer, consumer)
    }
}

impl<'a, K: fmt::Debug, V: fmt::Debug> fmt::Debug for IntoParIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entries = SplitBuckets::new(&self.table).map(|bucket| unsafe {
            let (ref k, ref v) = *bucket.as_ref();
            (k, v)
        });

        f.debug_list().entries(entries).finish()
    }
}

struct IntoParIterProducer<'a, K: 'a, V: 'a> {
    iter: SplitBuckets<'a, K, V>,
}

impl<'a, K, V> From<SplitBuckets<'a, K, V>> for IntoParIterProducer<'a, K, V> {
    #[inline]
    fn from(iter: SplitBuckets<'a, K, V>) -> Self {
        Self { iter }
    }
}

unsafe impl<'a, K: Send, V: Send> Send for IntoParIterProducer<'a, K, V> {}

impl<'a, K: Send, V: Send> UnindexedProducer for IntoParIterProducer<'a, K, V> {
    type Item = (K, V);

    fn split(self) -> (Self, Option<Self>) {
        self.iter.split()
    }

    fn fold_with<F>(mut self, mut folder: F) -> F
    where
        F: Folder<Self::Item>,
    {
        for bucket in self.iter.by_ref() {
            unsafe {
                self.iter.table.set_empty_bucket(&bucket);
                folder = folder.consume(bucket.read());
            }

            if folder.full() {
                break;
            }
        }

        folder
    }
}

// Use the default `Drop` impl for `IntoParIter` which recursively drops the
// inner `RawTable`
