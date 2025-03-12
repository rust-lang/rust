//! A hash table with lock-free reads.
//!
//! It is based on the table from the `hashbrown` crate.

use crate::{
    collect::{self, Pin, pin},
    raw::{bitmask::BitMask, imp::Group},
    scopeguard::guard,
    util::{cold_path, make_insert_hash},
};
use core::ptr::NonNull;
use parking_lot::{Mutex, MutexGuard};
use std::{
    alloc::{Allocator, Global, Layout, LayoutError, handle_alloc_error},
    cell::UnsafeCell,
    cmp, fmt,
    hash::BuildHasher,
    intrinsics::{likely, unlikely},
    iter::{FromIterator, FusedIterator},
    marker::PhantomData,
    mem,
    sync::atomic::{AtomicU8, Ordering},
};
use std::{borrow::Borrow, hash::Hash};
use std::{collections::hash_map::RandomState, sync::Arc};
use std::{ops::Deref, sync::atomic::AtomicPtr};
use std::{ops::DerefMut, sync::atomic::AtomicUsize};

mod code;
mod tests;

#[inline]
fn hasher<K: Hash, V, S: BuildHasher>(hash_builder: &S, val: &(K, V)) -> u64 {
    make_insert_hash(hash_builder, &val.0)
}

#[inline]
fn eq<Q, K, V>(key: &Q) -> impl Fn(&(K, V)) -> bool + '_
where
    K: Borrow<Q>,
    Q: ?Sized + Eq,
{
    move |x| key.eq(x.0.borrow())
}

/// A reference to a hash table bucket containing a `T`.
///
/// This is usually just a pointer to the element itself. However if the element
/// is a ZST, then we instead track the index of the element in the table so
/// that `erase` works properly.
struct Bucket<T> {
    // Actually it is pointer to next element than element itself
    // this is needed to maintain pointer arithmetic invariants
    // keeping direct pointer to element introduces difficulty.
    // Using `NonNull` for variance and niche layout
    ptr: NonNull<T>,
}

impl<T> Clone for Bucket<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T> Bucket<T> {
    #[inline]
    fn as_ptr(&self) -> *mut T {
        if mem::size_of::<T>() == 0 {
            // Just return an arbitrary ZST pointer which is properly aligned
            mem::align_of::<T>() as *mut T
        } else {
            unsafe { self.ptr.as_ptr().sub(1) }
        }
    }
    #[inline]
    unsafe fn next_n(&self, offset: usize) -> Self {
        unsafe {
            let ptr = if mem::size_of::<T>() == 0 {
                (self.ptr.as_ptr() as usize + offset) as *mut T
            } else {
                self.ptr.as_ptr().sub(offset)
            };
            Self {
                ptr: NonNull::new_unchecked(ptr),
            }
        }
    }
    #[inline]
    unsafe fn drop(&self) {
        unsafe {
            self.as_ptr().drop_in_place();
        }
    }
    #[inline]
    unsafe fn write(&self, val: T) {
        unsafe {
            self.as_ptr().write(val);
        }
    }
    #[inline]
    unsafe fn as_ref<'a>(&self) -> &'a T {
        unsafe { &*self.as_ptr() }
    }
    #[inline]
    unsafe fn as_mut<'a>(&self) -> &'a mut T {
        unsafe { &mut *self.as_ptr() }
    }
}

impl<K, V> Bucket<(K, V)> {
    #[inline]
    pub unsafe fn as_pair_ref<'a>(&self) -> (&'a K, &'a V) {
        unsafe {
            let pair = &*self.as_ptr();
            (&pair.0, &pair.1)
        }
    }
}

/// A handle to a [SyncTable] with read access.
///
/// It is acquired either by a pin, or by exclusive access to the table.
pub struct Read<'a, K, V, S = DefaultHashBuilder> {
    table: &'a SyncTable<K, V, S>,
}

impl<K, V, S> Copy for Read<'_, K, V, S> {}
impl<K, V, S> Clone for Read<'_, K, V, S> {
    fn clone(&self) -> Self {
        Self { table: self.table }
    }
}

/// A handle to a [SyncTable] with write access.
pub struct Write<'a, K, V, S = DefaultHashBuilder> {
    table: &'a SyncTable<K, V, S>,
}

/// A handle to a [SyncTable] with write access protected by a lock.
pub struct LockedWrite<'a, K, V, S = DefaultHashBuilder> {
    table: Write<'a, K, V, S>,
    _guard: MutexGuard<'a, ()>,
}

impl<'a, K, V, S> Deref for LockedWrite<'a, K, V, S> {
    type Target = Write<'a, K, V, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl<'a, K, V, S> DerefMut for LockedWrite<'a, K, V, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

/// Default hash builder for [SyncTable].
pub type DefaultHashBuilder = RandomState;

/// A hash table with lock-free reads.
///
/// It is based on the table from the `hashbrown` crate.
pub struct SyncTable<K, V, S = DefaultHashBuilder> {
    hash_builder: S,

    current: AtomicPtr<TableInfo>,

    lock: Mutex<()>,

    old: UnsafeCell<Vec<Arc<DestroyTable<(K, V)>>>>,

    // Tell dropck that we own instances of K, V.
    marker: PhantomData<(K, V)>,
}

struct TableInfo {
    // Mask to get an index from a hash value. The value is one less than the
    // number of buckets in the table.
    bucket_mask: usize,

    // Number of elements that can be inserted before we need to grow the table
    growth_left: AtomicUsize,

    // Number of elements that has been removed from the table
    tombstones: AtomicUsize,
}

impl TableInfo {
    #[inline]
    fn num_ctrl_bytes(&self) -> usize {
        self.buckets() + Group::WIDTH
    }

    /// Returns the number of buckets in the table.
    #[inline]
    fn buckets(&self) -> usize {
        self.bucket_mask + 1
    }

    #[inline]
    fn items(&self) -> usize {
        // FIXME: May overflow and return wrong value.
        // TODO: Or will they synchronize due to the lock?
        // NO: A concurrent write / remove may happen which puts them out of sync?
        bucket_mask_to_capacity(self.bucket_mask)
            - self.growth_left.load(Ordering::Acquire)
            - self.tombstones.load(Ordering::Acquire)
    }

    /// Returns a pointer to a control byte.
    #[inline]
    unsafe fn ctrl(&self, index: usize) -> *mut u8 {
        unsafe {
            debug_assert!(index < self.num_ctrl_bytes());

            let info = Layout::new::<TableInfo>();
            let control = Layout::new::<Group>();
            let offset = info.extend(control).unwrap().1;

            let ctrl = (self as *const TableInfo as *mut u8).add(offset);

            ctrl.add(index)
        }
    }

    /// Sets a control byte, and possibly also the replicated control byte at
    /// the end of the array.
    #[inline]
    unsafe fn set_ctrl(&self, index: usize, ctrl: u8) {
        unsafe {
            // Replicate the first Group::WIDTH control bytes at the end of
            // the array without using a branch:
            // - If index >= Group::WIDTH then index == index2.
            // - Otherwise index2 == self.bucket_mask + 1 + index.
            //
            // The very last replicated control byte is never actually read because
            // we mask the initial index for unaligned loads, but we write it
            // anyways because it makes the set_ctrl implementation simpler.
            //
            // If there are fewer buckets than Group::WIDTH then this code will
            // replicate the buckets at the end of the trailing group. For example
            // with 2 buckets and a group size of 4, the control bytes will look
            // like this:
            //
            //     Real    |             Replicated
            // ---------------------------------------------
            // | [A] | [B] | [EMPTY] | [EMPTY] | [A] | [B] |
            // ---------------------------------------------
            let index2 = ((index.wrapping_sub(Group::WIDTH)) & self.bucket_mask) + Group::WIDTH;

            *self.ctrl(index) = ctrl;
            *self.ctrl(index2) = ctrl;
        }
    }

    /// Sets a control byte, and possibly also the replicated control byte at
    /// the end of the array. Same as set_ctrl, but uses release stores.
    #[inline]
    unsafe fn set_ctrl_release(&self, index: usize, ctrl: u8) {
        unsafe {
            let index2 = ((index.wrapping_sub(Group::WIDTH)) & self.bucket_mask) + Group::WIDTH;

            (*(self.ctrl(index) as *mut AtomicU8)).store(ctrl, Ordering::Release);
            (*(self.ctrl(index2) as *mut AtomicU8)).store(ctrl, Ordering::Release);
        }
    }

    /// Sets a control byte to the hash, and possibly also the replicated control byte at
    /// the end of the array.
    #[inline]
    unsafe fn set_ctrl_h2(&self, index: usize, hash: u64) {
        unsafe { self.set_ctrl(index, h2(hash)) }
    }

    #[inline]
    unsafe fn record_item_insert_at(&self, index: usize, hash: u64) {
        unsafe {
            self.growth_left.store(
                self.growth_left.load(Ordering::Relaxed) - 1,
                Ordering::Release,
            );
            self.set_ctrl_release(index, h2(hash));
        }
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting
    /// a new element and sets the hash for that slot.
    ///
    /// There must be at least 1 empty bucket in the table.
    #[inline]
    unsafe fn prepare_insert_slot(&self, hash: u64) -> (usize, u8) {
        unsafe {
            let index = self.find_insert_slot(hash);
            let old_ctrl = *self.ctrl(index);
            self.set_ctrl_h2(index, hash);
            (index, old_ctrl)
        }
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting
    /// a new element.
    ///
    /// There must be at least 1 empty bucket in the table.
    #[inline]
    unsafe fn find_insert_slot(&self, hash: u64) -> usize {
        unsafe {
            let mut probe_seq = self.probe_seq(hash);
            loop {
                let group = Group::load(self.ctrl(probe_seq.pos));
                if let Some(bit) = group.match_empty().lowest_set_bit() {
                    let result = (probe_seq.pos + bit) & self.bucket_mask;

                    return result;
                }
                probe_seq.move_next(self.bucket_mask);
            }
        }
    }

    /// Returns an iterator-like object for a probe sequence on the table.
    ///
    /// This iterator never terminates, but is guaranteed to visit each bucket
    /// group exactly once. The loop using `probe_seq` must terminate upon
    /// reaching a group containing an empty bucket.
    #[inline]
    unsafe fn probe_seq(&self, hash: u64) -> ProbeSeq {
        ProbeSeq {
            pos: h1(hash) & self.bucket_mask,
            stride: 0,
        }
    }
}

#[repr(transparent)]
struct TableRef<T> {
    data: NonNull<TableInfo>,

    marker: PhantomData<*mut T>,
}

impl<T> Copy for TableRef<T> {}
impl<T> Clone for TableRef<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            marker: self.marker,
        }
    }
}

impl<T> TableRef<T> {
    #[inline]
    fn empty() -> Self {
        #[repr(C)]
        struct EmptyTable {
            info: TableInfo,
            control_bytes: [Group; 1],
        }

        static EMPTY: EmptyTable = EmptyTable {
            info: TableInfo {
                bucket_mask: 0,
                growth_left: AtomicUsize::new(0),
                tombstones: AtomicUsize::new(0),
            },
            control_bytes: [Group::EMPTY; 1],
        };

        Self {
            data: unsafe { NonNull::new_unchecked(&EMPTY as *const EmptyTable as *mut TableInfo) },
            marker: PhantomData,
        }
    }

    #[inline]
    fn layout(bucket_count: usize) -> Result<(Layout, usize), LayoutError> {
        let buckets = Layout::new::<T>().repeat(bucket_count)?.0;
        let info = Layout::new::<TableInfo>();
        let control =
            Layout::array::<u8>(bucket_count + Group::WIDTH)?.align_to(mem::align_of::<Group>())?;
        let (total, info_offset) = buckets.extend(info)?;
        Ok((total.extend(control)?.0, info_offset))
    }

    #[inline]
    fn allocate(bucket_count: usize) -> Self {
        let (layout, info_offset) = Self::layout(bucket_count).expect("capacity overflow");

        let ptr: NonNull<u8> = Global
            .allocate(layout)
            .map(|ptr| ptr.cast())
            .unwrap_or_else(|_| handle_alloc_error(layout));

        let info =
            unsafe { NonNull::new_unchecked(ptr.as_ptr().add(info_offset) as *mut TableInfo) };

        let mut result = Self {
            data: info,
            marker: PhantomData,
        };

        unsafe {
            *result.info_mut() = TableInfo {
                bucket_mask: bucket_count - 1,
                growth_left: AtomicUsize::new(bucket_mask_to_capacity(bucket_count - 1)),
                tombstones: AtomicUsize::new(0),
            };

            result
                .info()
                .ctrl(0)
                .write_bytes(EMPTY, result.info().num_ctrl_bytes());
        }

        result
    }

    #[inline]
    unsafe fn free(self) {
        unsafe {
            if self.info().bucket_mask > 0 {
                if mem::needs_drop::<T>() {
                    for item in self.iter() {
                        item.drop();
                    }
                }

                // TODO: Document why we don't need to account for padding when adjusting
                // the pointer. Sizes allowed can't result in padding?
                Global.deallocate(
                    NonNull::new_unchecked(self.bucket_before_first() as *mut u8),
                    Self::layout(self.info().buckets()).unwrap_unchecked().0,
                )
            }
        }
    }

    fn from_maybe_empty_iter<
        S,
        I: Iterator<Item = T>,
        H: Fn(&S, &T) -> u64,
        const CHECK_LEN: bool,
    >(
        iter: I,
        iter_size: usize,
        capacity: usize,
        hash_builder: &S,
        hasher: H,
    ) -> TableRef<T> {
        if iter_size == 0 {
            TableRef::empty()
        } else {
            let buckets =
                capacity_to_buckets(cmp::max(iter_size, capacity)).expect("capacity overflow");
            unsafe {
                TableRef::from_iter::<_, _, _, CHECK_LEN>(iter, buckets, hash_builder, hasher)
            }
        }
    }

    /// Allocates a new table and fills it with the content of an iterator
    unsafe fn from_iter<S, I: Iterator<Item = T>, H: Fn(&S, &T) -> u64, const CHECK_LEN: bool>(
        iter: I,
        buckets: usize,
        hash_builder: &S,
        hasher: H,
    ) -> TableRef<T> {
        unsafe {
            let mut new_table = TableRef::allocate(buckets);

            let mut guard = guard(Some(new_table), |new_table| {
                new_table.map(|new_table| new_table.free());
            });

            let mut growth_left = *new_table.info_mut().growth_left.get_mut();

            // Copy all elements to the new table.
            for item in iter {
                if CHECK_LEN && growth_left == 0 {
                    break;
                }

                // This may panic.
                let hash = hasher(hash_builder, &item);

                // We can use a simpler version of insert() here since:
                // - we know there is enough space in the table.
                // - all elements are unique.
                let (index, _) = new_table.info().prepare_insert_slot(hash);

                new_table.bucket(index).write(item);

                growth_left -= 1;
            }

            *new_table.info_mut().growth_left.get_mut() = growth_left;

            *guard = None;

            new_table
        }
    }

    unsafe fn info(&self) -> &TableInfo {
        unsafe { self.data.as_ref() }
    }

    unsafe fn info_mut(&mut self) -> &mut TableInfo {
        unsafe { self.data.as_mut() }
    }

    #[inline]
    unsafe fn bucket_before_first(&self) -> *mut T {
        unsafe { self.bucket_past_last().sub(self.info().buckets()) }
    }

    #[inline]
    unsafe fn bucket_past_last(&self) -> *mut T {
        self.data.as_ptr() as *mut T
    }

    /// Returns a pointer to an element in the table.
    #[inline]
    unsafe fn bucket(&self, index: usize) -> Bucket<T> {
        unsafe {
            debug_assert!(index < self.info().buckets());

            Bucket {
                ptr: NonNull::new_unchecked(self.bucket_past_last().sub(index)),
            }
        }
    }

    /// Returns an iterator over every element in the table. It is up to
    /// the caller to ensure that the table outlives the `RawIterRange`.
    /// Because we cannot make the `next` method unsafe on the `RawIterRange`
    /// struct, we have to make the `iter` method unsafe.
    #[inline]
    unsafe fn iter(&self) -> RawIterRange<T> {
        unsafe {
            let data = Bucket {
                ptr: NonNull::new_unchecked(self.bucket_past_last()),
            };
            RawIterRange::new(self.info().ctrl(0), data, self.info().buckets())
        }
    }

    /// Searches for an element in the table.
    #[inline]
    unsafe fn search<R>(
        &self,
        hash: u64,
        mut eq: impl FnMut(&T) -> bool,
        mut stop: impl FnMut(&Group, &ProbeSeq) -> Option<R>,
    ) -> Result<(usize, Bucket<T>), R> {
        unsafe {
            let h2_hash = h2(hash);
            let mut probe_seq = self.info().probe_seq(hash);
            let mut group = Group::load(self.info().ctrl(probe_seq.pos));
            let mut bitmask = group.match_byte(h2_hash).into_iter();

            loop {
                if let Some(bit) = bitmask.next() {
                    let index = (probe_seq.pos + bit) & self.info().bucket_mask;

                    let bucket = self.bucket(index);
                    let elm = self.bucket(index).as_ref();
                    if likely(eq(elm)) {
                        return Ok((index, bucket));
                    }

                    // Look at the next bit
                    continue;
                }

                if let Some(stop) = stop(&group, &probe_seq) {
                    return Err(stop);
                }

                probe_seq.move_next(self.info().bucket_mask);
                group = Group::load(self.info().ctrl(probe_seq.pos));
                bitmask = group.match_byte(h2_hash).into_iter();
            }
        }
    }

    /// Searches for an element in the table.
    #[inline]
    unsafe fn find(&self, hash: u64, eq: impl FnMut(&T) -> bool) -> Option<(usize, Bucket<T>)> {
        unsafe {
            self.search(hash, eq, |group, _| {
                if likely(group.match_empty().any_bit_set()) {
                    Some(())
                } else {
                    None
                }
            })
            .ok()
        }
    }

    /// Searches for an element in the table.
    #[inline]
    unsafe fn find_potential(
        &self,
        hash: u64,
        eq: impl FnMut(&T) -> bool,
    ) -> Result<(usize, Bucket<T>), PotentialSlot<'static>> {
        unsafe {
            self.search(hash, eq, |group, probe_seq| {
                let bit = group.match_empty().lowest_set_bit();
                if likely(bit.is_some()) {
                    let index = (probe_seq.pos + bit.unwrap_unchecked()) & self.info().bucket_mask;
                    Some(PotentialSlot {
                        table_info: &*self.data.as_ptr(),
                        index,
                    })
                } else {
                    None
                }
            })
        }
    }
}

impl<T: Clone> TableRef<T> {
    /// Allocates a new table of a different size and moves the contents of the
    /// current table into it.
    unsafe fn clone_table<S>(
        &self,
        hash_builder: &S,
        buckets: usize,
        hasher: impl Fn(&S, &T) -> u64,
    ) -> TableRef<T> {
        unsafe {
            debug_assert!(buckets >= self.info().buckets());

            TableRef::from_iter::<_, _, _, false>(
                self.iter().map(|bucket| bucket.as_ref().clone()),
                buckets,
                hash_builder,
                hasher,
            )
        }
    }
}

struct DestroyTable<T> {
    table: TableRef<T>,
    lock: Mutex<bool>,
}

unsafe impl<T> Sync for DestroyTable<T> {}
unsafe impl<T: Send> Send for DestroyTable<T> {}

impl<T> DestroyTable<T> {
    unsafe fn run(&self) {
        unsafe {
            let mut status = self.lock.lock();
            if !*status {
                *status = true;
                self.table.free();
            }
        }
    }
}

unsafe impl<#[may_dangle] K, #[may_dangle] V, S> Drop for SyncTable<K, V, S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.current().free();
            for table in self.old.get_mut() {
                table.run();
            }
        }
    }
}

unsafe impl<K: Send, V: Send, S: Send> Send for SyncTable<K, V, S> {}
unsafe impl<K: Sync, V: Sync, S: Sync> Sync for SyncTable<K, V, S> {}

impl<K, V, S: Default> Default for SyncTable<K, V, S> {
    #[inline]
    fn default() -> Self {
        Self::new_with(Default::default(), 0)
    }
}

impl<K, V> SyncTable<K, V, DefaultHashBuilder> {
    /// Creates an empty [SyncTable].
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<K, V, S> SyncTable<K, V, S> {
    /// Creates an empty [SyncTable] with the specified capacity, using `hash_builder`
    /// to hash the elements or keys.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    #[inline]
    pub fn new_with(hash_builder: S, capacity: usize) -> Self {
        Self {
            hash_builder,
            current: AtomicPtr::new(
                if capacity > 0 {
                    TableRef::<(K, V)>::allocate(
                        capacity_to_buckets(capacity).expect("capacity overflow"),
                    )
                } else {
                    TableRef::empty()
                }
                .data
                .as_ptr(),
            ),
            old: UnsafeCell::new(Vec::new()),
            marker: PhantomData,
            lock: Mutex::new(()),
        }
    }

    /// Returns a reference to the table's `BuildHasher`.
    #[inline]
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Gets a reference to the underlying mutex that protects writes.
    #[inline]
    pub fn mutex(&self) -> &Mutex<()> {
        &self.lock
    }

    /// Creates a [Read] handle from a pinned region.
    ///
    /// Use [crate::collect::pin] to get a `Pin` instance.
    #[inline]
    pub fn read<'a>(&'a self, pin: Pin<'a>) -> Read<'a, K, V, S> {
        let _pin = pin;
        Read { table: self }
    }

    /// Creates a [Write] handle without checking for exclusive access.
    ///
    /// # Safety
    /// It's up to the caller to ensure only one thread writes to the vector at a time.
    #[inline]
    pub unsafe fn unsafe_write(&self) -> Write<'_, K, V, S> {
        Write { table: self }
    }

    /// Creates a [Write] handle from a mutable reference.
    #[inline]
    pub fn write(&mut self) -> Write<'_, K, V, S> {
        Write { table: self }
    }

    /// Creates a [LockedWrite] handle by taking the underlying mutex that protects writes.
    #[inline]
    pub fn lock(&self) -> LockedWrite<'_, K, V, S> {
        LockedWrite {
            table: Write { table: self },
            _guard: self.lock.lock(),
        }
    }

    /// Creates a [LockedWrite] handle from a guard protecting the underlying mutex that protects writes.
    #[inline]
    pub fn lock_from_guard<'a>(&'a self, guard: MutexGuard<'a, ()>) -> LockedWrite<'a, K, V, S> {
        // Verify that we are target of the guard
        assert_eq!(
            &self.lock as *const _,
            MutexGuard::mutex(&guard) as *const _
        );

        LockedWrite {
            table: Write { table: self },
            _guard: guard,
        }
    }

    #[inline]
    fn current(&self) -> TableRef<(K, V)> {
        TableRef {
            data: unsafe { NonNull::new_unchecked(self.current.load(Ordering::Acquire)) },
            marker: PhantomData,
        }
    }
}

impl<K, V, S: BuildHasher> SyncTable<K, V, S> {
    #[inline]
    fn unwrap_hash<Q>(&self, key: &Q, hash: Option<u64>) -> u64
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash,
    {
        hash.unwrap_or_else(|| self.hash_key(key))
    }

    /// Hashes a key with the table's hasher.
    #[inline]
    pub fn hash_key<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash,
    {
        make_insert_hash(&self.hash_builder, key)
    }

    /// Gets a mutable reference to an element in the table.
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q, hash: Option<u64>) -> Option<(&mut K, &mut V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Eq + Hash,
    {
        let hash = self.unwrap_hash(key, hash);

        unsafe {
            self.current().find(hash, eq(key)).map(|(_, bucket)| {
                let pair = bucket.as_mut();
                (&mut pair.0, &mut pair.1)
            })
        }
    }
}

impl<'a, K, V, S: BuildHasher> Read<'a, K, V, S> {
    /// Gets a reference to an element in the table or a potential location
    /// where that element could be.
    #[inline]
    pub fn get_potential<Q>(
        self,
        key: &Q,
        hash: Option<u64>,
    ) -> Result<(&'a K, &'a V), PotentialSlot<'a>>
    where
        K: Borrow<Q>,
        Q: ?Sized + Eq + Hash,
    {
        let hash = self.table.unwrap_hash(key, hash);

        unsafe {
            match self.table.current().find_potential(hash, eq(key)) {
                Ok((_, bucket)) => Ok(bucket.as_pair_ref()),
                Err(slot) => Err(slot),
            }
        }
    }

    /// Gets a reference to an element in the table.
    #[inline]
    pub fn get<Q>(self, key: &Q, hash: Option<u64>) -> Option<(&'a K, &'a V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Eq + Hash,
    {
        let hash = self.table.unwrap_hash(key, hash);

        unsafe {
            self.table
                .current()
                .find(hash, eq(key))
                .map(|(_, bucket)| bucket.as_pair_ref())
        }
    }
}

impl<'a, K, V, S> Read<'a, K, V, S> {
    /// Gets a reference to an element in the table with a custom equality function.
    #[inline]
    pub fn get_with_eq(
        self,
        hash: u64,
        mut eq: impl FnMut(&K, &V) -> bool,
    ) -> Option<(&'a K, &'a V)> {
        unsafe {
            self.table
                .current()
                .find(hash, |(k, v)| eq(k, v))
                .map(|(_, bucket)| bucket.as_pair_ref())
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    #[inline]
    pub fn capacity(self) -> usize {
        unsafe { bucket_mask_to_capacity(self.table.current().info().bucket_mask) }
    }

    /// Returns the number of elements in the table.
    #[inline]
    pub fn len(self) -> usize {
        unsafe { self.table.current().info().items() }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    #[inline]
    pub fn iter(self) -> Iter<'a, K, V> {
        let table = self.table.current();

        // Here we tie the lifetime of self to the iter.
        unsafe {
            Iter {
                inner: table.iter(),
                marker: PhantomData,
            }
        }
    }

    #[allow(dead_code)]
    fn dump(self)
    where
        K: std::fmt::Debug,
        V: std::fmt::Debug,
    {
        let table = self.table.current();

        println!("Table dump:");

        unsafe {
            for i in 0..table.info().buckets() {
                if *table.info().ctrl(i) == EMPTY {
                    println!("[#{:x}]", i);
                } else {
                    println!(
                        "[#{:x}, ${:x}, {:?}]",
                        i,
                        *table.info().ctrl(i),
                        table.bucket(i).as_ref()
                    );
                }
            }
        }

        println!("--------");
    }
}

impl<K: Hash + Clone, V: Clone, S: Clone + BuildHasher> Clone for SyncTable<K, V, S> {
    fn clone(&self) -> SyncTable<K, V, S> {
        pin(|_pin| {
            let table = self.current();

            unsafe {
                let buckets = table.info().buckets();

                SyncTable {
                    hash_builder: self.hash_builder.clone(),
                    current: AtomicPtr::new(
                        if buckets > 0 {
                            table.clone_table(&self.hash_builder, buckets, hasher)
                        } else {
                            TableRef::empty()
                        }
                        .data
                        .as_ptr(),
                    ),
                    old: UnsafeCell::new(Vec::new()),
                    marker: PhantomData,
                    lock: Mutex::new(()),
                }
            }
        })
    }
}

impl<'a, K, V, S> Write<'a, K, V, S> {
    /// Creates a [Read] handle which gives access to read operations.
    #[inline]
    pub fn read(&self) -> Read<'_, K, V, S> {
        Read { table: self.table }
    }

    /// Returns a reference to the table's `BuildHasher`.
    #[inline]
    pub fn hasher(&self) -> &'a S {
        &self.table.hash_builder
    }
}

impl<'a, K: Send, V: Send + Clone, S: BuildHasher> Write<'a, K, V, S> {
    /// Removes an element from the table, and returns a reference to it if was present.
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q, hash: Option<u64>) -> Option<(&'a K, &'a V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Eq + Hash,
    {
        let hash = self.table.unwrap_hash(key, hash);

        let table = self.table.current();

        unsafe {
            table.find(hash, eq(key)).map(|(index, bucket)| {
                debug_assert!(is_full(*table.info().ctrl(index)));
                table.info().set_ctrl_release(index, DELETED);
                table.info().tombstones.store(
                    table.info().tombstones.load(Ordering::Relaxed) + 1,
                    Ordering::Release,
                );
                bucket.as_pair_ref()
            })
        }
    }
}

impl<'a, K: Hash + Eq + Send + Clone, V: Send + Clone, S: BuildHasher> Write<'a, K, V, S> {
    /// Inserts a element into the table.
    /// Returns `false` if it already exists and doesn't update the value.
    #[inline]
    pub fn insert(&mut self, key: K, value: V, hash: Option<u64>) -> bool {
        let hash = self.table.unwrap_hash(&key, hash);

        let mut table = self.table.current();

        unsafe {
            if unlikely(table.info().growth_left.load(Ordering::Relaxed) == 0) {
                table = self.expand_by_one();
            }

            match table.find_potential(hash, eq(&key)) {
                Ok(_) => false,
                Err(slot) => {
                    slot.insert_new_unchecked(self, key, value, Some(hash));
                    true
                }
            }
        }
    }
}

impl<'a, K: Hash + Send + Clone, V: Send + Clone, S: BuildHasher> Write<'a, K, V, S> {
    /// Inserts a new element into the table, and returns a reference to it.
    ///
    /// This does not check if the given element already exists in the table.
    #[inline]
    pub fn insert_new(&mut self, key: K, value: V, hash: Option<u64>) -> (&'a K, &'a V) {
        let hash = self.table.unwrap_hash(&key, hash);

        let mut table = self.table.current();

        unsafe {
            if unlikely(table.info().growth_left.load(Ordering::Relaxed) == 0) {
                table = self.expand_by_one();
            }

            let index = table.info().find_insert_slot(hash);

            let bucket = table.bucket(index);
            bucket.write((key, value));

            table.info().record_item_insert_at(index, hash);

            bucket.as_pair_ref()
        }
    }

    /// Reserve room for one more element.
    #[inline]
    pub fn reserve_one(&mut self) {
        let table = self.table.current();

        if unlikely(unsafe { table.info().growth_left.load(Ordering::Relaxed) } == 0) {
            self.expand_by_one();
        }
    }

    #[cold]
    #[inline(never)]
    fn expand_by_one(&mut self) -> TableRef<(K, V)> {
        self.expand_by(1)
    }

    /// Out-of-line slow path for `reserve` and `try_reserve`.
    fn expand_by(&mut self, additional: usize) -> TableRef<(K, V)> {
        let table = self.table.current();

        // Avoid `Option::ok_or_else` because it bloats LLVM IR.
        let new_items = match unsafe { table.info() }.items().checked_add(additional) {
            Some(new_items) => new_items,
            None => panic!("capacity overflow"),
        };

        let full_capacity = bucket_mask_to_capacity(unsafe { table.info().bucket_mask });

        let new_capacity = usize::max(new_items, full_capacity + 1);

        unsafe {
            debug_assert!(table.info().items() <= new_capacity);
        }

        let buckets = capacity_to_buckets(new_capacity).expect("capacity overflow");

        let new_table = unsafe { table.clone_table(&self.table.hash_builder, buckets, hasher) };

        self.replace_table(new_table);

        new_table
    }
}

impl<K: Hash + Send, V: Send, S: BuildHasher> Write<'_, K, V, S> {
    fn replace_table(&mut self, new_table: TableRef<(K, V)>) {
        let table = self.table.current();

        self.table
            .current
            .store(new_table.data.as_ptr(), Ordering::Release);

        let destroy = Arc::new(DestroyTable {
            table,
            lock: Mutex::new(false),
        });

        unsafe {
            (*self.table.old.get()).push(destroy.clone());

            collect::defer_unchecked(move || destroy.run());
        }
    }

    /// Replaces the content of the table with the content of the iterator.
    /// All the elements must be unique.
    /// `capacity` specifies the new capacity if it's greater than the length of the iterator.
    #[inline]
    pub fn replace<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I, capacity: usize) {
        let iter = iter.into_iter();

        let table = if let Some(max) = iter.size_hint().1 {
            TableRef::from_maybe_empty_iter::<_, _, _, true>(
                iter,
                max,
                capacity,
                &self.table.hash_builder,
                hasher,
            )
        } else {
            let elements: Vec<_> = iter.collect();
            let len = elements.len();
            TableRef::from_maybe_empty_iter::<_, _, _, false>(
                elements.into_iter(),
                len,
                capacity,
                &self.table.hash_builder,
                hasher,
            )
        };

        self.replace_table(table);
    }
}

impl<K: Eq + Hash + Clone + Send, V: Clone + Send, S: BuildHasher + Default> FromIterator<(K, V)>
    for SyncTable<K, V, S>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut map = Self::new_with(S::default(), iter.size_hint().0);
        {
            let mut write = map.write();
            iter.for_each(|(k, v)| {
                write.insert(k, v, None);
            });
        }
        map
    }
}

/// Represents where a value would be if inserted.
///
/// Created by calling `get_potential` on [Read]. All methods on this type takes a table handle
/// and this must be a handle to the same table `get_potential` was called on. Operations also must
/// be on the same element as given to `get_potential`. The operations have a fast path for when
/// the element is still missing.
#[derive(Copy, Clone)]
pub struct PotentialSlot<'a> {
    table_info: &'a TableInfo,
    index: usize,
}

impl<'a> PotentialSlot<'a> {
    #[inline]
    unsafe fn is_empty<T>(self, table: TableRef<T>) -> bool {
        unsafe {
            let table_info = table.info() as *const TableInfo;
            let index = self.index;

            // Check that we are still looking at the same table,
            // otherwise our index could be out of date due to expansion
            // or a `replace` call.
            table_info == (self.table_info as *const TableInfo)
                && *self.table_info.ctrl(index) == EMPTY
        }
    }

    /// Gets a reference to an element in the table.
    #[inline]
    pub fn get<'b, Q, K, V, S: BuildHasher>(
        self,
        table: Read<'b, K, V, S>,
        key: &Q,
        hash: Option<u64>,
    ) -> Option<(&'b K, &'b V)>
    where
        K: Borrow<Q>,
        Q: ?Sized + Eq + Hash,
    {
        unsafe {
            if likely(self.is_empty(table.table.current())) {
                return None;
            }
        }

        cold_path(|| table.get(key, hash))
    }

    /// Returns a new up-to-date potential slot.
    /// This can be useful if there could have been insertions since the slot was derived
    /// and you want to use `try_insert_new` or `insert_new_unchecked`.
    #[inline]
    pub fn refresh<Q, K, V, S: BuildHasher>(
        self,
        table: Read<'a, K, V, S>,
        key: &Q,
        hash: Option<u64>,
    ) -> Result<(&'a K, &'a V), PotentialSlot<'a>>
    where
        K: Borrow<Q>,
        Q: ?Sized + Eq + Hash,
    {
        unsafe {
            if likely(self.is_empty(table.table.current())) {
                return Err(self);
            }
        }

        cold_path(|| table.get_potential(key, hash))
    }

    #[inline]
    unsafe fn insert<'b, K, V>(
        self,
        table: TableRef<(K, V)>,
        value: (K, V),
        hash: u64,
    ) -> (&'b K, &'b V) {
        unsafe {
            let index = self.index;
            let bucket = table.bucket(index);
            bucket.write(value);

            table.info().record_item_insert_at(index, hash);

            let pair = bucket.as_ref();
            (&pair.0, &pair.1)
        }
    }

    /// Inserts a new element into the table, and returns a reference to it.
    ///
    /// This does not check if the given element already exists in the table.
    #[inline]
    pub fn insert_new<'b, K: Hash + Send + Clone, V: Send + Clone, S: BuildHasher>(
        self,
        table: &mut Write<'b, K, V, S>,
        key: K,
        value: V,
        hash: Option<u64>,
    ) -> (&'b K, &'b V) {
        let hash = table.table.unwrap_hash(&key, hash);

        unsafe {
            let table = table.table.current();

            if likely(self.is_empty(table) && table.info().growth_left.load(Ordering::Relaxed) > 0)
            {
                debug_assert_eq!(self.index, table.info().find_insert_slot(hash));

                return self.insert(table, (key, value), hash);
            }
        }

        cold_path(|| table.insert_new(key, value, Some(hash)))
    }

    /// Inserts a new element into the table, and returns a reference to it.
    /// Returns [None] if the potential slot is taken by other insertions or if
    /// there's no spare capacity in the table.
    ///
    /// This does not check if the given element already exists in the table.
    #[inline]
    pub fn try_insert_new<'b, K: Hash, V, S: BuildHasher>(
        self,
        table: &mut Write<'b, K, V, S>,
        key: K,
        value: V,
        hash: Option<u64>,
    ) -> Option<(&'b K, &'b V)> {
        let hash = table.table.unwrap_hash(&key, hash);

        unsafe {
            let table = table.table.current();

            if likely(self.is_empty(table) && table.info().growth_left.load(Ordering::Relaxed) > 0)
            {
                Some(self.insert(table, (key, value), hash))
            } else {
                None
            }
        }
    }

    /// Inserts a new element into the table, and returns a reference to it.
    ///
    /// This does not check if the given element already exists in the table.
    ///
    /// # Safety
    /// Derived refers here to either a value returned by `get_potential` or by a `refresh` call.
    ///
    /// The following conditions must hold for this function to be safe:
    /// - `table` must be the same table that `self` is derived from.
    /// - `hash`, `key` and `value` must match the value used when `self` was derived.
    /// - There must not have been any insertions or `replace` calls to the table since `self`
    ///   was derived.
    #[inline]
    pub unsafe fn insert_new_unchecked<'b, K: Hash, V, S: BuildHasher>(
        self,
        table: &mut Write<'b, K, V, S>,
        key: K,
        value: V,
        hash: Option<u64>,
    ) -> (&'b K, &'b V) {
        unsafe {
            let hash = table.table.unwrap_hash(&key, hash);

            let table = table.table.current();

            debug_assert!(self.is_empty(table));
            debug_assert!(table.info().growth_left.load(Ordering::Relaxed) > 0);

            self.insert(table, (key, value), hash)
        }
    }
}

/// An iterator over the entries of a `HashMap`.
///
/// This `struct` is created by the [`iter`] method on [`HashMap`]. See its
/// documentation for more.
///
/// [`iter`]: struct.HashMap.html#method.iter
/// [`HashMap`]: struct.HashMap.html
pub struct Iter<'a, K, V> {
    inner: RawIterRange<(K, V)>,
    marker: PhantomData<&'a (K, V)>,
}

impl<K, V> Clone for Iter<'_, K, V> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner
            .next()
            .map(|bucket| unsafe { bucket.as_pair_ref() })
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

/// Returns the maximum effective capacity for the given bucket mask, taking
/// the maximum load factor into account.
#[inline]
fn bucket_mask_to_capacity(bucket_mask: usize) -> usize {
    if bucket_mask < 8 {
        // For tables with 1/2/4/8 buckets, we always reserve one empty slot.
        // Keep in mind that the bucket mask is one less than the bucket count.
        bucket_mask
    } else {
        // For larger tables we reserve 12.5% of the slots as empty.
        ((bucket_mask + 1) / 8) * 7
    }
}

/// Returns the number of buckets needed to hold the given number of items,
/// taking the maximum load factor into account.
///
/// Returns `None` if an overflow occurs.
// Workaround for emscripten bug emscripten-core/emscripten-fastcomp#258
#[cfg_attr(target_os = "emscripten", inline(never))]
#[cfg_attr(not(target_os = "emscripten"), inline)]
fn capacity_to_buckets(cap: usize) -> Option<usize> {
    debug_assert_ne!(cap, 0);

    // For small tables we require at least 1 empty bucket so that lookups are
    // guaranteed to terminate if an element doesn't exist in the table.
    let result = if cap < 8 {
        // We don't bother with a table size of 2 buckets since that can only
        // hold a single element. Instead we skip directly to a 4 bucket table
        // which can hold 3 elements.
        if cap < 4 { 4 } else { 8 }
    } else {
        // Otherwise require 1/8 buckets to be empty (87.5% load)
        //
        // Be careful when modifying this, calculate_layout relies on the
        // overflow check here.
        let adjusted_cap = cap.checked_mul(8)? / 7;

        // Any overflows will have been caught by the checked_mul. Also, any
        // rounding errors from the division above will be cleaned up by
        // next_power_of_two (which can't overflow because of the previous divison).
        adjusted_cap.next_power_of_two()
    };

    // Have at least the number of buckets required to fill a group.
    // This avoids logic to deal with control bytes not associated with a bucket
    // when batch processing a group.
    Some(usize::max(result, Group::WIDTH))
}

/// Primary hash function, used to select the initial bucket to probe from.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn h1(hash: u64) -> usize {
    // On 32-bit platforms we simply ignore the higher hash bits.
    hash as usize
}

/// Secondary hash function, saved in the low 7 bits of the control byte.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn h2(hash: u64) -> u8 {
    // Grab the top 7 bits of the hash. While the hash is normally a full 64-bit
    // value, some hash functions (such as FxHash) produce a usize result
    // instead, which means that the top 32 bits are 0 on 32-bit platforms.
    let hash_len = usize::min(mem::size_of::<usize>(), mem::size_of::<u64>());
    let top7 = hash >> (hash_len * 8 - 7);
    (top7 & 0x7f) as u8 // truncation
}

/// Control byte value for an empty bucket.
const EMPTY: u8 = 0b1111_1111;

/// Control byte value for a deleted bucket.
const DELETED: u8 = 0b1000_0000;

/// Checks whether a control byte represents a full bucket (top bit is clear).
#[inline]
fn is_full(ctrl: u8) -> bool {
    ctrl & 0x80 == 0
}

/// Probe sequence based on triangular numbers, which is guaranteed (since our
/// table size is a power of two) to visit every group of elements exactly once.
///
/// A triangular probe has us jump by 1 more group every time. So first we
/// jump by 1 group (meaning we just continue our linear scan), then 2 groups
/// (skipping over 1 group), then 3 groups (skipping over 2 groups), and so on.
///
/// Proof that the probe will visit every group in the table:
/// <https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/>
struct ProbeSeq {
    pos: usize,
    stride: usize,
}

impl ProbeSeq {
    #[inline]
    fn move_next(&mut self, bucket_mask: usize) {
        // We should have found an empty bucket by now and ended the probe.
        debug_assert!(
            self.stride <= bucket_mask,
            "Went past end of probe sequence"
        );

        self.stride += Group::WIDTH;
        self.pos += self.stride;
        self.pos &= bucket_mask;
    }
}

/// Iterator over a sub-range of a table.
struct RawIterRange<T> {
    // Mask of full buckets in the current group. Bits are cleared from this
    // mask as each element is processed.
    current_group: BitMask,

    // Pointer to the buckets for the current group.
    data: Bucket<T>,

    // Pointer to the next group of control bytes,
    // Must be aligned to the group size.
    next_ctrl: *const u8,

    // Pointer one past the last control byte of this range.
    end: *const u8,
}

impl<T> RawIterRange<T> {
    /// Returns a `RawIterRange` covering a subset of a table.
    ///
    /// The control byte address must be aligned to the group size.
    #[inline]
    unsafe fn new(ctrl: *const u8, data: Bucket<T>, len: usize) -> Self {
        unsafe {
            debug_assert_ne!(len, 0);
            debug_assert_eq!(ctrl as usize % Group::WIDTH, 0);
            let end = ctrl.add(len);

            // Load the first group and advance ctrl to point to the next group
            let current_group = Group::load_aligned(ctrl).match_full();
            let next_ctrl = ctrl.add(Group::WIDTH);

            Self {
                current_group,
                data,
                next_ctrl,
                end,
            }
        }
    }
}

// We make raw iterators unconditionally Send and Sync, and let the PhantomData
// in the actual iterator implementations determine the real Send/Sync bounds.
unsafe impl<T> Send for RawIterRange<T> {}
unsafe impl<T> Sync for RawIterRange<T> {}

impl<T> Clone for RawIterRange<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            next_ctrl: self.next_ctrl,
            current_group: self.current_group,
            end: self.end,
        }
    }
}

impl<T> Iterator for RawIterRange<T> {
    type Item = Bucket<T>;

    #[inline]
    fn next(&mut self) -> Option<Bucket<T>> {
        unsafe {
            loop {
                if let Some(index) = self.current_group.lowest_set_bit() {
                    self.current_group = self.current_group.remove_lowest_bit();
                    return Some(self.data.next_n(index));
                }

                if self.next_ctrl >= self.end {
                    return None;
                }

                // We might read past self.end up to the next group boundary,
                // but this is fine because it only occurs on tables smaller
                // than the group size where the trailing control bytes are all
                // EMPTY. On larger tables self.end is guaranteed to be aligned
                // to the group size (since tables are power-of-two sized).
                self.current_group = Group::load_aligned(self.next_ctrl).match_full();
                self.data = self.data.next_n(Group::WIDTH);
                self.next_ctrl = self.next_ctrl.add(Group::WIDTH);
            }
        }
    }
}

impl<T> FusedIterator for RawIterRange<T> {}

/// Get a suitable shard index from a hash when sharding the hash table.
///
/// `shards` must be a power of 2.
#[inline]
pub fn shard_index_by_hash(hash: u64, shards: usize) -> usize {
    assert!(shards.is_power_of_two());
    let shard_bits = shards - 1;

    let hash_len = mem::size_of::<usize>();
    // Ignore the top 7 bits as hashbrown uses these and get the next SHARD_BITS highest bits.
    // hashbrown also uses the lowest bits, so we can't use those
    let bits = (hash >> (hash_len * 8 - 7 - shard_bits)) as usize;
    bits % shards
}
