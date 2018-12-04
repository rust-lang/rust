use alloc::alloc::{alloc, dealloc, handle_alloc_error};
use core::alloc::Layout;
use core::hint;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem;
use core::mem::ManuallyDrop;
use core::ops::Range;
use core::ptr::NonNull;
use scopeguard::guard;
use CollectionAllocErr;

// Branch prediction hint. This is currently only available on nightly but it
// consistently improves performance by 10-15%.
#[cfg(feature = "nightly")]
use core::intrinsics::{likely, unlikely};
#[cfg(not(feature = "nightly"))]
#[inline]
fn likely(b: bool) -> bool {
    b
}
#[cfg(not(feature = "nightly"))]
#[inline]
fn unlikely(b: bool) -> bool {
    b
}

#[cfg(feature = "nightly")]
#[inline]
unsafe fn offset_from<T>(to: *const T, from: *const T) -> usize {
    to.offset_from(from) as usize
}
#[cfg(not(feature = "nightly"))]
#[inline]
unsafe fn offset_from<T>(to: *const T, from: *const T) -> usize {
    (to as usize - from as usize) / mem::size_of::<T>()
}

// Use the SSE2 implementation if possible: it allows us to scan 16 buckets at
// once instead of 8. We don't bother with AVX since it would require runtime
// dispatch and wouldn't gain us much anyways: the probability of finding a
// match drops off drastically after the first few buckets.
//
// I attempted an implementation on ARM using NEON instructions, but it turns
// out that most NEON instructions have multi-cycle latency, which in the end
// outweighs any gains over the generic implementation.
#[cfg(all(
    target_feature = "sse2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[path = "sse2.rs"]
mod imp;
#[cfg(not(all(
    target_feature = "sse2",
    any(target_arch = "x86", target_arch = "x86_64")
)))]
#[path = "generic.rs"]
mod imp;

mod bitmask;

use self::bitmask::BitMask;
use self::imp::Group;

/// Whether memory allocation errors should return an error or abort.
enum Fallibility {
    Fallible,
    Infallible,
}

impl Fallibility {
    /// Error to return on capacity overflow.
    #[inline]
    fn capacity_overflow(&self) -> CollectionAllocErr {
        match *self {
            Fallibility::Fallible => CollectionAllocErr::CapacityOverflow,
            Fallibility::Infallible => panic!("Hash table capacity overflow"),
        }
    }

    /// Error to return on allocation error.
    #[inline]
    fn alloc_err(&self, layout: Layout) -> CollectionAllocErr {
        match *self {
            Fallibility::Fallible => CollectionAllocErr::AllocErr,
            Fallibility::Infallible => handle_alloc_error(layout),
        }
    }
}

/// Control byte value for an empty bucket.
const EMPTY: u8 = 0b11111111;

/// Control byte value for a deleted bucket.
const DELETED: u8 = 0b10000000;

/// Checks whether a control byte represents a full bucket (top bit is clear).
#[inline]
fn is_full(ctrl: u8) -> bool {
    ctrl & 0x80 == 0
}

/// Checks whether a control byte represents a special value (top bit is set).
#[inline]
fn is_special(ctrl: u8) -> bool {
    ctrl & 0x80 != 0
}

/// Checks whether a special control value is EMPTY (just check 1 bit).
#[inline]
fn special_is_empty(ctrl: u8) -> bool {
    debug_assert!(is_special(ctrl));
    ctrl & 0x01 != 0
}

/// Primary hash function, used to select the initial bucket to probe from.
#[inline]
fn h1(hash: u64) -> usize {
    hash as usize
}

/// Secondary hash function, saved in the low 7 bits of the control byte.
#[inline]
fn h2(hash: u64) -> u8 {
    // Grab the top 7 bits of the hash. While the hash is normally a full 64-bit
    // value, some hash functions (such as FxHash) produce a usize result
    // instead, which means that the top 32 bits are 0 on 32-bit platforms.
    let hash_len = usize::min(mem::size_of::<usize>(), mem::size_of::<u64>());
    let top7 = hash >> (hash_len * 8 - 7);
    (top7 & 0x7f) as u8
}

/// Probe sequence based on triangular numbers, which is guaranteed (since our
/// table size is a power of two) to visit every group of elements exactly once.
struct ProbeSeq {
    mask: usize,
    offset: usize,
    index: usize,
}

impl Iterator for ProbeSeq {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        // We should have found an empty bucket by now and ended the probe.
        debug_assert!(self.index <= self.mask, "Went past end of probe sequence");

        let result = self.offset;
        self.index += Group::WIDTH;
        self.offset += self.index;
        self.offset &= self.mask;
        Some(result)
    }
}

/// Returns the number of buckets needed to hold the given number of items,
/// taking the maximum load factor into account.
///
/// Returns `None` if an overflow occurs.
#[inline]
fn capacity_to_buckets(cap: usize) -> Option<usize> {
    let adjusted_cap = if cap < 8 {
        // Need at least 1 free bucket on small tables
        cap + 1
    } else {
        // Otherwise require 1/8 buckets to be empty (87.5% load)
        //
        // Be careful when modifying this, calculate_layout relies on the
        // overflow check here.
        cap.checked_mul(8)? / 7
    };

    // Any overflows will have been caught by the checked_mul.
    Some(adjusted_cap.next_power_of_two())
}

/// Returns the maximum effective capacity for the given bucket mask, taking
/// the maximum load factor into account.
#[inline]
fn bucket_mask_to_capacity(bucket_mask: usize) -> usize {
    if bucket_mask < 8 {
        bucket_mask
    } else {
        ((bucket_mask + 1) / 8) * 7
    }
}

// Returns a Layout which describes the allocation required for a hash table,
// and the offset of the buckets in the allocation.
///
/// Returns `None` if an overflow occurs.
#[inline]
#[cfg(feature = "nightly")]
fn calculate_layout<T>(buckets: usize) -> Option<(Layout, usize)> {
    debug_assert!(buckets.is_power_of_two());

    // Array of buckets
    let data = Layout::array::<T>(buckets).ok()?;

    // Array of control bytes. This must be aligned to the group size.
    //
    // We add `Group::WIDTH` control bytes at the end of the array which
    // replicate the bytes at the start of the array and thus avoids the need to
    // perform bounds-checking while probing.
    //
    // There is no possible overflow here since buckets is a power of two and
    // Group::WIDTH is a small number.
    let ctrl = unsafe { Layout::from_size_align_unchecked(buckets + Group::WIDTH, Group::WIDTH) };

    ctrl.extend(data).ok()
}

// Returns a Layout which describes the allocation required for a hash table,
// and the offset of the buckets in the allocation.
#[inline]
#[cfg(not(feature = "nightly"))]
fn calculate_layout<T>(buckets: usize) -> Option<(Layout, usize)> {
    debug_assert!(buckets.is_power_of_two());

    // Manual layout calculation since Layout methods are not yet stable.
    let data_align = usize::max(mem::align_of::<T>(), Group::WIDTH);
    let data_offset = (buckets + Group::WIDTH).checked_add(data_align - 1)? & !(data_align - 1);
    let len = data_offset.checked_add(mem::size_of::<T>().checked_mul(buckets)?)?;

    Some((
        unsafe { Layout::from_size_align_unchecked(len, data_align) },
        data_offset,
    ))
}

/// A reference to a hash table bucket containing a `T`.
pub struct Bucket<T> {
    ptr: NonNull<T>,
}

// This Send impl is needed for rayon support. This is safe since Bucket is
// never exposed in a public API.
unsafe impl<T> Send for Bucket<T> {}

impl<T> Clone for Bucket<T> {
    #[inline]
    fn clone(&self) -> Self {
        Bucket { ptr: self.ptr }
    }
}

impl<T> Bucket<T> {
    #[inline]
    unsafe fn from_ptr(ptr: *const T) -> Self {
        Bucket {
            ptr: NonNull::new_unchecked(ptr as *mut T),
        }
    }
    #[inline]
    pub unsafe fn drop(&self) {
        self.ptr.as_ptr().drop_in_place();
    }
    #[inline]
    pub unsafe fn read(&self) -> T {
        self.ptr.as_ptr().read()
    }
    #[inline]
    pub unsafe fn write(&self, val: T) {
        self.ptr.as_ptr().write(val);
    }
    #[inline]
    pub unsafe fn as_ref<'a>(&self) -> &'a T {
        &*self.ptr.as_ptr()
    }
    #[inline]
    pub unsafe fn as_mut<'a>(&self) -> &'a mut T {
        &mut *self.ptr.as_ptr()
    }
}

/// A raw hash table with an unsafe API.
pub struct RawTable<T> {
    ctrl: NonNull<u8>,
    bucket_mask: usize,
    data: NonNull<T>,
    items: usize,
    growth_left: usize,
}

impl<T> RawTable<T> {
    /// Creates a new empty hash table without allocating any memory.
    ///
    /// In effect this returns a table with exactly 1 bucket. However we can
    /// leave the data pointer dangling since that bucket is never written to
    /// due to our load factor forcing us to always have at least 1 free bucket.
    #[inline]
    pub fn new() -> RawTable<T> {
        RawTable {
            data: NonNull::dangling(),
            ctrl: NonNull::from(&Group::static_empty()[0]),
            bucket_mask: 0,
            items: 0,
            growth_left: 0,
        }
    }

    /// Allocates a new hash table with the given number of buckets.
    ///
    /// The control bytes are left uninitialized.
    #[inline]
    unsafe fn new_uninitialized(
        buckets: usize,
        fallability: Fallibility,
    ) -> Result<RawTable<T>, CollectionAllocErr> {
        let (layout, data_offset) =
            calculate_layout::<T>(buckets).ok_or_else(|| fallability.capacity_overflow())?;
        let ctrl = NonNull::new(alloc(layout)).ok_or_else(|| fallability.alloc_err(layout))?;
        let data = NonNull::new_unchecked(ctrl.as_ptr().add(data_offset) as *mut T);
        Ok(RawTable {
            data,
            ctrl,
            bucket_mask: buckets - 1,
            items: 0,
            growth_left: bucket_mask_to_capacity(buckets - 1),
        })
    }

    /// Attempts to allocate a new hash table with at least enough capacity
    /// for inserting the given number of elements without reallocating.
    fn try_with_capacity(
        capacity: usize,
        fallability: Fallibility,
    ) -> Result<RawTable<T>, CollectionAllocErr> {
        if capacity == 0 {
            Ok(RawTable::new())
        } else {
            unsafe {
                let buckets =
                    capacity_to_buckets(capacity).ok_or_else(|| fallability.capacity_overflow())?;
                let result = RawTable::new_uninitialized(buckets, fallability)?;
                result
                    .ctrl(0)
                    .write_bytes(EMPTY, result.buckets() + Group::WIDTH);

                // If we have fewer buckets than the group width then we need to
                // fill in unused spaces in the trailing control bytes with
                // DELETED entries. See the comments in set_ctrl.
                if result.buckets() < Group::WIDTH {
                    result
                        .ctrl(result.buckets())
                        .write_bytes(DELETED, Group::WIDTH - result.buckets());
                }

                Ok(result)
            }
        }
    }

    /// Allocates a new hash table with at least enough capacity for inserting
    /// the given number of elements without reallocating.
    pub fn with_capacity(capacity: usize) -> RawTable<T> {
        RawTable::try_with_capacity(capacity, Fallibility::Infallible)
            .unwrap_or_else(|_| unsafe { hint::unreachable_unchecked() })
    }

    /// Deallocates the table without dropping any entries.
    #[inline]
    unsafe fn free_buckets(&mut self) {
        let (layout, _) =
            calculate_layout::<T>(self.buckets()).unwrap_or_else(|| hint::unreachable_unchecked());
        dealloc(self.ctrl.as_ptr(), layout);
    }

    /// Returns the index of a bucket from a `Bucket`.
    #[inline]
    unsafe fn bucket_index(&self, bucket: &Bucket<T>) -> usize {
        offset_from(bucket.ptr.as_ptr(), self.data.as_ptr())
    }

    /// Returns a pointer to a control byte.
    #[inline]
    unsafe fn ctrl(&self, index: usize) -> *mut u8 {
        debug_assert!(index < self.buckets() + Group::WIDTH);
        self.ctrl.as_ptr().add(index)
    }

    /// Returns a pointer to an element in the table.
    #[inline]
    pub unsafe fn bucket(&self, index: usize) -> Bucket<T> {
        debug_assert_ne!(self.bucket_mask, 0);
        debug_assert!(index < self.buckets());
        Bucket::from_ptr(self.data.as_ptr().add(index))
    }

    /// Erases an element from the table without dropping it.
    #[inline]
    pub unsafe fn erase_no_drop(&mut self, item: &Bucket<T>) {
        let index = self.bucket_index(item);
        let index_before = index.wrapping_sub(Group::WIDTH) & self.bucket_mask;
        let empty_before = Group::load(self.ctrl(index_before)).match_empty();
        let empty_after = Group::load(self.ctrl(index)).match_empty();

        // If we are inside a continuous block of Group::WIDTH full or deleted
        // cells then a probe window may have seen a full block when trying to
        // insert. We therefore need to keep that block non-empty so that
        // lookups will continue searching to the next probe window.
        let ctrl = if empty_before.leading_zeros() + empty_after.trailing_zeros() >= Group::WIDTH {
            DELETED
        } else {
            self.growth_left += 1;
            EMPTY
        };
        self.set_ctrl(index, ctrl);
        self.items -= 1;
    }

    /// Returns an iterator for a probe sequence on the table.
    ///
    /// This iterator never terminates, but is guaranteed to visit each bucket
    /// group exactly once.
    #[inline]
    fn probe_seq(&self, hash: u64) -> ProbeSeq {
        ProbeSeq {
            mask: self.bucket_mask,
            offset: h1(hash) & self.bucket_mask,
            index: 0,
        }
    }

    /// Sets a control byte, and possibly also the replicated control byte at
    /// the end of the array.
    #[inline]
    unsafe fn set_ctrl(&self, index: usize, ctrl: u8) {
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
        // -------------------------------------------------
        // | [A] | [B] | [DELETED] | [DELETED] | [A] | [B] |
        // -------------------------------------------------
        let index2 = ((index.wrapping_sub(Group::WIDTH)) & self.bucket_mask) + Group::WIDTH;

        *self.ctrl(index) = ctrl;
        *self.ctrl(index2) = ctrl;
    }

    /// Searches for an empty or deleted bucket which is suitable for inserting
    /// a new element.
    ///
    /// There must be at least 1 empty bucket in the table.
    #[inline]
    fn find_insert_slot(&self, hash: u64) -> usize {
        for pos in self.probe_seq(hash) {
            unsafe {
                let group = Group::load(self.ctrl(pos));
                if let Some(bit) = group.match_empty_or_deleted().lowest_set_bit() {
                    let result = (pos + bit) & self.bucket_mask;

                    // In tables smaller than the group width, trailing control
                    // bytes outside the range of the table are filled with
                    // DELETED entries. These will unfortunately trigger a
                    // match, but once masked will point to a full bucket that
                    // is already occupied. We detect this situation here and
                    // perform a second scan starting at the begining of the
                    // table. This second scan is guaranteed to find an empty
                    // slot (due to the load factor) before hitting the trailing
                    // control bytes (containing DELETED).
                    if unlikely(is_full(*self.ctrl(result))) {
                        debug_assert!(self.bucket_mask < Group::WIDTH);
                        debug_assert_ne!(pos, 0);
                        return Group::load_aligned(self.ctrl(0))
                            .match_empty_or_deleted()
                            .lowest_set_bit_nonzero();
                    } else {
                        return result;
                    }
                }
            }
        }

        // probe_seq never returns.
        unreachable!();
    }

    /// Marks all table buckets as empty without dropping their contents.
    #[inline]
    pub fn clear_no_drop(&mut self) {
        if self.bucket_mask != 0 {
            unsafe {
                self.ctrl(0)
                    .write_bytes(EMPTY, self.buckets() + Group::WIDTH);
            }
        }
        self.items = 0;
        self.growth_left = bucket_mask_to_capacity(self.bucket_mask);
    }

    /// Removes all elements from the table without freeing the backing memory.
    #[inline]
    pub fn clear(&mut self) {
        // Ensure that the table is reset even if one of the drops panic
        let self_ = guard(self, |self_| self_.clear_no_drop());

        if mem::needs_drop::<T>() {
            unsafe {
                for item in self_.iter() {
                    item.drop();
                }
            }
        }
    }

    /// Shrinks the table to fit `max(self.len(), min_size)` elements.
    #[inline]
    pub fn shrink_to(&mut self, min_size: usize, hasher: impl Fn(&T) -> u64) {
        let min_size = usize::max(self.items, min_size);
        if self.bucket_mask != 0 && bucket_mask_to_capacity(self.bucket_mask) >= min_size * 2 {
            self.resize(min_size, hasher, Fallibility::Infallible)
                .unwrap_or_else(|_| unsafe { hint::unreachable_unchecked() });
        }
    }

    /// Ensures that at least `additional` items can be inserted into the table
    /// without reallocation.
    #[inline]
    pub fn reserve(&mut self, additional: usize, hasher: impl Fn(&T) -> u64) {
        if additional > self.growth_left {
            self.reserve_rehash(additional, hasher, Fallibility::Infallible)
                .unwrap_or_else(|_| unsafe { hint::unreachable_unchecked() });
        }
    }

    /// Tries to ensure that at least `additional` items can be inserted into
    /// the table without reallocation.
    #[inline]
    pub fn try_reserve(
        &mut self,
        additional: usize,
        hasher: impl Fn(&T) -> u64,
    ) -> Result<(), CollectionAllocErr> {
        if additional > self.growth_left {
            self.reserve_rehash(additional, hasher, Fallibility::Fallible)
        } else {
            Ok(())
        }
    }

    /// Out-of-line slow path for `reserve` and `try_reserve`.
    #[cold]
    #[inline(never)]
    fn reserve_rehash(
        &mut self,
        additional: usize,
        hasher: impl Fn(&T) -> u64,
        fallability: Fallibility,
    ) -> Result<(), CollectionAllocErr> {
        let new_items = self
            .items
            .checked_add(additional)
            .ok_or_else(|| fallability.capacity_overflow())?;

        // Rehash in-place without re-allocating if we have plenty of spare
        // capacity that is locked up due to DELETED entries.
        if new_items < bucket_mask_to_capacity(self.bucket_mask) / 2 {
            self.rehash_in_place(hasher);
            Ok(())
        } else {
            self.resize(new_items, hasher, fallability)
        }
    }

    /// Rehashes the contents of the table in place (i.e. without changing the
    /// allocation).
    ///
    /// If `hasher` panics then some the table's contents may be lost.
    fn rehash_in_place(&mut self, hasher: impl Fn(&T) -> u64) {
        unsafe {
            // Bulk convert all full control bytes to DELETED, and all DELETED
            // control bytes to EMPTY. This effectively frees up all buckets
            // containing a DELETED entry.
            for i in (0..self.buckets()).step_by(Group::WIDTH) {
                let group = Group::load_aligned(self.ctrl(i));
                let group = group.convert_special_to_empty_and_full_to_deleted();
                group.store_aligned(self.ctrl(i));
            }

            // Fix up the trailing control bytes. See the comments in set_ctrl.
            if self.buckets() < Group::WIDTH {
                self.ctrl(0)
                    .copy_to(self.ctrl(Group::WIDTH), self.buckets());
                self.ctrl(self.buckets())
                    .write_bytes(DELETED, Group::WIDTH - self.buckets());
            } else {
                self.ctrl(0)
                    .copy_to(self.ctrl(self.buckets()), Group::WIDTH);
            }

            // If the hash function panics then properly clean up any elements
            // that we haven't rehashed yet. We unfortunately can't preserve the
            // element since we lost their hash and have no way of recovering it
            // without risking another panic.
            let mut guard = guard(self, |self_| {
                if mem::needs_drop::<T>() {
                    for i in 0..self_.buckets() {
                        if *self_.ctrl(i) == DELETED {
                            self_.set_ctrl(i, EMPTY);
                            self_.bucket(i).drop();
                            self_.items -= 1;
                        }
                    }
                }
                self_.growth_left = bucket_mask_to_capacity(self_.bucket_mask) - self_.items;
            });

            // At this point, DELETED elements are elements that we haven't
            // rehashed yet. Find them and re-insert them at their ideal
            // position.
            'outer: for i in 0..guard.buckets() {
                if *guard.ctrl(i) != DELETED {
                    continue;
                }
                'inner: loop {
                    // Hash the current item
                    let item = guard.bucket(i);
                    let hash = hasher(item.as_ref());

                    // Search for a suitable place to put it
                    let new_i = guard.find_insert_slot(hash);

                    // Probing works by scanning through all of the control
                    // bytes in groups, which may not be aligned to the group
                    // size. If both the new and old position fall within the
                    // same unaligned group, then there is no benefit in moving
                    // it and we can just continue to the next item.
                    let probe_index = |pos: usize| {
                        (pos.wrapping_sub(guard.probe_seq(hash).offset) & guard.bucket_mask)
                            / Group::WIDTH
                    };
                    if likely(probe_index(i) == probe_index(new_i)) {
                        guard.set_ctrl(i, h2(hash));
                        continue 'outer;
                    }

                    // We are moving the current item to a new position. Write
                    // our H2 to the control byte of the new position.
                    let prev_ctrl = *guard.ctrl(new_i);
                    guard.set_ctrl(new_i, h2(hash));

                    if prev_ctrl == EMPTY {
                        // If the target slot is empty, simply move the current
                        // element into the new slot and clear the old control
                        // byte.
                        guard.set_ctrl(i, EMPTY);
                        guard.bucket(new_i).write(item.read());
                        continue 'outer;
                    } else {
                        // If the target slot is occupied, swap the two elements
                        // and then continue processing the element that we just
                        // swapped into the old slot.
                        debug_assert_eq!(prev_ctrl, DELETED);
                        mem::swap(guard.bucket(new_i).as_mut(), item.as_mut());
                        continue 'inner;
                    }
                }
            }

            guard.growth_left = bucket_mask_to_capacity(guard.bucket_mask) - guard.items;
            mem::forget(guard);
        }
    }

    /// Allocates a new table of a different size and moves the contents of the
    /// current table into it.
    fn resize(
        &mut self,
        capacity: usize,
        hasher: impl Fn(&T) -> u64,
        fallability: Fallibility,
    ) -> Result<(), CollectionAllocErr> {
        unsafe {
            debug_assert!(self.items <= capacity);

            // Allocate and initialize the new table.
            let mut new_table = RawTable::try_with_capacity(capacity, fallability)?;
            new_table.growth_left -= self.items;
            new_table.items = self.items;

            // The hash function may panic, in which case we simply free the new
            // table without dropping any elements that may have been copied into
            // it.
            let mut new_table = guard(ManuallyDrop::new(new_table), |new_table| {
                if new_table.bucket_mask != 0 {
                    new_table.free_buckets();
                }
            });

            // Copy all elements to the new table.
            for item in self.iter() {
                // This may panic.
                let hash = hasher(item.as_ref());

                // We can use a simpler version of insert() here since there are no
                // DELETED entries.
                let index = new_table.find_insert_slot(hash);
                new_table.set_ctrl(index, h2(hash));
                new_table.bucket(index).write(item.read());
            }

            // We successfully copied all elements without panicking. Now replace
            // self with the new table. The old table will have its memory freed but
            // the items will not be dropped (since they have been moved into the
            // new table).
            mem::swap(self, &mut new_table);

            Ok(())
        }
    }

    /// Inserts a new element into the table.
    ///
    /// This does not check if the given element already exists in the table.
    #[inline]
    pub fn insert(&mut self, hash: u64, value: T, hasher: impl Fn(&T) -> u64) -> Bucket<T> {
        self.reserve(1, hasher);

        unsafe {
            let index = self.find_insert_slot(hash);
            let bucket = self.bucket(index);

            // If we are replacing a DELETED entry then we don't need to update
            // the load counter.
            let old_ctrl = *self.ctrl(index);
            self.growth_left -= special_is_empty(old_ctrl) as usize;

            self.set_ctrl(index, h2(hash));
            bucket.write(value);
            self.items += 1;
            bucket
        }
    }

    /// Searches for an element in the table.
    #[inline]
    pub fn find(&self, hash: u64, mut eq: impl FnMut(&T) -> bool) -> Option<Bucket<T>> {
        unsafe {
            for pos in self.probe_seq(hash) {
                let group = Group::load(self.ctrl(pos));
                for bit in group.match_byte(h2(hash)) {
                    let index = (pos + bit) & self.bucket_mask;
                    let bucket = self.bucket(index);
                    if likely(eq(bucket.as_ref())) {
                        return Some(bucket);
                    }
                }
                if likely(group.match_empty().any_bit_set()) {
                    return None;
                }
            }
        }

        // probe_seq never returns.
        unreachable!();
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the table might be able to hold
    /// more, but is guaranteed to be able to hold at least this many.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.items + self.growth_left
    }

    /// Returns the number of elements in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.items
    }

    /// Returns the number of buckets in the table.
    #[inline]
    pub fn buckets(&self) -> usize {
        self.bucket_mask + 1
    }

    /// Returns an iterator over every element in the table. It is up to
    /// the caller to ensure that the `RawTable` outlives the `RawIter`.
    /// Because we cannot make the `next` method unsafe on the `RawIter`
    /// struct, we have to make the `iter` method unsafe.
    #[inline]
    pub unsafe fn iter(&self) -> RawIter<T> {
        RawIter {
            iter: RawIterRange::new(self.ctrl.as_ptr(), self.data.as_ptr(), 0..self.buckets()),
            items: self.items,
        }
    }

    /// Returns an iterator which removes all elements from the table without
    /// freeing the memory. It is up to the caller to ensure that the `RawTable`
    /// outlives the `RawDrain`. Because we cannot make the `next` method unsafe
    /// on the `RawDrain`, we have to make the `drain` method unsafe.
    #[inline]
    pub unsafe fn drain(&mut self) -> RawDrain<T> {
        RawDrain {
            iter: self.iter(),
            table: NonNull::from(self),
            _marker: PhantomData,
        }
    }

    /// Converts the table into a raw allocation. The contents of the table
    /// should be dropped using a `RawIter` before freeing the allocation.
    #[inline]
    pub fn into_alloc(self) -> Option<(NonNull<u8>, Layout)> {
        let alloc = if self.bucket_mask != 0 {
            let (layout, _) = calculate_layout::<T>(self.buckets())
                .unwrap_or_else(|| unsafe { hint::unreachable_unchecked() });
            Some((self.ctrl.cast(), layout))
        } else {
            None
        };
        mem::forget(self);
        alloc
    }
}

unsafe impl<T> Send for RawTable<T> where T: Send {}
unsafe impl<T> Sync for RawTable<T> where T: Sync {}

impl<T: Clone> Clone for RawTable<T> {
    fn clone(&self) -> Self {
        if self.bucket_mask == 0 {
            Self::new()
        } else {
            unsafe {
                let mut new_table = ManuallyDrop::new(
                    Self::new_uninitialized(self.buckets(), Fallibility::Infallible)
                        .unwrap_or_else(|_| hint::unreachable_unchecked()),
                );

                // Copy the control bytes unchanged. We do this in a single pass
                self.ctrl(0)
                    .copy_to_nonoverlapping(new_table.ctrl(0), self.buckets() + Group::WIDTH);

                {
                    // The cloning of elements may panic, in which case we need
                    // to make sure we drop only the elements that have been
                    // cloned so far.
                    let mut guard = guard((0, &mut new_table), |(index, new_table)| {
                        if mem::needs_drop::<T>() {
                            for i in 0..=*index {
                                if is_full(*new_table.ctrl(i)) {
                                    new_table.bucket(i).drop();
                                }
                            }
                        }
                        new_table.free_buckets();
                    });

                    for from in self.iter() {
                        let index = self.bucket_index(&from);
                        let to = guard.1.bucket(index);
                        to.write(from.as_ref().clone());

                        // Update the index in case we need to unwind.
                        guard.0 = index;
                    }

                    // Successfully cloned all items, no need to clean up.
                    mem::forget(guard);
                }

                // Return the newly created table.
                new_table.items = self.items;
                new_table.growth_left = self.growth_left;
                ManuallyDrop::into_inner(new_table)
            }
        }
    }
}

#[cfg(feature = "nightly")]
unsafe impl<#[may_dangle] T> Drop for RawTable<T> {
    #[inline]
    fn drop(&mut self) {
        if self.bucket_mask != 0 {
            unsafe {
                if mem::needs_drop::<T>() {
                    for item in self.iter() {
                        item.drop();
                    }
                }
                self.free_buckets();
            }
        }
    }
}
#[cfg(not(feature = "nightly"))]
impl<T> Drop for RawTable<T> {
    #[inline]
    fn drop(&mut self) {
        if self.bucket_mask != 0 {
            unsafe {
                if mem::needs_drop::<T>() {
                    for item in self.iter() {
                        item.drop();
                    }
                }
                self.free_buckets();
            }
        }
    }
}

impl<T> IntoIterator for RawTable<T> {
    type Item = T;
    type IntoIter = RawIntoIter<T>;

    #[inline]
    fn into_iter(self) -> RawIntoIter<T> {
        unsafe {
            let iter = self.iter();
            let alloc = self.into_alloc();
            RawIntoIter { iter, alloc }
        }
    }
}

/// Iterator over a a sub-range of a table. Unlike `RawIter` this iterator does
/// not track an item count.
pub struct RawIterRange<T> {
    // Using *const here for covariance
    data: *const T,
    ctrl: *const u8,
    current_group: BitMask,
    end: *const u8,
}

impl<T> RawIterRange<T> {
    /// Returns a `RawIterRange` covering a subset of a table.
    ///
    /// The start offset must be aligned to the group width.
    #[inline]
    unsafe fn new(
        input_ctrl: *const u8,
        input_data: *const T,
        range: Range<usize>,
    ) -> RawIterRange<T> {
        debug_assert_eq!(range.start % Group::WIDTH, 0);
        let ctrl = input_ctrl.add(range.start);
        let data = input_data.add(range.start);
        let end = input_ctrl.add(range.end);
        debug_assert_eq!(offset_from(end, ctrl), range.end - range.start);
        let current_group = Group::load_aligned(ctrl).match_empty_or_deleted().invert();
        RawIterRange {
            data,
            ctrl,
            current_group,
            end,
        }
    }

    /// Splits a `RawIterRange` into two halves.
    ///
    /// This will fail if the total range is smaller than the group width.
    #[inline]
    #[cfg(feature = "rayon")]
    pub fn split(mut self) -> (RawIterRange<T>, Option<RawIterRange<T>>) {
        unsafe {
            let len = offset_from(self.end, self.ctrl);
            debug_assert!(len.is_power_of_two());
            if len <= Group::WIDTH {
                (self, None)
            } else {
                debug_assert_eq!(len % (Group::WIDTH * 2), 0);
                let mid = len / 2;
                let tail = RawIterRange::new(self.ctrl, self.data, mid..len);
                debug_assert_eq!(self.data.add(mid), tail.data);
                debug_assert_eq!(self.end, tail.end);
                self.end = self.ctrl.add(mid);
                debug_assert_eq!(self.end, tail.ctrl);
                (self, Some(tail))
            }
        }
    }
}

unsafe impl<T> Send for RawIterRange<T> where T: Send {}
unsafe impl<T> Sync for RawIterRange<T> where T: Sync {}

impl<T> Clone for RawIterRange<T> {
    #[inline]
    fn clone(&self) -> Self {
        RawIterRange {
            data: self.data,
            ctrl: self.ctrl,
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
                    return Some(Bucket::from_ptr(self.data.add(index)));
                }

                self.ctrl = self.ctrl.add(Group::WIDTH);
                if self.ctrl >= self.end {
                    return None;
                }

                self.data = self.data.add(Group::WIDTH);
                self.current_group = Group::load_aligned(self.ctrl)
                    .match_empty_or_deleted()
                    .invert();
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't have an item count, so just guess based on the range size.
        (0, Some(unsafe { offset_from(self.end, self.ctrl) }))
    }
}

impl<T> FusedIterator for RawIterRange<T> {}

/// Iterator which returns a raw pointer to every full bucket in the table.
pub struct RawIter<T> {
    pub iter: RawIterRange<T>,
    items: usize,
}

impl<T> Clone for RawIter<T> {
    #[inline]
    fn clone(&self) -> Self {
        RawIter {
            iter: self.iter.clone(),
            items: self.items,
        }
    }
}

impl<T> Iterator for RawIter<T> {
    type Item = Bucket<T>;

    #[inline]
    fn next(&mut self) -> Option<Bucket<T>> {
        match self.iter.next() {
            Some(b) => {
                self.items -= 1;
                Some(b)
            }
            None => {
                // We don't check against items == 0 here to allow the
                // compiler to optimize away the item count entirely if the
                // iterator length is never queried.
                debug_assert_eq!(self.items, 0);
                None
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.items, Some(self.items))
    }
}

impl<T> ExactSizeIterator for RawIter<T> {}
impl<T> FusedIterator for RawIter<T> {}

/// Iterator which consumes a table and returns elements.
pub struct RawIntoIter<T> {
    iter: RawIter<T>,
    alloc: Option<(NonNull<u8>, Layout)>,
}

impl<'a, T> RawIntoIter<T> {
    #[inline]
    pub fn iter(&self) -> RawIter<T> {
        self.iter.clone()
    }
}

unsafe impl<T> Send for RawIntoIter<T> where T: Send {}
unsafe impl<T> Sync for RawIntoIter<T> where T: Sync {}

impl<T> Drop for RawIntoIter<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Drop all remaining elements
            if mem::needs_drop::<T>() {
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

impl<T> Iterator for RawIntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe { Some(self.iter.next()?.read()) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> ExactSizeIterator for RawIntoIter<T> {}
impl<T> FusedIterator for RawIntoIter<T> {}

/// Iterator which consumes elements without freeing the table storage.
pub struct RawDrain<'a, T: 'a> {
    iter: RawIter<T>,

    // We don't use a &'a RawTable<T> because we want RawDrain to be covariant
    // over 'a.
    table: NonNull<RawTable<T>>,
    _marker: PhantomData<&'a RawTable<T>>,
}

impl<'a, T> RawDrain<'a, T> {
    #[inline]
    pub fn iter(&self) -> RawIter<T> {
        self.iter.clone()
    }
}

unsafe impl<'a, T> Send for RawDrain<'a, T> where T: Send {}
unsafe impl<'a, T> Sync for RawDrain<'a, T> where T: Sync {}

impl<'a, T> Drop for RawDrain<'a, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Ensure that the table is reset even if one of the drops panic
            let _guard = guard(self.table, |table| table.as_mut().clear_no_drop());

            // Drop all remaining elements
            if mem::needs_drop::<T>() {
                while let Some(item) = self.iter.next() {
                    item.drop();
                }
            }
        }
    }
}

impl<'a, T> Iterator for RawDrain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            let item = self.iter.next()?;

            // Mark the item as DELETED in the table and decrement the item
            // counter. We don't need to use the full delete algorithm like
            // erase_no_drop since we will just clear the control bytes when
            // the RawDrain is dropped.
            let index = self.table.as_ref().bucket_index(&item);
            *self.table.as_mut().ctrl(index) = DELETED;
            self.table.as_mut().items -= 1;

            Some(item.read())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for RawDrain<'a, T> {}
impl<'a, T> FusedIterator for RawDrain<'a, T> {}
