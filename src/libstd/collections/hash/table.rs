// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::BucketState::*;

use borrow::{Borrow, BorrowMut};
use clone::Clone;
use cmp;
use hash::{Hash, Hasher};
use iter::Iterator;
use marker::{Copy, Sized, PhantomData};
use mem::{self, min_align_of, size_of};
use ops::Drop;
use option::Option::{self, Some, None};
use ptr::{self, Unique};
use rt::heap::{allocate, deallocate};
use collections::hash_state::HashState;
use core::nonzero::NonZero;

static EMPTY: u8 = 0;

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, keys, and values.
///
/// This design uses less memory and is a lot faster than the naive
/// `Vec<Option<u64, K, V>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Essential invariants of this structure:
///
///   - if t.hashes[i] == None, then `Bucket::new(&t, i).raw`
///     points to 'undefined' contents. Don't read from it. This invariant is
///     enforced outside this module with the `EmptyBucket`, `FullBucket`,
///     and `SafeHash` types.
///
///   - An `EmptyBucket` is only constructed at an index with
///     a hash of None.
///
///   - A `FullBucket` is only constructed at an index with a hash.
///
///   - A `SafeHash` is only constructed for non-zero hash. We get
///     around hashes of zero by changing them to 0x8000_0000_0000_0000,
///     which will likely map to the same bucket, while not being confused
///     with "empty".
///
///   - Both "arrays represented by pointers" have the same length:
///     `capacity`. This is set at creation and never changes. The arrays
///     are unzipped to be more cache aware (scanning through 8 hashes brings
///     in at most 2 cache lines, since they're all right beside each other).
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<u64, K, V>>`.
#[unsafe_no_drop_flag]
pub struct RawTable<K, V> {
    capacity: usize,
    size:     usize,
    // NB. The table will probably need manual impls of Send and Sync if this
    // field ever changes.
    middle: Unique<(K, V)>,
}

struct RawBucket<K, V> {
    hash: *mut Option<SafeHash>,
    kval: *mut (K, V),
}

pub struct Bucket<K, V, M, S = bucket::EmptyOrFull> {
    raw: RawBucket<K, V>,
    idx: usize,
    range: usize,
    table: M,
    marker: PhantomData<S>,

}

impl<K, V> Copy for RawBucket<K, V> {}
impl<K, V> Clone for RawBucket<K, V> {
    fn clone(&self) -> RawBucket<K, V> { *self }
}

impl<K, V, M: Copy, S> Copy for Bucket<K, V, M, S> where M: Borrow<RawTable<K, V>> {}
impl<K, V, M: Copy, S> Clone for Bucket<K, V, M, S> where M: Borrow<RawTable<K, V>> {
    fn clone(&self) -> Bucket<K, V, M, S> { *self }
}

mod bucket {
    pub enum Empty {}
    pub enum Full {}
    pub enum EmptyOrFull {}
}

pub type EmptyBucket<K, V, M> = Bucket<K, V, M, bucket::Empty>;
pub type EmptyBucketImm<'t, K, V> = EmptyBucket<K, V, &'t RawTable<K, V>>;
pub type EmptyBucketMut<'t, K, V> = EmptyBucket<K, V, &'t mut RawTable<K, V>>;

pub type FullBucket<K, V, M> = Bucket<K, V, M, bucket::Full>;
pub type FullBucketImm<'t, K, V> = FullBucket<K, V, &'t RawTable<K, V>>;
pub type FullBucketMut<'t, K, V> = FullBucket<K, V, &'t mut RawTable<K, V>>;

pub enum BucketState<K, V, M> {
    Empty(EmptyBucket<K, V, M>),
    Full(FullBucket<K, V, M>),
}

// A GapThenFull encapsulates the state of two consecutive buckets at once.
// The first bucket, called the gap, is known to be empty.
// The second bucket is full.
struct GapThenFull<K, V, M> {
    gap: EmptyBucket<K, V, ()>,
    full: FullBucket<K, V, M>,
}

/// A hash that is not zero, since we use a hash of zero to represent empty
/// buckets.
pub type SafeHash = NonZero<u64>;

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
pub fn make_hash<T: ?Sized, S>(hash_state: &S, t: &T) -> SafeHash
    where T: Hash, S: HashState
{
    let mut state = hash_state.hasher();
    t.hash(&mut state);
    // We need to avoid 0 in order to prevent collisions with
    // EMPTY_HASH. We can maintain our precious uniform distribution
    // of initial indexes by unconditionally setting the MSB,
    // effectively reducing 64-bits hashes to 63 bits.
    unsafe { NonZero::new(0x8000_0000_0000_0000 | state.finish()) }
}

// `read` casts a `*mut SafeHash` to a `*mut Option<SafeHash>`. Since we
// statically ensure that a `FullBucket` points to an index with a non-zero
// hash, and a `SafeHash` is NonZero, this is safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `Option<SafeHash>`. If you need to change the nullability of `SafeHash`,
// some functions need to be modified to no longer assume this.
#[test]
fn can_alias_safehash_as_option() {
    assert_eq!(size_of::<SafeHash>(), size_of::<Option<SafeHash>>())
}

impl<K, V> RawBucket<K, V> {
    unsafe fn offset(self, count: isize) -> RawBucket<K, V> {
        RawBucket {
            hash: self.hash.offset(count),
            kval: self.kval.offset(count),
        }
    }
}

// It is safe to access the table through any number of buckets as long
// as operations on the outer bucket `Bucket` can't invalidate inner `Bucket`s.
impl<K, V, M, S> Borrow<RawTable<K, V>> for Bucket<K, V, M, S>
    where M: Borrow<RawTable<K, V>>
{
    fn borrow(&self) -> &RawTable<K, V> {
        self.table.borrow().borrow()
    }
}

impl<K, V, M, S> BorrowMut<RawTable<K, V>> for Bucket<K, V, M, S>
    where M: BorrowMut<RawTable<K, V>>
{
    fn borrow_mut(&mut self) -> &mut RawTable<K, V> {
        self.table.borrow_mut().borrow_mut()
    }
}

/// `Put` is implemented for types which provide access to a table and cannot be invalidated
///  by filling a bucket. A similar implementation for `Take` is possible.
pub trait Put {}
impl<K, V> Put for RawTable<K, V> {}
impl<'t, K, V> Put for &'t mut RawTable<K, V> {}
impl<K, V, M: Put> Put for Bucket<K, V, M> {}
impl<K, V, M: Put> Put for FullBucket<K, V, M> {}

// Buckets hold references to the table.
impl<K, V, M, S> Bucket<K, V, M, S> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
    /// Get the raw index.
    pub fn index(&self) -> usize {
        self.idx
    }
}

impl<K, V, M> Bucket<K, V, M> where M: Borrow<RawTable<K, V>> {
    pub fn new(table: M, ib_index: usize) -> Option<Bucket<K, V, M>> {
        unsafe {
            let capacity = table.borrow().capacity();
            if capacity == 0 {
                None
            } else {
                let idx = ib_index & (capacity - 1);
                let bucket: Bucket<K, V, M> = Bucket {
                    raw: table.borrow().first_bucket_raw().offset(idx as isize),
                    idx: idx,
                    range: capacity,
                    table: table,
                    marker: PhantomData,
                };
                Some(bucket.state_cast())
            }
        }
    }

    pub fn raw_full_buckets(table: M) -> RawFullBuckets<K, V, M> {
        let first_bucket_raw = table.borrow().first_bucket_raw();
        RawFullBuckets {
            raw: first_bucket_raw,
            hashes_end: unsafe {
                first_bucket_raw.hash.offset(table.borrow().capacity as isize)
            },
            table: table,
        }
    }

    /// Reads a bucket at a given index, returning an enum indicating whether
    /// it's initialized or not. You need to match on this enum to get
    /// the appropriate types to call most of the other functions in
    /// this module.
    pub fn peek(self) -> BucketState<K, V, M> {
        match unsafe { *self.raw.hash } {
            None => Empty(self.state_cast()),
            _ => Full(self.state_cast()),
        }
    }

    /// Modifies the bucket pointer in place to make it point to the next slot.
    pub fn next(&mut self) {
        self.idx += 1;

        let dist = if self.idx & (self.range - 1) == 0 {
            1 - self.range as isize
        } else {
            1
        };

        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }
}

impl<K, V, M, S> Bucket<K, V, M, S> where M: Borrow<RawTable<K, V>> {
    /// Transmutes the state of a bucket. This method can't be public.
    fn state_cast<S2>(self) -> Bucket<K, V, M, S2> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            range: self.range,
            table: self.table,
            marker: PhantomData,
        }
    }

    /// Erases information about the state of a bucket.
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        self.state_cast()
    }

    /// Erases information about the state of a bucket and advance it.
    pub fn into_next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    /// Duplicates the current position. This can be useful for operations
    /// on two or more buckets.
    pub fn stash(self) -> Bucket<K, V, Bucket<K, V, M, S>, S> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            range: self.range,
            table: self,
            marker: PhantomData,
        }
    }
}

impl<K, V, M> EmptyBucket<K, V, M> where M: Borrow<RawTable<K, V>> {
    pub fn gap_peek(self) -> Option<GapThenFull<K, V, M>> {
        let gap = Bucket {
            table: (),
            idx: self.idx,
            range: self.range,
            raw: self.raw,
            marker: PhantomData,
        };

        match self.into_next().peek() {
            Full(bucket) => {
                Some(GapThenFull {
                    gap: gap,
                    full: bucket
                })
            }
            Empty(..) => None
        }
    }
}

impl<K, V, M> EmptyBucket<K, V, M> where M: BorrowMut<RawTable<K, V>>, M: Put {
    /// Puts given key and value pair, along with the key's hash,
    /// into this bucket in the hashtable. Note how `self` is 'moved' into
    /// this function, because this slot will no longer be empty when
    /// we return! A `FullBucket` is returned for later use, pointing to
    /// the newly-filled slot in the hashtable.
    ///
    /// Use `make_hash` to construct a `SafeHash` to pass to this function.
    pub fn put(mut self, hash: SafeHash, key: K, value: V)
               -> FullBucket<K, V, M> {
        unsafe {
            *self.raw.hash = Some(hash);
            ptr::write(self.raw.kval, (key, value));
        }

        self.table.borrow_mut().size += 1;

        self.state_cast()
    }
}

impl<'t, K, V, M: 't> FullBucket<K, V, M> where M: Borrow<RawTable<K, V>> {
    /// Get the distance between this bucket and the 'ideal' location
    /// as determined by the key's hash stored in it.
    ///
    /// Blog posts cited in the other module call the "distance to
    /// initial bucket", or DIB. Also known as "probe count".
    pub fn displacement(&self) -> usize {
        // Calculates the distance one has to travel when going from
        // `hash mod capacity` onwards to `idx mod capacity`, wrapping around
        // if the destination is not reached before the end of the table.
        (self.idx.wrapping_sub(**self.read().0 as usize)) & (self.table.borrow().capacity() - 1)
    }

    /// Gets references to the key and value at a given index.
    pub fn read(&self) -> (&SafeHash, &K, &V) {
        let (&ref h, &(ref k, ref v)) = unsafe {
            (&*(self.raw.hash as *mut SafeHash), &*self.raw.kval)
        };
        (h, k, v)
    }

    /// Exchange a bucket state for immutable references into the table.
    /// Because the underlying reference to the table is also consumed,
    /// no further changes to the structure of the table are possible;
    /// in exchange for this, the returned references have a longer lifetime
    /// than the references returned by `read()`.
    pub fn into_refs(self) -> (&'t K, &'t V) {
        unsafe { (&(*self.raw.kval).0, &(*self.raw.kval).1) }
    }
}

impl<'t, K, V, M: 't> FullBucket<K, V, M> where M: BorrowMut<RawTable<K, V>> {
    /// Gets mutable references to the key and value at a given index.
    pub fn read_mut(&mut self) -> (&mut SafeHash, &mut K, &mut V) {
        unsafe {
            let &mut (ref mut k, ref mut v) = &mut *self.raw.kval;
            (&mut *(self.raw.hash as *mut SafeHash), k, v)
        }
    }

    /// This works similarly to `into_refs`, exchanging a bucket state
    /// for mutable references into the table.
    pub fn into_mut_refs(self) -> (&'t mut K, &'t mut V) {
        unsafe { (&mut (*self.raw.kval).0, &mut (*self.raw.kval).1) }
    }

    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(mut self) -> (EmptyBucket<K, V, M>, K, V) {
        self.table.borrow_mut().size -= 1;

        unsafe {
            *self.raw.hash = None;
            let (k, v) = ptr::read(self.raw.kval);
            (self.state_cast(), k, v)
        }
    }
}

impl<K, V, M> GapThenFull<K, V, M> where M: Borrow<RawTable<K, V>> {
    #[inline]
    pub fn full(&self) -> &FullBucket<K, V, M> {
        &self.full
    }

    /// Advances `GapThenFull` by one bucket.
    pub fn shift(mut self) -> Option<GapThenFull<K, V, M>> {
        unsafe {
            *self.gap.raw.hash = mem::replace(&mut *self.full.raw.hash, None);
            ptr::copy_nonoverlapping(self.full.raw.kval, self.gap.raw.kval, 1);
        }

        let Bucket { raw: prev_raw, idx: prev_idx, .. } = self.full;

        match self.full.into_next().peek() {
            Full(bucket) => {
                self.gap.raw = prev_raw;
                self.gap.idx = prev_idx;

                self.full = bucket;

                Some(self)
            }
            Empty(..) => None
        }
    }
}

impl<K, V> RawTable<K, V> {
    /// Does not initialize the buckets.
    pub fn new_uninitialized(capacity: usize) -> PartialRawTable<K, V> {
        unsafe {
            let table = if capacity == 0 {
                RawTable {
                    capacity: 0,
                    size:     0,
                    middle:   Unique::new(&EMPTY as *const _ as *mut (K, V)),
                }
            } else {
                let alloc = allocate(checked_size_generic::<K, V>(capacity), align::<K, V>());
                if alloc.is_null() { ::alloc::oom() }

                RawTable {
                    capacity: capacity,
                    size:     0,
                    middle:   Unique::new((alloc as *mut (K, V)).offset(capacity as isize)),
                }
            };

            PartialRawTable {
                front: table.first_bucket_raw(),
                back: table.first_bucket_raw(),
                front_num: 0,
                back_num: capacity,
                table: table,
            }
        }
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    pub fn new(capacity: usize) -> RawTable<K, V> {
        RawTable::new_uninitialized(capacity).unwrap()
    }

    #[inline]
    fn first_bucket_raw(&self) -> RawBucket<K, V> {
        unsafe {
            RawBucket {
                hash: self.as_mut_ptr() as *mut Option<SafeHash>,
                kval: self.as_mut_ptr().offset(-(self.capacity as isize)),
            }
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Pointer to one-past-the-last key-value pair.
    pub fn as_mut_ptr(&self) -> *mut (K, V) {
        unsafe { self.middle.get() as *const _ as *mut _ }
    }
}

/// Rounds up to a multiple of a power of two. Returns the closest multiple
/// of `target_alignment` that is higher or equal to `unrounded`.
///
/// # Panics
///
/// Panics if `target_alignment` is not a power of two.
fn round_up_to_next(unrounded: usize, target_alignment: usize) -> usize {
    assert!(target_alignment.is_power_of_two());
    (unrounded + target_alignment - 1) & !(target_alignment - 1)
}

#[test]
fn test_rounding() {
    assert_eq!(round_up_to_next(0, 4), 0);
    assert_eq!(round_up_to_next(1, 4), 4);
    assert_eq!(round_up_to_next(2, 4), 4);
    assert_eq!(round_up_to_next(3, 4), 4);
    assert_eq!(round_up_to_next(4, 4), 4);
    assert_eq!(round_up_to_next(5, 4), 8);
    assert_eq!(round_up_to_next(5, 8), 8);
}

#[inline]
fn size_generic<K, V>(capacity: usize) -> usize {
    let hash_align = min_align_of::<Option<SafeHash>>();
    round_up_to_next(size_of::<(K, V)>() * capacity, hash_align) + size_of::<SafeHash>() * capacity
}

fn checked_size_generic<K, V>(capacity: usize) -> usize {
    let size = size_generic::<K, V>(capacity);
    let elem_size = size_of::<(K, V)>() + size_of::<SafeHash>();
    assert!(size >= capacity.checked_mul(elem_size).expect("capacity overflow"),
            "capacity overflow");
    size
}

#[inline]
fn align<K, V>() -> usize {
    cmp::max(mem::min_align_of::<(K, V)>(), mem::min_align_of::<u64>())
}

/// A newtyped RawBucket. Not copyable.
pub struct RawFullBucket<K, V, M>(RawBucket<K, V>, PhantomData<M>);

impl<'t, K, V, M: 't> RawFullBucket<K, V, M> where M: Borrow<RawTable<K, V>> {
    pub fn into_refs(self) -> (&'t K, &'t V) {
        unsafe { (&(*self.0.kval).0, &(*self.0.kval).1) }
    }
}

impl<'t, K, V, M: 't> RawFullBucket<K, V, M> where M: BorrowMut<RawTable<K, V>> {
    pub fn into_mut_refs(self) -> (&'t mut K, &'t mut V) {
        unsafe { (&mut (*self.0.kval).0, &mut (*self.0.kval).1) }
    }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
pub struct RawFullBuckets<K, V, M> {
    raw: RawBucket<K, V>,
    hashes_end: *mut Option<SafeHash>,
    table: M,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<K, V, M: Clone> Clone for RawFullBuckets<K, V, M> {
    fn clone(&self) -> RawFullBuckets<K, V, M> {
        RawFullBuckets {
            raw: self.raw,
            hashes_end: self.hashes_end,
            table: self.table.clone(),
        }
    }
}

impl<K, V, M> Iterator for RawFullBuckets<K, V, M> {
    type Item = RawFullBucket<K, V, M>;

    fn next(&mut self) -> Option<RawFullBucket<K, V, M>> {
        while self.raw.hash != self.hashes_end {
            unsafe {
                // We are swapping out the pointer to a bucket and replacing
                // it with the pointer to the next one.
                let prev = ptr::replace(&mut self.raw, self.raw.offset(1));
                if *prev.hash != None {
                    return Some(RawFullBucket(prev, PhantomData));
                }
            }
        }

        None
    }
}

impl<K, V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.capacity == 0 || self.capacity == mem::POST_DROP_USIZE {
            return;
        }
        // Check if the size is 0, so we don't do a useless scan when
        // dropping empty tables such as on resize.
        // Avoid double drop of elements that have been already moved out.
        unsafe {
            if self.size != 0 {
                for bucket in Bucket::raw_full_buckets(&mut *self) {
                    ptr::read(bucket.0.kval);
                }
            }

            let ptr = self.as_mut_ptr().offset(-(self.capacity as isize)) as *mut u8;
            deallocate(ptr, size_generic::<K, V>(self.capacity), align::<K, V>());
        }
    }
}

/// A partial table provides safe and cheap draining and incremental construction.
pub struct PartialRawTable<K, V> {
    table: RawTable<K, V>,
    front: RawBucket<K, V>,
    back: RawBucket<K, V>,
    front_num: usize,
    back_num: usize,
}

impl<K, V> PartialRawTable<K, V> {
    /// Turn a table into a partial table. All buckets are already initialized.
    pub fn new(table: RawTable<K, V>) -> PartialRawTable<K, V> {
        unsafe {
            PartialRawTable {
                front: table.first_bucket_raw(),
                back: table.first_bucket_raw().offset(table.capacity() as isize),
                front_num: 0,
                back_num: 0,
                table: table,
            }
        }
    }

    /// Initialize a bucket. Has no effect if there are no uninitialized buckets at the back.
    pub fn push_back(&mut self, bucket: Option<(SafeHash, K, V)>) {
        unsafe {
            if self.back_num != 0 {
                self.back_num -= 1;
                let back = ptr::replace(&mut self.back, self.back.offset(1));
                if let Some((h, k, v)) = bucket {
                    *back.hash = Some(h);
                    ptr::write(back.kval, (k, v));
                    self.table.size += 1;
                } else {
                    *back.hash = None;
                }
            }
        }
    }

    /// Takes out an initialized bucket. Returns None if all buckets are uninitialized.
    pub fn take_front(&mut self) -> Option<(SafeHash, K, V)> {
        unsafe {
            while self.front.hash != self.back.hash {
                self.front_num += 1;
                let front = ptr::replace(&mut self.front, self.front.offset(1));
                if let Some(h) = *front.hash {
                    self.table.size -= 1;
                    let (k, v) = ptr::read(front.kval);
                    return Some((h, k, v));
                }
            }
        }

        None
    }

    /// Unwrap the table by zeroing uninitialized ranges.
    pub fn unwrap(self) -> RawTable<K, V> {
        unsafe {
            ptr::write_bytes(self.table.first_bucket_raw().hash, 0, self.front_num);
            ptr::write_bytes(self.back.hash, 0, self.back_num);
            let table = ptr::read(&self.table);
            mem::forget(self);
            table
        }
    }

    pub fn size(&self) -> usize {
        self.table.size()
    }
}

/// Drops all initialized buckets in the partial table.
impl<K, V> Drop for PartialRawTable<K, V> {
    fn drop(&mut self) {
        while let Some(_) = self.take_front() {}
        debug_assert_eq!(self.table.size, 0);
    }
}
