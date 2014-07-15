// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15883

use clone::Clone;
use cmp;
use hash::{Hash, Hasher};
use iter::{Iterator, count};
use mem::{min_align_of, size_of};
use mem;
use num::{CheckedMul, is_power_of_two};
use ops::{Deref, DerefMut, Drop};
use option::{Some, None, Option};
use ptr::RawPtr;
use ptr::set_memory;
use ptr::write;
use ptr;
use rt::heap::{allocate, deallocate};

static EMPTY_BUCKET: u64 = 0u64;

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, keys, and values.
///
/// This design uses less memory and is a lot faster than the naive
/// `Vec<Option<u64, K, V>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Key invariants of this structure:
///
///   - if hashes[i] == EMPTY_BUCKET, then keys[i] and vals[i] have
///     'undefined' contents. Don't read from them. This invariant is
///     enforced outside this module with the `EmptyIndex`, `FullIndex`,
///     and `SafeHash` types.
///
///   - An `EmptyIndex` is only constructed for a bucket at an index with
///     a hash of EMPTY_BUCKET.
///
///   - A `FullIndex` is only constructed for a bucket at an index with a
///     non-EMPTY_BUCKET hash.
///
///   - A `SafeHash` is only constructed for non-`EMPTY_BUCKET` hash. We get
///     around hashes of zero by changing them to 0x8000_0000_0000_0000,
///     which will likely map to the same bucket, while not being confused
///     with "empty".
///
///   - All three "arrays represented by pointers" are the same length:
///     `capacity`. This is set at creation and never changes. The arrays
///     are unzipped to save space (we don't have to pay for the padding
///     between odd sized elements, such as in a map from u64 to u8), and
///     be more cache aware (scanning through 8 hashes brings in 2 cache
///     lines, since they're all right beside each other).
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<u64, K, V>>`.
///
/// FIXME(cgaebel):
///
/// Feb 11, 2014: This hashtable was just implemented, and, hard as I tried,
/// isn't yet totally safe. There's a "known exploit" that you can create
/// multiple FullIndexes for a bucket, `take` one, and then still `take`
/// the other causing undefined behavior. Currently, there's no story
/// for how to protect against this statically. Therefore, there are asserts
/// on `take`, `get`, `get_mut`, and `put` which check the bucket state.
/// With time, and when we're confident this works correctly, they should
/// be removed. Also, the bounds check in `peek` is especially painful,
/// as that's called in the innermost loops of the hashtable and has the
/// potential to be a major performance drain. Remove this too.
///
/// Or, better than remove, only enable these checks for debug builds.
/// There's currently no "debug-only" asserts in rust, so if you're reading
/// this and going "what? of course there are debug-only asserts!", then
/// please make this use them!
#[unsafe_no_drop_flag]
pub struct RawTable<K, V> {
    capacity: uint,
    size:     uint,
    hashes:   *mut u64
}

/// A bucket that holds a reference to the table
pub trait BucketWithTable<M> {
    /// A bucket that holds a reference to the table
    fn table<'a>(&'a self) -> &'a M;

    /// Move out the reference to the table.
    fn into_table(self) -> M;

    /// Get the raw index.
    fn index(&self) -> uint;
}

struct RawBucket<K, V> {
    hash: *mut u64,
    key:  *mut K,
    val:  *mut V
}

pub struct Bucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   uint,
    table: M
}

pub struct EmptyBucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   uint,
    table: M
}

pub struct FullBucket<K, V, M> {
    raw:   RawBucket<K, V>,
    idx:   uint,
    table: M
}

pub type EmptyBucketImm<'table,K,V> = EmptyBucket<K, V, &'table RawTable<K,V>>;
pub type  FullBucketImm<'table,K,V> =  FullBucket<K, V, &'table RawTable<K,V>>;

pub type EmptyBucketMut<'table,K,V> = EmptyBucket<K, V, &'table mut RawTable<K,V>>;
pub type  FullBucketMut<'table,K,V> =  FullBucket<K, V, &'table mut RawTable<K,V>>;

struct GapThenFull<K, V, M> {
    gap: EmptyBucket<K, V, ()>,
    full: FullBucket<K, V, M>
}

impl<K, V, M: Deref<RawTable<K,V>>> GapThenFull<K, V, M> {
    pub fn full<'a>(&'a self) -> &'a FullBucket<K, V, M> {
        &self.full
    }

    pub fn shift(mut self) -> Option<GapThenFull<K, V, M>> {
        unsafe {
            *self.gap.raw.hash = mem::replace(&mut *self.full.raw.hash, EMPTY_BUCKET);
            mem::overwrite(self.gap.raw.key, ptr::read(self.full.raw.key as *const K));
            mem::overwrite(self.gap.raw.val, ptr::read(self.full.raw.val as *const V));
        }

        let FullBucket { raw, idx, .. } = self.full;

        match self.full.next().peek() {
            Empty(_) => None,
            Full(bucket) => {
                self.gap.raw = raw;
                self.gap.idx = idx;

                self.full = bucket;
                self.full.idx &= self.full.table.capacity - 1;

                Some(self)
            }
        }
    }
}

impl<K, V> RawPtr<u64> for RawBucket<K, V> {
    unsafe fn offset(self, count: int) -> RawBucket<K, V> {
        RawBucket {
            hash: self.hash.offset(count),
            key:  self.key.offset(count),
            val:  self.val.offset(count),
        }
    }

    fn null() -> RawBucket<K, V> {
        RawBucket {
            hash: RawPtr::null(),
            key:  RawPtr::null(),
            val:  RawPtr::null()
        }
    }

    fn is_null(&self) -> bool {
        self.hash.is_null()
    }

    fn to_uint(&self) -> uint {
        self.hash.to_uint()
    }

    unsafe fn to_option(&self) -> Option<&u64> {
        self.hash.to_option()
    }
}

impl<K, V, M: Deref<RawTable<K,V>>> EmptyBucket<K, V, M> {
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table
        }
    }

    pub fn gap_peek(self) -> Option<GapThenFull<K, V, M>> {
        let gap = EmptyBucket {
            raw: self.raw,
            idx: self.idx,
            table: ()
        };

        match self.next().peek() {
            Empty(_) => None,
            Full(bucket) => {
                Some(GapThenFull {
                    gap: gap,
                    full: bucket
                })
            }
        }
    }
}

impl<K, V, M: DerefMut<RawTable<K,V>>> EmptyBucket<K, V, M> {
    pub fn put(mut self, hash: SafeHash, key: K, value: V)
               -> FullBucket<K, V, M> {
        unsafe {
            *self.raw.hash = hash.inspect();
            write(self.raw.key, key);
            write(self.raw.val, value);
        }

        self.table.size += 1;

        FullBucket { raw: self.raw, idx: self.idx, table: self.table }
    }
}

impl<K, V, M: Deref<RawTable<K,V>>> FullBucket<K, V, M> {
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            idx: self.idx,
            table: self.table
        }
    }

    pub fn distance(&self) -> uint {
        (self.idx - self.hash().inspect() as uint) & (self.table.capacity() - 1)
    }

    pub fn hash(&self) -> SafeHash {
        unsafe {
            SafeHash {
                hash: *self.raw.hash
            }
        }
    }

    pub fn read<'a>(&'a self) -> (&'a K, &'a V) {
        unsafe {
            (&*self.raw.key,
             &*self.raw.val)
        }
    }

    pub fn into_refs(self) -> (&K, &V) {
        unsafe {
            // debug_assert!(*self.raw.hash != EMPTY_BUCKET);
            (&*self.raw.key,
             &*self.raw.val)
        }
    }
}

impl<K, V, M: DerefMut<RawTable<K,V>>> FullBucket<K, V, M> {
    pub fn take(mut self) -> (EmptyBucket<K, V, M>, K, V) {
        let key = self.raw.key as *const K;
        let val = self.raw.val as *const V;

        self.table.size -= 1;

        unsafe {
            *self.raw.hash = EMPTY_BUCKET;
            (
                EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                },
                ptr::read(key),
                ptr::read(val)
            )
        }
    }

    pub fn replace(&mut self, h: SafeHash, k: K, v: V) -> (SafeHash, K, V) {
        unsafe {
            let old_hash = ptr::replace(self.raw.hash as *mut SafeHash, h);
            let old_key  = ptr::replace(self.raw.key,  k);
            let old_val  = ptr::replace(self.raw.val,  v);

            (old_hash, old_key, old_val)
        }
    }

    pub fn read_mut<'a>(&'a self) -> (&'a mut K, &'a mut V) {
        unsafe {
            // debug_assert!(*self.raw.hash != EMPTY_BUCKET);
            (&mut *self.raw.key,
             &mut *self.raw.val)
        }
    }

    pub fn into_mut_refs(self) -> (&mut K, &mut V) {
        unsafe {
            // debug_assert!(*self.raw.hash != EMPTY_BUCKET);
            (&mut *self.raw.key,
             &mut *self.raw.val)
        }
    }
}

impl<K, V, M: Deref<RawTable<K,V>>> Bucket<K, V, M> {
    pub fn new(table: M, hash: &SafeHash) -> Bucket<K, V, M> {
        let ib_index = (hash.inspect() as uint) & (table.capacity() - 1);
        Bucket {
            raw: unsafe {
               table.as_mut_ptrs().offset(ib_index as int)
            },
            idx: ib_index,
            table: table
        }
    }

    pub fn at_index(table: M, ib_index: uint) -> Bucket<K, V, M> {
        let ib_index = ib_index & (table.capacity() - 1);
        Bucket {
            raw: unsafe {
               table.as_mut_ptrs().offset(ib_index as int)
            },
            idx: ib_index,
            table: table
        }
    }

    pub fn first(table: M) -> Bucket<K, V, M> {
        Bucket {
            raw: table.as_mut_ptrs(),
            idx: 0,
            table: table
        }
    }

    pub fn peek(self) -> BucketState<K, V, M> {
        match unsafe { *self.raw.hash } {
            EMPTY_BUCKET =>
                Empty(EmptyBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                }),
            _ =>
                Full(FullBucket {
                    raw: self.raw,
                    idx: self.idx,
                    table: self.table
                })
        }
    }

    pub fn next(&mut self) {
        self.idx += 1;

        let dist = if self.idx == self.table.capacity() {
            -(self.table.capacity() as int - 1)
        } else {
            1i
        };

        unsafe {
            self.raw = self.raw.offset(dist);
        }
    }
}

impl<K, V, M> BucketWithTable<M> for FullBucket<K, V, M> {
    fn table<'a>(&'a self) -> &'a M {
        &self.table
    }

    fn into_table(self) -> M {
        self.table
    }

    fn index(&self) -> uint {
        self.idx
    }
}

impl<K, V, M> BucketWithTable<M> for EmptyBucket<K, V, M> {
    fn table<'a>(&'a self) -> &'a M {
        &self.table
    }

    fn into_table(self) -> M {
        self.table
    }

    fn index(&self) -> uint {
        self.idx
    }
}

impl<K, V, M> BucketWithTable<M> for Bucket<K, V, M> {
    fn table<'a>(&'a self) -> &'a M {
        &self.table
    }

    fn into_table(self) -> M {
        self.table
    }

    fn index(&self) -> uint {
        self.idx
    }
}

impl<'table,K,V> Deref<RawTable<K,V>> for &'table RawTable<K,V> {
    fn deref<'a>(&'a self) -> &'a RawTable<K,V> {
        &**self
    }
}

impl<'table,K,V> Deref<RawTable<K,V>> for &'table mut RawTable<K,V> {
    fn deref<'a>(&'a self) -> &'a RawTable<K,V> {
        &**self
    }
}

impl<'table,K,V> DerefMut<RawTable<K,V>> for &'table mut RawTable<K,V> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut RawTable<K,V> {
        &mut **self
    }
}

pub enum BucketState<K, V, M> {
    Empty(EmptyBucket<K, V, M>),
    Full(FullBucket<K, V, M>),
}

/// A hash that is not zero, since we use a hash of zero to represent empty
/// buckets.
#[deriving(PartialEq)]
pub struct SafeHash {
    hash: u64,
}

impl SafeHash {
    /// Peek at the hash value, which is guaranteed to be non-zero.
    #[inline(always)]
    pub fn inspect(&self) -> u64 { self.hash }
}

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
pub fn make_hash<T: Hash<S>, S, H: Hasher<S>>(hasher: &H, t: &T) -> SafeHash {
    match hasher.hash(t) {
        // This constant is exceedingly likely to hash to the same
        // bucket, but it won't be counted as empty!
        EMPTY_BUCKET => SafeHash { hash: 0x8000_0000_0000_0000 },
        h            => SafeHash { hash: h },
    }
}

fn round_up_to_next(unrounded: uint, target_alignment: uint) -> uint {
    assert!(is_power_of_two(target_alignment));
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
}

// Returns a tuple of (minimum required malloc alignment, hash_offset,
// key_offset, val_offset, array_size), from the start of a mallocated array.
fn calculate_offsets(
    hash_size: uint, hash_align: uint,
    keys_size: uint, keys_align: uint,
    vals_size: uint, vals_align: uint) -> (uint, uint, uint, uint, uint) {

    let hash_offset   = 0;
    let end_of_hashes = hash_offset + hash_size;

    let keys_offset   = round_up_to_next(end_of_hashes, keys_align);
    let end_of_keys   = keys_offset + keys_size;

    let vals_offset   = round_up_to_next(end_of_keys, vals_align);
    let end_of_vals   = vals_offset + vals_size;

    let min_align = cmp::max(hash_align, cmp::max(keys_align, vals_align));

    (min_align, hash_offset, keys_offset, vals_offset, end_of_vals)
}

#[test]
fn test_offset_calculation() {
    assert_eq!(calculate_offsets(128, 8, 15, 1, 4, 4 ), (8, 0, 128, 144, 148));
    assert_eq!(calculate_offsets(3,   1, 2,  1, 1, 1 ), (1, 0, 3,   5,   6));
    assert_eq!(calculate_offsets(6,   2, 12, 4, 24, 8), (8, 0, 8,   24,  48));
}

impl<K, V> RawTable<K, V> {

    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    unsafe fn new_uninitialized(capacity: uint) -> RawTable<K, V> {
        if capacity == 0 {
            return RawTable {
                size: 0,
                capacity: 0,
                hashes: 0 as *mut u64,
            };
        }
        let hashes_size = capacity.checked_mul(&size_of::<u64>())
                                  .expect("capacity overflow");
        let keys_size = capacity.checked_mul(&size_of::< K >())
                                .expect("capacity overflow");
        let vals_size = capacity.checked_mul(&size_of::< V >())
                                .expect("capacity overflow");

        // Allocating hashmaps is a little tricky. We need to allocate three
        // arrays, but since we know their sizes and alignments up front,
        // we just allocate a single array, and then have the subarrays
        // point into it.
        //
        // This is great in theory, but in practice getting the alignment
        // right is a little subtle. Therefore, calculating offsets has been
        // factored out into a different function.
        let (malloc_alignment, hash_offset, _, _, size) =
            calculate_offsets(
                hashes_size, min_align_of::<u64>(),
                keys_size,   min_align_of::< K >(),
                vals_size,   min_align_of::< V >());

        let buffer = allocate(size, malloc_alignment);

        let hashes = buffer.offset(hash_offset as int) as *mut u64;

        RawTable {
            capacity: capacity,
            size:     0,
            hashes:   hashes,
        }
    }

    fn as_mut_ptrs(&self) -> RawBucket<K, V> {
        let hashes_size = self.capacity * size_of::<u64>();
        let keys_size = self.capacity * size_of::<K>();

        let keys_offset = (hashes_size + min_align_of::< K >() - 1) & !(min_align_of::< K >() - 1);
        let end_of_keys = keys_offset + keys_size;

        let vals_offset = (end_of_keys + min_align_of::< V >() - 1) & !(min_align_of::< V >() - 1);

        let buffer = self.hashes as *mut u8;

        unsafe {
            RawBucket {
                hash: self.hashes,
                key:  buffer.offset(keys_offset as int) as *mut K,
                val:  buffer.offset(vals_offset as int) as *mut V
            }
        }
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    #[allow(experimental)]
    pub fn new(capacity: uint) -> RawTable<K, V> {
        unsafe {
            let ret = RawTable::new_uninitialized(capacity);
            set_memory(ret.hashes, 0u8, capacity);
            ret
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> uint {
        self.capacity
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> uint {
        self.size
    }

    fn ptrs<'a>(&'a self) -> RawBuckets<'a, K, V> {
        RawBuckets {
            raw: self.as_mut_ptrs(),
            hashes_end: unsafe {
                self.hashes.offset(self.capacity as int)
            }
        }
    }

    pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
        Entries {
            iter: self.ptrs(),
            elems_left: self.size(),
        }
    }

    pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        MutEntries {
            iter: self.ptrs(),
            elems_left: self.size(),
        }
    }

    pub fn move_iter(self) -> MoveEntries<K, V> {
        MoveEntries {
            iter: self.ptrs(),
            table: self,
        }
    }

    pub fn rev_move_buckets<'a>(&'a mut self) -> RevMoveBuckets<'a, K, V> {
        let raw_bucket = self.as_mut_ptrs();
        unsafe {
            RevMoveBuckets {
                raw: raw_bucket.offset(self.capacity as int),
                hashes_end: raw_bucket.hash,
                elems_left: self.size
            }
        }
    }
}

pub struct RawBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64
}

impl<'a, K, V> Iterator<RawBucket<K, V>> for RawBuckets<'a, K, V> {
    fn next(&mut self) -> Option<RawBucket<K, V>> {
        while self.raw.hash != self.hashes_end {
            unsafe {
                let prev = ptr::replace(&mut self.raw, self.raw.offset(1));
                if *prev.hash != EMPTY_BUCKET {
                    return Some(prev);
                }
            }
        }

        None
    }
}

pub struct RevMoveBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64,
    elems_left: uint
}

impl<'a, K, V> Iterator<(K, V)> for RevMoveBuckets<'a, K, V> {
    fn next(&mut self) -> Option<(K, V)> {
        if self.elems_left == 0 {
            return None;
        }

        loop {
            debug_assert!(self.raw.hash != self.hashes_end);

            unsafe {
                self.raw = self.raw.offset(-1);

                if *self.raw.hash != EMPTY_BUCKET {
                    self.elems_left -= 1;
                    return Some((
                        ptr::read(self.raw.key as *const K),
                        ptr::read(self.raw.val as *const V)
                    ));
                }
            }
        }
    }
}

// `read_all_mut` casts a `*u64` to a `*SafeHash`. Since we statically
// ensure that a `FullIndex` points to an index with a non-zero hash,
// and a `SafeHash` is just a `u64` with a different name, this is
// safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `u64`. If you need to change the size of `SafeHash` (and
// consequently made this test fail), `read_all_mut` needs to be
// modified to no longer assume this.
#[test]
fn can_alias_safehash_as_u64() {
    assert_eq!(size_of::<SafeHash>(), size_of::<u64>())
}

/// Iterator over shared references to entries in a table.
pub struct Entries<'a, K, V> {
    iter: RawBuckets<'a, K, V>,
    elems_left: uint,
}

/// Iterator over mutable references to entries in a table.
pub struct MutEntries<'a, K, V> {
    iter: RawBuckets<'a, K, V>,
    elems_left: uint,
}

/// Iterator over the entries in a table, consuming the table.
pub struct MoveEntries<K, V> {
    table: RawTable<K, V>,
    iter: RawBuckets<'static, K, V>
}

impl<'a, K, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                (&*bucket.key,
                 &*bucket.val)
            }
        })
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.elems_left, Some(self.elems_left))
    }
}

impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.iter.next().map(|bucket| {
            self.elems_left -= 1;
            unsafe {
                (&*bucket.key,
                 &mut *bucket.val)
            }
        })
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.elems_left, Some(self.elems_left))
    }
}

impl<K, V> Iterator<(SafeHash, K, V)> for MoveEntries<K, V> {
    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|bucket| {
            self.table.size -= 1;
            unsafe {
                (
                    SafeHash {
                        hash: *bucket.hash,
                    },
                    ptr::read(bucket.key as *const K),
                    ptr::read(bucket.val as *const V)
                )
            }
        })
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        let size = self.table.size();
        (size, Some(size))
    }
}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> RawTable<K, V> {
        unsafe {
            let mut new_ht = RawTable::new_uninitialized(self.capacity());

            {
                let cap = self.capacity();
                let mut new_buckets = Bucket::first(&mut new_ht);
                let mut buckets = Bucket::first(self);
                while buckets.index() != cap {
                    match buckets.peek() {
                        Full(full) => {
                            let (h, k, v) = {
                                let (k, v) = full.read();
                                (full.hash(), k.clone(), v.clone())
                            };
                            *new_buckets.raw.hash = h.inspect();
                            mem::overwrite(new_buckets.raw.key, k);
                            mem::overwrite(new_buckets.raw.val, v);
                        }
                        _  => {
                            *new_buckets.raw.hash = EMPTY_BUCKET;
                        }
                    }
                    new_buckets.next();
                    buckets.next();
                }
            }

            new_ht.size = self.size();

            new_ht
        }
    }
}

#[unsafe_destructor]
impl<K, V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.hashes.is_null() {
            return;
        }
        // This is in reverse because we're likely to have partially taken
        // some elements out with `.move_iter()` from the front.
        // Check if the size is 0, so we don't do a useless scan when
        // dropping empty tables such as on resize.
        // Avoid double free of elements already moved out.
        for _ in self.rev_move_buckets() {}

        let hashes_size = self.capacity * size_of::<u64>();
        let keys_size = self.capacity * size_of::<K>();
        let vals_size = self.capacity * size_of::<V>();
        let (align, _, _, _, size) = calculate_offsets(hashes_size, min_align_of::<u64>(),
                                                       keys_size, min_align_of::<K>(),
                                                       vals_size, min_align_of::<V>());

        unsafe {
            deallocate(self.hashes as *mut u8, size, align);
            // Remember how everything was allocated out of one buffer
            // during initialization? We only need one call to free here.
        }

        self.hashes = RawPtr::null();
    }
}
