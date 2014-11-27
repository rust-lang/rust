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

pub use self::BucketState::*;

use clone::Clone;
use cmp;
use hash::{Hash, Hasher};
use iter::{Iterator, count};
use kinds::{Sized, marker};
use mem::{min_align_of, size_of};
use mem;
use num::{Int, UnsignedInt};
use ops::{Deref, DerefMut, Drop};
use option::{Some, None, Option};
use ptr::{RawPtr, copy_nonoverlapping_memory, zero_memory};
use ptr;
use rt::heap::{allocate, deallocate};

const EMPTY_BUCKET: u64 = 0u64;

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, keys, and values.
///
/// This design uses less memory and is a lot faster than the naive
/// `Vec<Option<u64, K, V>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Essential invariants of this structure:
///
///   - if t.hashes[i] == EMPTY_BUCKET, then `Bucket::at_index(&t, i).raw`
///     points to 'undefined' contents. Don't read from it. This invariant is
///     enforced outside this module with the `EmptyBucket`, `FullBucket`,
///     and `SafeHash` types.
///
///   - An `EmptyBucket` is only constructed at an index with
///     a hash of EMPTY_BUCKET.
///
///   - A `FullBucket` is only constructed at an index with a
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
///     be more cache aware (scanning through 8 hashes brings in at most
///     2 cache lines, since they're all right beside each other).
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<u64, K, V>>`.
#[unsafe_no_drop_flag]
pub struct RawTable<K, V> {
    capacity: uint,
    size:     uint,
    hashes:   *mut u64,
    // Because K/V do not appear directly in any of the types in the struct,
    // inform rustc that in fact instances of K and V are reachable from here.
    marker:   marker::CovariantType<(K,V)>,
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

pub type EmptyBucketImm<'table, K, V> = EmptyBucket<K, V, &'table RawTable<K, V>>;
pub type  FullBucketImm<'table, K, V> =  FullBucket<K, V, &'table RawTable<K, V>>;

pub type EmptyBucketMut<'table, K, V> = EmptyBucket<K, V, &'table mut RawTable<K, V>>;
pub type  FullBucketMut<'table, K, V> =  FullBucket<K, V, &'table mut RawTable<K, V>>;

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
#[deriving(PartialEq)]
pub struct SafeHash {
    hash: u64,
}

impl SafeHash {
    /// Peek at the hash value, which is guaranteed to be non-zero.
    #[inline(always)]
    pub fn inspect(&self) -> u64 { unimplemented!() }
}

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
pub fn make_hash<Sized? T: Hash<S>, S, H: Hasher<S>>(hasher: &H, t: &T) -> SafeHash { unimplemented!() }

// `replace` casts a `*u64` to a `*SafeHash`. Since we statically
// ensure that a `FullBucket` points to an index with a non-zero hash,
// and a `SafeHash` is just a `u64` with a different name, this is
// safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `u64`. If you need to change the size of `SafeHash` (and
// consequently made this test fail), `replace` needs to be
// modified to no longer assume this.
#[test]
fn can_alias_safehash_as_u64() { unimplemented!() }

impl<K, V> RawBucket<K, V> {
    unsafe fn offset(self, count: int) -> RawBucket<K, V> { unimplemented!() }
}

// Buckets hold references to the table.
impl<K, V, M> FullBucket<K, V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M { unimplemented!() }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M { unimplemented!() }
    /// Get the raw index.
    pub fn index(&self) -> uint { unimplemented!() }
}

impl<K, V, M> EmptyBucket<K, V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M { unimplemented!() }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M { unimplemented!() }
}

impl<K, V, M> Bucket<K, V, M> {
    /// Move out the reference to the table.
    pub fn into_table(self) -> M { unimplemented!() }
    /// Get the raw index.
    pub fn index(&self) -> uint { unimplemented!() }
}

impl<K, V, M: Deref<RawTable<K, V>>> Bucket<K, V, M> {
    pub fn new(table: M, hash: &SafeHash) -> Bucket<K, V, M> { unimplemented!() }

    pub fn at_index(table: M, ib_index: uint) -> Bucket<K, V, M> { unimplemented!() }

    pub fn first(table: M) -> Bucket<K, V, M> { unimplemented!() }

    /// Reads a bucket at a given index, returning an enum indicating whether
    /// it's initialized or not. You need to match on this enum to get
    /// the appropriate types to call most of the other functions in
    /// this module.
    pub fn peek(self) -> BucketState<K, V, M> { unimplemented!() }

    /// Modifies the bucket pointer in place to make it point to the next slot.
    pub fn next(&mut self) { unimplemented!() }
}

impl<K, V, M: Deref<RawTable<K, V>>> EmptyBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> { unimplemented!() }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> { unimplemented!() }

    pub fn gap_peek(self) -> Option<GapThenFull<K, V, M>> { unimplemented!() }
}

impl<K, V, M: DerefMut<RawTable<K, V>>> EmptyBucket<K, V, M> {
    /// Puts given key and value pair, along with the key's hash,
    /// into this bucket in the hashtable. Note how `self` is 'moved' into
    /// this function, because this slot will no longer be empty when
    /// we return! A `FullBucket` is returned for later use, pointing to
    /// the newly-filled slot in the hashtable.
    ///
    /// Use `make_hash` to construct a `SafeHash` to pass to this function.
    pub fn put(mut self, hash: SafeHash, key: K, value: V)
               -> FullBucket<K, V, M> { unimplemented!() }
}

impl<K, V, M: Deref<RawTable<K, V>>> FullBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> { unimplemented!() }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> { unimplemented!() }

    /// Get the distance between this bucket and the 'ideal' location
    /// as determined by the key's hash stored in it.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// initial bucket", or DIB. Also known as "probe count".
    pub fn distance(&self) -> uint { unimplemented!() }

    #[inline]
    pub fn hash(&self) -> SafeHash { unimplemented!() }

    /// Gets references to the key and value at a given index.
    pub fn read(&self) -> (&K, &V) { unimplemented!() }
}

impl<K, V, M: DerefMut<RawTable<K, V>>> FullBucket<K, V, M> {
    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(mut self) -> (EmptyBucket<K, V, M>, K, V) { unimplemented!() }

    pub fn replace(&mut self, h: SafeHash, k: K, v: V) -> (SafeHash, K, V) { unimplemented!() }

    /// Gets mutable references to the key and value at a given index.
    pub fn read_mut(&mut self) -> (&mut K, &mut V) { unimplemented!() }
}

impl<'t, K, V, M: Deref<RawTable<K, V>> + 't> FullBucket<K, V, M> {
    /// Exchange a bucket state for immutable references into the table.
    /// Because the underlying reference to the table is also consumed,
    /// no further changes to the structure of the table are possible;
    /// in exchange for this, the returned references have a longer lifetime
    /// than the references returned by `read()`.
    pub fn into_refs(self) -> (&'t K, &'t V) { unimplemented!() }
}

impl<'t, K, V, M: DerefMut<RawTable<K, V>> + 't> FullBucket<K, V, M> {
    /// This works similarly to `into_refs`, exchanging a bucket state
    /// for mutable references into the table.
    pub fn into_mut_refs(self) -> (&'t mut K, &'t mut V) { unimplemented!() }
}

impl<K, V, M> BucketState<K, V, M> {
    // For convenience.
    pub fn expect_full(self) -> FullBucket<K, V, M> { unimplemented!() }
}

impl<K, V, M: Deref<RawTable<K, V>>> GapThenFull<K, V, M> {
    #[inline]
    pub fn full(&self) -> &FullBucket<K, V, M> { unimplemented!() }

    pub fn shift(mut self) -> Option<GapThenFull<K, V, M>> { unimplemented!() }
}


/// Rounds up to a multiple of a power of two. Returns the closest multiple
/// of `target_alignment` that is higher or equal to `unrounded`.
///
/// # Panics
///
/// Panics if `target_alignment` is not a power of two.
fn round_up_to_next(unrounded: uint, target_alignment: uint) -> uint { unimplemented!() }

#[test]
fn test_rounding() { unimplemented!() }

// Returns a tuple of (key_offset, val_offset),
// from the start of a mallocated array.
fn calculate_offsets(hashes_size: uint,
                     keys_size: uint, keys_align: uint,
                     vals_align: uint)
                     -> (uint, uint) { unimplemented!() }

// Returns a tuple of (minimum required malloc alignment, hash_offset,
// array_size), from the start of a mallocated array.
fn calculate_allocation(hash_size: uint, hash_align: uint,
                        keys_size: uint, keys_align: uint,
                        vals_size: uint, vals_align: uint)
                        -> (uint, uint, uint) { unimplemented!() }

#[test]
fn test_offset_calculation() { unimplemented!() }

impl<K, V> RawTable<K, V> {
    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    unsafe fn new_uninitialized(capacity: uint) -> RawTable<K, V> { unimplemented!() }

    fn first_bucket_raw(&self) -> RawBucket<K, V> { unimplemented!() }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    #[allow(experimental)]
    pub fn new(capacity: uint) -> RawTable<K, V> { unimplemented!() }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> uint { unimplemented!() }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> uint { unimplemented!() }

    fn raw_buckets(&self) -> RawBuckets<K, V> { unimplemented!() }

    pub fn iter(&self) -> Entries<K, V> { unimplemented!() }

    pub fn iter_mut(&mut self) -> MutEntries<K, V> { unimplemented!() }

    pub fn into_iter(self) -> MoveEntries<K, V> { unimplemented!() }

    /// Returns an iterator that copies out each entry. Used while the table
    /// is being dropped.
    unsafe fn rev_move_buckets(&mut self) -> RevMoveBuckets<K, V> { unimplemented!() }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
struct RawBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64,
    marker: marker::ContravariantLifetime<'a>,
}

impl<'a, K, V> Iterator<RawBucket<K, V>> for RawBuckets<'a, K, V> {
    fn next(&mut self) -> Option<RawBucket<K, V>> { unimplemented!() }
}

/// An iterator that moves out buckets in reverse order. It leaves the table
/// in an inconsistent state and should only be used for dropping
/// the table's remaining entries. It's used in the implementation of Drop.
struct RevMoveBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    hashes_end: *mut u64,
    elems_left: uint,
    marker: marker::ContravariantLifetime<'a>,
}

impl<'a, K, V> Iterator<(K, V)> for RevMoveBuckets<'a, K, V> {
    fn next(&mut self) -> Option<(K, V)> { unimplemented!() }
}

/// Iterator over shared references to entries in a table.
pub struct Entries<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: uint,
}

/// Iterator over mutable references to entries in a table.
pub struct MutEntries<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    elems_left: uint,
}

/// Iterator over the entries in a table, consuming the table.
pub struct MoveEntries<K, V> {
    table: RawTable<K, V>,
    iter: RawBuckets<'static, K, V>
}

impl<'a, K, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a V)> { unimplemented!() }

    fn size_hint(&self) -> (uint, Option<uint>) { unimplemented!() }
}

impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> { unimplemented!() }

    fn size_hint(&self) -> (uint, Option<uint>) { unimplemented!() }
}

impl<K, V> Iterator<(SafeHash, K, V)> for MoveEntries<K, V> {
    fn next(&mut self) -> Option<(SafeHash, K, V)> { unimplemented!() }

    fn size_hint(&self) -> (uint, Option<uint>) { unimplemented!() }
}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> RawTable<K, V> { unimplemented!() }
}

#[unsafe_destructor]
impl<K, V> Drop for RawTable<K, V> {
    fn drop(&mut self) { unimplemented!() }
}
