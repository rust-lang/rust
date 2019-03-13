use crate::alloc::{Global, Alloc, Layout, LayoutErr, handle_alloc_error};
use crate::collections::CollectionAllocErr;
use crate::hash::{BuildHasher, Hash, Hasher};
use crate::marker;
use crate::mem::{self, size_of, needs_drop};
use crate::ops::{Deref, DerefMut};
use crate::ptr::{self, Unique, NonNull};
use crate::hint;

use self::BucketState::*;

/// Integer type used for stored hash values.
///
/// No more than bit_width(usize) bits are needed to select a bucket.
///
/// The most significant bit is ours to use for tagging `SafeHash`.
///
/// (Even if we could have usize::MAX bytes allocated for buckets,
/// each bucket stores at least a `HashUint`, so there can be no more than
/// usize::MAX / size_of(usize) buckets.)
type HashUint = usize;

const EMPTY_BUCKET: HashUint = 0;
const EMPTY: usize = 1;

/// Special `Unique<HashUint>` that uses the lower bit of the pointer
/// to expose a boolean tag.
/// Note: when the pointer is initialized to EMPTY `.ptr()` will return
/// null and the tag functions shouldn't be used.
struct TaggedHashUintPtr(Unique<HashUint>);

impl TaggedHashUintPtr {
    #[inline]
    unsafe fn new(ptr: *mut HashUint) -> Self {
        debug_assert!(ptr as usize & 1 == 0 || ptr as usize == EMPTY as usize);
        TaggedHashUintPtr(Unique::new_unchecked(ptr))
    }

    #[inline]
    fn set_tag(&mut self, value: bool) {
        let mut usize_ptr = self.0.as_ptr() as usize;
        unsafe {
            if value {
                usize_ptr |= 1;
            } else {
                usize_ptr &= !1;
            }
            self.0 = Unique::new_unchecked(usize_ptr as *mut HashUint)
        }
    }

    #[inline]
    fn tag(&self) -> bool {
        (self.0.as_ptr() as usize) & 1 == 1
    }

    #[inline]
    fn ptr(&self) -> *mut HashUint {
        (self.0.as_ptr() as usize & !1) as *mut HashUint
    }
}

/// The raw hashtable, providing safe-ish access to the unzipped and highly
/// optimized arrays of hashes, and key-value pairs.
///
/// This design is a lot faster than the naive
/// `Vec<Option<(u64, K, V)>>`, because we don't pay for the overhead of an
/// option on every element, and we get a generally more cache-aware design.
///
/// Essential invariants of this structure:
///
///   - if `t.hashes[i] == EMPTY_BUCKET`, then `Bucket::at_index(&t, i).raw`
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
///   - Both "arrays represented by pointers" are the same length:
///     `capacity`. This is set at creation and never changes. The arrays
///     are unzipped and are more cache aware (scanning through 8 hashes
///     brings in at most 2 cache lines, since they're all right beside each
///     other). This layout may waste space in padding such as in a map from
///     u64 to u8, but is a more cache conscious layout as the key-value pairs
///     are only very shortly probed and the desired value will be in the same
///     or next cache line.
///
/// You can kind of think of this module/data structure as a safe wrapper
/// around just the "table" part of the hashtable. It enforces some
/// invariants at the type level and employs some performance trickery,
/// but in general is just a tricked out `Vec<Option<(u64, K, V)>>`.
///
/// The hashtable also exposes a special boolean tag. The tag defaults to false
/// when the RawTable is created and is accessible with the `tag` and `set_tag`
/// functions.
pub struct RawTable<K, V> {
    capacity_mask: usize,
    size: usize,
    hashes: TaggedHashUintPtr,

    // Because K/V do not appear directly in any of the types in the struct,
    // inform rustc that in fact instances of K and V are reachable from here.
    marker: marker::PhantomData<(K, V)>,
}

// An unsafe view of a RawTable bucket
// Valid indexes are within [0..table_capacity)
pub struct RawBucket<K, V> {
    hash_start: *mut HashUint,
    // We use *const to ensure covariance with respect to K and V
    pair_start: *const (K, V),
    idx: usize,
    _marker: marker::PhantomData<(K, V)>,
}

impl<K, V> Copy for RawBucket<K, V> {}
impl<K, V> Clone for RawBucket<K, V> {
    fn clone(&self) -> RawBucket<K, V> {
        *self
    }
}

pub struct Bucket<K, V, M> {
    raw: RawBucket<K, V>,
    table: M,
}

impl<K, V, M: Copy> Copy for Bucket<K, V, M> {}
impl<K, V, M: Copy> Clone for Bucket<K, V, M> {
    fn clone(&self) -> Bucket<K, V, M> {
        *self
    }
}

pub struct EmptyBucket<K, V, M> {
    raw: RawBucket<K, V>,
    table: M,
}

pub struct FullBucket<K, V, M> {
    raw: RawBucket<K, V>,
    table: M,
}

pub type FullBucketMut<'table, K, V> = FullBucket<K, V, &'table mut RawTable<K, V>>;

pub enum BucketState<K, V, M> {
    Empty(EmptyBucket<K, V, M>),
    Full(FullBucket<K, V, M>),
}

// A GapThenFull encapsulates the state of two consecutive buckets at once.
// The first bucket, called the gap, is known to be empty.
// The second bucket is full.
pub struct GapThenFull<K, V, M> {
    gap: EmptyBucket<K, V, ()>,
    full: FullBucket<K, V, M>,
}

/// A hash that is not zero, since we use a hash of zero to represent empty
/// buckets.
#[derive(PartialEq, Copy, Clone)]
pub struct SafeHash {
    hash: HashUint,
}

impl SafeHash {
    /// Peek at the hash value, which is guaranteed to be non-zero.
    #[inline(always)]
    pub fn inspect(&self) -> HashUint {
        self.hash
    }

    #[inline(always)]
    pub fn new(hash: u64) -> Self {
        // We need to avoid 0 in order to prevent collisions with
        // EMPTY_HASH. We can maintain our precious uniform distribution
        // of initial indexes by unconditionally setting the MSB,
        // effectively reducing the hashes by one bit.
        //
        // Truncate hash to fit in `HashUint`.
        let hash_bits = size_of::<HashUint>() * 8;
        SafeHash { hash: (1 << (hash_bits - 1)) | (hash as HashUint) }
    }
}

/// We need to remove hashes of 0. That's reserved for empty buckets.
/// This function wraps up `hash_keyed` to be the only way outside this
/// module to generate a SafeHash.
pub fn make_hash<T: ?Sized, S>(hash_state: &S, t: &T) -> SafeHash
    where T: Hash,
          S: BuildHasher
{
    let mut state = hash_state.build_hasher();
    t.hash(&mut state);
    SafeHash::new(state.finish())
}

// `replace` casts a `*HashUint` to a `*SafeHash`. Since we statically
// ensure that a `FullBucket` points to an index with a non-zero hash,
// and a `SafeHash` is just a `HashUint` with a different name, this is
// safe.
//
// This test ensures that a `SafeHash` really IS the same size as a
// `HashUint`. If you need to change the size of `SafeHash` (and
// consequently made this test fail), `replace` needs to be
// modified to no longer assume this.
#[test]
fn can_alias_safehash_as_hash() {
    assert_eq!(size_of::<SafeHash>(), size_of::<HashUint>())
}

// RawBucket methods are unsafe as it's possible to
// make a RawBucket point to invalid memory using safe code.
impl<K, V> RawBucket<K, V> {
    unsafe fn hash(&self) -> *mut HashUint {
        self.hash_start.add(self.idx)
    }
    unsafe fn pair(&self) -> *mut (K, V) {
        self.pair_start.add(self.idx) as *mut (K, V)
    }
    unsafe fn hash_pair(&self) -> (*mut HashUint, *mut (K, V)) {
        (self.hash(), self.pair())
    }
}

// Buckets hold references to the table.
impl<K, V, M> FullBucket<K, V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Borrow a mutable reference to the table.
    pub fn table_mut(&mut self) -> &mut M {
        &mut self.table
    }
    /// Move out the reference to the table.
    pub fn into_table(self) -> M {
        self.table
    }
    /// Gets the raw index.
    pub fn index(&self) -> usize {
        self.raw.idx
    }
    /// Gets the raw bucket.
    pub fn raw(&self) -> RawBucket<K, V> {
        self.raw
    }
}

impl<K, V, M> EmptyBucket<K, V, M> {
    /// Borrow a reference to the table.
    pub fn table(&self) -> &M {
        &self.table
    }
    /// Borrow a mutable reference to the table.
    pub fn table_mut(&mut self) -> &mut M {
        &mut self.table
    }
}

impl<K, V, M> Bucket<K, V, M> {
    /// Gets the raw index.
    pub fn index(&self) -> usize {
        self.raw.idx
    }
    /// get the table.
    pub fn into_table(self) -> M {
        self.table
    }
}

impl<K, V, M> Deref for FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>>
{
    type Target = RawTable<K, V>;
    fn deref(&self) -> &RawTable<K, V> {
        &self.table
    }
}

/// `Put` is implemented for types which provide access to a table and cannot be invalidated
///  by filling a bucket. A similar implementation for `Take` is possible.
pub trait Put<K, V> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V>;
}


impl<K, V> Put<K, V> for &mut RawTable<K, V> {
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V> {
        *self
    }
}

impl<K, V, M> Put<K, V> for Bucket<K, V, M>
    where M: Put<K, V>
{
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V> {
        self.table.borrow_table_mut()
    }
}

impl<K, V, M> Put<K, V> for FullBucket<K, V, M>
    where M: Put<K, V>
{
    unsafe fn borrow_table_mut(&mut self) -> &mut RawTable<K, V> {
        self.table.borrow_table_mut()
    }
}

impl<K, V, M: Deref<Target = RawTable<K, V>>> Bucket<K, V, M> {
    #[inline]
    pub fn new(table: M, hash: SafeHash) -> Bucket<K, V, M> {
        Bucket::at_index(table, hash.inspect() as usize)
    }

    pub fn new_from(r: RawBucket<K, V>, t: M)
        -> Bucket<K, V, M>
    {
        Bucket {
            raw: r,
            table: t,
        }
    }

    #[inline]
    pub fn at_index(table: M, ib_index: usize) -> Bucket<K, V, M> {
        // if capacity is 0, then the RawBucket will be populated with bogus pointers.
        // This is an uncommon case though, so avoid it in release builds.
        debug_assert!(table.capacity() > 0,
                      "Table should have capacity at this point");
        let ib_index = ib_index & table.capacity_mask;
        Bucket {
            raw: table.raw_bucket_at(ib_index),
            table,
        }
    }

    pub fn first(table: M) -> Bucket<K, V, M> {
        Bucket {
            raw: table.raw_bucket_at(0),
            table,
        }
    }

    // "So a few of the first shall be last: for many be called,
    // but few chosen."
    //
    // We'll most likely encounter a few buckets at the beginning that
    // have their initial buckets near the end of the table. They were
    // placed at the beginning as the probe wrapped around the table
    // during insertion. We must skip forward to a bucket that won't
    // get reinserted too early and won't unfairly steal others spot.
    // This eliminates the need for robin hood.
    pub fn head_bucket(table: M) -> Bucket<K, V, M> {
        let mut bucket = Bucket::first(table);

        loop {
            bucket = match bucket.peek() {
                Full(full) => {
                    if full.displacement() == 0 {
                        // This bucket occupies its ideal spot.
                        // It indicates the start of another "cluster".
                        bucket = full.into_bucket();
                        break;
                    }
                    // Leaving this bucket in the last cluster for later.
                    full.into_bucket()
                }
                Empty(b) => {
                    // Encountered a hole between clusters.
                    b.into_bucket()
                }
            };
            bucket.next();
        }
        bucket
    }

    /// Reads a bucket at a given index, returning an enum indicating whether
    /// it's initialized or not. You need to match on this enum to get
    /// the appropriate types to call most of the other functions in
    /// this module.
    pub fn peek(self) -> BucketState<K, V, M> {
        match unsafe { *self.raw.hash() } {
            EMPTY_BUCKET => {
                Empty(EmptyBucket {
                    raw: self.raw,
                    table: self.table,
                })
            }
            _ => {
                Full(FullBucket {
                    raw: self.raw,
                    table: self.table,
                })
            }
        }
    }

    /// Modifies the bucket in place to make it point to the next slot.
    pub fn next(&mut self) {
        self.raw.idx = self.raw.idx.wrapping_add(1) & self.table.capacity_mask;
    }

    /// Modifies the bucket in place to make it point to the previous slot.
    pub fn prev(&mut self) {
        self.raw.idx = self.raw.idx.wrapping_sub(1) & self.table.capacity_mask;
    }
}

impl<K, V, M: Deref<Target = RawTable<K, V>>> EmptyBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            table: self.table,
        }
    }

    pub fn gap_peek(self) -> Result<GapThenFull<K, V, M>, Bucket<K, V, M>> {
        let gap = EmptyBucket {
            raw: self.raw,
            table: (),
        };

        match self.next().peek() {
            Full(bucket) => {
                Ok(GapThenFull {
                    gap,
                    full: bucket,
                })
            }
            Empty(e) => Err(e.into_bucket()),
        }
    }
}

impl<K, V, M> EmptyBucket<K, V, M>
    where M: Put<K, V>
{
    /// Puts given key and value pair, along with the key's hash,
    /// into this bucket in the hashtable. Note how `self` is 'moved' into
    /// this function, because this slot will no longer be empty when
    /// we return! A `FullBucket` is returned for later use, pointing to
    /// the newly-filled slot in the hashtable.
    ///
    /// Use `make_hash` to construct a `SafeHash` to pass to this function.
    pub fn put(mut self, hash: SafeHash, key: K, value: V) -> FullBucket<K, V, M> {
        unsafe {
            *self.raw.hash() = hash.inspect();
            ptr::write(self.raw.pair(), (key, value));

            self.table.borrow_table_mut().size += 1;
        }

        FullBucket {
            raw: self.raw,
            table: self.table,
        }
    }
}

impl<K, V, M: Deref<Target = RawTable<K, V>>> FullBucket<K, V, M> {
    #[inline]
    pub fn next(self) -> Bucket<K, V, M> {
        let mut bucket = self.into_bucket();
        bucket.next();
        bucket
    }

    #[inline]
    pub fn into_bucket(self) -> Bucket<K, V, M> {
        Bucket {
            raw: self.raw,
            table: self.table,
        }
    }

    /// Duplicates the current position. This can be useful for operations
    /// on two or more buckets.
    pub fn stash(self) -> FullBucket<K, V, Self> {
        FullBucket {
            raw: self.raw,
            table: self,
        }
    }

    /// Gets the distance between this bucket and the 'ideal' location
    /// as determined by the key's hash stored in it.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// initial bucket", or DIB. Also known as "probe count".
    pub fn displacement(&self) -> usize {
        // Calculates the distance one has to travel when going from
        // `hash mod capacity` onwards to `idx mod capacity`, wrapping around
        // if the destination is not reached before the end of the table.
        (self.raw.idx.wrapping_sub(self.hash().inspect() as usize)) & self.table.capacity_mask
    }

    #[inline]
    pub fn hash(&self) -> SafeHash {
        unsafe { SafeHash { hash: *self.raw.hash() } }
    }

    /// Gets references to the key and value at a given index.
    pub fn read(&self) -> (&K, &V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&(*pair_ptr).0, &(*pair_ptr).1)
        }
    }
}

// We take a mutable reference to the table instead of accepting anything that
// implements `DerefMut` to prevent fn `take` from being called on `stash`ed
// buckets.
impl<'t, K, V> FullBucket<K, V, &'t mut RawTable<K, V>> {
    /// Removes this bucket's key and value from the hashtable.
    ///
    /// This works similarly to `put`, building an `EmptyBucket` out of the
    /// taken bucket.
    pub fn take(self) -> (EmptyBucket<K, V, &'t mut RawTable<K, V>>, K, V) {
        self.table.size -= 1;

        unsafe {
            *self.raw.hash() = EMPTY_BUCKET;
            let (k, v) = ptr::read(self.raw.pair());
            (EmptyBucket {
                 raw: self.raw,
                 table: self.table,
             },
            k,
            v)
        }
    }
}

// This use of `Put` is misleading and restrictive, but safe and sufficient for our use cases
// where `M` is a full bucket or table reference type with mutable access to the table.
impl<K, V, M> FullBucket<K, V, M>
    where M: Put<K, V>
{
    pub fn replace(&mut self, h: SafeHash, k: K, v: V) -> (SafeHash, K, V) {
        unsafe {
            let old_hash = ptr::replace(self.raw.hash() as *mut SafeHash, h);
            let (old_key, old_val) = ptr::replace(self.raw.pair(), (k, v));

            (old_hash, old_key, old_val)
        }
    }
}

impl<K, V, M> FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>> + DerefMut
{
    /// Gets mutable references to the key and value at a given index.
    pub fn read_mut(&mut self) -> (&mut K, &mut V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&mut (*pair_ptr).0, &mut (*pair_ptr).1)
        }
    }
}

impl<'t, K, V, M> FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>> + 't
{
    /// Exchange a bucket state for immutable references into the table.
    /// Because the underlying reference to the table is also consumed,
    /// no further changes to the structure of the table are possible;
    /// in exchange for this, the returned references have a longer lifetime
    /// than the references returned by `read()`.
    pub fn into_refs(self) -> (&'t K, &'t V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&(*pair_ptr).0, &(*pair_ptr).1)
        }
    }
}

impl<'t, K, V, M> FullBucket<K, V, M>
    where M: Deref<Target = RawTable<K, V>> + DerefMut + 't
{
    /// This works similarly to `into_refs`, exchanging a bucket state
    /// for mutable references into the table.
    pub fn into_mut_refs(self) -> (&'t mut K, &'t mut V) {
        unsafe {
            let pair_ptr = self.raw.pair();
            (&mut (*pair_ptr).0, &mut (*pair_ptr).1)
        }
    }
}

impl<K, V, M> GapThenFull<K, V, M>
    where M: Deref<Target = RawTable<K, V>>
{
    #[inline]
    pub fn full(&self) -> &FullBucket<K, V, M> {
        &self.full
    }

    pub fn into_table(self) -> M {
        self.full.into_table()
    }

    pub fn shift(mut self) -> Result<GapThenFull<K, V, M>, Bucket<K, V, M>> {
        unsafe {
            let (gap_hash, gap_pair) = self.gap.raw.hash_pair();
            let (full_hash, full_pair) = self.full.raw.hash_pair();
            *gap_hash = mem::replace(&mut *full_hash, EMPTY_BUCKET);
            ptr::copy_nonoverlapping(full_pair, gap_pair, 1);
        }

        let FullBucket { raw: prev_raw, .. } = self.full;

        match self.full.next().peek() {
            Full(bucket) => {
                self.gap.raw = prev_raw;

                self.full = bucket;

                Ok(self)
            }
            Empty(b) => Err(b.into_bucket()),
        }
    }
}

// Returns a Layout which describes the allocation required for a hash table,
// and the offset of the array of (key, value) pairs in the allocation.
#[inline(always)]
fn calculate_layout<K, V>(capacity: usize) -> Result<(Layout, usize), LayoutErr> {
    let hashes = Layout::array::<HashUint>(capacity)?;
    let pairs = Layout::array::<(K, V)>(capacity)?;
    hashes.extend(pairs).map(|(layout, _)| {
        // LLVM seems to have trouble properly const-propagating pairs.align(),
        // possibly due to the use of NonZeroUsize. This little hack allows it
        // to generate optimal code.
        //
        // See https://github.com/rust-lang/rust/issues/51346 for more details.
        (
            layout,
            hashes.size() + hashes.padding_needed_for(mem::align_of::<(K, V)>()),
        )
    })
}

pub(crate) enum Fallibility {
    Fallible,
    Infallible,
}

use self::Fallibility::*;

impl<K, V> RawTable<K, V> {
    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    /// Returns an error if it cannot allocate or capacity overflows.
    unsafe fn new_uninitialized_internal(
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<RawTable<K, V>, CollectionAllocErr> {
        if capacity == 0 {
            return Ok(RawTable {
                size: 0,
                capacity_mask: capacity.wrapping_sub(1),
                hashes: TaggedHashUintPtr::new(EMPTY as *mut HashUint),
                marker: marker::PhantomData,
            });
        }

        // Allocating hashmaps is a little tricky. We need to allocate two
        // arrays, but since we know their sizes and alignments up front,
        // we just allocate a single array, and then have the subarrays
        // point into it.
        let (layout, _) = calculate_layout::<K, V>(capacity)?;
        let buffer = Global.alloc(layout).map_err(|e| match fallibility {
            Infallible => handle_alloc_error(layout),
            Fallible => e,
        })?;

        Ok(RawTable {
            capacity_mask: capacity.wrapping_sub(1),
            size: 0,
            hashes: TaggedHashUintPtr::new(buffer.cast().as_ptr()),
            marker: marker::PhantomData,
        })
    }

    /// Does not initialize the buckets. The caller should ensure they,
    /// at the very least, set every hash to EMPTY_BUCKET.
    unsafe fn new_uninitialized(capacity: usize) -> RawTable<K, V> {
        match Self::new_uninitialized_internal(capacity, Infallible) {
            Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
            Err(CollectionAllocErr::AllocErr) => unreachable!(),
            Ok(table) => { table }
        }
    }

    #[inline(always)]
    fn raw_bucket_at(&self, index: usize) -> RawBucket<K, V> {
        let (_, pairs_offset) = calculate_layout::<K, V>(self.capacity())
            .unwrap_or_else(|_| unsafe { hint::unreachable_unchecked() });
        let buffer = self.hashes.ptr() as *mut u8;
        unsafe {
            RawBucket {
                hash_start: buffer as *mut HashUint,
                pair_start: buffer.add(pairs_offset) as *const (K, V),
                idx: index,
                _marker: marker::PhantomData,
            }
        }
    }

    #[inline]
    fn new_internal(
        capacity: usize,
        fallibility: Fallibility,
    ) -> Result<RawTable<K, V>, CollectionAllocErr> {
        unsafe {
            let ret = RawTable::new_uninitialized_internal(capacity, fallibility)?;
            if capacity > 0 {
                ptr::write_bytes(ret.hashes.ptr(), 0, capacity);
            }
            Ok(ret)
        }
    }

    /// Tries to create a new raw table from a given capacity. If it cannot allocate,
    /// it returns with AllocErr.
    #[inline]
    pub fn try_new(capacity: usize) -> Result<RawTable<K, V>, CollectionAllocErr> {
        Self::new_internal(capacity, Fallible)
    }

    /// Creates a new raw table from a given capacity. All buckets are
    /// initially empty.
    #[inline]
    pub fn new(capacity: usize) -> RawTable<K, V> {
        match Self::new_internal(capacity, Infallible) {
            Err(CollectionAllocErr::CapacityOverflow) => panic!("capacity overflow"),
            Err(CollectionAllocErr::AllocErr) => unreachable!(),
            Ok(table) => { table }
        }
    }

    /// The hashtable's capacity, similar to a vector's.
    pub fn capacity(&self) -> usize {
        self.capacity_mask.wrapping_add(1)
    }

    /// The number of elements ever `put` in the hashtable, minus the number
    /// of elements ever `take`n.
    pub fn size(&self) -> usize {
        self.size
    }

    fn raw_buckets(&self) -> RawBuckets<K, V> {
        RawBuckets {
            raw: self.raw_bucket_at(0),
            elems_left: self.size,
            marker: marker::PhantomData,
        }
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.raw_buckets(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            iter: self.raw_buckets(),
            _marker: marker::PhantomData,
        }
    }

    pub fn into_iter(self) -> IntoIter<K, V> {
        let RawBuckets { raw, elems_left, .. } = self.raw_buckets();
        // Replace the marker regardless of lifetime bounds on parameters.
        IntoIter {
            iter: RawBuckets {
                raw,
                elems_left,
                marker: marker::PhantomData,
            },
            table: self,
        }
    }

    pub fn drain(&mut self) -> Drain<K, V> {
        let RawBuckets { raw, elems_left, .. } = self.raw_buckets();
        // Replace the marker regardless of lifetime bounds on parameters.
        Drain {
            iter: RawBuckets {
                raw,
                elems_left,
                marker: marker::PhantomData,
            },
            table: NonNull::from(self),
            marker: marker::PhantomData,
        }
    }

    /// Drops buckets in reverse order. It leaves the table in an inconsistent
    /// state and should only be used for dropping the table's remaining
    /// entries. It's used in the implementation of Drop.
    unsafe fn rev_drop_buckets(&mut self) {
        // initialize the raw bucket past the end of the table
        let mut raw = self.raw_bucket_at(self.capacity());
        let mut elems_left = self.size;

        while elems_left != 0 {
            raw.idx -= 1;

            if *raw.hash() != EMPTY_BUCKET {
                elems_left -= 1;
                ptr::drop_in_place(raw.pair());
            }
        }
    }

    /// Sets the table tag.
    pub fn set_tag(&mut self, value: bool) {
        self.hashes.set_tag(value)
    }

    /// Gets the table tag.
    pub fn tag(&self) -> bool {
        self.hashes.tag()
    }
}

/// A raw iterator. The basis for some other iterators in this module. Although
/// this interface is safe, it's not used outside this module.
struct RawBuckets<'a, K, V> {
    raw: RawBucket<K, V>,
    elems_left: usize,

    // Strictly speaking, this should be &'a (K,V), but that would
    // require that K:'a, and we often use RawBuckets<'static...> for
    // move iterations, so that messes up a lot of other things. So
    // just use `&'a (K,V)` as this is not a publicly exposed type
    // anyway.
    marker: marker::PhantomData<&'a ()>,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for RawBuckets<'_, K, V> {
    fn clone(&self) -> Self {
        RawBuckets {
            raw: self.raw,
            elems_left: self.elems_left,
            marker: marker::PhantomData,
        }
    }
}


impl<'a, K, V> Iterator for RawBuckets<'a, K, V> {
    type Item = RawBucket<K, V>;

    fn next(&mut self) -> Option<RawBucket<K, V>> {
        if self.elems_left == 0 {
            return None;
        }

        loop {
            unsafe {
                let item = self.raw;
                self.raw.idx += 1;
                if *item.hash() != EMPTY_BUCKET {
                    self.elems_left -= 1;
                    return Some(item);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}

impl<K, V> ExactSizeIterator for RawBuckets<'_, K, V> {
    fn len(&self) -> usize {
        self.elems_left
    }
}

/// Iterator over shared references to entries in a table.
pub struct Iter<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
}

unsafe impl<K: Sync, V: Sync> Sync for Iter<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Send for Iter<'_, K, V> {}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

/// Iterator over mutable references to entries in a table.
pub struct IterMut<'a, K: 'a, V: 'a> {
    iter: RawBuckets<'a, K, V>,
    // To ensure invariance with respect to V
    _marker: marker::PhantomData<&'a mut V>,
}

unsafe impl<K: Sync, V: Sync> Sync for IterMut<'_, K, V> {}
// Both K: Sync and K: Send are correct for IterMut's Send impl,
// but Send is the more useful bound
unsafe impl<K: Send, V: Send> Send for IterMut<'_, K, V> {}

impl<'a, K: 'a, V: 'a> IterMut<'a, K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

/// Iterator over the entries in a table, consuming the table.
pub struct IntoIter<K, V> {
    table: RawTable<K, V>,
    iter: RawBuckets<'static, K, V>,
}

unsafe impl<K: Sync, V: Sync> Sync for IntoIter<K, V> {}
unsafe impl<K: Send, V: Send> Send for IntoIter<K, V> {}

impl<K, V> IntoIter<K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

/// Iterator over the entries in a table, clearing the table.
pub struct Drain<'a, K: 'a, V: 'a> {
    table: NonNull<RawTable<K, V>>,
    iter: RawBuckets<'static, K, V>,
    marker: marker::PhantomData<&'a RawTable<K, V>>,
}

unsafe impl<K: Sync, V: Sync> Sync for Drain<'_, K, V> {}
unsafe impl<K: Send, V: Send> Send for Drain<'_, K, V> {}

impl<'a, K, V> Drain<'a, K, V> {
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.iter.next().map(|raw| unsafe {
            let pair_ptr = raw.pair();
            (&(*pair_ptr).0, &(*pair_ptr).1)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.iter.next().map(|raw| unsafe {
            let pair_ptr = raw.pair();
            (&(*pair_ptr).0, &mut (*pair_ptr).1)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (SafeHash, K, V);

    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|raw| {
            self.table.size -= 1;
            unsafe {
                let (k, v) = ptr::read(raw.pair());
                (SafeHash { hash: *raw.hash() }, k, v)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.iter().len()
    }
}

impl<'a, K, V> Iterator for Drain<'a, K, V> {
    type Item = (SafeHash, K, V);

    #[inline]
    fn next(&mut self) -> Option<(SafeHash, K, V)> {
        self.iter.next().map(|raw| {
            unsafe {
                self.table.as_mut().size -= 1;
                let (k, v) = ptr::read(raw.pair());
                (SafeHash { hash: ptr::replace(&mut *raw.hash(), EMPTY_BUCKET) }, k, v)
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<K, V> ExactSizeIterator for Drain<'_, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<K, V> Drop for Drain<'_, K, V> {
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> RawTable<K, V> {
        unsafe {
            let cap = self.capacity();
            let mut new_ht = RawTable::new_uninitialized(cap);

            let mut new_buckets = new_ht.raw_bucket_at(0);
            let mut buckets = self.raw_bucket_at(0);
            while buckets.idx < cap {
                *new_buckets.hash() = *buckets.hash();
                if *new_buckets.hash() != EMPTY_BUCKET {
                    let pair_ptr = buckets.pair();
                    let kv = ((*pair_ptr).0.clone(), (*pair_ptr).1.clone());
                    ptr::write(new_buckets.pair(), kv);
                }
                buckets.idx += 1;
                new_buckets.idx += 1;
            }

            new_ht.size = self.size();
            new_ht.set_tag(self.tag());

            new_ht
        }
    }
}

unsafe impl<#[may_dangle] K, #[may_dangle] V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.capacity() == 0 {
            return;
        }

        // This is done in reverse because we've likely partially taken
        // some elements out with `.into_iter()` from the front.
        // Check if the size is 0, so we don't do a useless scan when
        // dropping empty tables such as on resize.
        // Also avoid double drop of elements that have been already moved out.
        unsafe {
            if needs_drop::<(K, V)>() {
                // avoid linear runtime for types that don't need drop
                self.rev_drop_buckets();
            }
        }

        let (layout, _) = calculate_layout::<K, V>(self.capacity())
            .unwrap_or_else(|_| unsafe { hint::unreachable_unchecked() });
        unsafe {
            Global.dealloc(NonNull::new_unchecked(self.hashes.ptr()).cast(), layout);
            // Remember how everything was allocated out of one buffer
            // during initialization? We only need one call to free here.
        }
    }
}
