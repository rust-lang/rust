// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unordered containers, implemented as hash-tables (`HashSet` and `HashMap` types)

use std::container::{Container, Mutable, Map, MutableMap, Set, MutableSet};
use std::clone::Clone;
use std::cmp::{Eq, TotalEq, Equiv, max};
use std::default::Default;
use std::fmt;
use std::fmt::Show;
use std::hash::{Hash, Hasher, sip};
use std::iter;
use std::iter::{Iterator, FromIterator, Extendable};
use std::iter::{FilterMap, Chain, Repeat, Zip};
use std::iter::{range, range_inclusive};
use std::mem::replace;
use std::num;
use std::option::{Option, Some, None};
use rand;
use rand::Rng;
use std::result::{Ok, Err};
use std::slice::ImmutableVector;

mod table {
    use std::clone::Clone;
    use std::cmp::Eq;
    use std::hash::{Hash, Hasher};
    use std::kinds::marker;
    use std::libc;
    use std::num::CheckedMul;
    use std::option::{Option, Some, None};
    use std::prelude::Drop;
    use std::ptr;
    use std::ptr::RawPtr;
    use std::rt::global_heap;
    use std::intrinsics::{size_of, transmute, move_val_init};
    use std::iter::{Iterator, range_step_inclusive};

    static EMPTY_BUCKET: u64 = 0u64;

    /// The raw hashtable, providing safe-ish access to the unzipped and highly
    /// optimized arrays of hashes, keys, and values.
    ///
    /// This design uses less memory and is a lot faster than the naive
    /// `~[Option<u64, K, V>]`, because we don't pay for the overhead of an
    /// option on every element, and we get a generally more cache-aware design.
    ///
    /// Key invariants of this structure:
    ///
    ///   - if hashes[i] == EMPTY_BUCKET, then keys[i] and vals[i] have
    ///     'undefined' contents. Don't read from them. This invariant is
    ///     enforced outside this module with the [EmptyIndex], [FullIndex],
    ///     and [SafeHash] types/concepts.
    ///
    ///   - An `EmptyIndex` is only constructed for a bucket at an index with
    ///     a hash of EMPTY_BUCKET.
    ///
    ///   - A `FullIndex` is only constructed for a bucket at an index with a
    ///     non-EMPTY_BUCKET hash.
    ///
    ///   - A `SafeHash` is only constructed for non-`EMPTY_BUCKET` hash. We get
    ///     around hashes of zero by changing them to 0x800_0000, which will
    ///     likely hash to the same bucket, but not be represented as "empty".
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
    /// but in general is just a tricked out `~[Option<u64, K, V>]`.
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
    pub struct RawTable<K, V> {
        priv capacity: uint,
        priv size:     uint,
        priv hashes:   *mut u64,
        priv keys:     *mut K,
        priv vals:     *mut V,
    }

    /// Represents an index into a `RawTable` with no key or value in it.
    pub struct EmptyIndex {
        priv idx:   int,
        priv nocopy: marker::NoCopy,
    }

    /// Represents an index into a `RawTable` with a key, value, and hash
    /// in it.
    pub struct FullIndex {
        priv idx:   int,
        priv hash:  SafeHash,
        priv nocopy: marker::NoCopy,
    }

    impl FullIndex {
        /// Since we get the hash for free whenever we check the bucket state,
        /// this function is provided for fast access, letting us avoid making
        /// redundant trips back to the hashtable.
        pub fn hash(&self) -> SafeHash { self.hash }

        /// Same comment as with `hash`.
        pub fn raw_index(&self) -> uint { self.idx as uint }
    }

    /// Represents the state of a bucket: it can either have a key/value
    /// pair (be full) or not (be empty). You cannot `take` empty buckets,
    /// and you cannot `put` into full buckets.
    pub enum BucketState {
        Empty(EmptyIndex),
        Full(FullIndex),
    }

    /// A hash that is not zero, since we use that to represent empty buckets.
    #[deriving(Eq)]
    pub struct SafeHash {
        priv hash: u64,
    }

    impl SafeHash {
        /// Peek at the hash value, which is guaranteed to be non-zero.
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

    impl<K, V> RawTable<K, V> {

        /// Does not initialize the buckets. The caller should ensure they,
        /// at the very least, set every hash to EMPTY_BUCKET.
        unsafe fn new_uninitialized(capacity: uint) -> RawTable<K, V> {
            let hashes_size =
                capacity.checked_mul(&size_of::<u64>()).expect("capacity overflow");
            let keys_size   =
                capacity.checked_mul(&size_of::< K >()).expect("capacity overflow");
            let vals_size   =
                capacity.checked_mul(&size_of::< V >()).expect("capacity overflow");

            /*
            The following code was my first pass at making RawTable only
            allocate a single buffer, since that's all it needs. There's
            no logical reason for this to require three calls to malloc.

            However, I'm not convinced the code below is correct. If you
            want to take a stab at it, please do! The alignment is
            especially tricky to get right, especially if you need more
            alignment than malloc guarantees.

            let hashes_offset = 0;
            let keys_offset   = align_size(hashes_offset + hashes_size, keys_align);
            let vals_offset   = align_size(keys_offset + keys_size, vals_align);
            let end = vals_offset + vals_size;

            let buffer = global_heap::malloc_raw(end);

            let hashes = buffer.offset(hashes_offset) as *mut u64;
            let keys   = buffer.offset(keys_offset)   as *mut K;
            let vals   = buffer.offset(vals_offset)   as *mut V;

            */

            let hashes = global_heap::malloc_raw(hashes_size) as *mut u64;
            let keys   = global_heap::malloc_raw(keys_size)   as *mut K;
            let vals   = global_heap::malloc_raw(vals_size)   as *mut V;

            RawTable {
                capacity: capacity,
                size:     0,
                hashes:   hashes,
                keys:     keys,
                vals:     vals,
            }
        }



        /// Creates a new raw table from a given capacity. All buckets are
        /// initially empty.
        pub fn new(capacity: uint) -> RawTable<K, V> {
            unsafe {
                let ret = RawTable::new_uninitialized(capacity);

                for i in range(0, ret.capacity() as int) {
                    *ret.hashes.offset(i) = EMPTY_BUCKET;
                }

                ret
            }
        }

        /// Reads a bucket at a given index, returning an enum indicating whether
        /// there's anything there or not. You need to match on this enum to get
        /// the appropriate types to pass on to most of the rest of the functions
        /// in this module.
        pub fn peek(&self, index: uint) -> BucketState {
            // FIXME #12049
            if cfg!(test) { assert!(index < self.capacity) }

            let idx  = index as int;
            let hash = unsafe { *self.hashes.offset(idx) };

            let nocopy = marker::NoCopy;

            match hash {
                EMPTY_BUCKET =>
                    Empty(EmptyIndex {
                        idx: idx,
                        nocopy: nocopy
                    }),
                full_hash =>
                    Full(FullIndex {
                        idx:   idx,
                        hash:  SafeHash { hash: full_hash },
                        nocopy: nocopy,
                    })
            }
        }

        /// Gets references to the key and value at a given index.
        pub fn read<'a>(&'a self, index: &FullIndex) -> (&'a K, &'a V) {
            let idx = index.idx;

            unsafe {
                // FIXME #12049
                if cfg!(test) { assert!(*self.hashes.offset(idx) != EMPTY_BUCKET) }
                (&'a *self.keys.offset(idx),
                 &'a *self.vals.offset(idx))
            }
        }

        /// Gets references to the key and value at a given index, with the
        /// value's reference being mutable.
        pub fn read_mut<'a>(&'a mut self, index: &FullIndex) -> (&'a K, &'a mut V) {
            let idx = index.idx;

            unsafe {
                // FIXME #12049
                if cfg!(test) { assert!(*self.hashes.offset(idx) != EMPTY_BUCKET) }
                (&'a     *self.keys.offset(idx),
                 &'a mut *self.vals.offset(idx))
            }
        }

        /// Read everything, mutably.
        pub fn read_all_mut<'a>(&'a mut self, index: &FullIndex)
            -> (&'a mut SafeHash, &'a mut K, &'a mut V) {
            let idx = index.idx;

            // I'm totally abusing the fact that a pointer to any u64 in the
            // hashtable at a full index is a safe hash. Thanks to `SafeHash`
            // just being a wrapper around u64, this is true. It's just really
            // really really really unsafe. However, the exposed API is now
            // impossible to get wrong. You cannot insert an empty hash into
            // this slot now.

            unsafe {
                // FIXME #12049
                if cfg!(test) { assert!(*self.hashes.offset(idx) != EMPTY_BUCKET) }
                (transmute(self.hashes.offset(idx)),
                 &'a mut *self.keys.offset(idx),
                 &'a mut *self.vals.offset(idx))
            }
        }

        /// Puts a key and value pair, along with the key's hash, into a given
        /// index in the hashtable. Note how the `EmptyIndex` is 'moved' into this
        /// function, because that slot will no longer be empty when we return!
        /// Because we know this, a FullIndex is returned for later use, pointing
        /// to the newly-filled slot in the hashtable.
        ///
        /// Use `make_hash` to construct a `SafeHash` to pass to this function.
        pub fn put(&mut self, index: EmptyIndex, hash: SafeHash, k: K, v: V) -> FullIndex {
            let idx = index.idx;

            unsafe {
                // FIXME #12049
                if cfg!(test) { assert!(*self.hashes.offset(idx) == EMPTY_BUCKET) }
                *self.hashes.offset(idx) = hash.inspect();
                move_val_init(&mut *self.keys.offset(idx), k);
                move_val_init(&mut *self.vals.offset(idx), v);
            }

            self.size += 1;

            FullIndex { idx: idx, hash: hash, nocopy: marker::NoCopy }
        }

        /// Removes a key and value from the hashtable.
        ///
        /// This works similarly to `put`, building an `EmptyIndex` out of the
        /// taken FullIndex.
        pub fn take(&mut self, index: FullIndex) -> (EmptyIndex, K, V) {
            let idx  = index.idx;

            unsafe {
                // FIXME #12049
                if cfg!(test) { assert!(*self.hashes.offset(idx) != EMPTY_BUCKET) }

                let hash_ptr = self.hashes.offset(idx);

                *hash_ptr = EMPTY_BUCKET;

                // Drop the mutable constraint.
                let keys = self.keys as *K;
                let vals = self.vals as *V;

                let k = ptr::read(keys.offset(idx));
                let v = ptr::read(vals.offset(idx));

                self.size -= 1;

                (EmptyIndex { idx: idx, nocopy: marker::NoCopy }, k, v)
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

        pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
            Entries { table: self, idx: 0 }
        }

        pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, V> {
            MutEntries { table: self, idx: 0 }
        }

        pub fn move_iter(self) -> MoveEntries<K, V> {
            MoveEntries { table: self, idx: 0 }
        }
    }

    pub struct Entries<'a, K, V> {
        priv table: &'a RawTable<K, V>,
        priv idx: uint,
    }

    pub struct MutEntries<'a, K, V> {
        priv table: &'a mut RawTable<K, V>,
        priv idx: uint,
    }

    pub struct MoveEntries<K, V> {
        priv table: RawTable<K, V>,
        priv idx: uint,
    }

    impl<'a, K, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
        fn next(&mut self) -> Option<(&'a K, &'a V)> {
            while self.idx < self.table.capacity() {
                let i = self.idx;
                self.idx += 1;

                match self.table.peek(i) {
                    Empty(_)  => {},
                    Full(idx) => return Some(self.table.read(&idx))
                }
            }

            None
        }

        fn size_hint(&self) -> (uint, Option<uint>) {
            let size = self.table.size() - self.idx;
            (size, Some(size))
        }
    }

    impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
        fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
            while self.idx < self.table.capacity() {
                let i = self.idx;
                self.idx += 1;

                match self.table.peek(i) {
                    Empty(_)  => {},
                    // the transmute here fixes:
                    // error: lifetime of `self` is too short to guarantee its contents
                    //        can be safely reborrowed
                    Full(idx) => unsafe {
                        return Some(transmute(self.table.read_mut(&idx)))
                    }
                }
            }

            None
        }

        fn size_hint(&self) -> (uint, Option<uint>) {
            let size = self.table.size() - self.idx;
            (size, Some(size))
        }
    }

    impl<K, V> Iterator<(SafeHash, K, V)> for MoveEntries<K, V> {
        fn next(&mut self) -> Option<(SafeHash, K, V)> {
            while self.idx < self.table.capacity() {
                let i = self.idx;
                self.idx += 1;

                match self.table.peek(i) {
                    Empty(_) => {},
                    Full(idx) => {
                        let h = idx.hash();
                        let (_, k, v) = self.table.take(idx);
                        return Some((h, k, v));
                    }
                }
            }

            None
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

                for i in range(0, self.capacity()) {
                    match self.peek(i) {
                        Empty(_)  => {
                            *new_ht.hashes.offset(i as int) = EMPTY_BUCKET;
                        },
                        Full(idx) => {
                            let hash = idx.hash().inspect();
                            let (k, v) = self.read(&idx);
                            *new_ht.hashes.offset(i as int) = hash;
                            move_val_init(&mut *new_ht.keys.offset(i as int), (*k).clone());
                            move_val_init(&mut *new_ht.vals.offset(i as int), (*v).clone());
                        }
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
            // Ideally, this should be in reverse, since we're likely to have
            // partially taken some elements out with `.move_iter()` from the
            // front.
            for i in range_step_inclusive(self.capacity as int - 1, 0, -1) {
                // Check if the size is 0, so we don't do a useless scan when
                // dropping empty tables such as on resize.

                if self.size == 0 { break }

                match self.peek(i as uint) {
                    Empty(_)  => {},
                    Full(idx) => { self.take(idx); }
                }
            }

            assert!(self.size == 0);

            unsafe {
                libc::free(self.vals   as *mut libc::c_void);
                libc::free(self.keys   as *mut libc::c_void);
                libc::free(self.hashes as *mut libc::c_void);
            }
        }
    }
}

// We use this type for the load factor, to avoid floating point operations
// which might not be supported efficiently on some hardware.
//
// We use small u16s here to save space in the hashtable. They get upcasted
// to u64s when we actually use them.
type Fraction = (u16, u16); // (numerator, denominator)

// multiplication by a fraction, in a way that won't generally overflow for
// array sizes outside a factor of 10 of U64_MAX.
fn fraction_mul(lhs: uint, (num, den): Fraction) -> uint {
    (((lhs as u64) * (num as u64)) / (den as u64)) as uint
}

static INITIAL_LOG2_CAP: uint = 5;
static INITIAL_CAPACITY: uint = 1 << INITIAL_LOG2_CAP; // 2^5
static INITIAL_LOAD_FACTOR: Fraction = (9, 10);

// The main performance trick in this hashmap is called Robin Hood Hashing.
// It gains its excellent performance from one key invariant:
//
//    If an insertion collides with an existing element, and that elements
//    "probe distance" (how far away the element is from its ideal location)
//    is higher than how far we've already probed, swap the elements.
//
// This massively lowers variance in probe distance, and allows us to get very
// high load factors with good performance. The 90% load factor I use is rather
// conservative.
//
// > Why a load factor of 90%?
//
// In general, all the distances to inital buckets will converge on the mean.
// At a load factor of α, the odds of finding the target bucket after k
// probes is approximately 1-α^k. If we set this equal to 50% (since we converge
// on the mean) and set k=8 (64-byte cache line / 8-byte hash), α=0.92. I round
// this down to 0.90 to make the math easier on the CPU and avoid its FPU.
// Since on average we start the probing in the middle of a cache line, this
// strategy pulls in two cache lines of hashes on every lookup. I think that's
// pretty good, but if you want to trade off some space, it could go down to one
// cache line on average with an α of 0.84.
//
// > Wait, what? Where did you get 1-α^k from?
//
// On the first probe, your odds of a collision with an existing element is α.
// The odds of doing this twice in a row is approximatelly α^2. For three times,
// α^3, etc. Therefore, the odds of colliding k times is α^k. The odds of NOT
// colliding after k tries is 1-α^k.
//
// Future Improvements (FIXME!)
// ============================
//
// Allow the load factor to be changed dynamically and/or at initialization.
// I'm having trouble figuring out a sane API for this without exporting my
// hackish fraction type, while still avoiding floating point.
//
// Also, would it be possible for us to reuse storage when growing the
// underlying table? This is exactly the use case for 'realloc', and may
// be worth exploring.
//
// Future Optimizations (FIXME!)
// =============================
//
// The paper cited below mentions an implementation which keeps track of the
// distance-to-initial-bucket histogram. I'm suspicious of this approach because
// it requires maintaining an internal map. If this map were replaced with a
// hashmap, it would be faster, but now our data structure is self-referential
// and blows up. Also, this allows very good first guesses, but array accesses
// are no longer linear and in one direction, as we have now. There is also
// memory and cache pressure that this map would entail that would be very
// difficult to properly see in a microbenchmark.
//
// Another possible design choice that I made without any real reason is
// parameterizing the raw table over keys and values. Technically, all we need
// is the size and alignment of keys and values, and the code should be just as
// efficient (well, we might need one for power-of-two size and one for not...).
// This has the potential to reduce code bloat in rust executables, without
// really losing anything except 4 words (key size, key alignment, val size,
// val alignment) which can be passed in to every call of a `RawTable` function.
// This would definitely be an avenue worth exploring if people start complaining
// about the size of rust executables.
//
// There's also two optimizations that have been omitted regarding how the
// hashtable allocates. The first is that a hashtable which never has an element
// inserted should not allocate. I'm suspicious of this one, because supporting
// that internally gains no performance over just using an
// `Option<HashMap<K, V>>`, and is significantly more complicated.
//
// The second omitted allocation optimization is that right now we allocate three
// arrays to back the hashtable. This is wasteful. In theory, we only need one
// array, and each of the three original arrays can just be slices of it. This
// would reduce the pressure on the allocator, and will play much nicer with the
// rest of the system. An initial implementation is commented out in
// `table::RawTable::new`, but I'm not confident it works for all sane alignments,
// especially if a type needs more alignment than `malloc` provides.

/// A hash map implementation which uses linear probing with Robin
/// Hood bucket stealing.
///
/// The hashes are all keyed by the task-local random number generator
/// on creation by default, this means the ordering of the keys is
/// randomized, but makes the tables more resistant to
/// denial-of-service attacks (Hash DoS). This behaviour can be
/// overriden with one of the constructors.
///
/// It is required that the keys implement the `Eq` and `Hash` traits, although
/// this can frequently be achieved by using `#[deriving(Eq, Hash)]`.
///
/// Relevant papers/articles:
///
/// 1. Pedro Celis. ["Robin Hood Hashing"](https://cs.uwaterloo.ca/research/tr/1986/CS-86-14.pdf)
/// 2. Emmanuel Goossaert. ["Robin Hood
///    hashing"](http://codecapsule.com/2013/11/11/robin-hood-hashing/)
/// 3. Emmanuel Goossaert. ["Robin Hood hashing: backward shift
///    deletion"](http://codecapsule.com/2013/11/17/robin-hood-hashing-backward-shift-deletion/)
///
/// # Example
///
/// ```rust
/// use collections::HashMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `HashMap<&str, &str>` in this example).
/// let mut book_reviews = HashMap::new();
///
/// // review some books.
/// book_reviews.insert("Adventures of Hucklebury Fin",      "My favorite book.");
/// book_reviews.insert("Grimms' Fairy Tales",               "Masterpiece.");
/// book_reviews.insert("Pride and Prejudice",               "Very enjoyable.");
/// book_reviews.insert("The Adventures of Sherlock Holmes", "Eye lyked it alot.");
///
/// // check for a specific one.
/// if !book_reviews.contains_key(& &"Les Misérables") {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              book_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// book_reviews.remove(& &"The Adventures of Sherlock Holmes");
///
/// // look up the values associated with some keys.
/// let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
/// for book in to_find.iter() {
///     match book_reviews.find(book) {
///         Some(review) => println!("{}: {}", *book, *review),
///         None => println!("{} is unreviewed.", *book)
///     }
/// }
///
/// // iterate over everything.
/// for (book, review) in book_reviews.iter() {
///     println!("{}: \"{}\"", *book, *review);
/// }
/// ```
#[deriving(Clone)]
pub struct HashMap<K, V, H = sip::SipHasher> {
    // All hashes are keyed on these values, to prevent hash collision attacks.
    priv hasher: H,

    // When size == grow_at, we double the capacity.
    priv grow_at: uint,

    // The capacity must never drop below this.
    priv minimum_capacity: uint,

    priv table: table::RawTable<K, V>,

    // We keep this at the end since it's 4-bytes, unlike everything else
    // in this struct. Might as well save a word of padding!
    priv load_factor: Fraction,
}

/// Get the number of elements which will force the capacity to grow.
fn grow_at(capacity: uint, load_factor: Fraction) -> uint {
    fraction_mul(capacity, load_factor)
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S>> HashMap<K, V, H> {
    /// Get the number of elements which will force the capacity to shrink.
    /// When size == self.shrink_at(), we halve the capacity.
    fn shrink_at(&self) -> uint {
        self.table.capacity() >> 2
    }

    // Probe the `idx`th bucket for a given hash, returning the index of the
    // target bucket.
    //
    // This exploits the power-of-two size of the hashtable. As long as this
    // is always true, we can use a bitmask of cap-1 to do modular arithmetic.
    //
    // Prefer to use this with increasing values of `idx` rather than repeatedly
    // calling `probe_next`. This reduces data-dependencies between loops, which
    // can help the optimizer, and certainly won't hurt it. `probe_next` is
    // simply for convenience, and is no more efficient than `probe`.
    fn probe(&self, hash: &table::SafeHash, idx: uint) -> uint {
        let hash_mask = self.table.capacity() - 1;

        // So I heard a rumor that unsigned overflow is safe in rust..
        ((hash.inspect() as uint) + idx) & hash_mask
    }

    // Generate the next probe in a sequence. Prefer to use 'probe' by itself,
    // but this can sometimes be useful.
    fn probe_next(&self, probe: uint) -> uint {
        let hash_mask = self.table.capacity() - 1;
        (probe + 1) & hash_mask
    }

    fn make_hash<X: Hash<S>>(&self, x: &X) -> table::SafeHash {
        table::make_hash(&self.hasher, x)
    }

    /// Get the distance of the bucket at the given index that it lies
    /// from its 'ideal' location.
    ///
    /// In the cited blog posts above, this is called the "distance to
    /// inital bucket", or DIB.
    fn bucket_distance(&self, index_of_elem: &table::FullIndex) -> uint {
        // where the hash of the element that happens to reside at
        // `index_of_elem` tried to place itself first.
        let first_probe_index = self.probe(&index_of_elem.hash(), 0);

        let raw_index = index_of_elem.raw_index();

        if first_probe_index <= raw_index {
             // probe just went forward
            raw_index - first_probe_index
        } else {
            // probe wrapped around the hashtable
            raw_index + (self.table.capacity() - first_probe_index)
        }
    }

    /// Search for a pre-hashed key.
    fn search_hashed_generic(&self, hash: &table::SafeHash, is_match: |&K| -> bool)
        -> Option<table::FullIndex> {
        for num_probes in range(0u, self.table.size()) {
            let probe = self.probe(hash, num_probes);

            let idx = match self.table.peek(probe) {
                table::Empty(_)  => return None, // hit an empty bucket
                table::Full(idx) => idx
            };

            // We can finish the search early if we hit any bucket
            // with a lower distance to initial bucket than we've probed.
            if self.bucket_distance(&idx) < num_probes { return None }

            // If the hash doesn't match, it can't be this one..
            if hash != &idx.hash() { continue }

            let (k, _) = self.table.read(&idx);

            // If the key doesn't match, it can't be this one..
            if !is_match(k) { continue }

            return Some(idx);
        }

        return None
    }

    fn search_hashed(&self, hash: &table::SafeHash, k: &K) -> Option<table::FullIndex> {
        self.search_hashed_generic(hash, |k_| *k == *k_)
    }

    fn search_equiv<Q: Hash<S> + Equiv<K>>(&self, q: &Q) -> Option<table::FullIndex> {
        self.search_hashed_generic(&self.make_hash(q), |k| q.equiv(k))
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    fn search(&self, k: &K) -> Option<table::FullIndex> {
        self.search_hashed(&self.make_hash(k), k)
    }
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S>> Container for HashMap<K, V, H> {
    /// Return the number of elements in the map
    fn len(&self) -> uint { self.table.size() }
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S>> Mutable for HashMap<K, V, H> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) {
        self.minimum_capacity = self.table.size();

        for i in range(0, self.table.capacity()) {
            match self.table.peek(i) {
                table::Empty(_)  => {},
                table::Full(idx) => { self.table.take(idx); }
            }
        }
    }
}


impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S>> Map<K, V> for HashMap<K, V, H> {
    fn find<'a>(&'a self, k: &K) -> Option<&'a V> {
        self.search(k).map(|idx| {
            let (_, v) = self.table.read(&idx);
            v
        })
    }

    fn contains_key(&self, k: &K) -> bool {
        self.search(k).is_some()
    }
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S>> MutableMap<K, V> for HashMap<K, V, H> {
    fn find_mut<'a>(&'a mut self, k: &K) -> Option<&'a mut V> {
        match self.search(k) {
            None => None,
            Some(idx) => {
                let (_, v) = self.table.read_mut(&idx);
                Some(v)
            }
        }
    }

    fn swap(&mut self, k: K, v: V) -> Option<V> {
        let hash = self.make_hash(&k);
        let potential_new_size = self.table.size() + 1;
        self.make_some_room(potential_new_size);

        for dib in range_inclusive(0u, self.table.size()) {
            let probe = self.probe(&hash, dib);

            let idx = match self.table.peek(probe) {
                table::Empty(idx) => {
                    // Found a hole!
                    self.table.put(idx, hash, k, v);
                    return None;
                },
                table::Full(idx) => idx
            };

            if idx.hash() == hash {
                let (bucket_k, bucket_v) = self.table.read_mut(&idx);
                if k == *bucket_k {
                    // Found an existing value.
                    return Some(replace(bucket_v, v));
                }
            }

            let probe_dib = self.bucket_distance(&idx);

            if probe_dib < dib {
                // Found a luckier bucket. This implies that the key does not
                // already exist in the hashtable. Just do a robin hood
                // insertion, then.
                self.robin_hood(idx, probe_dib, hash, k, v);
                return None;
            }
        }

        // We really shouldn't be here.
        fail!("Internal HashMap error: Out of space.");
    }

    fn pop(&mut self, k: &K) -> Option<V> {
        if self.table.size() == 0 {
            return None
        }

        let potential_new_size = self.table.size() - 1;
        self.make_some_room(potential_new_size);

        let starting_index = match self.search(k) {
            Some(idx) => idx,
            None      => return None,
        };

        let starting_probe = starting_index.raw_index();

        let ending_probe = {
            let mut probe = self.probe_next(starting_probe);
            for _ in range(0u, self.table.size()) {
                match self.table.peek(probe) {
                    table::Empty(_) => {}, // empty bucket. this is the end of our shifting.
                    table::Full(idx) => {
                        // Bucket that isn't us, which has a non-zero probe distance.
                        // This isn't the ending index, so keep searching.
                        if self.bucket_distance(&idx) != 0 {
                            probe = self.probe_next(probe);
                            continue;
                        }

                        // if we do have a bucket_distance of zero, we're at the end
                        // of what we need to shift.
                    }
                }
                break;
            }

            probe
        };

        let (_, _, retval) = self.table.take(starting_index);

        let mut      probe = starting_probe;
        let mut next_probe = self.probe_next(probe);

        // backwards-shift all the elements after our newly-deleted one.
        while next_probe != ending_probe {
            match self.table.peek(next_probe) {
                table::Empty(_) => {
                    // nothing to shift in. just empty it out.
                    match self.table.peek(probe) {
                        table::Empty(_) => {},
                        table::Full(idx) => { self.table.take(idx); }
                    }
                },
                table::Full(next_idx) => {
                    // something to shift. move it over!
                    let next_hash = next_idx.hash();
                    let (_, next_key, next_val) = self.table.take(next_idx);
                    match self.table.peek(probe) {
                        table::Empty(idx) => {
                            self.table.put(idx, next_hash, next_key, next_val);
                        },
                        table::Full(idx) => {
                            let (emptyidx, _, _) = self.table.take(idx);
                            self.table.put(emptyidx, next_hash, next_key, next_val);
                        }
                    }
                }
            }

            probe = next_probe;
            next_probe = self.probe_next(next_probe);
        }

        // Done the backwards shift, but there's still an element left!
        // Empty it out.
        match self.table.peek(probe) {
            table::Empty(_) => {},
            table::Full(idx) => { self.table.take(idx); }
        }

        // Now we're done all our shifting. Return the value we grabbed
        // earlier.
        return Some(retval);
    }
}

impl<K: Hash + TotalEq, V> HashMap<K, V, sip::SipHasher> {
    /// Create an empty HashMap.
    pub fn new() -> HashMap<K, V, sip::SipHasher> {
        HashMap::with_capacity(INITIAL_CAPACITY)
    }

    pub fn with_capacity(capacity: uint) -> HashMap<K, V, sip::SipHasher> {
        let mut r = rand::task_rng();
        let r0 = r.gen();
        let r1 = r.gen();
        let hasher = sip::SipHasher::new_with_keys(r0, r1);
        HashMap::with_capacity_and_hasher(capacity, hasher)
    }
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S>> HashMap<K, V, H> {
    pub fn with_hasher(hasher: H) -> HashMap<K, V, H> {
        HashMap::with_capacity_and_hasher(INITIAL_CAPACITY, hasher)
    }

    /// Create an empty HashMap with space for at least `capacity`
    /// elements, using `hasher` to hash the keys.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    pub fn with_capacity_and_hasher(capacity: uint, hasher: H) -> HashMap<K, V, H> {
        let cap = num::next_power_of_two(max(INITIAL_CAPACITY, capacity));
        HashMap {
            hasher:           hasher,
            load_factor:      INITIAL_LOAD_FACTOR,
            grow_at:          grow_at(cap, INITIAL_LOAD_FACTOR),
            minimum_capacity: cap,
            table:            table::RawTable::new(cap),
        }
    }

    /// The hashtable will never try to shrink below this size. You can use
    /// this function to reduce reallocations if your hashtable frequently
    /// grows and shrinks by large amounts.
    ///
    /// This function has no effect on the operational semantics of the
    /// hashtable, only on performance.
    pub fn reserve(&mut self, new_minimum_capacity: uint) {
        let cap = num::next_power_of_two(
            max(INITIAL_CAPACITY, new_minimum_capacity));

        self.minimum_capacity = cap;

        if self.table.capacity() < cap {
            self.resize(cap);
        }
    }

    /// Resizes the internal vectors to a new capacity. It's your responsibility to:
    ///   1) Make sure the new capacity is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure new_capacity is a power of two.
    fn resize(&mut self, new_capacity: uint) {
        assert!(self.table.size() <= new_capacity);
        assert!((new_capacity - 1) & new_capacity == 0);

        self.grow_at = grow_at(new_capacity, self.load_factor);

        let old_table = replace(&mut self.table, table::RawTable::new(new_capacity));
        let old_size  = old_table.size();

        for (h, k, v) in old_table.move_iter() {
            self.manual_insert_hashed_nocheck(h, k, v);
        }

        assert_eq!(self.table.size(), old_size);
    }

    /// Performs any necessary resize operations, such that there's space for
    /// new_size elements.
    fn make_some_room(&mut self, new_size: uint) {
        let should_shrink = new_size <= self.shrink_at();
        let should_grow   = self.grow_at <= new_size;

        if should_grow {
            let new_capacity = self.table.capacity() << 1;
            self.resize(new_capacity);
        } else if should_shrink {
            let new_capacity = self.table.capacity() >> 1;

            // Never shrink below the minimum capacity
            if self.minimum_capacity <= new_capacity {
                self.resize(new_capacity);
            }
        }
    }

    /// Perform robin hood bucket stealing at the given 'index'. You must
    /// also pass that probe's "distance to initial bucket" so we don't have
    /// to recalculate it, as well as the total number of probes already done
    /// so we have some sort of upper bound on the number of probes to do.
    ///
    /// 'hash', 'k', and 'v' are the elements to robin hood into the hashtable.
    fn robin_hood(&mut self, mut index: table::FullIndex, mut dib_param: uint,
                  mut hash: table::SafeHash, mut k: K, mut v: V) {
        'outer: loop {
            let (old_hash, old_key, old_val) = {
                let (old_hash_ref, old_key_ref, old_val_ref) =
                        self.table.read_all_mut(&index);

                let old_hash = replace(old_hash_ref, hash);
                let old_key  = replace(old_key_ref,  k);
                let old_val  = replace(old_val_ref,  v);

                (old_hash, old_key, old_val)
            };

            let mut probe = self.probe_next(index.raw_index());

            for dib in range(dib_param + 1, self.table.size()) {
                let full_index = match self.table.peek(probe) {
                    table::Empty(idx) => {
                        // Finally. A hole!
                        self.table.put(idx, old_hash, old_key, old_val);
                        return;
                    },
                    table::Full(idx) => idx
                };

                let probe_dib = self.bucket_distance(&full_index);

                // Robin hood! Steal the spot.
                if probe_dib < dib {
                    index = full_index;
                    dib_param = probe_dib;
                    hash = old_hash;
                    k = old_key;
                    v = old_val;
                    continue 'outer;
                }

                probe = self.probe_next(probe);
            }

            fail!("HashMap fatal error: 100% load factor?");
        }
    }

    /// Manually insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    ///
    /// If the key already exists, the hashtable will be returned untouched
    /// and a reference to the existing element will be returned.
    fn manual_insert_hashed_nocheck<'a>(
        &'a mut self, hash: table::SafeHash, k: K, v: V) -> &'a mut V {

        for dib in range_inclusive(0u, self.table.size()) {
            let probe = self.probe(&hash, dib);

            let idx = match self.table.peek(probe) {
                table::Empty(idx) => {
                    // Found a hole!
                    let fullidx  = self.table.put(idx, hash, k, v);
                    let (_, val) = self.table.read_mut(&fullidx);
                    return val;
                },
                table::Full(idx) => idx
            };

            if idx.hash() == hash {
                let (bucket_k, bucket_v) = self.table.read_mut(&idx);
                // FIXME #12147 the conditional return confuses
                // borrowck if we return bucket_v directly
                let bv: *mut V = bucket_v;
                if k == *bucket_k {
                    // Key already exists. Get its reference.
                    return unsafe {&mut *bv};
                }
            }

            let probe_dib = self.bucket_distance(&idx);

            if  probe_dib < dib {
                // Found a luckier bucket than me. Better steal his spot.
                self.robin_hood(idx, probe_dib, hash, k, v);

                // Now that it's stolen, just read the value's pointer
                // right out of the table!
                match self.table.peek(probe) {
                    table::Empty(_)  => fail!("Just stole a spot, but now that spot's empty."),
                    table::Full(idx) => {
                        let (_, v) = self.table.read_mut(&idx);
                        return v;
                    }
                }
            }
        }

        // We really shouldn't be here.
        fail!("Internal HashMap error: Out of space.");
    }

    fn manual_insert_hashed<'a>(&'a mut self, hash: table::SafeHash, k: K, v: V) -> &'a mut V {
        let potential_new_size = self.table.size() + 1;
        self.make_some_room(potential_new_size);
        self.manual_insert_hashed_nocheck(hash, k, v)
    }

    /// Inserts an element, returning a reference to that element inside the
    /// hashtable.
    fn manual_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        let hash = self.make_hash(&k);
        self.manual_insert_hashed(hash, k, v)
    }

    /// Return the value corresponding to the key in the map, or insert
    /// and return the value if it doesn't exist.
    pub fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        match self.search(&k) {
            Some(idx) => {
                let (_, v_ref) = self.table.read_mut(&idx);
                v_ref
            },
            None => self.manual_insert(k, v)
        }
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    pub fn find_or_insert_with<'a>(&'a mut self, k: K, f: |&K| -> V)
                               -> &'a mut V {
        match self.search(&k) {
            Some(idx) => {
                let (_, v_ref) = self.table.read_mut(&idx);
                v_ref
            },
            None      => {
                let v = f(&k);
                self.manual_insert(k, v)
            }
        }
    }

    /// Insert a key-value pair into the map if the key is not already present.
    /// Otherwise, modify the existing value for the key.
    /// Returns the new or modified value for the key.
    pub fn insert_or_update_with<'a>(
                                 &'a mut self,
                                 k: K,
                                 v: V,
                                 f: |&K, &mut V|)
                                 -> &'a mut V {
        match self.search(&k) {
            None      => self.manual_insert(k, v),
            Some(idx) => {
                let (_, v_ref) = self.table.read_mut(&idx);
                f(&k, v_ref);
                v_ref
            }
        }
    }

    /// Retrieves a value for the given key, failing if the key is not present.
    pub fn get<'a>(&'a self, k: &K) -> &'a V {
        match self.find(k) {
            Some(v) => v,
            None => fail!("No entry found for key: {:?}", k)
        }
    }

    /// Retrieves a (mutable) value for the given key, failing if the key is not present.
    pub fn get_mut<'a>(&'a mut self, k: &K) -> &'a mut V {
        match self.find_mut(k) {
            Some(v) => v,
            None => fail!("No entry found for key: {:?}", k)
        }
    }

    /// Return true if the map contains a value for the specified key,
    /// using equivalence.
    pub fn contains_key_equiv<Q: Hash<S> + Equiv<K>>(&self, key: &Q) -> bool {
        self.search_equiv(key).is_some()
    }

    /// Return the value corresponding to the key in the map, using
    /// equivalence.
    pub fn find_equiv<'a, Q: Hash<S> + Equiv<K>>(&'a self, k: &Q) -> Option<&'a V> {
        match self.search_equiv(k) {
            None      => None,
            Some(idx) => {
                let (_, v_ref) = self.table.read(&idx);
                Some(v_ref)
            }
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// Iterator element type is &'a K.
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        self.iter().map(|(k, _v)| k)
    }

    /// An iterator visiting all values in arbitrary order.
    /// Iterator element type is &'a V.
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        self.iter().map(|(_k, v)| v)
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// Iterator element type is (&'a K, &'a V).
    pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
        self.table.iter()
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// Iterator element type is (&'a K, &'a mut V).
    pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        self.table.mut_iter()
    }

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    pub fn move_iter(self) -> MoveEntries<K, V> {
        self.table.move_iter().map(|(_, k, v)| (k, v))
    }
}

impl<K: TotalEq + Hash<S>, V: Clone, S, H: Hasher<S>> HashMap<K, V, H> {
    /// Like `find`, but returns a copy of the value.
    pub fn find_copy(&self, k: &K) -> Option<V> {
        self.find(k).map(|v| (*v).clone())
    }

    /// Like `get`, but returns a copy of the value.
    pub fn get_copy(&self, k: &K) -> V {
        (*self.get(k)).clone()
    }
}

impl<K: TotalEq + Hash<S>, V: Eq, S, H: Hasher<S>> Eq for HashMap<K, V, H> {
    fn eq(&self, other: &HashMap<K, V, H>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter().all(|(key, value)| {
            match other.find(key) {
                None    => false,
                Some(v) => *value == *v
            }
        })
    }
}

impl<K: TotalEq + Hash<S> + Show, V: Show, S, H: Hasher<S>> Show for HashMap<K, V, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f.buf, r"\{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f.buf, ", ")); }
            try!(write!(f.buf, "{}: {}", *k, *v));
        }

        write!(f.buf, r"\}")
    }
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S> + Default> Default for HashMap<K, V, H> {
    fn default() -> HashMap<K, V, H> {
        HashMap::with_capacity_and_hasher(INITIAL_CAPACITY, Default::default())
    }
}

/// HashMap iterator
pub type Entries<'a, K, V> = table::Entries<'a, K, V>;

/// HashMap mutable values iterator
pub type MutEntries<'a, K, V> = table::MutEntries<'a, K, V>;

/// HashMap move iterator
pub type MoveEntries<K, V> =
    iter::Map<'static, (table::SafeHash, K, V), (K, V), table::MoveEntries<K, V>>;

/// HashMap keys iterator
pub type Keys<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a K, Entries<'a, K, V>>;

/// HashMap values iterator
pub type Values<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a V, Entries<'a, K, V>>;

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S> + Default> FromIterator<(K, V)> for HashMap<K, V, H> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> HashMap<K, V, H> {
        let (lower, _) = iter.size_hint();
        let mut map = HashMap::with_capacity_and_hasher(lower, Default::default());
        map.extend(iter);
        map
    }
}

impl<K: TotalEq + Hash<S>, V, S, H: Hasher<S> + Default> Extendable<(K, V)> for HashMap<K, V, H> {
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

/// HashSet iterator
pub type SetItems<'a, K> =
    iter::Map<'static, (&'a K, &'a ()), &'a K, Entries<'a, K, ()>>;

/// HashSet move iterator
pub type SetMoveItems<K> =
    iter::Map<'static, (K, ()), K, MoveEntries<K, ()>>;

/// An implementation of a hash set using the underlying representation of a
/// HashMap where the value is (). As with the `HashMap` type, a `HashSet`
/// requires that the elements implement the `Eq` and `Hash` traits.
#[deriving(Clone)]
pub struct HashSet<T, H = sip::SipHasher> {
    priv map: HashMap<T, (), H>
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S>> Eq for HashSet<T, H> {
    // FIXME #11998: Since the value is a (), and `find` returns a Some(&()),
    // we trigger #11998 when matching on it. I've fallen back to manual
    // iteration until this is fixed.
    fn eq(&self, other: &HashSet<T, H>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter().all(|key| other.map.contains_key(key))
    }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S>> Container for HashSet<T, H> {
    /// Return the number of elements in the set
    fn len(&self) -> uint { self.map.len() }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S>> Mutable for HashSet<T, H> {
    /// Clear the set, removing all values.
    fn clear(&mut self) { self.map.clear() }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S>> Set<T> for HashSet<T, H> {
    /// Return true if the set contains a value
    fn contains(&self, value: &T) -> bool { self.map.search(value).is_some() }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    fn is_disjoint(&self, other: &HashSet<T, H>) -> bool {
        self.iter().all(|v| !other.contains(v))
    }

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &HashSet<T, H>) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &HashSet<T, H>) -> bool {
        other.is_subset(self)
    }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S>> MutableSet<T> for HashSet<T, H> {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
}

impl<T: Hash + TotalEq> HashSet<T, sip::SipHasher> {
    /// Create an empty HashSet
    pub fn new() -> HashSet<T, sip::SipHasher> {
        HashSet::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty HashSet with space for at least `n` elements in
    /// the hash table.
    pub fn with_capacity(capacity: uint) -> HashSet<T, sip::SipHasher> {
        HashSet { map: HashMap::with_capacity(capacity) }
    }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S>> HashSet<T, H> {
    pub fn with_hasher(hasher: H) -> HashSet<T, H> {
        HashSet::with_capacity_and_hasher(INITIAL_CAPACITY, hasher)
    }

    /// Create an empty HashSet with space for at least `capacity`
    /// elements in the hash table, using `hasher` to hash the keys.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow `HashSet`s to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    pub fn with_capacity_and_hasher(capacity: uint, hasher: H) -> HashSet<T, H> {
        HashSet { map: HashMap::with_capacity_and_hasher(capacity, hasher) }
    }

    /// Reserve space for at least `n` elements in the hash table.
    pub fn reserve(&mut self, n: uint) {
        self.map.reserve(n)
    }

    /// Returns true if the hash set contains a value equivalent to the
    /// given query value.
    pub fn contains_equiv<Q: Hash<S> + Equiv<T>>(&self, value: &Q) -> bool {
      self.map.contains_key_equiv(value)
    }

    /// An iterator visiting all elements in arbitrary order.
    /// Iterator element type is &'a T.
    pub fn iter<'a>(&'a self) -> SetItems<'a, T> {
        self.map.keys()
    }

    /// Creates a consuming iterator, that is, one that moves each value out
    /// of the set in arbitrary order. The set cannot be used after calling
    /// this.
    pub fn move_iter(self) -> SetMoveItems<T> {
        self.map.move_iter().map(|(k, _)| k)
    }

    /// Visit the values representing the difference
    pub fn difference<'a>(&'a self, other: &'a HashSet<T, H>) -> SetAlgebraItems<'a, T, H> {
        Repeat::new(other)
            .zip(self.iter())
            .filter_map(|(other, elt)| {
                if !other.contains(elt) { Some(elt) } else { None }
            })
    }

    /// Visit the values representing the symmetric difference
    pub fn symmetric_difference<'a>(&'a self, other: &'a HashSet<T, H>)
        -> Chain<SetAlgebraItems<'a, T, H>, SetAlgebraItems<'a, T, H>> {
        self.difference(other).chain(other.difference(self))
    }

    /// Visit the values representing the intersection
    pub fn intersection<'a>(&'a self, other: &'a HashSet<T, H>)
        -> SetAlgebraItems<'a, T, H> {
        Repeat::new(other)
            .zip(self.iter())
            .filter_map(|(other, elt)| {
                if other.contains(elt) { Some(elt) } else { None }
            })
    }

    /// Visit the values representing the union
    pub fn union<'a>(&'a self, other: &'a HashSet<T, H>)
        -> Chain<SetItems<'a, T>, SetAlgebraItems<'a, T, H>> {
        self.iter().chain(other.difference(self))
    }

}

impl<T: TotalEq + Hash<S> + fmt::Show, S, H: Hasher<S>> fmt::Show for HashSet<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f.buf, r"\{"));

        for (i, x) in self.iter().enumerate() {
            if i != 0 { try!(write!(f.buf, ", ")); }
            try!(write!(f.buf, "{}", *x));
        }

        write!(f.buf, r"\}")
    }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S> + Default> FromIterator<T> for HashSet<T, H> {
    fn from_iter<I: Iterator<T>>(iter: I) -> HashSet<T, H> {
        let (lower, _) = iter.size_hint();
        let mut set = HashSet::with_capacity_and_hasher(lower, Default::default());
        set.extend(iter);
        set
    }
}

impl<T: TotalEq + Hash<S>, S, H: Hasher<S> + Default> Extendable<T> for HashSet<T, H> {
    fn extend<I: Iterator<T>>(&mut self, mut iter: I) {
        for k in iter {
            self.insert(k);
        }
    }
}

impl<T: TotalEq + Hash> Default for HashSet<T, sip::SipHasher> {
    fn default() -> HashSet<T> { HashSet::new() }
}

// `Repeat` is used to feed the filter closure an explicit capture
// of a reference to the other set
/// Set operations iterator
pub type SetAlgebraItems<'a, T, H> =
    FilterMap<'static, (&'a HashSet<T, H>, &'a T), &'a T,
              Zip<Repeat<&'a HashSet<T, H>>, SetItems<'a, T>>>;

#[cfg(test)]
mod test_map {
    use super::HashMap;
    use std::iter::{Iterator,range_inclusive,range_step_inclusive};
    use std::local_data;
    use std::vec;

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

        assert!(m.insert(1, 1));

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2));
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4));
        assert_eq!(m.len(), 2);
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert_eq!(*m.find(&2).unwrap(), 4);
    }

    local_data_key!(drop_vector: vec::Vec<int>)

    #[deriving(Hash, Eq, TotalEq)]
    struct Dropable {
        k: int
    }


    impl Dropable {
        fn new(k: int) -> Dropable {
            local_data::get_mut(drop_vector,
                |v| { v.unwrap().as_mut_slice()[k] += 1; });

            Dropable { k: k }
        }
    }

    impl Drop for Dropable {
        fn drop(&mut self) {
            local_data::get_mut(drop_vector, |v|
                { v.unwrap().as_mut_slice()[self.k] -= 1; });
        }
    }

    #[test]
    fn test_drops() {
        local_data::set(drop_vector, vec::Vec::from_elem(200, 0));

        {
            let mut m = HashMap::new();

            local_data::get(drop_vector, |v| {
                for i in range(0, 200) {
                    assert_eq!(v.unwrap().as_slice()[i], 0);
                }
            });

            for i in range(0, 100) {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                m.insert(d1, d2);
            }

            local_data::get(drop_vector, |v| {
                for i in range(0, 200) {
                    assert_eq!(v.unwrap().as_slice()[i], 1);
                }
            });

            for i in range(0, 50) {
                let k = Dropable::new(i);
                let v = m.pop(&k);

                assert!(v.is_some());

                local_data::get(drop_vector, |v| {
                    assert_eq!(v.unwrap().as_slice()[i], 1);
                    assert_eq!(v.unwrap().as_slice()[i+100], 1);
                });
            }

            local_data::get(drop_vector, |v| {
                for i in range(0, 50) {
                    assert_eq!(v.unwrap().as_slice()[i], 0);
                    assert_eq!(v.unwrap().as_slice()[i+100], 0);
                }

                for i in range(50, 100) {
                    assert_eq!(v.unwrap().as_slice()[i], 1);
                    assert_eq!(v.unwrap().as_slice()[i+100], 1);
                }
            });
        }

        local_data::get(drop_vector, |v| {
            for i in range(0, 200) {
                assert_eq!(v.unwrap().as_slice()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_pop() {
        let mut m: HashMap<int, bool> = HashMap::new();
        assert_eq!(m.pop(&0), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = HashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in range(0, 10) {
            assert!(m.is_empty());

            for i in range_inclusive(1, 1000) {
                assert!(m.insert(i, i));

                for j in range_inclusive(1, i) {
                    let r = m.find(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in range_inclusive(i+1, 1000) {
                    let r = m.find(&j);
                    assert_eq!(r, None);
                }
            }

            for i in range_inclusive(1001, 2000) {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in range_inclusive(1, 1000) {
                assert!(m.remove(&i));

                for j in range_inclusive(1, i) {
                    assert!(!m.contains_key(&j));
                }

                for j in range_inclusive(i+1, 1000) {
                    assert!(m.contains_key(&j));
                }
            }

            for i in range_inclusive(1, 1000) {
                assert!(!m.contains_key(&i));
            }

            for i in range_inclusive(1, 1000) {
                assert!(m.insert(i, i));
            }

            // remove backwards
            for i in range_step_inclusive(1000, 1, -1) {
                assert!(m.remove(&i));

                for j in range_inclusive(i, 1000) {
                    assert!(!m.contains_key(&j));
                }

                for j in range_inclusive(1, i-1) {
                    assert!(m.contains_key(&j));
                }
            }
        }
    }

    #[test]
    fn test_find_mut() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 12));
        assert!(m.insert(2, 8));
        assert!(m.insert(5, 14));
        let new = 100;
        match m.find_mut(&5) {
            None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2));
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert!(!m.insert(1, 3));
        assert_eq!(*m.find(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert_eq!(*m.find(&9).unwrap(), 4);
        assert_eq!(*m.find(&5).unwrap(), 3);
        assert_eq!(*m.find(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert!(m.insert(5, 3));
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert_eq!(*m.find(&5).unwrap(), 3);
        assert!(m.insert(9, 4));
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert_eq!(*m.find(&5).unwrap(), 3);
        assert_eq!(*m.find(&9).unwrap(), 4);
        assert!(m.remove(&1));
        assert_eq!(*m.find(&9).unwrap(), 4);
        assert_eq!(*m.find(&5).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(!m.is_empty());
        assert!(m.remove(&1));
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = HashMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_swap() {
        let mut m = HashMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }

    #[test]
    fn test_move_iter() {
        let hm = {
            let mut hm = HashMap::new();

            hm.insert('a', 1);
            hm.insert('b', 2);

            hm
        };

        let v = hm.move_iter().collect::<~[(char, int)]>();
        assert!([('a', 1), ('b', 2)] == v || [('b', 2), ('a', 1)] == v);
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in range(0u, 32) {
            assert!(m.insert(i, i*2));
        }
        assert_eq!(m.len(), 32);

        let mut observed = 0;

        for (k, v) in m.iter() {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = ~[(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.move_iter().collect::<HashMap<int, char>>();
        let keys = map.keys().map(|&k| k).collect::<~[int]>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = ~[(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.move_iter().collect::<HashMap<int, char>>();
        let values = map.values().map(|&v| v).collect::<~[char]>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.find(&1).is_none());
        m.insert(1, 2);
        match m.find(&1) {
            None => fail!(),
            Some(v) => assert!(*v == 2)
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = HashMap::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = HashMap::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_expand() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0u;
        let old_resize_at = m.grow_at;
        while old_resize_at == m.grow_at {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_find_equiv() {
        let mut m = HashMap::new();

        let (foo, bar, baz) = (1,2,3);
        m.insert(~"foo", foo);
        m.insert(~"bar", bar);
        m.insert(~"baz", baz);


        assert_eq!(m.find_equiv(&("foo")), Some(&foo));
        assert_eq!(m.find_equiv(&("bar")), Some(&bar));
        assert_eq!(m.find_equiv(&("baz")), Some(&baz));

        assert_eq!(m.find_equiv(&("qux")), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
        }
    }
}

#[cfg(test)]
mod test_set {
    use super::HashSet;
    use std::container::Container;
    use std::slice::ImmutableEqVector;

    #[test]
    fn test_disjoint() {
        let mut xs = HashSet::new();
        let mut ys = HashSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(ys.insert(-11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = HashSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = HashSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_iterate() {
        let mut a = HashSet::new();
        for i in range(0u, 32) {
            assert!(a.insert(i));
        }
        let mut observed = 0;
        for k in a.iter() {
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_intersection() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));
        assert!(a.insert(-5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(-9));
        assert!(b.insert(-42));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for x in a.intersection(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for x in a.difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(-2));
        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [-2, 1, 5, 11, 14, 22];
        for x in a.symmetric_difference(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(-2));
        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24];
        for x in a.union(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: HashSet<int> = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_move_iter() {
        let hs = {
            let mut hs = HashSet::new();

            hs.insert('a');
            hs.insert('b');

            hs
        };

        let v = hs.move_iter().collect::<~[char]>();
        assert!(['a', 'b'] == v || ['b', 'a'] == v);
    }

    #[test]
    fn test_eq() {
        // These constants once happened to expose a bug in insert().
        // I'm keeping them around to prevent a regression.
        let mut s1 = HashSet::new();

        s1.insert(1);
        s1.insert(2);
        s1.insert(3);

        let mut s2 = HashSet::new();

        s2.insert(1);
        s2.insert(2);

        assert!(s1 != s2);

        s2.insert(3);

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_show() {
        let mut set: HashSet<int> = HashSet::new();
        let empty: HashSet<int> = HashSet::new();

        set.insert(1);
        set.insert(2);

        let set_str = format!("{}", set);

        assert!(set_str == ~"{1, 2}" || set_str == ~"{2, 1}");
        assert_eq!(format!("{}", empty), ~"{}");
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::BenchHarness;
    use std::iter::{range_inclusive};

    #[bench]
    fn insert(b: &mut BenchHarness) {
        use super::HashMap;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1001;

        b.iter(|| {
            m.insert(k, k);
            k += 1;
        });
    }

    #[bench]
    fn find_existing(b: &mut BenchHarness) {
        use super::HashMap;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        b.iter(|| {
            m.contains_key(&412);
        });
    }

    #[bench]
    fn find_nonexisting(b: &mut BenchHarness) {
        use super::HashMap;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        b.iter(|| {
            m.contains_key(&2048);
        });
    }

    #[bench]
    fn hashmap_as_queue(b: &mut BenchHarness) {
        use super::HashMap;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1;

        b.iter(|| {
            m.pop(&k);
            m.insert(k + 1000, k + 1000);
            k += 1;
        });
    }

    #[bench]
    fn find_pop_insert(b: &mut BenchHarness) {
        use super::HashMap;

        let mut m = HashMap::new();

        for i in range_inclusive(1, 1000) {
            m.insert(i, i);
        }

        let mut k = 1;

        b.iter(|| {
            m.find(&(k + 400));
            m.find(&(k + 2000));
            m.pop(&k);
            m.insert(k + 1000, k + 1000);
            k += 1;
        })
    }
}
