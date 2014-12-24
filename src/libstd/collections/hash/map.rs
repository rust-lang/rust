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

use self::Entry::*;
use self::SearchResult::*;
use self::VacantEntryState::*;

use borrow::BorrowFrom;
use clone::Clone;
use cmp::{max, Eq, Equiv, PartialEq};
use default::Default;
use fmt::{mod, Show};
use hash::{Hash, Hasher, RandomSipHasher};
use iter::{mod, Iterator, IteratorExt, FromIterator, Extend, Map};
use kinds::Sized;
use mem::{mod, replace};
use num::{Int, UnsignedInt};
use ops::{Deref, FnMut, Index, IndexMut};
use option::Option;
use option::Option::{Some, None};
use result::Result;
use result::Result::{Ok, Err};

use super::table::{
    mod,
    Bucket,
    EmptyBucket,
    FullBucket,
    FullBucketImm,
    FullBucketMut,
    RawTable,
    SafeHash
};
use super::table::BucketState::{
    Empty,
    Full,
};

const INITIAL_LOG2_CAP: uint = 5;
pub const INITIAL_CAPACITY: uint = 1 << INITIAL_LOG2_CAP; // 2^5

/// The default behavior of HashMap implements a load factor of 90.9%.
/// This behavior is characterized by the following condition:
///
/// - if size > 0.909 * capacity: grow the map
#[deriving(Clone)]
struct DefaultResizePolicy;

impl DefaultResizePolicy {
    fn new() -> DefaultResizePolicy {
        DefaultResizePolicy
    }

    #[inline]
    fn min_capacity(&self, usable_size: uint) -> uint {
        // Here, we are rephrasing the logic by specifying the lower limit
        // on capacity:
        //
        // - if `cap < size * 1.1`: grow the map
        usable_size * 11 / 10
    }

    /// An inverse of `min_capacity`, approximately.
    #[inline]
    fn usable_capacity(&self, cap: uint) -> uint {
        // As the number of entries approaches usable capacity,
        // min_capacity(size) must be smaller than the internal capacity,
        // so that the map is not resized:
        // `min_capacity(usable_capacity(x)) <= x`.
        // The lef-hand side can only be smaller due to flooring by integer
        // division.
        //
        // This doesn't have to be checked for overflow since allocation size
        // in bytes will overflow earlier than multiplication by 10.
        cap * 10 / 11
    }
}

#[test]
fn test_resize_policy() {
    use prelude::*;
    let rp = DefaultResizePolicy;
    for n in range(0u, 1000) {
        assert!(rp.min_capacity(rp.usable_capacity(n)) <= n);
        assert!(rp.usable_capacity(rp.min_capacity(n)) <= n);
    }
}

// The main performance trick in this hashmap is called Robin Hood Hashing.
// It gains its excellent performance from one essential operation:
//
//    If an insertion collides with an existing element, and that element's
//    "probe distance" (how far away the element is from its ideal location)
//    is higher than how far we've already probed, swap the elements.
//
// This massively lowers variance in probe distance, and allows us to get very
// high load factors with good performance. The 90% load factor I use is rather
// conservative.
//
// > Why a load factor of approximately 90%?
//
// In general, all the distances to initial buckets will converge on the mean.
// At a load factor of α, the odds of finding the target bucket after k
// probes is approximately 1-α^k. If we set this equal to 50% (since we converge
// on the mean) and set k=8 (64-byte cache line / 8-byte hash), α=0.92. I round
// this down to make the math easier on the CPU and avoid its FPU.
// Since on average we start the probing in the middle of a cache line, this
// strategy pulls in two cache lines of hashes on every lookup. I think that's
// pretty good, but if you want to trade off some space, it could go down to one
// cache line on average with an α of 0.84.
//
// > Wait, what? Where did you get 1-α^k from?
//
// On the first probe, your odds of a collision with an existing element is α.
// The odds of doing this twice in a row is approximately α^2. For three times,
// α^3, etc. Therefore, the odds of colliding k times is α^k. The odds of NOT
// colliding after k tries is 1-α^k.
//
// The paper from 1986 cited below mentions an implementation which keeps track
// of the distance-to-initial-bucket histogram. This approach is not suitable
// for modern architectures because it requires maintaining an internal data
// structure. This allows very good first guesses, but we are most concerned
// with guessing entire cache lines, not individual indexes. Furthermore, array
// accesses are no longer linear and in one direction, as we have now. There
// is also memory and cache pressure that this would entail that would be very
// difficult to properly see in a microbenchmark.
//
// ## Future Improvements (FIXME!)
//
// Allow the load factor to be changed dynamically and/or at initialization.
//
// Also, would it be possible for us to reuse storage when growing the
// underlying table? This is exactly the use case for 'realloc', and may
// be worth exploring.
//
// ## Future Optimizations (FIXME!)
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
// Annotate exceedingly likely branches in `table::make_hash`
// and `search_hashed` to reduce instruction cache pressure
// and mispredictions once it becomes possible (blocked on issue #11092).
//
// Shrinking the table could simply reallocate in place after moving buckets
// to the first half.
//
// The growth algorithm (fragment of the Proof of Correctness)
// --------------------
//
// The growth algorithm is basically a fast path of the naive reinsertion-
// during-resize algorithm. Other paths should never be taken.
//
// Consider growing a robin hood hashtable of capacity n. Normally, we do this
// by allocating a new table of capacity `2n`, and then individually reinsert
// each element in the old table into the new one. This guarantees that the
// new table is a valid robin hood hashtable with all the desired statistical
// properties. Remark that the order we reinsert the elements in should not
// matter. For simplicity and efficiency, we will consider only linear
// reinsertions, which consist of reinserting all elements in the old table
// into the new one by increasing order of index. However we will not be
// starting our reinsertions from index 0 in general. If we start from index
// i, for the purpose of reinsertion we will consider all elements with real
// index j < i to have virtual index n + j.
//
// Our hash generation scheme consists of generating a 64-bit hash and
// truncating the most significant bits. When moving to the new table, we
// simply introduce a new bit to the front of the hash. Therefore, if an
// elements has ideal index i in the old table, it can have one of two ideal
// locations in the new table. If the new bit is 0, then the new ideal index
// is i. If the new bit is 1, then the new ideal index is n + i. Intutively,
// we are producing two independent tables of size n, and for each element we
// independently choose which table to insert it into with equal probability.
// However the rather than wrapping around themselves on overflowing their
// indexes, the first table overflows into the first, and the first into the
// second. Visually, our new table will look something like:
//
// [yy_xxx_xxxx_xxx|xx_yyy_yyyy_yyy]
//
// Where x's are elements inserted into the first table, y's are elements
// inserted into the second, and _'s are empty sections. We now define a few
// key concepts that we will use later. Note that this is a very abstract
// perspective of the table. A real resized table would be at least half
// empty.
//
// Theorem: A linear robin hood reinsertion from the first ideal element
// produces identical results to a linear naive reinsertion from the same
// element.
//
// FIXME(Gankro, pczarn): review the proof and put it all in a separate doc.rs

/// A hash map implementation which uses linear probing with Robin
/// Hood bucket stealing.
///
/// The hashes are all keyed by the task-local random number generator
/// on creation by default. This means that the ordering of the keys is
/// randomized, but makes the tables more resistant to
/// denial-of-service attacks (Hash DoS). This behaviour can be
/// overridden with one of the constructors.
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
/// ```
/// use std::collections::HashMap;
///
/// // type inference lets us omit an explicit type signature (which
/// // would be `HashMap<&str, &str>` in this example).
/// let mut book_reviews = HashMap::new();
///
/// // review some books.
/// book_reviews.insert("Adventures of Huckleberry Finn",    "My favorite book.");
/// book_reviews.insert("Grimms' Fairy Tales",               "Masterpiece.");
/// book_reviews.insert("Pride and Prejudice",               "Very enjoyable.");
/// book_reviews.insert("The Adventures of Sherlock Holmes", "Eye lyked it alot.");
///
/// // check for a specific one.
/// if !book_reviews.contains_key(&("Les Misérables")) {
///     println!("We've got {} reviews, but Les Misérables ain't one.",
///              book_reviews.len());
/// }
///
/// // oops, this review has a lot of spelling mistakes, let's delete it.
/// book_reviews.remove(&("The Adventures of Sherlock Holmes"));
///
/// // look up the values associated with some keys.
/// let to_find = ["Pride and Prejudice", "Alice's Adventure in Wonderland"];
/// for book in to_find.iter() {
///     match book_reviews.get(book) {
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
///
/// The easiest way to use `HashMap` with a custom type is to derive `Eq` and `Hash`.
/// We must also derive `PartialEq`.
///
/// ```
/// use std::collections::HashMap;
///
/// #[deriving(Hash, Eq, PartialEq, Show)]
/// struct Viking<'a> {
///     name: &'a str,
///     power: uint,
/// }
///
/// let mut vikings = HashMap::new();
///
/// vikings.insert("Norway", Viking { name: "Einar", power: 9u });
/// vikings.insert("Denmark", Viking { name: "Olaf", power: 4u });
/// vikings.insert("Iceland", Viking { name: "Harald", power: 8u });
///
/// // Use derived implementation to print the vikings.
/// for (land, viking) in vikings.iter() {
///     println!("{} at {}", viking, land);
/// }
/// ```
#[deriving(Clone)]
pub struct HashMap<K, V, H = RandomSipHasher> {
    // All hashes are keyed on these values, to prevent hash collision attacks.
    hasher: H,

    table: RawTable<K, V>,

    resize_policy: DefaultResizePolicy,
}

/// Search for a pre-hashed key.
fn search_hashed<K, V, M, F>(table: M,
                             hash: SafeHash,
                             mut is_match: F)
                             -> SearchResult<K, V, M> where
    M: Deref<RawTable<K, V>>,
    F: FnMut(&K) -> bool,
{
    let size = table.size();
    let mut probe = Bucket::new(table, hash);
    let ib = probe.index();

    while probe.index() != ib + size {
        let full = match probe.peek() {
            Empty(b) => return TableRef(b.into_table()), // hit an empty bucket
            Full(b) => b
        };

        if full.distance() + ib < full.index() {
            // We can finish the search early if we hit any bucket
            // with a lower distance to initial bucket than we've probed.
            return TableRef(full.into_table());
        }

        // If the hash doesn't match, it can't be this one..
        if hash == full.hash() {
            // If the key doesn't match, it can't be this one..
            if is_match(full.read().0) {
                return FoundExisting(full);
            }
        }

        probe = full.next();
    }

    TableRef(probe.into_table())
}

fn pop_internal<K, V>(starting_bucket: FullBucketMut<K, V>) -> (K, V) {
    let (empty, retkey, retval) = starting_bucket.take();
    let mut gap = match empty.gap_peek() {
        Some(b) => b,
        None => return (retkey, retval)
    };

    while gap.full().distance() != 0 {
        gap = match gap.shift() {
            Some(b) => b,
            None => break
        };
    }

    // Now we've done all our shifting. Return the value we grabbed earlier.
    (retkey, retval)
}

/// Perform robin hood bucket stealing at the given `bucket`. You must
/// also pass the position of that bucket's initial bucket so we don't have
/// to recalculate it.
///
/// `hash`, `k`, and `v` are the elements to "robin hood" into the hashtable.
fn robin_hood<'a, K: 'a, V: 'a>(mut bucket: FullBucketMut<'a, K, V>,
                        mut ib: uint,
                        mut hash: SafeHash,
                        mut k: K,
                        mut v: V)
                        -> &'a mut V {
    let starting_index = bucket.index();
    let size = {
        let table = bucket.table(); // FIXME "lifetime too short".
        table.size()
    };
    // There can be at most `size - dib` buckets to displace, because
    // in the worst case, there are `size` elements and we already are
    // `distance` buckets away from the initial one.
    let idx_end = starting_index + size - bucket.distance();

    loop {
        let (old_hash, old_key, old_val) = bucket.replace(hash, k, v);
        loop {
            let probe = bucket.next();
            assert!(probe.index() != idx_end);

            let full_bucket = match probe.peek() {
                Empty(bucket) => {
                    // Found a hole!
                    let b = bucket.put(old_hash, old_key, old_val);
                    // Now that it's stolen, just read the value's pointer
                    // right out of the table!
                    return Bucket::at_index(b.into_table(), starting_index)
                               .peek()
                               .expect_full()
                               .into_mut_refs()
                               .1;
                },
                Full(bucket) => bucket
            };

            let probe_ib = full_bucket.index() - full_bucket.distance();

            bucket = full_bucket;

            // Robin hood! Steal the spot.
            if ib < probe_ib {
                ib = probe_ib;
                hash = old_hash;
                k = old_key;
                v = old_val;
                break;
            }
        }
    }
}

/// A result that works like Option<FullBucket<..>> but preserves
/// the reference that grants us access to the table in any case.
enum SearchResult<K, V, M> {
    // This is an entry that holds the given key:
    FoundExisting(FullBucket<K, V, M>),

    // There was no such entry. The reference is given back:
    TableRef(M)
}

impl<K, V, M> SearchResult<K, V, M> {
    fn into_option(self) -> Option<FullBucket<K, V, M>> {
        match self {
            FoundExisting(bucket) => Some(bucket),
            TableRef(_) => None
        }
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> HashMap<K, V, H> {
    fn make_hash<Sized? X: Hash<S>>(&self, x: &X) -> SafeHash {
        table::make_hash(&self.hasher, x)
    }

    #[allow(deprecated)]
    fn search_equiv<'a, Sized? Q: Hash<S> + Equiv<K>>(&'a self, q: &Q)
                    -> Option<FullBucketImm<'a, K, V>> {
        let hash = self.make_hash(q);
        search_hashed(&self.table, hash, |k| q.equiv(k)).into_option()
    }

    #[allow(deprecated)]
    fn search_equiv_mut<'a, Sized? Q: Hash<S> + Equiv<K>>(&'a mut self, q: &Q)
                    -> Option<FullBucketMut<'a, K, V>> {
        let hash = self.make_hash(q);
        search_hashed(&mut self.table, hash, |k| q.equiv(k)).into_option()
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    fn search<'a, Sized? Q>(&'a self, q: &Q) -> Option<FullBucketImm<'a, K, V>>
        where Q: BorrowFrom<K> + Eq + Hash<S>
    {
        let hash = self.make_hash(q);
        search_hashed(&self.table, hash, |k| q.eq(BorrowFrom::borrow_from(k)))
            .into_option()
    }

    fn search_mut<'a, Sized? Q>(&'a mut self, q: &Q) -> Option<FullBucketMut<'a, K, V>>
        where Q: BorrowFrom<K> + Eq + Hash<S>
    {
        let hash = self.make_hash(q);
        search_hashed(&mut self.table, hash, |k| q.eq(BorrowFrom::borrow_from(k)))
            .into_option()
    }

    // The caller should ensure that invariants by Robin Hood Hashing hold.
    fn insert_hashed_ordered(&mut self, hash: SafeHash, k: K, v: V) {
        let cap = self.table.capacity();
        let mut buckets = Bucket::new(&mut self.table, hash);
        let ib = buckets.index();

        while buckets.index() != ib + cap {
            // We don't need to compare hashes for value swap.
            // Not even DIBs for Robin Hood.
            buckets = match buckets.peek() {
                Empty(empty) => {
                    empty.put(hash, k, v);
                    return;
                }
                Full(b) => b.into_bucket()
            };
            buckets.next();
        }
        panic!("Internal HashMap error: Out of space.");
    }
}

impl<K: Hash + Eq, V> HashMap<K, V, RandomSipHasher> {
    /// Create an empty HashMap.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, int> = HashMap::new();
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn new() -> HashMap<K, V, RandomSipHasher> {
        let hasher = RandomSipHasher::new();
        HashMap::with_hasher(hasher)
    }

    /// Creates an empty hash map with the given initial capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, int> = HashMap::with_capacity(10);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn with_capacity(capacity: uint) -> HashMap<K, V, RandomSipHasher> {
        let hasher = RandomSipHasher::new();
        HashMap::with_capacity_and_hasher(capacity, hasher)
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> HashMap<K, V, H> {
    /// Creates an empty hashmap which will use the given hasher to hash keys.
    ///
    /// The creates map has the default initial capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use std::hash::sip::SipHasher;
    ///
    /// let h = SipHasher::new();
    /// let mut map = HashMap::with_hasher(h);
    /// map.insert(1i, 2u);
    /// ```
    #[inline]
    pub fn with_hasher(hasher: H) -> HashMap<K, V, H> {
        HashMap {
            hasher:        hasher,
            resize_policy: DefaultResizePolicy::new(),
            table:         RawTable::new(0),
        }
    }

    /// Create an empty HashMap with space for at least `capacity`
    /// elements, using `hasher` to hash the keys.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use std::hash::sip::SipHasher;
    ///
    /// let h = SipHasher::new();
    /// let mut map = HashMap::with_capacity_and_hasher(10, h);
    /// map.insert(1i, 2u);
    /// ```
    #[inline]
    pub fn with_capacity_and_hasher(capacity: uint, hasher: H) -> HashMap<K, V, H> {
        let resize_policy = DefaultResizePolicy::new();
        let min_cap = max(INITIAL_CAPACITY, resize_policy.min_capacity(capacity));
        let internal_cap = min_cap.checked_next_power_of_two().expect("capacity overflow");
        assert!(internal_cap >= capacity, "capacity overflow");
        HashMap {
            hasher:        hasher,
            resize_policy: resize_policy,
            table:         RawTable::new(internal_cap),
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let map: HashMap<int, int> = HashMap::with_capacity(100);
    /// assert!(map.capacity() >= 100);
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn capacity(&self) -> uint {
        self.resize_policy.usable_capacity(self.table.capacity())
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `uint`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, int> = HashMap::new();
    /// map.reserve(10);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn reserve(&mut self, additional: uint) {
        let new_size = self.len().checked_add(additional).expect("capacity overflow");
        let min_cap = self.resize_policy.min_capacity(new_size);

        // An invalid value shouldn't make us run out of space. This includes
        // an overflow check.
        assert!(new_size <= min_cap);

        if self.table.capacity() < min_cap {
            let new_capacity = max(min_cap.next_power_of_two(), INITIAL_CAPACITY);
            self.resize(new_capacity);
        }
    }

    /// Resizes the internal vectors to a new capacity. It's your responsibility to:
    ///   1) Make sure the new capacity is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure new_capacity is a power of two or zero.
    fn resize(&mut self, new_capacity: uint) {
        assert!(self.table.size() <= new_capacity);
        assert!(new_capacity.is_power_of_two() || new_capacity == 0);

        let mut old_table = replace(&mut self.table, RawTable::new(new_capacity));
        let old_size = old_table.size();

        if old_table.capacity() == 0 || old_table.size() == 0 {
            return;
        }

        // Grow the table.
        // Specialization of the other branch.
        let mut bucket = Bucket::first(&mut old_table);

        // "So a few of the first shall be last: for many be called,
        // but few chosen."
        //
        // We'll most likely encounter a few buckets at the beginning that
        // have their initial buckets near the end of the table. They were
        // placed at the beginning as the probe wrapped around the table
        // during insertion. We must skip forward to a bucket that won't
        // get reinserted too early and won't unfairly steal others spot.
        // This eliminates the need for robin hood.
        loop {
            bucket = match bucket.peek() {
                Full(full) => {
                    if full.distance() == 0 {
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

        // This is how the buckets might be laid out in memory:
        // ($ marks an initialized bucket)
        //  ________________
        // |$$$_$$$$$$_$$$$$|
        //
        // But we've skipped the entire initial cluster of buckets
        // and will continue iteration in this order:
        //  ________________
        //     |$$$$$$_$$$$$
        //                  ^ wrap around once end is reached
        //  ________________
        //  $$$_____________|
        //    ^ exit once table.size == 0
        loop {
            bucket = match bucket.peek() {
                Full(bucket) => {
                    let h = bucket.hash();
                    let (b, k, v) = bucket.take();
                    self.insert_hashed_ordered(h, k, v);
                    {
                        let t = b.table(); // FIXME "lifetime too short".
                        if t.size() == 0 { break }
                    };
                    b.into_bucket()
                }
                Empty(b) => b.into_bucket()
            };
            bucket.next();
        }

        assert_eq!(self.table.size(), old_size);
    }

    /// Shrinks the capacity of the map as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map: HashMap<int, int> = HashMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to_fit();
    /// assert!(map.capacity() >= 2);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn shrink_to_fit(&mut self) {
        let min_capacity = self.resize_policy.min_capacity(self.len());
        let min_capacity = max(min_capacity.next_power_of_two(), INITIAL_CAPACITY);

        // An invalid value shouldn't make us run out of space.
        debug_assert!(self.len() <= min_capacity);

        if self.table.capacity() != min_capacity {
            let old_table = replace(&mut self.table, RawTable::new(min_capacity));
            let old_size = old_table.size();

            // Shrink the table. Naive algorithm for resizing:
            for (h, k, v) in old_table.into_iter() {
                self.insert_hashed_nocheck(h, k, v);
            }

            debug_assert_eq!(self.table.size(), old_size);
        }
    }

    /// Insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    ///
    /// If the key already exists, the hashtable will be returned untouched
    /// and a reference to the existing element will be returned.
    fn insert_hashed_nocheck(&mut self, hash: SafeHash, k: K, v: V) -> &mut V {
        self.insert_or_replace_with(hash, k, v, |_, _, _| ())
    }

    fn insert_or_replace_with<'a, F>(&'a mut self,
                                     hash: SafeHash,
                                     k: K,
                                     v: V,
                                     mut found_existing: F)
                                     -> &'a mut V where
        F: FnMut(&mut K, &mut V, V),
    {
        // Worst case, we'll find one empty bucket among `size + 1` buckets.
        let size = self.table.size();
        let mut probe = Bucket::new(&mut self.table, hash);
        let ib = probe.index();

        loop {
            let mut bucket = match probe.peek() {
                Empty(bucket) => {
                    // Found a hole!
                    return bucket.put(hash, k, v).into_mut_refs().1;
                }
                Full(bucket) => bucket
            };

            // hash matches?
            if bucket.hash() == hash {
                // key matches?
                if k == *bucket.read_mut().0 {
                    let (bucket_k, bucket_v) = bucket.into_mut_refs();
                    debug_assert!(k == *bucket_k);
                    // Key already exists. Get its reference.
                    found_existing(bucket_k, bucket_v, v);
                    return bucket_v;
                }
            }

            let robin_ib = bucket.index() as int - bucket.distance() as int;

            if (ib as int) < robin_ib {
                // Found a luckier bucket than me. Better steal his spot.
                return robin_hood(bucket, robin_ib as uint, hash, k, v);
            }

            probe = bucket.next();
            assert!(probe.index() != ib + size + 1);
        }
    }

    /// Deprecated: use `contains_key` and `BorrowFrom` instead.
    #[deprecated = "use contains_key and BorrowFrom instead"]
    pub fn contains_key_equiv<Sized? Q: Hash<S> + Equiv<K>>(&self, key: &Q) -> bool {
        self.search_equiv(key).is_some()
    }

    /// Deprecated: use `get` and `BorrowFrom` instead.
    #[deprecated = "use get and BorrowFrom instead"]
    pub fn find_equiv<'a, Sized? Q: Hash<S> + Equiv<K>>(&'a self, k: &Q) -> Option<&'a V> {
        self.search_equiv(k).map(|bucket| bucket.into_refs().1)
    }

    /// Deprecated: use `remove` and `BorrowFrom` instead.
    #[deprecated = "use remove and BorrowFrom instead"]
    pub fn pop_equiv<Sized? Q:Hash<S> + Equiv<K>>(&mut self, k: &Q) -> Option<V> {
        if self.table.size() == 0 {
            return None
        }

        self.reserve(1);

        self.search_equiv_mut(k).map(|bucket| pop_internal(bucket).1)
    }

    /// An iterator visiting all keys in arbitrary order.
    /// Iterator element type is `&'a K`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((&'a K,&'a V)) -> &'a K = first; // coerce to fn ptr

        Keys { inner: self.iter().map(first) }
    }

    /// An iterator visiting all values in arbitrary order.
    /// Iterator element type is `&'a V`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for key in map.values() {
    ///     println!("{}", key);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        fn second<A, B>((_, b): (A, B)) -> B { b }
        let second: fn((&'a K,&'a V)) -> &'a V = second; // coerce to fn ptr

        Values { inner: self.iter().map(second) }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// Iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter(&self) -> Entries<K, V> {
        Entries { inner: self.table.iter() }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// Iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Update all values
    /// for (_, val) in map.iter_mut() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut { inner: self.table.iter_mut() }
    }

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(&str, int)> = map.into_iter().collect();
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn into_iter(self) -> IntoIter<K, V> {
        fn last_two<A, B, C>((_, b, c): (A, B, C)) -> (B, C) { (b, c) }
        let last_two: fn((SafeHash, K, V)) -> (K, V) = last_two;

        IntoIter {
            inner: self.table.into_iter().map(last_two)
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation
    pub fn entry<'a>(&'a mut self, key: K) -> Entry<'a, K, V> {
        // Gotta resize now.
        self.reserve(1);

        let hash = self.make_hash(&key);
        search_entry_hashed(&mut self.table, hash, key)
    }

    /// Return the number of elements in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1u, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn len(&self) -> uint { self.table.size() }

    /// Return true if the map contains no elements.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1u, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1u, "a");
    /// a.insert(2u, "b");
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == "a" || v == "b");
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn drain(&mut self) -> Drain<K, V> {
        fn last_two<A, B, C>((_, b, c): (A, B, C)) -> (B, C) { (b, c) }
        let last_two: fn((SafeHash, K, V)) -> (K, V) = last_two; // coerce to fn pointer

        Drain {
            inner: self.table.drain().map(last_two),
        }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1u, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    #[inline]
    pub fn clear(&mut self) {
        self.drain();
    }

    /// Deprecated: Renamed to `get`.
    #[deprecated = "Renamed to `get`"]
    pub fn find(&self, k: &K) -> Option<&V> {
        self.get(k)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1u, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get<Sized? Q>(&self, k: &Q) -> Option<&V>
        where Q: Hash<S> + Eq + BorrowFrom<K>
    {
        self.search(k).map(|bucket| bucket.into_refs().1)
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1u, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn contains_key<Sized? Q>(&self, k: &Q) -> bool
        where Q: Hash<S> + Eq + BorrowFrom<K>
    {
        self.search(k).is_some()
    }

    /// Deprecated: Renamed to `get_mut`.
    #[deprecated = "Renamed to `get_mut`"]
    pub fn find_mut(&mut self, k: &K) -> Option<&mut V> {
        self.get_mut(k)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1u, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[1], "b");
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn get_mut<Sized? Q>(&mut self, k: &Q) -> Option<&mut V>
        where Q: Hash<S> + Eq + BorrowFrom<K>
    {
        self.search_mut(k).map(|bucket| bucket.into_mut_refs().1)
    }

    /// Deprecated: Renamed to `insert`.
    #[deprecated = "Renamed to `insert`"]
    pub fn swap(&mut self, k: K, v: V) -> Option<V> {
        self.insert(k, v)
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37u, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[37], "c");
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        let hash = self.make_hash(&k);
        self.reserve(1);

        let mut retval = None;
        self.insert_or_replace_with(hash, k, v, |_, val_ref, val| {
            retval = Some(replace(val_ref, val));
        });
        retval
    }

    /// Deprecated: Renamed to `remove`.
    #[deprecated = "Renamed to `remove`"]
    pub fn pop(&mut self, k: &K) -> Option<V> {
        self.remove(k)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1u, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[unstable = "matches collection reform specification, waiting for dust to settle"]
    pub fn remove<Sized? Q>(&mut self, k: &Q) -> Option<V>
        where Q: Hash<S> + Eq + BorrowFrom<K>
    {
        if self.table.size() == 0 {
            return None
        }

        self.search_mut(k).map(|bucket| pop_internal(bucket).1)
    }
}

fn search_entry_hashed<'a, K: Eq, V>(table: &'a mut RawTable<K,V>, hash: SafeHash, k: K)
        -> Entry<'a, K, V> {
    // Worst case, we'll find one empty bucket among `size + 1` buckets.
    let size = table.size();
    let mut probe = Bucket::new(table, hash);
    let ib = probe.index();

    loop {
        let bucket = match probe.peek() {
            Empty(bucket) => {
                // Found a hole!
                return Vacant(VacantEntry {
                    hash: hash,
                    key: k,
                    elem: NoElem(bucket),
                });
            },
            Full(bucket) => bucket
        };

        // hash matches?
        if bucket.hash() == hash {
            // key matches?
            if k == *bucket.read().0 {
                return Occupied(OccupiedEntry{
                    elem: bucket,
                });
            }
        }

        let robin_ib = bucket.index() as int - bucket.distance() as int;

        if (ib as int) < robin_ib {
            // Found a luckier bucket than me. Better steal his spot.
            return Vacant(VacantEntry {
                hash: hash,
                key: k,
                elem: NeqElem(bucket, robin_ib as uint),
            });
        }

        probe = bucket.next();
        assert!(probe.index() != ib + size + 1);
    }
}

impl<K: Eq + Hash<S>, V: Clone, S, H: Hasher<S>> HashMap<K, V, H> {
    /// Deprecated: Use `map.get(k).cloned()`.
    ///
    /// Return a copy of the value corresponding to the key.
    #[deprecated = "Use `map.get(k).cloned()`"]
    pub fn find_copy(&self, k: &K) -> Option<V> {
        self.get(k).cloned()
    }

    /// Deprecated: Use `map[k].clone()`.
    ///
    /// Return a copy of the value corresponding to the key.
    #[deprecated = "Use `map[k].clone()`"]
    pub fn get_copy(&self, k: &K) -> V {
        self[*k].clone()
    }
}

impl<K: Eq + Hash<S>, V: PartialEq, S, H: Hasher<S>> PartialEq for HashMap<K, V, H> {
    fn eq(&self, other: &HashMap<K, V, H>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter().all(|(key, value)|
            other.get(key).map_or(false, |v| *value == *v)
        )
    }
}

impl<K: Eq + Hash<S>, V: Eq, S, H: Hasher<S>> Eq for HashMap<K, V, H> {}

impl<K: Eq + Hash<S> + Show, V: Show, S, H: Hasher<S>> Show for HashMap<K, V, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", *k, *v));
        }

        write!(f, "}}")
    }
}

#[stable]
impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> Default for HashMap<K, V, H> {
    #[stable]
    fn default() -> HashMap<K, V, H> {
        HashMap::with_hasher(Default::default())
    }
}

impl<K: Hash<S> + Eq, Sized? Q, V, S, H: Hasher<S>> Index<Q, V> for HashMap<K, V, H>
    where Q: BorrowFrom<K> + Hash<S> + Eq
{
    #[inline]
    fn index<'a>(&'a self, index: &Q) -> &'a V {
        self.get(index).expect("no entry found for key")
    }
}

impl<K: Hash<S> + Eq, Sized? Q, V, S, H: Hasher<S>> IndexMut<Q, V> for HashMap<K, V, H>
    where Q: BorrowFrom<K> + Hash<S> + Eq
{
    #[inline]
    fn index_mut<'a>(&'a mut self, index: &Q) -> &'a mut V {
        self.get_mut(index).expect("no entry found for key")
    }
}

/// HashMap iterator
pub struct Entries<'a, K: 'a, V: 'a> {
    inner: table::Entries<'a, K, V>
}

/// HashMap mutable values iterator
pub struct IterMut<'a, K: 'a, V: 'a> {
    inner: table::IterMut<'a, K, V>
}

/// HashMap move iterator
pub struct IntoIter<K, V> {
    inner: iter::Map<
        (SafeHash, K, V),
        (K, V),
        table::IntoIter<K, V>,
        fn((SafeHash, K, V)) -> (K, V),
    >
}

/// HashMap keys iterator
pub struct Keys<'a, K: 'a, V: 'a> {
    inner: Map<(&'a K, &'a V), &'a K, Entries<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>
}

/// HashMap values iterator
pub struct Values<'a, K: 'a, V: 'a> {
    inner: Map<(&'a K, &'a V), &'a V, Entries<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>
}

/// HashMap drain iterator
pub struct Drain<'a, K: 'a, V: 'a> {
    inner: iter::Map<
        (SafeHash, K, V),
        (K, V),
        table::Drain<'a, K, V>,
        fn((SafeHash, K, V)) -> (K, V),
    >
}

/// A view into a single occupied location in a HashMap
pub struct OccupiedEntry<'a, K:'a, V:'a> {
    elem: FullBucket<K, V, &'a mut RawTable<K, V>>,
}

/// A view into a single empty location in a HashMap
pub struct VacantEntry<'a, K:'a, V:'a> {
    hash: SafeHash,
    key: K,
    elem: VacantEntryState<K,V, &'a mut RawTable<K, V>>,
}

/// A view into a single location in a map, which may be vacant or occupied
pub enum Entry<'a, K:'a, V:'a> {
    /// An occupied Entry
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant Entry
    Vacant(VacantEntry<'a, K, V>),
}

/// Possible states of a VacantEntry
enum VacantEntryState<K, V, M> {
    /// The index is occupied, but the key to insert has precedence,
    /// and will kick the current one out on insertion
    NeqElem(FullBucket<K, V, M>, uint),
    /// The index is genuinely vacant
    NoElem(EmptyBucket<K, V, M>),
}

impl<'a, K, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    #[inline] fn next(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}

impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for IterMut<'a, K, V> {
    #[inline] fn next(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}

impl<K, V> Iterator<(K, V)> for IntoIter<K, V> {
    #[inline] fn next(&mut self) -> Option<(K, V)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}

impl<'a, K, V> Iterator<&'a K> for Keys<'a, K, V> {
    #[inline] fn next(&mut self) -> Option<(&'a K)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}

impl<'a, K, V> Iterator<&'a V> for Values<'a, K, V> {
    #[inline] fn next(&mut self) -> Option<(&'a V)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}

impl<'a, K: 'a, V: 'a> Iterator<(K, V)> for Drain<'a, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry
    pub fn get(&self) -> &V {
        self.elem.read().1
    }

    /// Gets a mutable reference to the value in the entry
    pub fn get_mut(&mut self) -> &mut V {
        self.elem.read_mut().1
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself
    pub fn into_mut(self) -> &'a mut V {
        self.elem.into_mut_refs().1
    }

    /// Sets the value of the entry, and returns the entry's old value
    pub fn set(&mut self, mut value: V) -> V {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }

    /// Takes the value out of the entry, and returns it
    pub fn take(self) -> V {
        pop_internal(self.elem).1
    }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it
    pub fn set(self, value: V) -> &'a mut V {
        match self.elem {
            NeqElem(bucket, ib) => {
                robin_hood(bucket, ib, self.hash, self.key, value)
            }
            NoElem(bucket) => {
                bucket.put(self.hash, self.key, value).into_mut_refs().1
            }
        }
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> FromIterator<(K, V)> for HashMap<K, V, H> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> HashMap<K, V, H> {
        let lower = iter.size_hint().0;
        let mut map = HashMap::with_capacity_and_hasher(lower, Default::default());
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> Extend<(K, V)> for HashMap<K, V, H> {
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

#[cfg(test)]
mod test_map {
    use prelude::*;

    use super::HashMap;
    use super::Entry::{Occupied, Vacant};
    use hash;
    use iter::{range_inclusive, range_step_inclusive};
    use cell::RefCell;
    use rand::{weak_rng, Rng};

    struct KindaIntLike(int);

    impl Equiv<int> for KindaIntLike {
        fn equiv(&self, other: &int) -> bool {
            let KindaIntLike(this) = *self;
            this == *other
        }
    }
    impl<S: hash::Writer> hash::Hash<S> for KindaIntLike {
        fn hash(&self, state: &mut S) {
            let KindaIntLike(this) = *self;
            this.hash(state)
        }
    }

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

        assert!(m.insert(1i, 1i).is_none());

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1i, 2i).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2i, 4i).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&2).unwrap(), 4);
    }

    thread_local! { static DROP_VECTOR: RefCell<Vec<int>> = RefCell::new(Vec::new()) }

    #[deriving(Hash, PartialEq, Eq)]
    struct Dropable {
        k: uint
    }

    impl Dropable {
        fn new(k: uint) -> Dropable {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[k] += 1;
            });

            Dropable { k: k }
        }
    }

    impl Drop for Dropable {
        fn drop(&mut self) {
            DROP_VECTOR.with(|slot| {
                slot.borrow_mut()[self.k] -= 1;
            });
        }
    }

    impl Clone for Dropable {
        fn clone(&self) -> Dropable {
            Dropable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        DROP_VECTOR.with(|slot| {
            *slot.borrow_mut() = Vec::from_elem(200, 0i);
        });

        {
            let mut m = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in range(0u, 200) {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in range(0u, 100) {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                m.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in range(0u, 200) {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for i in range(0u, 50) {
                let k = Dropable::new(i);
                let v = m.remove(&k);

                assert!(v.is_some());

                DROP_VECTOR.with(|v| {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i+100], 1);
                });
            }

            DROP_VECTOR.with(|v| {
                for i in range(0u, 50) {
                    assert_eq!(v.borrow()[i], 0);
                    assert_eq!(v.borrow()[i+100], 0);
                }

                for i in range(50u, 100) {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i+100], 1);
                }
            });
        }

        DROP_VECTOR.with(|v| {
            for i in range(0u, 200) {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_move_iter_drops() {
        DROP_VECTOR.with(|v| {
            *v.borrow_mut() = Vec::from_elem(200, 0i);
        });

        let hm = {
            let mut hm = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in range(0u, 200) {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in range(0u, 100) {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                hm.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in range(0u, 200) {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            hm
        };

        // By the way, ensure that cloning doesn't screw up the dropping.
        drop(hm.clone());

        {
            let mut half = hm.into_iter().take(50);

            DROP_VECTOR.with(|v| {
                for i in range(0u, 200) {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for _ in half {}

            DROP_VECTOR.with(|v| {
                let nk = range(0u, 100).filter(|&i| {
                    v.borrow()[i] == 1
                }).count();

                let nv = range(0u, 100).filter(|&i| {
                    v.borrow()[i+100] == 1
                }).count();

                assert_eq!(nk, 50);
                assert_eq!(nv, 50);
            });
        };

        DROP_VECTOR.with(|v| {
            for i in range(0u, 200) {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_pop() {
        let mut m: HashMap<int, bool> = HashMap::new();
        assert_eq!(m.remove(&0), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = HashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in range(0i, 10) {
            assert!(m.is_empty());

            for i in range_inclusive(1i, 1000) {
                assert!(m.insert(i, i).is_none());

                for j in range_inclusive(1, i) {
                    let r = m.get(&j);
                    assert_eq!(r, Some(&j));
                }

                for j in range_inclusive(i+1, 1000) {
                    let r = m.get(&j);
                    assert_eq!(r, None);
                }
            }

            for i in range_inclusive(1001i, 2000) {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in range_inclusive(1i, 1000) {
                assert!(m.remove(&i).is_some());

                for j in range_inclusive(1, i) {
                    assert!(!m.contains_key(&j));
                }

                for j in range_inclusive(i+1, 1000) {
                    assert!(m.contains_key(&j));
                }
            }

            for i in range_inclusive(1i, 1000) {
                assert!(!m.contains_key(&i));
            }

            for i in range_inclusive(1i, 1000) {
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in range_step_inclusive(1000i, 1, -1) {
                assert!(m.remove(&i).is_some());

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
        assert!(m.insert(1i, 12i).is_none());
        assert!(m.insert(2i, 8i).is_none());
        assert!(m.insert(5i, 14i).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(), Some(x) => *x = new
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1i, 2i).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(!m.insert(1i, 3i).is_none());
        assert_eq!(*m.get(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1i, 2i).is_none());
        assert!(m.insert(5i, 3i).is_none());
        assert!(m.insert(9i, 4i).is_none());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1i, 2i).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(m.insert(5, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert!(m.remove(&1).is_some());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1i, 2i).is_none());
        assert!(!m.is_empty());
        assert!(m.remove(&1).is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = HashMap::new();
        m.insert(1i, 2i);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    #[allow(experimental)]
    fn test_pop_equiv() {
        let mut m = HashMap::new();
        m.insert(1i, 2i);
        assert_eq!(m.pop_equiv(&KindaIntLike(1)), Some(2));
        assert_eq!(m.pop_equiv(&KindaIntLike(1)), None);
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in range(0u, 32) {
            assert!(m.insert(i, i*2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;

        for (k, v) in m.iter() {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1i, 'a'), (2i, 'b'), (3i, 'c')];
        let map = vec.into_iter().collect::<HashMap<int, char>>();
        let keys = map.keys().map(|&k| k).collect::<Vec<int>>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1i, 'a'), (2i, 'b'), (3i, 'c')];
        let map = vec.into_iter().collect::<HashMap<int, char>>();
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.get(&1i).is_none());
        m.insert(1i, 2i);
        match m.get(&1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2)
        }
    }

    #[test]
    #[allow(deprecated)]
    fn test_find_copy() {
        let mut m = HashMap::new();
        assert!(m.get(&1i).is_none());

        for i in range(1i, 10000) {
            m.insert(i, i + 7);
            match m.find_copy(&i) {
                None => panic!(),
                Some(v) => assert_eq!(v, i + 7)
            }
            for j in range(1i, i/100) {
                match m.find_copy(&j) {
                    None => panic!(),
                    Some(v) => assert_eq!(v, j + 7)
                }
            }
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = HashMap::new();
        m1.insert(1i, 2i);
        m1.insert(2i, 3i);
        m1.insert(3i, 4i);

        let mut m2 = HashMap::new();
        m2.insert(1i, 2i);
        m2.insert(2i, 3i);

        assert!(m1 != m2);

        m2.insert(3i, 4i);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_show() {
        let mut map: HashMap<int, int> = HashMap::new();
        let empty: HashMap<int, int> = HashMap::new();

        map.insert(1i, 2i);
        map.insert(3i, 4i);

        let map_str = format!("{}", map);

        assert!(map_str == "{1: 2, 3: 4}" || map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{}", empty), "{}");
    }

    #[test]
    fn test_expand() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0u;
        let old_cap = m.table.capacity();
        while old_cap == m.table.capacity() {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_behavior_resize_policy() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert_eq!(m.table.capacity(), 0);
        assert!(m.is_empty());

        m.insert(0, 0);
        m.remove(&0);
        assert!(m.is_empty());
        let initial_cap = m.table.capacity();
        m.reserve(initial_cap);
        let cap = m.table.capacity();

        assert_eq!(cap, initial_cap * 2);

        let mut i = 0u;
        for _ in range(0, cap * 3 / 4) {
            m.insert(i, i);
            i += 1;
        }
        // three quarters full

        assert_eq!(m.len(), i);
        assert_eq!(m.table.capacity(), cap);

        for _ in range(0, cap / 4) {
            m.insert(i, i);
            i += 1;
        }
        // half full

        let new_cap = m.table.capacity();
        assert_eq!(new_cap, cap * 2);

        for _ in range(0, cap / 2 - 1) {
            i -= 1;
            m.remove(&i);
            assert_eq!(m.table.capacity(), new_cap);
        }
        // A little more than one quarter full.
        m.shrink_to_fit();
        assert_eq!(m.table.capacity(), cap);
        // again, a little more than half full
        for _ in range(0, cap / 2 - 1) {
            i -= 1;
            m.remove(&i);
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
        assert_eq!(m.table.capacity(), initial_cap);
    }

    #[test]
    fn test_reserve_shrink_to_fit() {
        let mut m = HashMap::new();
        m.insert(0u, 0u);
        m.remove(&0);
        assert!(m.capacity() >= m.len());
        for i in range(0, 128) {
            m.insert(i, i);
        }
        m.reserve(256);

        let usable_cap = m.capacity();
        for i in range(128, 128+256) {
            m.insert(i, i);
            assert_eq!(m.capacity(), usable_cap);
        }

        for i in range(100, 128+256) {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), 100);
        assert!(!m.is_empty());
        assert!(m.capacity() >= m.len());

        for i in range(0, 100) {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();
        m.insert(0, 0);

        assert_eq!(m.len(), 1);
        assert!(m.capacity() >= m.len());
        assert_eq!(m.remove(&0), Some(0));
    }

    #[test]
    fn test_find_equiv() {
        let mut m = HashMap::new();

        let (foo, bar, baz) = (1i,2i,3i);
        m.insert("foo".to_string(), foo);
        m.insert("bar".to_string(), bar);
        m.insert("baz".to_string(), baz);


        assert_eq!(m.get("foo"), Some(&foo));
        assert_eq!(m.get("bar"), Some(&bar));
        assert_eq!(m.get("baz"), Some(&baz));

        assert_eq!(m.get("qux"), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1i, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1i, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_mut_size_hint() {
        let xs = [(1i, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_index() {
        let mut map: HashMap<int, int> = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[2], 1);
    }

    #[test]
    #[should_fail]
    fn test_index_nonexistent() {
        let mut map: HashMap<int, int> = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[4];
    }

    #[test]
    fn test_entry(){
        let xs = [(1i, 10i), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.set(100), 10);
            }
        }
        assert_eq!(map.get(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);


        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        assert_eq!(map.get(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.take(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);


        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.set(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_entry_take_doesnt_corrupt() {
        // Test for #19292
        fn check(m: &HashMap<int, ()>) {
            for k in m.keys() {
                assert!(m.contains_key(k),
                        "{} is in keys() but not in the map?", k);
            }
        }

        let mut m = HashMap::new();
        let mut rng = weak_rng();

        // Populate the map with some items.
        for _ in range(0u, 50) {
            let x = rng.gen_range(-10, 10);
            m.insert(x, ());
        }

        for i in range(0u, 1000) {
            let x = rng.gen_range(-10, 10);
            match m.entry(x) {
                Vacant(_) => {},
                Occupied(e) => {
                    println!("{}: remove {}", i, x);
                    e.take();
                },
            }

            check(&m);
        }
    }
}
