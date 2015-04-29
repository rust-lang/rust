// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::Entry::*;
use self::VacantEntryState::*;

use borrow::{Borrow, BorrowMut};
use clone::Clone;
use cmp::{max, Eq, PartialEq};
use default::Default;
use fmt::{self, Debug};
use hash::{Hash, SipHasher};
use iter::{Iterator, ExactSizeIterator, IntoIterator, FromIterator, Extend, Map};
use marker::Sized;
use mem::{self, swap, replace};
use ops::{Drop, FnMut, FnOnce, Index};
use option::Option::{self, Some, None};
use rand::{self, Rng};
use result::Result::{self, Ok, Err};

use super::table::{
    self,
    Bucket,
    EmptyBucket,
    FullBucket,
    FullBucketMut,
    RawTable,
    SafeHash,
    PartialRawTable,
    Put,
};
use super::table::BucketState::{
    Empty,
    Full,
};
use super::state::HashState;

const INITIAL_LOG2_CAP: usize = 5;
#[unstable(feature = "std_misc")]
pub const INITIAL_CAPACITY: usize = 1 << INITIAL_LOG2_CAP; // 2^5

/// The default behavior of HashMap implements a load factor of 90.9%.
/// This behavior is characterized by the following condition:
///
/// - if size > 0.909 * capacity: grow the map
#[derive(Clone)]
struct DefaultResizePolicy;

impl DefaultResizePolicy {
    fn new() -> DefaultResizePolicy {
        DefaultResizePolicy
    }

    #[inline]
    fn min_capacity(&self, usable_size: usize) -> usize {
        // Here, we are rephrasing the logic by specifying the lower limit
        // on capacity:
        //
        // - if `cap < size * 1.1`: grow the map
        usable_size * 11 / 10
    }

    /// An inverse of `min_capacity`, approximately.
    #[inline]
    fn usable_capacity(&self, cap: usize) -> usize {
        // As the number of entries approaches usable capacity,
        // min_capacity(size) must be smaller than the internal capacity,
        // so that the map is not resized:
        // `min_capacity(usable_capacity(x)) <= x`.
        // The left-hand side can only be smaller due to flooring by integer
        // division.
        //
        // This doesn't have to be checked for overflow since allocation size
        // in bytes will overflow earlier than multiplication by 10.
        cap * 10 / 11
    }
}

#[test]
fn test_resize_policy() {
    let rp = DefaultResizePolicy;
    for n in 0..1000 {
        assert!(rp.min_capacity(rp.usable_capacity(n)) <= n);
        assert!(rp.usable_capacity(rp.min_capacity(n)) <= n);
    }
}

// The main performance trick in this hashmap is called Robin Hood hashing with
// linear probing. It gains its excellent performance from one essential
// operation:
//
//    If an insertion collides with an existing element, and that element's
//    "displacement" (how far away the element is from its ideal location)
//    is lower than how far we've already probed, swap the elements.
//
// This massively lowers variance in displacement and allows us to get very
// high load factors with good performance. The 90% load factor I use is rather
// conservative.
//
// > Why a load factor of approximately 90%?
//
// In general, all the distances to initial buckets will converge on the mean.
// At a load factor of α, the odds of finding the target bucket after exactly n
// unsuccesful probes[4] are
//
// Pr{displacement = n} =
// (1 - α) / α * ∑_{k≥1} e^(-kα) * (kα)^(k+n) / (k + n)! * (1 - kα / (k + n + 1))
//
// These odds can be approximated with this code:
//
// ```
// use std::num::Float;
// use std::iter::AdditiveIterator;
//
// fn factorial(value: f64) -> f64 {
//     if value == 0.0 { 1.0 } else { value * factorial(value - 1.0) }
// }
//
// fn psi(a: f64, displacement: u32) -> f64 {
//     let n = displacement as f64;
//     (1.0 / a - 1.0) * (0..130).map(|k_| {
//         let k = k_ as f64;
//         (-k * a).exp() * (1.0 - k * a / (k + n + 1.0)) *
//         (k * a).powi((k + n) as i32) / factorial(k + n) as f64
//     }).sum()
// }
// ```
//
// If we set α=0.909, then Pr{displacement < 4} = 0.51 and Pr{displacement <
// 8} = 0.77. The exact value of 0.909 is chosen to make the math easier on
// the CPU and avoid its FPU. Since on average we start the probing in the
// middle of a cache line, this strategy pulls in one cache line of hashes on
// most lookups (64-byte cache line with 8-byte hash). I think this choice is
// pretty good, but α could go up to 0.95, or down to 0.84 to trade off some
// space.
//
// ## Future Improvements (FIXME!)
//
// Allow the load factor to be changed dynamically and/or at initialization.
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
// Annotate the exceedingly likely branch in `search_hashed` to reduce
// instruction cache pressure and mispredictions once it becomes possible
// (blocked on issue #11092).
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
// is i. If the new bit is 1, then the new ideal index is n + i. Intuitively,
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
// FIXME(Gankro, pczarn): review the proof and put it all in a separate README.md

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
/// this can frequently be achieved by using `#[derive(Eq, Hash)]`. If you
/// implement these yourself, it is important that the following property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must be equal.
///
/// It is a logic error for a key to be modified in such a way that the key's
/// hash, as determined by the `Hash` trait, or its equality, as determined by
/// the `Eq` trait, changes while it is in the map. This is normally only
/// possible through `Cell`, `RefCell`, global state, I/O, or unsafe code.
///
/// Relevant papers/articles:
///
/// 1. Pedro Celis. ["Robin Hood Hashing"](https://cs.uwaterloo.ca/research/tr/1986/CS-86-14.pdf)
/// 2. Emmanuel Goossaert. ["Robin Hood
///    hashing"](http://codecapsule.com/2013/11/11/robin-hood-hashing/)
/// 3. Emmanuel Goossaert. ["Robin Hood hashing: backward shift
///    deletion"](http://codecapsule.com/2013/11/17/robin-hood-hashing-backward-shift-deletion/)
/// 4. Alfredo Viola (2005). Distributional analysis of Robin Hood linear probing
///    hashing with buckets.
///
/// # Examples
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
/// The easiest way to use `HashMap` with a custom type as key is to derive `Eq` and `Hash`.
/// We must also derive `PartialEq`.
///
/// ```
/// use std::collections::HashMap;
///
/// #[derive(Hash, Eq, PartialEq, Debug)]
/// struct Viking {
///     name: String,
///     country: String,
/// }
///
/// impl Viking {
///     /// Create a new Viking.
///     fn new(name: &str, country: &str) -> Viking {
///         Viking { name: name.to_string(), country: country.to_string() }
///     }
/// }
///
/// // Use a HashMap to store the vikings' health points.
/// let mut vikings = HashMap::new();
///
/// vikings.insert(Viking::new("Einar", "Norway"), 25);
/// vikings.insert(Viking::new("Olaf", "Denmark"), 24);
/// vikings.insert(Viking::new("Harald", "Iceland"), 12);
///
/// // Use derived implementation to print the status of the vikings.
/// for (viking, health) in vikings.iter() {
///     println!("{:?} has {} hp", viking, health);
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub struct HashMap<K, V, S = RandomState> {
    // All hashes are keyed on these values, to prevent hash collision attacks.
    hash_state: S,

    table: RawTable<K, V>,

    resize_policy: DefaultResizePolicy,
}

/// Search for a pre-hashed key.
fn search_hashed<K, V, M, F>(table: M,
                             hash: SafeHash,
                             mut is_match: F)
                             -> InternalEntry<K, V, M> where
    M: Borrow<RawTable<K, V>>,
    F: FnMut(&K) -> bool,
{
    // Worst case, we'll find one empty bucket among `size + 1` buckets.
    let size = table.borrow().size();
    let mut probe = match Bucket::new(table, *hash as usize) {
        Some(probe) => probe,
        // This is the only function where capacity can be zero.
        None => return InternalEntry::TableIsEmpty,
    };
    let ib = probe.index();

    loop {
        let bucket = match probe.peek() {
            Empty(empty) => {
                // Found a hole!
                return InternalEntry::Vacant(NoElem(empty));
            }
            Full(bucket) => bucket
        };

        let robin_ib = bucket.index() as isize - bucket.displacement() as isize;

        if (ib as isize) < robin_ib {
            // We can finish the search early if we hit any bucket
            // with a lower distance to initial bucket than we've probed.
            return InternalEntry::Vacant(NeqElem(bucket, robin_ib as usize));
        }

        // If the hash doesn't match, it can't be this one..
        if hash == *bucket.read().0 {
            // If the key doesn't match, it can't be this one..
            if is_match(bucket.read().1) {
                return InternalEntry::Occupied(OccupiedEntryState {
                    elem: bucket,
                });
            }
        }

        probe = bucket.into_next();
        debug_assert!(probe.index() != ib + size + 1);
    }
}

fn pop_internal<K, V>(starting_bucket: FullBucketMut<K, V>) -> (K, V) {
    let (empty, retkey, retval) = starting_bucket.take();
    let mut gap = match empty.gap_peek() {
        Some(b) => b,
        None => return (retkey, retval)
    };

    while gap.full().displacement() != 0 {
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
fn robin_hood<'a, K: 'a, V: 'a>(bucket: FullBucketMut<'a, K, V>,
                        mut ib: usize,
                        mut hash: SafeHash,
                        mut key: K,
                        mut val: V)
                        -> &'a mut V {
    let starting_index = bucket.index();
    let size = {
        let table = bucket.table(); // FIXME "lifetime too short".
        table.size()
    };
    let mut bucket = bucket.stash();
    // There can be at most `size - displacement` buckets to displace, because
    // in the worst case, there are `size` elements and we already are
    // `displacement` buckets away from the initial one.
    let idx_end = starting_index + size - bucket.displacement();

    loop {
        {
            let (h_ref, k_ref, v_ref) = bucket.read_mut();
            swap(h_ref, &mut hash);
            swap(k_ref, &mut key);
            swap(v_ref, &mut val);
        };
        loop {
            let probe = bucket.into_next();
            assert!(probe.index() != idx_end);

            let full_bucket = match probe.peek() {
                Empty(bucket) => {
                    // Found a hole!
                    let b = bucket.put(hash, key, val);
                    // Now that it's stolen, just read the value's pointer
                    // right out of the table!
                    return b.into_table().into_mut_refs().1;
                },
                Full(bucket) => bucket
            };

            let probe_ib = full_bucket.index() - full_bucket.displacement();

            bucket = full_bucket;

            // Robin hood! Steal the spot.
            if ib < probe_ib {
                ib = probe_ib;
                break;
            }
        }
    }
}

// Performs insertion with relaxed requirements.
// The caller should ensure that invariants of Robin Hood linear probing hold.
fn insert_hashed_ordered<M: Put, K, V>(table: M, hash: SafeHash, key: K, val: V) -> M
    where M: BorrowMut<RawTable<K, V>>
{
    let cap = table.borrow().capacity();
    let mut buckets = Bucket::new(table, *hash as usize).unwrap();
    let ib = buckets.index();

    while buckets.index() != ib + cap {
        // We don't need to compare hashes for value swap.
        // Not even DIBs for Robin Hood.
        buckets = match buckets.peek() {
            Empty(empty) => {
                return empty.put(hash, key, val).into_table();
            }
            Full(full) => full.into_bucket()
        };
        buckets.next();
    }

    panic!("Internal HashMap error: Out of space.");
}

impl<K, V, S> HashMap<K, V, S>
    where K: Eq + Hash, S: HashState
{
    fn make_hash<X: ?Sized>(&self, x: &X) -> SafeHash where X: Hash {
        table::make_hash(&self.hash_state, x)
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    fn search<'a, Q: ?Sized>(&'a self, q: &Q)
                            -> InternalEntry<K, V, &'a RawTable<K, V>>
        where K: Borrow<Q>, Q: Eq + Hash
    {
        let hash = self.make_hash(q);
        search_hashed(&self.table, hash, |k| q.eq(k.borrow()))
    }

    fn search_mut<'a, Q: ?Sized>(&'a mut self, q: &Q)
                                -> InternalEntry<K, V, &'a mut RawTable<K, V>>
        where K: Borrow<Q>, Q: Eq + Hash
    {
        let hash = self.make_hash(q);
        search_hashed(&mut self.table, hash, |k| q.eq(k.borrow()))
    }
}

impl<K: Hash + Eq, V> HashMap<K, V, RandomState> {
    /// Creates an empty HashMap.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, isize> = HashMap::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> HashMap<K, V, RandomState> {
        Default::default()
    }

    /// Creates an empty hash map with the given initial capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, isize> = HashMap::with_capacity(10);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> HashMap<K, V, RandomState> {
        HashMap::with_capacity_and_hash_state(capacity, Default::default())
    }
}

impl<K, V, S> HashMap<K, V, S>
    where K: Eq + Hash, S: HashState
{
    /// Creates an empty hashmap which will use the given hasher to hash keys.
    ///
    /// The creates map has the default initial capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::collections::HashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = HashMap::with_hash_state(s);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    #[unstable(feature = "std_misc", reason = "hasher stuff is unclear")]
    pub fn with_hash_state(hash_state: S) -> HashMap<K, V, S> {
        HashMap {
            hash_state:    hash_state,
            resize_policy: DefaultResizePolicy::new(),
            table:         RawTable::new(0),
        }
    }

    /// Creates an empty HashMap with space for at least `capacity`
    /// elements, using `hasher` to hash the keys.
    ///
    /// Warning: `hasher` is normally randomly generated, and
    /// is designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting it
    /// manually using this function can expose a DoS attack vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::collections::HashMap;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let s = RandomState::new();
    /// let mut map = HashMap::with_capacity_and_hash_state(10, s);
    /// map.insert(1, 2);
    /// ```
    #[inline]
    #[unstable(feature = "std_misc", reason = "hasher stuff is unclear")]
    pub fn with_capacity_and_hash_state(capacity: usize, hash_state: S)
                                        -> HashMap<K, V, S> {
        let resize_policy = DefaultResizePolicy::new();
        let min_cap = max(INITIAL_CAPACITY, resize_policy.min_capacity(capacity));
        let internal_cap = min_cap.checked_next_power_of_two().expect("capacity overflow");
        assert!(internal_cap >= capacity, "capacity overflow");
        HashMap {
            hash_state:    hash_state,
            resize_policy: resize_policy,
            table:         RawTable::new(internal_cap),
        }
    }

    /// Returns the number of elements the map can hold without reallocating.
    /// This value may be lower than the real number of elements the map will
    /// hold before reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let map: HashMap<isize, isize> = HashMap::with_capacity(100);
    /// assert!(map.capacity() >= 100);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.resize_policy.usable_capacity(self.table.capacity())
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the `HashMap`. The collection may reserve more space to avoid
    /// frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new allocation size overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, isize> = HashMap::new();
    /// map.reserve(10);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        let new_size = self.len().checked_add(additional).expect("capacity overflow");
        let min_cap = self.resize_policy.min_capacity(new_size);

        // An invalid value shouldn't make us run out of space. This includes
        // an overflow check.
        assert!(new_size <= min_cap);

        if self.table.capacity() < min_cap {
            let new_capacity = max(min_cap.next_power_of_two(), INITIAL_CAPACITY);
            let old_size = self.table.size();
            self.resize(new_capacity);
            assert_eq!(self.table.size(), old_size);
        }
    }

    /// Resizes the internal vectors to a new capacity. It's your responsibility to:
    ///   1) Make sure the new capacity is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure new_capacity is a power of two or zero.
    fn resize(&mut self, new_capacity: usize) {
        assert!(self.table.size() <= new_capacity);
        assert!(new_capacity.is_power_of_two() || new_capacity == 0);

        let old_capacity = self.table.capacity();

        if self.table.size() == 0 {
            self.table = RawTable::new(new_capacity);
            return;
        }

        let mut destination = RawTable::new(new_capacity);

        // Iterate over `old_capacity` buckets, which constitute half of
        // the table which was resized in-place, or the entire
        // `old_table`.
        let mut bucket = Bucket::new(&mut self.table, 0).unwrap();

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
        let idx_end = bucket.index() + old_capacity;

        while bucket.index() != idx_end {
            bucket = match bucket.peek() {
                Full(bucket) => {
                    let h = *bucket.read().0;
                    let (b, k, v) = bucket.take();
                    insert_hashed_ordered(&mut destination, h, k, v);
                    b.into_bucket()
                }
                Empty(b) => b.into_bucket()
            };
            bucket.next(); // wraps at old_capacity
        }

        replace(bucket.into_table(), destination);
    }

    /// Shrinks the capacity of the map as much as possible. It will drop
    /// down as much as possible while maintaining the internal rules
    /// and possibly leaving some space in accordance with the resize policy.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map: HashMap<isize, isize> = HashMap::with_capacity(100);
    /// map.insert(1, 2);
    /// map.insert(3, 4);
    /// assert!(map.capacity() >= 100);
    /// map.shrink_to_fit();
    /// assert!(map.capacity() >= 2);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        let min_capacity = self.resize_policy.min_capacity(self.len());
        let min_capacity = max(min_capacity.next_power_of_two(), INITIAL_CAPACITY);

        // An invalid value shouldn't make us run out of space.
        debug_assert!(self.len() <= min_capacity);

        if self.table.capacity() != min_capacity {
            let mut old_table = PartialRawTable::new(replace(&mut self.table,
                                                             RawTable::new(min_capacity)));
            let old_size = old_table.size();

            // Shrink the table. Naive algorithm for resizing:
            while let Some((h, k, v)) = old_table.take_front() {
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
    fn insert_hashed_nocheck(&mut self, hash: SafeHash, k: K, v: V) -> Option<V> {
        match search_hashed(&mut self.table, hash, |key| *key == k) {
            InternalEntry::Vacant(entry) => {
                entry.insert(hash, k, v);
                return None;
            }
            InternalEntry::Occupied(mut entry) => {
                return Some(entry.insert(v));
            }
            InternalEntry::TableIsEmpty => unreachable!()
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// Iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for key in map.keys() {
    ///     println!("{}", key);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((&'a K,&'a V)) -> &'a K = first; // coerce to fn ptr

        Keys { inner: self.iter().map(first) }
    }

    /// An iterator visiting all values in arbitrary order.
    /// Iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for val in map.values() {
    ///     println!("{}", val);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        fn second<A, B>((_, b): (A, B)) -> B { b }
        let second: fn((&'a K,&'a V)) -> &'a V = second; // coerce to fn ptr

        Values { inner: self.iter().map(second) }
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// Iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            elems_left: self.table.size(),
            inner: Bucket::raw_full_buckets(&self.table),
        }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// Iterator element type is `(&'a K, &'a mut V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            elems_left: self.table.size(),
            inner: Bucket::raw_full_buckets(&mut self.table),
        }
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        // Gotta resize now.
        self.reserve(1);
        let hash = self.make_hash(&key);
        match search_hashed(&mut self.table, hash, |k| key.eq(k.borrow())) {
            InternalEntry::Occupied(state) => Occupied(state),
            InternalEntry::Vacant(bucket) => Vacant(VacantEntry {
                key: key,
                hash: hash,
                elem: bucket,
            }),
            InternalEntry::TableIsEmpty => unreachable!()
        }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.table.size() }

    /// Returns true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Clears the map, returning all key-value pairs as an iterator. Keeps the
    /// allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(std_misc)]
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// for (k, v) in a.drain().take(1) {
    ///     assert!(k == 1 || k == 2);
    ///     assert!(v == "a" || v == "b");
    /// }
    ///
    /// assert!(a.is_empty());
    /// ```
    #[inline]
    #[unstable(feature = "std_misc",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn drain(&mut self) -> Drain<K, V> {
        Drain {
            inner: Bucket::new(&mut self.table, 0)
        }
    }

    /// Clears the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut a = HashMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn clear(&mut self) {
        self.drain();
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
        where K: Borrow<Q>, Q: Hash + Eq
    {
        self.search(k).into_option().map(|bucket| bucket.into_refs().1)
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains_key<Q: ?Sized>(&self, k: &Q) -> bool
        where K: Borrow<Q>, Q: Hash + Eq
    {
        self.search(k).into_option().is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
        where K: Borrow<Q>, Q: Hash + Eq
    {
        self.search_mut(k).into_option().map(|bucket| bucket.into_mut_refs().1)
    }

    /// Inserts a key-value pair into the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.reserve(1);
        let hash = self.make_hash(&key);
        self.insert_hashed_nocheck(hash, key, value)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<V>
        where K: Borrow<Q>, Q: Hash + Eq
    {
        if self.table.size() == 0 {
            return None
        }

        self.search_mut(k).into_option().map(|bucket| pop_internal(bucket).1)
    }
}

impl<K, V, S> PartialEq for HashMap<K, V, S>
    where K: Eq + Hash, V: PartialEq, S: HashState
{
    fn eq(&self, other: &HashMap<K, V, S>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter().all(|(key, value)|
            other.get(key).map_or(false, |v| *value == *v)
        )
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, S> Eq for HashMap<K, V, S>
    where K: Eq + Hash, V: Eq, S: HashState
{}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, S> Debug for HashMap<K, V, S>
    where K: Eq + Hash + Debug, V: Debug, S: HashState
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.iter().fold(f.debug_map(), |b, (k, v)| b.entry(k, v)).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, S> Default for HashMap<K, V, S>
    where K: Eq + Hash,
          S: HashState + Default,
{
    fn default() -> HashMap<K, V, S> {
        HashMap::with_hash_state(Default::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, Q: ?Sized, V, S> Index<&'a Q> for HashMap<K, V, S>
    where K: Eq + Hash + Borrow<Q>,
          Q: Eq + Hash,
          S: HashState,
{
    type Output = V;

    #[inline]
    fn index(&self, index: &Q) -> &V {
        self.get(index).expect("no entry found for key")
    }
}

impl<K: Clone, V: Clone, S: Clone> Clone for HashMap<K, V, S> {
    fn clone(&self) -> HashMap<K, V, S> {
        let mut new_ht = RawTable::new_uninitialized(self.table.capacity());

        let cap = self.table.capacity();
        if let Some(mut buckets) = Bucket::new(&self.table, 0) {
            while buckets.index() != cap {
                match buckets.peek() {
                    Full(full) => {
                        let (h, k, v) = {
                            let (h, k, v) = full.read();
                            (*h, k.clone(), v.clone())
                        };
                        new_ht.push_back(Some((h, k, v)));
                    }
                    Empty(..) => {
                        new_ht.push_back(None);
                    }
                }
                buckets.next();
            }
        }

        HashMap {
            table: new_ht.unwrap(),
            hash_state: self.hash_state.clone(),
            resize_policy: self.resize_policy.clone(),
        }
    }
}

/// HashMap iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, K: 'a, V: 'a> {
    inner: table::RawFullBuckets<K, V, &'a RawTable<K, V>>,
    elems_left: usize,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, K, V> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Iter<'a, K, V> {
        Iter {
            inner: self.inner.clone(),
            elems_left: self.elems_left,
        }
    }
}

/// HashMap mutable values iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, K: 'a, V: 'a> {
    inner: table::RawFullBuckets<K, V, &'a mut RawTable<K, V>>,
    elems_left: usize,
}

/// HashMap move iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<K, V> {
    table: PartialRawTable<K, V>,
}

/// HashMap keys iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Keys<'a, K: 'a, V: 'a> {
    inner: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, K, V> Clone for Keys<'a, K, V> {
    fn clone(&self) -> Keys<'a, K, V> {
        Keys {
            inner: self.inner.clone()
        }
    }
}

/// HashMap values iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Values<'a, K: 'a, V: 'a> {
    inner: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
impl<'a, K, V> Clone for Values<'a, K, V> {
    fn clone(&self) -> Values<'a, K, V> {
        Values {
            inner: self.inner.clone()
        }
    }
}

/// HashMap drain iterator.
#[unstable(feature = "std_misc",
           reason = "matches collection reform specification, waiting for dust to settle")]
pub struct Drain<'a, K: 'a, V: 'a> {
    inner: Option<table::Bucket<K, V, &'a mut RawTable<K, V>>>,
}

/// A view into a single occupied location in a HashMap.
#[stable(feature = "rust1", since = "1.0.0")]
pub type OccupiedEntry<'a, K: 'a, V: 'a> = OccupiedEntryState<K, V, &'a mut RawTable<K, V>>;

/// A view into a single empty location in a HashMap.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct VacantEntry<'a, K: 'a, V: 'a> {
    hash: SafeHash,
    key: K,
    elem: VacantEntryState<K, V, &'a mut RawTable<K, V>>,
}

/// A view into a single location in a map, which may be vacant or occupied.
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Entry<'a, K: 'a, V: 'a> {
    /// An occupied Entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    Occupied(OccupiedEntry<'a, K, V>),

    /// A vacant Entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    Vacant(VacantEntry<'a, K, V>),
}

/// A view into a single occupied location in a HashMap.
struct OccupiedEntryState<K, V, M> {
    elem: FullBucket<K, V, M>,
}

/// Possible states of a VacantEntry.
enum VacantEntryState<K, V, M> {
    /// The index is occupied, but the key to insert has precedence,
    /// and will kick the current one out on insertion.
    NeqElem(FullBucket<K, V, M>, usize),
    /// The index is genuinely vacant.
    NoElem(EmptyBucket<K, V, M>),
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V, S> IntoIterator for &'a HashMap<K, V, S>
    where K: Eq + Hash, S: HashState
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V, S> IntoIterator for &'a mut HashMap<K, V, S>
    where K: Eq + Hash, S: HashState
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(mut self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, S> IntoIterator for HashMap<K, V, S>
    where K: Eq + Hash, S: HashState
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Not possible with .iter()
    /// let vec: Vec<(&str, isize)> = map.into_iter().collect();
    /// ```
    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter {
            table: PartialRawTable::new(self.table)
        }
    }
}


enum InternalEntry<K, V, M> {
    Occupied(OccupiedEntryState<K, V, M>),
    Vacant(VacantEntryState<K, V, M>),
    /// The table is empty. Cannot create `EmptyBucket`.
    TableIsEmpty,
}

impl<K, V, M> InternalEntry<K, V, M> {
    fn into_option(self) -> Option<FullBucket<K, V, M>> {
        match self {
            InternalEntry::Occupied(bucket) => Some(bucket.elem),
            _ => None,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.inner.next().map(|bucket| {
            self.elems_left -= 1;
            bucket.into_refs()
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    #[inline] fn len(&self) -> usize { self.elems_left }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.inner.next().map(|bucket| {
            self.elems_left -= 1;
            let (k, v) = bucket.into_mut_refs();
            (&*k, v)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elems_left, Some(self.elems_left))
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    #[inline] fn len(&self) -> usize { self.elems_left }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        self.table.take_front().map(|(_, k, v)| (k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.table.size(), Some(self.table.size()))
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    #[inline] fn len(&self) -> usize { self.table.size() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline] fn next(&mut self) -> Option<(&'a K)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    #[inline] fn len(&self) -> usize { self.inner.len() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline] fn next(&mut self) -> Option<(&'a V)> { self.inner.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    #[inline] fn len(&self) -> usize { self.inner.len() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Drain<'a, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        if let Some(mut bucket) = self.inner.take() {
            let cap = bucket.table().capacity();
            while bucket.index() < cap {
                match bucket.peek() {
                    Full(bucket) => {
                        let (empty, k, v) = bucket.take();
                        self.inner = Some(empty.into_bucket());
                        return Some((k, v));
                    }
                    Empty(empty) => {
                        bucket = empty.into_next();
                    }
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.as_ref().map(|bucket| {
            let size = bucket.table().size();
            (size, Some(size))
        }).unwrap_or((0, Some(0)))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Drain<'a, K, V> {
    fn len(&self) -> usize { self.inner.as_ref().map(|bucket| bucket.table().size()).unwrap_or(0) }
}

impl<'a, K: 'a, V: 'a> Drop for Drain<'a, K, V> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<'a, K, V> Entry<'a, K, V> {
    /// Returns a mutable reference to the entry if occupied, or the VacantEntry if vacant
    #[unstable(feature = "std_misc",
               reason = "will soon be replaced by or_insert")]
    #[deprecated(since = "1.0",
                reason = "replaced with more ergonomic `or_insert` and `or_insert_with`")]
    pub fn get(self) -> Result<&'a mut V, VacantEntry<'a, K, V>> {
        match self {
            Occupied(entry) => Ok(entry.into_mut()),
            Vacant(entry) => Err(entry),
        }
    }

    /// Ensures a value is in the entry by inserting the default if empty, and returns
    /// a mutable reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the default function if empty,
    /// and returns a mutable reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.insert(default()),
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self) -> &V {
        self.elem.read().2
    }
}

#[unstable(feature = "std_misc",
           reason = "matches collection reform v2 specification, waiting for dust to settle")]
impl<'a, K: 'a, V: 'a, M: 'a> OccupiedEntryState<K, V, M> where M: BorrowMut<RawTable<K, V>> {
    /// Gets a mutable reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut V {
        self.elem.read_mut().2
    }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_mut(self) -> &'a mut V {
        self.elem.into_mut_refs().1
    }

    /// Sets the value of the entry, and returns the entry's old value
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, mut value: V) -> V {
        let old_value = self.get_mut();
        mem::swap(&mut value, old_value);
        value
    }
}

#[unstable(feature = "std_misc",
           reason = "matches collection reform v2 specification, waiting for dust to settle")]
impl<'a, K: 'a, V: 'a> OccupiedEntryState<K, V, &'a mut RawTable<K, V>> {
    /// Takes the value out of the entry, and returns it
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(self) -> V {
        pop_internal(self.elem).1
    }
}

impl<'a, K: 'a, V: 'a> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(self, value: V) -> &'a mut V {
        self.elem.insert(self.hash, self.key, value)
    }
}

impl<'a, K: 'a, V: 'a> VacantEntryState<K, V, &'a mut RawTable<K, V>> {
    /// Sets the value of the entry with the provided key,
    /// and returns a mutable reference to it
    fn insert(self, hash: SafeHash, key: K, value: V) -> &'a mut V {
        match self {
            NeqElem(bucket, ib) => {
                robin_hood(bucket, ib, hash, key, value)
            }
            NoElem(bucket) => {
                bucket.put(hash, key, value).into_mut_refs().1
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, S> FromIterator<(K, V)> for HashMap<K, V, S>
    where K: Eq + Hash, S: HashState + Default
{
    fn from_iter<T: IntoIterator<Item=(K, V)>>(iterable: T) -> HashMap<K, V, S> {
        let iter = iterable.into_iter();
        let lower = iter.size_hint().0;
        let mut map = HashMap::with_capacity_and_hash_state(lower,
                                                            Default::default());
        map.extend(iter);
        map
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V, S> Extend<(K, V)> for HashMap<K, V, S>
    where K: Eq + Hash, S: HashState
{
    fn extend<T: IntoIterator<Item=(K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}


/// `RandomState` is the default state for `HashMap` types.
///
/// A particular instance `RandomState` will create the same instances of
/// `Hasher`, but the hashers created by two different `RandomState`
/// instances are unlikely to produce the same result for the same values.
#[derive(Clone)]
#[unstable(feature = "std_misc",
           reason = "hashing an hash maps may be altered")]
pub struct RandomState {
    k0: u64,
    k1: u64,
}

#[unstable(feature = "std_misc",
           reason = "hashing an hash maps may be altered")]
impl RandomState {
    /// Constructs a new `RandomState` that is initialized with random keys.
    #[inline]
    #[allow(deprecated)]
    pub fn new() -> RandomState {
        let mut r = rand::thread_rng();
        RandomState { k0: r.gen(), k1: r.gen() }
    }
}

#[unstable(feature = "std_misc",
           reason = "hashing an hash maps may be altered")]
impl HashState for RandomState {
    type Hasher = SipHasher;
    fn hasher(&self) -> SipHasher {
        SipHasher::new_with_keys(self.k0, self.k1)
    }
}

#[unstable(feature = "std_misc",
           reason = "hashing an hash maps may be altered")]
impl Default for RandomState {
    #[inline]
    fn default() -> RandomState {
        RandomState::new()
    }
}

#[cfg(test)]
mod test_map {
    use prelude::v1::*;

    use super::HashMap;
    use super::Entry::{Occupied, Vacant};
    use iter::{range_inclusive, repeat};
    use cell::RefCell;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);

        assert!(m.insert(1, 1).is_none());

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert_eq!(*m.get(&2).unwrap(), 4);
    }

    #[test]
    fn test_clone() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        let m2 = m.clone();
        assert_eq!(*m2.get(&1).unwrap(), 2);
        assert_eq!(*m2.get(&2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }

    thread_local! { static DROP_VECTOR: RefCell<Vec<isize>> = RefCell::new(Vec::new()) }

    #[derive(Hash, PartialEq, Eq)]
    struct Dropable {
        k: usize
    }

    impl Dropable {
        fn new(k: usize) -> Dropable {
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
            *slot.borrow_mut() = repeat(0).take(200).collect();
        });

        {
            let mut m = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                m.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for i in 0..50 {
                let k = Dropable::new(i);
                let v = m.remove(&k);

                assert!(v.is_some());

                DROP_VECTOR.with(|v| {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i+100], 1);
                });
            }

            DROP_VECTOR.with(|v| {
                for i in 0..50 {
                    assert_eq!(v.borrow()[i], 0);
                    assert_eq!(v.borrow()[i+100], 0);
                }

                for i in 50..100 {
                    assert_eq!(v.borrow()[i], 1);
                    assert_eq!(v.borrow()[i+100], 1);
                }
            });
        }

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_into_iter_drops() {
        DROP_VECTOR.with(|v| {
            *v.borrow_mut() = repeat(0).take(200).collect();
        });

        let hm = {
            let mut hm = HashMap::new();

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 0);
                }
            });

            for i in 0..100 {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                hm.insert(d1, d2);
            }

            DROP_VECTOR.with(|v| {
                for i in 0..200 {
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
                for i in 0..200 {
                    assert_eq!(v.borrow()[i], 1);
                }
            });

            for _ in half.by_ref() {}

            DROP_VECTOR.with(|v| {
                let nk = (0..100).filter(|&i| {
                    v.borrow()[i] == 1
                }).count();

                let nv = (0..100).filter(|&i| {
                    v.borrow()[i+100] == 1
                }).count();

                assert_eq!(nk, 50);
                assert_eq!(nv, 50);
            });
        };

        DROP_VECTOR.with(|v| {
            for i in 0..200 {
                assert_eq!(v.borrow()[i], 0);
            }
        });
    }

    #[test]
    fn test_empty_remove() {
        let mut m: HashMap<isize, bool> = HashMap::new();
        assert_eq!(m.remove(&0), None);
    }

    #[test]
    fn test_empty_entry() {
        let mut m: HashMap<isize, bool> = HashMap::new();
        match m.entry(0) {
            Occupied(_) => panic!(),
            Vacant(_) => {}
        }
        assert!(*m.entry(0).or_insert(true));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_empty_iter() {
        let mut m: HashMap<isize, bool> = HashMap::new();
        assert_eq!(m.drain().next(), None);
        assert_eq!(m.keys().next(), None);
        assert_eq!(m.values().next(), None);
        assert_eq!(m.iter().next(), None);
        assert_eq!(m.iter_mut().next(), None);
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.into_iter().next(), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = HashMap::new();

        // Try this a few times to make sure we never screw up the hashmap's
        // internal state.
        for _ in 0..10 {
            assert!(m.is_empty());

            for i in range_inclusive(1, 1000) {
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

            for i in range_inclusive(1001, 2000) {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in range_inclusive(1, 1000) {
                assert!(m.remove(&i).is_some());

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
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in (1..1001).rev() {
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
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(), Some(x) => *x = new
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(&1).unwrap(), 2);
        assert!(!m.insert(1, 3).is_none());
        assert_eq!(*m.get(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(5, 3).is_none());
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(&9).unwrap(), 4);
        assert_eq!(*m.get(&5).unwrap(), 3);
        assert_eq!(*m.get(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
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
        assert!(m.insert(1, 2).is_none());
        assert!(!m.is_empty());
        assert!(m.remove(&1).is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = HashMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in 0..32 {
            assert!(m.insert(i, i*2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;

        for (k, v) in &m {
            assert_eq!(*v, *k * 2);
            observed |= 1 << *k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HashMap<_, _> = vec.into_iter().collect();
        let keys: Vec<_> = map.keys().cloned().collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: HashMap<_, _> = vec.into_iter().collect();
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.get(&1).is_none());
        m.insert(1, 2);
        match m.get(&1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2)
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
    fn test_show() {
        let mut map = HashMap::new();
        let empty: HashMap<i32, i32> = HashMap::new();

        map.insert(1, 2);
        map.insert(3, 4);

        let map_str = format!("{:?}", map);

        assert!(map_str == "{1: 2, 3: 4}" ||
                map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_expand() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0;
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

        let mut i = 0;
        for _ in 0..cap * 3 / 4 {
            m.insert(i, i);
            i += 1;
        }
        // three quarters full

        assert_eq!(m.len(), i);
        assert_eq!(m.table.capacity(), cap);

        for _ in 0..cap / 4 {
            m.insert(i, i);
            i += 1;
        }
        // half full

        let new_cap = m.table.capacity();
        assert_eq!(new_cap, cap * 2);

        for _ in 0..cap / 2 - 1 {
            i -= 1;
            m.remove(&i);
            assert_eq!(m.table.capacity(), new_cap);
        }
        // A little more than one quarter full.
        m.shrink_to_fit();
        assert_eq!(m.table.capacity(), cap);
        // again, a little more than half full
        for _ in 0..cap / 2 - 1 {
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
        m.insert(0, 0);
        m.remove(&0);
        assert!(m.capacity() >= m.len());
        for i in 0..128 {
            m.insert(i, i);
        }
        m.reserve(256);

        let usable_cap = m.capacity();
        for i in 128..(128 + 256) {
            m.insert(i, i);
            assert_eq!(m.capacity(), usable_cap);
        }

        for i in 100..(128 + 256) {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), 100);
        assert!(!m.is_empty());
        assert!(m.capacity() >= m.len());

        for i in 0..100 {
            assert_eq!(m.remove(&i), Some(i));
        }
        m.shrink_to_fit();
        m.insert(0, 0);

        assert_eq!(m.len(), 1);
        assert!(m.capacity() >= m.len());
        assert_eq!(m.remove(&0), Some(0));
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _> = xs.iter().cloned().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _>  = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<_, _>  = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_mut_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HashMap<_, _>  = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_mut_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: HashMap<_, _>  = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut map = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = HashMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[&4];
    }

    #[test]
    fn test_entry(){
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: HashMap<_, _> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
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
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(&3), None);
        assert_eq!(map.len(), 5);


        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_entry_take_doesnt_corrupt() {
        #![allow(deprecated)] //rand
        // Test for #19292
        fn check(m: &HashMap<isize, ()>) {
            for k in m.keys() {
                assert!(m.contains_key(k),
                        "{} is in keys() but not in the map?", k);
            }
        }

        let mut m = HashMap::new();
        let mut rng = thread_rng();

        // Populate the map with some items.
        for _ in 0..50 {
            let x = rng.gen_range(-10, 10);
            m.insert(x, ());
        }

        for i in 0..1000 {
            let x = rng.gen_range(-10, 10);
            match m.entry(x) {
                Vacant(_) => {},
                Occupied(e) => {
                    println!("{}: remove {}", i, x);
                    e.remove();
                },
            }

            check(&m);
        }
    }
}
