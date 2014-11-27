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

pub use self::Entry::*;
use self::SearchResult::*;
use self::VacantEntryState::*;

use borrow::BorrowFrom;
use clone::Clone;
use cmp::{max, Eq, Equiv, PartialEq};
use default::Default;
use fmt::{mod, Show};
use hash::{Hash, Hasher, RandomSipHasher};
use iter::{mod, Iterator, FromIterator, Extend};
use kinds::Sized;
use mem::{mod, replace};
use num::UnsignedInt;
use ops::{Deref, Index, IndexMut};
use option::{Some, None, Option};
use result::{Result, Ok, Err};

use super::table;
use super::table::{
    Bucket,
    Empty,
    EmptyBucket,
    Full,
    FullBucket,
    FullBucketImm,
    FullBucketMut,
    RawTable,
    SafeHash
};

// FIXME(conventions): update capacity management to match other collections (no auto-shrink)

const INITIAL_LOG2_CAP: uint = 5;
pub const INITIAL_CAPACITY: uint = 1 << INITIAL_LOG2_CAP; // 2^5

/// The default behavior of HashMap implements a load factor of 90.9%.
/// This behavior is characterized by the following conditions:
///
/// - if size > 0.909 * capacity: grow
/// - if size < 0.25 * capacity: shrink (if this won't bring capacity lower
///   than the minimum)
#[deriving(Clone)]
struct DefaultResizePolicy {
    /// Doubled minimal capacity. The capacity must never drop below
    /// the minimum capacity. (The check happens before the capacity
    /// is potentially halved.)
    minimum_capacity2: uint
}

impl DefaultResizePolicy {
    fn new(new_capacity: uint) -> DefaultResizePolicy { unimplemented!() }

    #[inline]
    fn capacity_range(&self, new_size: uint) -> (uint, uint) { unimplemented!() }

    #[inline]
    fn reserve(&mut self, new_capacity: uint) { unimplemented!() }
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

    // We keep this at the end since it might as well have tail padding.
    resize_policy: DefaultResizePolicy,
}

/// Search for a pre-hashed key.
fn search_hashed<K, V, M: Deref<RawTable<K, V>>>(table: M,
                                                 hash: &SafeHash,
                                                 is_match: |&K| -> bool)
                                                 -> SearchResult<K, V, M> { unimplemented!() }

fn pop_internal<K, V>(starting_bucket: FullBucketMut<K, V>) -> (K, V) { unimplemented!() }

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
                        -> &'a mut V { unimplemented!() }

/// A result that works like Option<FullBucket<..>> but preserves
/// the reference that grants us access to the table in any case.
enum SearchResult<K, V, M> {
    // This is an entry that holds the given key:
    FoundExisting(FullBucket<K, V, M>),

    // There was no such entry. The reference is given back:
    TableRef(M)
}

impl<K, V, M> SearchResult<K, V, M> {
    fn into_option(self) -> Option<FullBucket<K, V, M>> { unimplemented!() }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> HashMap<K, V, H> {
    fn make_hash<Sized? X: Hash<S>>(&self, x: &X) -> SafeHash { unimplemented!() }

    fn search_equiv<'a, Sized? Q: Hash<S> + Equiv<K>>(&'a self, q: &Q)
                    -> Option<FullBucketImm<'a, K, V>> { unimplemented!() }

    fn search_equiv_mut<'a, Sized? Q: Hash<S> + Equiv<K>>(&'a mut self, q: &Q)
                    -> Option<FullBucketMut<'a, K, V>> { unimplemented!() }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    fn search<'a, Sized? Q>(&'a self, q: &Q) -> Option<FullBucketImm<'a, K, V>>
        where Q: BorrowFrom<K> + Eq + Hash<S>
    { unimplemented!() }

    fn search_mut<'a, Sized? Q>(&'a mut self, q: &Q) -> Option<FullBucketMut<'a, K, V>>
        where Q: BorrowFrom<K> + Eq + Hash<S>
    { unimplemented!() }

    // The caller should ensure that invariants by Robin Hood Hashing hold.
    fn insert_hashed_ordered(&mut self, hash: SafeHash, k: K, v: V) { unimplemented!() }
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
    pub fn new() -> HashMap<K, V, RandomSipHasher> { unimplemented!() }

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
    pub fn with_capacity(capacity: uint) -> HashMap<K, V, RandomSipHasher> { unimplemented!() }
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
    pub fn with_hasher(hasher: H) -> HashMap<K, V, H> { unimplemented!() }

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
    pub fn with_capacity_and_hasher(capacity: uint, hasher: H) -> HashMap<K, V, H> { unimplemented!() }

    /// The hashtable will never try to shrink below this size. You can use
    /// this function to reduce reallocations if your hashtable frequently
    /// grows and shrinks by large amounts.
    ///
    /// This function has no effect on the operational semantics of the
    /// hashtable, only on performance.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, int> = HashMap::new();
    /// map.reserve(10);
    /// ```
    pub fn reserve(&mut self, new_minimum_capacity: uint) { unimplemented!() }

    /// Resizes the internal vectors to a new capacity. It's your responsibility to:
    ///   1) Make sure the new capacity is enough for all the elements, accounting
    ///      for the load factor.
    ///   2) Ensure new_capacity is a power of two.
    fn resize(&mut self, new_capacity: uint) { unimplemented!() }

    /// Performs any necessary resize operations, such that there's space for
    /// new_size elements.
    fn make_some_room(&mut self, new_size: uint) { unimplemented!() }

    /// Insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    ///
    /// If the key already exists, the hashtable will be returned untouched
    /// and a reference to the existing element will be returned.
    fn insert_hashed_nocheck(&mut self, hash: SafeHash, k: K, v: V) -> &mut V { unimplemented!() }

    fn insert_or_replace_with<'a>(&'a mut self,
                                  hash: SafeHash,
                                  k: K,
                                  v: V,
                                  found_existing: |&mut K, &mut V, V|)
                                  -> &'a mut V { unimplemented!() }

    /// Deprecated: use `contains_key` and `BorrowFrom` instead.
    #[deprecated = "use contains_key and BorrowFrom instead"]
    pub fn contains_key_equiv<Sized? Q: Hash<S> + Equiv<K>>(&self, key: &Q) -> bool { unimplemented!() }

    /// Deprecated: use `get` and `BorrowFrom` instead.
    #[deprecated = "use get and BorrowFrom instead"]
    pub fn find_equiv<'a, Sized? Q: Hash<S> + Equiv<K>>(&'a self, k: &Q) -> Option<&'a V> { unimplemented!() }

    /// Deprecated: use `remove` and `BorrowFrom` instead.
    #[deprecated = "use remove and BorrowFrom instead"]
    pub fn pop_equiv<Sized? Q:Hash<S> + Equiv<K>>(&mut self, k: &Q) -> Option<V> { unimplemented!() }

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
    pub fn keys(&self) -> Keys<K, V> { unimplemented!() }

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
    pub fn values(&self) -> Values<K, V> { unimplemented!() }

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
    pub fn iter(&self) -> Entries<K, V> { unimplemented!() }

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
    pub fn iter_mut(&mut self) -> MutEntries<K, V> { unimplemented!() }

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
    pub fn into_iter(self) -> MoveEntries<K, V> { unimplemented!() }

    /// Gets the given key's corresponding entry in the map for in-place manipulation
    pub fn entry<'a>(&'a mut self, key: K) -> Entry<'a, K, V> { unimplemented!() }

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
    pub fn len(&self) -> uint { unimplemented!() }

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
    pub fn is_empty(&self) -> bool { unimplemented!() }

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
    pub fn clear(&mut self) { unimplemented!() }

    /// Deprecated: Renamed to `get`.
    #[deprecated = "Renamed to `get`"]
    pub fn find(&self, k: &K) -> Option<&V> { unimplemented!() }

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
    { unimplemented!() }

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
    { unimplemented!() }

    /// Deprecated: Renamed to `get_mut`.
    #[deprecated = "Renamed to `get_mut`"]
    pub fn find_mut(&mut self, k: &K) -> Option<&mut V> { unimplemented!() }

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
    { unimplemented!() }

    /// Deprecated: Renamed to `insert`.
    #[deprecated = "Renamed to `insert`"]
    pub fn swap(&mut self, k: K, v: V) -> Option<V> { unimplemented!() }

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
    pub fn insert(&mut self, k: K, v: V) -> Option<V> { unimplemented!() }

    /// Deprecated: Renamed to `remove`.
    #[deprecated = "Renamed to `remove`"]
    pub fn pop(&mut self, k: &K) -> Option<V> { unimplemented!() }

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
    { unimplemented!() }
}

fn search_entry_hashed<'a, K: Eq, V>(table: &'a mut RawTable<K,V>, hash: SafeHash, k: K)
        -> Entry<'a, K, V> { unimplemented!() }

impl<K: Eq + Hash<S>, V: Clone, S, H: Hasher<S>> HashMap<K, V, H> {
    /// Deprecated: Use `map.get(k).cloned()`.
    ///
    /// Return a copy of the value corresponding to the key.
    #[deprecated = "Use `map.get(k).cloned()`"]
    pub fn find_copy(&self, k: &K) -> Option<V> { unimplemented!() }

    /// Deprecated: Use `map[k].clone()`.
    ///
    /// Return a copy of the value corresponding to the key.
    #[deprecated = "Use `map[k].clone()`"]
    pub fn get_copy(&self, k: &K) -> V { unimplemented!() }
}

impl<K: Eq + Hash<S>, V: PartialEq, S, H: Hasher<S>> PartialEq for HashMap<K, V, H> {
    fn eq(&self, other: &HashMap<K, V, H>) -> bool { unimplemented!() }
}

impl<K: Eq + Hash<S>, V: Eq, S, H: Hasher<S>> Eq for HashMap<K, V, H> {}

impl<K: Eq + Hash<S> + Show, V: Show, S, H: Hasher<S>> Show for HashMap<K, V, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { unimplemented!() }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> Default for HashMap<K, V, H> {
    fn default() -> HashMap<K, V, H> { unimplemented!() }
}

impl<K: Hash<S> + Eq, Sized? Q, V, S, H: Hasher<S>> Index<Q, V> for HashMap<K, V, H>
    where Q: BorrowFrom<K> + Hash<S> + Eq
{
    #[inline]
    fn index<'a>(&'a self, index: &Q) -> &'a V { unimplemented!() }
}

impl<K: Hash<S> + Eq, Sized? Q, V, S, H: Hasher<S>> IndexMut<Q, V> for HashMap<K, V, H>
    where Q: BorrowFrom<K> + Hash<S> + Eq
{
    #[inline]
    fn index_mut<'a>(&'a mut self, index: &Q) -> &'a mut V { unimplemented!() }
}

/// HashMap iterator
pub struct Entries<'a, K: 'a, V: 'a> {
    inner: table::Entries<'a, K, V>
}

/// HashMap mutable values iterator
pub struct MutEntries<'a, K: 'a, V: 'a> {
    inner: table::MutEntries<'a, K, V>
}

/// HashMap move iterator
pub struct MoveEntries<K, V> {
    inner: iter::Map<'static, (SafeHash, K, V), (K, V), table::MoveEntries<K, V>>
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
    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a V)> { unimplemented!() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { unimplemented!() }
}

impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> { unimplemented!() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { unimplemented!() }
}

impl<K, V> Iterator<(K, V)> for MoveEntries<K, V> {
    #[inline]
    fn next(&mut self) -> Option<(K, V)> { unimplemented!() }
    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { unimplemented!() }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry
    pub fn get(&self) -> &V { unimplemented!() }

    /// Gets a mutable reference to the value in the entry
    pub fn get_mut(&mut self) -> &mut V { unimplemented!() }

    /// Converts the OccupiedEntry into a mutable reference to the value in the entry
    /// with a lifetime bound to the map itself
    pub fn into_mut(self) -> &'a mut V { unimplemented!() }

    /// Sets the value of the entry, and returns the entry's old value
    pub fn set(&mut self, mut value: V) -> V { unimplemented!() }

    /// Takes the value out of the entry, and returns it
    pub fn take(self) -> V { unimplemented!() }
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it
    pub fn set(self, value: V) -> &'a mut V { unimplemented!() }
}

/// HashMap keys iterator
pub type Keys<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a K, Entries<'a, K, V>>;

/// HashMap values iterator
pub type Values<'a, K, V> =
    iter::Map<'static, (&'a K, &'a V), &'a V, Entries<'a, K, V>>;

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> FromIterator<(K, V)> for HashMap<K, V, H> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> HashMap<K, V, H> { unimplemented!() }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> Extend<(K, V)> for HashMap<K, V, H> {
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) { unimplemented!() }
}
