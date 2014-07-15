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
use cmp::{max, Eq, Equiv, PartialEq};
use collections::{Collection, Mutable, MutableSet, Map, MutableMap};
use default::Default;
use fmt::Show;
use fmt;
use RandomSipHasher;
use hash::{Hash, Hasher};
use iter::{Iterator, FromIterator, Extendable, range};
use iter;
use mem::replace;
use num;
use ops::Deref;
use option::{Some, None, Option};
use result::{Ok, Err};
use ops::Index;

use super::table::{BucketWithTable, FullBucketImm, RawTable, FullBucket, FullBucketMut, Bucket};
use super::table;

static INITIAL_LOG2_CAP: uint = 5;
pub static INITIAL_CAPACITY: uint = 1 << INITIAL_LOG2_CAP; // 2^5

/// The default behavior of HashMap implements a load factor of 90.9%.
/// This behavior is characterized by the following conditions:
///
/// - if `size * 1.1 < cap < size * 4` then shouldn't resize
/// - if `cap < minimum_capacity * 2` then shouldn't shrink
#[deriving(Clone)]
struct DefaultResizePolicy {
    /// Doubled minimal capacity. The capacity must never drop below
    /// the minimum capacity. (The check happens before the capacity
    /// is potentially halved.)
    minimum_capacity2: uint
}

impl DefaultResizePolicy {
    fn new(new_capacity: uint) -> DefaultResizePolicy {
        DefaultResizePolicy {
            minimum_capacity2: new_capacity << 1
        }
    }

    #[inline]
    fn capacity_range(&self, new_size: uint) -> (uint, uint) {
        ((new_size * 11) / 10, max(new_size << 3, self.minimum_capacity2))
    }

    #[inline]
    fn reserve(&mut self, new_capacity: uint) {
        self.minimum_capacity2 = new_capacity << 1;
    }
}

// The main performance trick in this hashmap is called Robin Hood Hashing.
// It gains its excellent performance from one crucial operation:
//
//    If an insertion collides with an existing element, and that elements
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
// Future Improvements (FIXME!)
// ============================
//
// Allow the load factor to be changed dynamically and/or at initialization.
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
// There's also an "optimization" that has been omitted regarding how the
// hashtable allocates. The vector type has set the expectation that a hashtable
// which never has an element inserted should not allocate. I'm suspicious of
// implementing this for hashtables, because supporting it has no performance
// benefit over using an `Option<HashMap<K, V>>`, and is significantly more
// complicated.

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

    table: table::RawTable<K, V>,

    // We keep this at the end since it might as well have tail padding.
    resize_policy: DefaultResizePolicy,
}

/// Search for a pre-hashed key.
fn search_hashed_generic<K, V, M: Deref<RawTable<K, V>>>(table: M, hash: &table::SafeHash, is_match: |&K| -> bool)
                        -> Option<FullBucket<K, V, M>> {
    let size = table.size();
    let mut probe = Bucket::new(table, hash);
    let ib = probe.index();

    while probe.index() != ib + size {
        let full = match probe.peek() {
            table::Empty(_) => return None, // hit an empty bucket
            table::Full(b) => b
        };

        if full.distance() + ib < full.index() {
            return None;
        }

        // If the hash doesn't match, it can't be this one..
        if *hash == full.hash() {
            let matched = {
                let (k, _) = full.read();
                is_match(k)
            };

            // If the key doesn't match, it can't be this one..
            if matched {
                return Some(full);
            }
        }

        probe = full.next();
    }

    None
}

fn search_hashed<K: Eq, V, M: Deref<RawTable<K, V>>>(table: M, hash: &table::SafeHash, k: &K)
                -> Option<table::FullBucket<K, V, M>> {
    search_hashed_generic(table, hash, |k_| *k == *k_)
}

fn pop_internal<K, V>(starting_bucket: FullBucketMut<K, V>) -> V {
    let size = {
        let table = starting_bucket.table();
        table.size()
    };
    let (empty, _k, retval) = starting_bucket.take();
    let mut gap = match empty.gap_peek() {
        Some(b) => b,
        None => return retval
    };
    // COMPILER error! wrong enum optimization. sets ptr to 0

    for _ in range(0, size) {
        if gap.full().distance() != 0 {
            gap = match gap.shift() {
                Some(b) => b,
                None => return retval
            };
            continue;
        }

        break;
    }

    // Now we're done all our shifting. Return the value we grabbed
    // earlier.
    return retval;
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> HashMap<K, V, H> {
    fn make_hash<X: Hash<S>>(&self, x: &X) -> table::SafeHash {
        table::make_hash(&self.hasher, x)
    }

    fn search_equiv<'a, Q: Hash<S> + Equiv<K>>(&'a self, q: &Q)
                    -> Option<FullBucketImm<'a, K, V>> {
        let hash = self.make_hash(q);
        search_hashed_generic(&self.table, &hash, |k| q.equiv(k))
    }

    fn search_equiv_mut<'a, Q: Hash<S> + Equiv<K>>(&'a mut self, q: &Q)
                    -> Option<FullBucketMut<'a, K, V>> {
        let hash = self.make_hash(q);
        search_hashed_generic(&mut self.table, &hash, |k| q.equiv(k))
    }

    /// Search for a key, yielding the index if it's found in the hashtable.
    /// If you already have the hash for the key lying around, use
    /// search_hashed.
    fn search<'a>(&'a self, k: &K) -> Option<FullBucketImm<'a, K, V>> {
        let hash = self.make_hash(k);
        search_hashed(&self.table, &hash, k)
    }

    fn search_mut<'a>(&'a mut self, k: &K) -> Option<FullBucketMut<'a, K, V>> {
        let hash = self.make_hash(k);
        search_hashed(&mut self.table, &hash, k)
    }

    fn insert_hashed_ordered(&mut self, hash: table::SafeHash, k: K, v: V) {
        let cap = self.table.capacity();
        let mut buckets = Bucket::new(&mut self.table, &hash);
        let ib = buckets.index();
        while buckets.index() != ib + cap {
            buckets = match buckets.peek() {
                table::Empty(empty) => {
                    empty.put(hash, k, v);
                    return;
                }
                table::Full(b) => b.into_bucket()
            };
            buckets.next();
        }
        fail!("Internal HashMap error: Out of space.");
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> Collection for HashMap<K, V, H> {
    /// Return the number of elements in the map.
    fn len(&self) -> uint { self.table.size() }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> Mutable for HashMap<K, V, H> {
    /// Clear the map, removing all key-value pairs. Keeps the allocated memory
    /// for reuse.
    fn clear(&mut self) {
        // Prevent reallocations from happening from now on. Makes it possible
        // for the map to be reused but has a downside: reserves permanently.
        self.resize_policy.reserve(self.table.size());

        let cap = self.table.capacity();
        let mut buckets = Bucket::first(&mut self.table);

        while buckets.index() != cap {
            buckets = match buckets.peek() {
                table::Empty(b)  => b.next(),
                table::Full(full) => {
                    let (b, _, _) = full.take();
                    b.next()
                }
            };
        }
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> Map<K, V> for HashMap<K, V, H> {
    fn find<'a>(&'a self, k: &K) -> Option<&'a V> {
        self.search(k).map(|bucket| {
            let (_, v) = bucket.into_refs();
            v
        })
    }

    fn contains_key(&self, k: &K) -> bool {
        self.search(k).is_some()
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> MutableMap<K, V> for HashMap<K, V, H> {
    fn find_mut<'a>(&'a mut self, k: &K) -> Option<&'a mut V> {
        match self.search_mut(k) {
            Some(bucket) => {
                let (_, v) = bucket.into_mut_refs();
                Some(v)
            }
            _ => None
        }
    }

    fn swap(&mut self, k: K, v: V) -> Option<V> {
        let hash = self.make_hash(&k);
        let potential_new_size = self.table.size() + 1;
        self.make_some_room(potential_new_size);

        let mut retval = None;
        self.insert_or_replace_with(hash, k, v, |val_ref, val| {
            retval = Some(replace(val_ref, val));
        });
        retval
    }


    fn pop(&mut self, k: &K) -> Option<V> {
        if self.table.size() == 0 {
            return None
        }

        let potential_new_size = self.table.size() - 1;
        self.make_some_room(potential_new_size);

        self.search_mut(k).map(|bucket| {
            pop_internal(bucket)
        })
    }
}

impl<K: Hash + Eq, V> HashMap<K, V, RandomSipHasher> {
    /// Create an empty HashMap.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map: HashMap<&str, int> = HashMap::with_capacity(10);
    /// ```
    #[inline]
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
            resize_policy: DefaultResizePolicy::new(INITIAL_CAPACITY),
            table:         table::RawTable::new(0),
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
        let cap = num::next_power_of_two(max(INITIAL_CAPACITY, capacity));
        HashMap {
            hasher:        hasher,
            resize_policy: DefaultResizePolicy::new(cap),
            table:         table::RawTable::new(cap),
        }
    }

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
    pub fn reserve(&mut self, new_minimum_capacity: uint) {
        let cap = num::next_power_of_two(
            max(INITIAL_CAPACITY, new_minimum_capacity));

        self.resize_policy.reserve(cap);

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
        assert!(num::is_power_of_two(new_capacity));

        let mut old_table = replace(&mut self.table, table::RawTable::new(new_capacity));
        let old_size = old_table.size();

        if old_table.capacity() == 0 {
            return;
        }

        if new_capacity < old_table.capacity() {
            for (h, k, v) in old_table.move_iter() {
                self.insert_hashed_nocheck(h, k, v);
            }
        } else {
            let mut bucket = Bucket::first(&mut old_table);

            loop {
                match bucket.peek() {
                    table::Full(full) => {
                        if full.distance() == 0 {
                            bucket = full.into_bucket();
                            break;
                        }
                        bucket = full.next();
                    }
                    table::Empty(b) => {
                        bucket = b.next();
                        break;
                    }
                };
            }

            loop {
                bucket = match bucket.peek() {
                    table::Full(bucket) => {
                        {
                            let t = bucket.table();
                            if t.size() == 0 { break }
                        }
                        let h = bucket.hash();
                        let (b, k, v) = bucket.take();
                        self.insert_hashed_ordered(h, k, v);
                        b.into_bucket()
                    }
                    table::Empty(b) => b.into_bucket()
                };
                bucket.next();
            }
        }

        assert_eq!(self.table.size(), old_size);
    }

    /// Performs any necessary resize operations, such that there's space for
    /// new_size elements.
    fn make_some_room(&mut self, new_size: uint) {
        let (grow_at, shrink_at) = self.resize_policy.capacity_range(new_size);
        let cap = self.table.capacity();

        // An invalid value shouldn't make us run out of space.
        debug_assert!(grow_at >= new_size);

        if cap <= grow_at {
            let new_capacity = max(cap << 1, INITIAL_CAPACITY);
            self.resize(new_capacity);
        } else if shrink_at <= cap {
            let new_capacity = cap >> 1;
            self.resize(new_capacity);
        }
    }

    /// Insert a pre-hashed key-value pair, without first checking
    /// that there's enough room in the buckets. Returns a reference to the
    /// newly insert value.
    ///
    /// If the key already exists, the hashtable will be returned untouched
    /// and a reference to the existing element will be returned.
    fn insert_hashed_nocheck<'a>(
        &'a mut self, hash: table::SafeHash, k: K, v: V) -> &'a mut V {
        self.insert_or_replace_with(hash, k, v, |_, _| ())
    }

    fn insert_or_replace_with<'a>(
        &'a mut self, hash: table::SafeHash, k: K, v: V,
        found_existing: |&mut V, V|
    ) -> &'a mut V {

        // Worst case, we'll find one empty bucket among `size + 1` buckets.
        let size = self.table.size();
        let mut rbucket = Bucket::new(&mut self.table, &hash);
        let ib = rbucket.index();

        loop {
            let mut bucket = match rbucket.peek() {
                table::Empty(bucket) => {
                    // Found a hole!
                    let bucket = bucket.put(hash, k, v);
                    let (_, val) = bucket.into_mut_refs();
                    return val;
                },
                table::Full(bucket) => bucket
            };

            if bucket.hash() == hash {
                let (bucket_k, bucket_v) = bucket.read_mut();
                // FIXME #12147 the conditional return confuses
                // borrowck if we return bucket_v directly
                let bv: *mut V = bucket_v;
                if k == *bucket_k {
                    // Key already exists. Get its reference.
                    found_existing(bucket_v, v);
                    return unsafe {&mut *bv};
                }
            }

            let robin_ib = bucket.index() as int - bucket.distance() as int;

            if (ib as int) < robin_ib {
                // Found a luckier bucket than me. Better steal his spot.
                let (mut hash, mut k, mut v) = bucket.replace(hash, k, v);
                let robin_index = bucket.index();
                let mut robin_ib = robin_ib as uint;
                let mut rbucket = bucket.next();
                loop {
                    let mut bucket = match rbucket.peek() {
                        table::Empty(bucket) => {
                            // Found a hole!
                            let b = bucket.put(hash, k, v);
                            // Now that it's stolen, just read the value's pointer
                            // right out of the table!
                            let (_, v) = match Bucket::at_index(b.into_table(), robin_index).peek() {
                                table::Full(b) => b.into_mut_refs(),
                                _ => fail!()
                            };
                            return v;
                        },
                        table::Full(bucket) => bucket
                    };

                    let probe_ib = bucket.index() - bucket.distance();

                    // Robin hood! Steal the spot.
                    if robin_ib < probe_ib {
                        robin_ib = probe_ib;
                        let (old_hash, old_key, old_val) = bucket.replace(hash, k, v);
                        hash = old_hash;
                        k = old_key;
                        v = old_val;
                    }
                    rbucket = bucket.next();
                    if rbucket.index() == ib + size + 1 {
                        fail!("HashMap fatal error: 100% load factor?")
                    }
                }
            }
            rbucket = bucket.next();
            if rbucket.index() == ib + size + 1 {
                fail!("Internal HashMap error: Out of space.")
            }
        }
    }

    /// Inserts an element which has already been hashed, returning a reference
    /// to that element inside the hashtable. This is more efficient that using
    /// `insert`, since the key will not be rehashed.
    fn insert_hashed<'a>(&'a mut self, hash: table::SafeHash, k: K, v: V) -> &'a mut V {
        let potential_new_size = self.table.size() + 1;
        self.make_some_room(potential_new_size);
        self.insert_hashed_nocheck(hash, k, v)
    }

    /// Return the value corresponding to the key in the map, or insert
    /// and return the value if it doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map = HashMap::new();
    ///
    /// // Insert 1i with key "a"
    /// assert_eq!(*map.find_or_insert("a", 1i), 1);
    ///
    /// // Find the existing key
    /// assert_eq!(*map.find_or_insert("a", -2), 1);
    /// ```
    pub fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        self.find_with_or_insert_with(k, v, |_k, _v, _a| (), |_k, a| a)
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map = HashMap::new();
    ///
    /// // Insert 10 with key 2
    /// assert_eq!(*map.find_or_insert_with(2i, |&key| 5 * key as uint), 10u);
    ///
    /// // Find the existing key
    /// assert_eq!(*map.find_or_insert_with(2, |&key| key as uint), 10);
    /// ```
    pub fn find_or_insert_with<'a>(&'a mut self, k: K, f: |&K| -> V)
                               -> &'a mut V {
        self.find_with_or_insert_with(k, (), |_k, _v, _a| (), |k, _a| f(k))
    }

    /// Insert a key-value pair into the map if the key is not already present.
    /// Otherwise, modify the existing value for the key.
    /// Returns the new or modified value for the key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// let mut map = HashMap::new();
    ///
    /// // Insert 2 with key "a"
    /// assert_eq!(*map.insert_or_update_with("a", 2u, |_key, val| *val = 3), 2);
    ///
    /// // Update and return the existing value
    /// assert_eq!(*map.insert_or_update_with("a", 9, |_key, val| *val = 7), 7);
    /// assert_eq!(map["a"], 7);
    /// ```
    pub fn insert_or_update_with<'a>(
                                 &'a mut self,
                                 k: K,
                                 v: V,
                                 f: |&K, &mut V|)
                                 -> &'a mut V {
        self.find_with_or_insert_with(k, v, |k, v, _a| f(k, v), |_k, a| a)
    }

    /// Modify and return the value corresponding to the key in the map, or
    /// insert and return a new value if it doesn't exist.
    ///
    /// This method allows for all insertion behaviours of a hashmap;
    /// see methods like
    /// [`insert`](../trait.MutableMap.html#tymethod.insert),
    /// [`find_or_insert`](#method.find_or_insert) and
    /// [`insert_or_update_with`](#method.insert_or_update_with)
    /// for less general and more friendly variations of this.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// // map some strings to vectors of strings
    /// let mut map = HashMap::new();
    /// map.insert("a key", vec!["value"]);
    /// map.insert("z key", vec!["value"]);
    ///
    /// let new = vec!["a key", "b key", "z key"];
    ///
    /// for k in new.move_iter() {
    ///     map.find_with_or_insert_with(
    ///         k, "new value",
    ///         // if the key does exist either prepend or append this
    ///         // new value based on the first letter of the key.
    ///         |key, already, new| {
    ///             if key.as_slice().starts_with("z") {
    ///                 already.insert(0, new);
    ///             } else {
    ///                 already.push(new);
    ///             }
    ///         },
    ///         // if the key doesn't exist in the map yet, add it in
    ///         // the obvious way.
    ///         |_k, v| vec![v]);
    /// }
    ///
    /// assert_eq!(map.len(), 3);
    /// assert_eq!(map["a key"], vec!["value", "new value"]);
    /// assert_eq!(map["b key"], vec!["new value"]);
    /// assert_eq!(map["z key"], vec!["new value", "value"]);
    /// ```
    pub fn find_with_or_insert_with<'a, A>(&'a mut self,
                                           k: K,
                                           a: A,
                                           found: |&K, &mut V, A|,
                                           not_found: |&K, A| -> V)
                                          -> &'a mut V {
        let hash = self.make_hash(&k);
        {
            match search_hashed(&mut self.table, &hash, &k) {
                Some(bucket) => {
                    let (_, v_ref) = bucket.into_mut_refs();
                    found(&k, v_ref, a);
                    return v_ref;
                }
                _ => {
                }
            };
        }
        let v = not_found(&k, a);
        self.insert_hashed(hash, k, v)
    }

    /// Retrieves a value for the given key.
    /// See [`find`](../trait.Map.html#tymethod.find) for a non-failing alternative.
    ///
    /// # Failure
    ///
    /// Fails if the key is not present.
    ///
    /// # Example
    ///
    /// ```
    /// #![allow(deprecated)]
    ///
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.get(&"a"), &1);
    /// ```
    #[deprecated = "prefer indexing instead, e.g., map[key]"]
    pub fn get<'a>(&'a self, k: &K) -> &'a V {
        match self.find(k) {
            Some(v) => v,
            None => fail!("no entry found for key")
        }
    }

    /// Retrieves a mutable value for the given key.
    /// See [`find_mut`](../trait.MutableMap.html#tymethod.find_mut) for a non-failing alternative.
    ///
    /// # Failure
    ///
    /// Fails if the key is not present.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// {
    ///     // val will freeze map to prevent usage during its lifetime
    ///     let val = map.get_mut(&"a");
    ///     *val = 40;
    /// }
    /// assert_eq!(map["a"], 40);
    ///
    /// // A more direct way could be:
    /// *map.get_mut(&"a") = -2;
    /// assert_eq!(map["a"], -2);
    /// ```
    pub fn get_mut<'a>(&'a mut self, k: &K) -> &'a mut V {
        match self.find_mut(k) {
            Some(v) => v,
            None => fail!("no entry found for key")
        }
    }

    /// Return true if the map contains a value for the specified key,
    /// using equivalence.
    ///
    /// See [pop_equiv](#method.pop_equiv) for an extended example.
    pub fn contains_key_equiv<Q: Hash<S> + Equiv<K>>(&self, key: &Q) -> bool {
        self.search_equiv(key).is_some()
    }

    /// Return the value corresponding to the key in the map, using
    /// equivalence.
    ///
    /// See [pop_equiv](#method.pop_equiv) for an extended example.
    pub fn find_equiv<'a, Q: Hash<S> + Equiv<K>>(&'a self, k: &Q) -> Option<&'a V> {
        match self.search_equiv(k) {
            None      => None,
            Some(bucket) => {
                let (_, v_ref) = bucket.into_refs();
                Some(v_ref)
            }
        }
    }

    /// Remove an equivalent key from the map, returning the value at the
    /// key if the key was previously in the map.
    ///
    /// # Example
    ///
    /// This is a slightly silly example where we define the number's
    /// parity as the equivalence class. It is important that the
    /// values hash the same, which is why we implement `Hash`.
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use std::hash::Hash;
    /// use std::hash::sip::SipState;
    ///
    /// #[deriving(Eq, PartialEq)]
    /// struct EvenOrOdd {
    ///     num: uint
    /// };
    ///
    /// impl Hash for EvenOrOdd {
    ///     fn hash(&self, state: &mut SipState) {
    ///         let parity = self.num % 2;
    ///         parity.hash(state);
    ///     }
    /// }
    ///
    /// impl Equiv<EvenOrOdd> for EvenOrOdd {
    ///     fn equiv(&self, other: &EvenOrOdd) -> bool {
    ///         self.num % 2 == other.num % 2
    ///     }
    /// }
    ///
    /// let mut map = HashMap::new();
    /// map.insert(EvenOrOdd { num: 3 }, "foo");
    ///
    /// assert!(map.contains_key_equiv(&EvenOrOdd { num: 1 }));
    /// assert!(!map.contains_key_equiv(&EvenOrOdd { num: 4 }));
    ///
    /// assert_eq!(map.find_equiv(&EvenOrOdd { num: 5 }), Some(&"foo"));
    /// assert_eq!(map.find_equiv(&EvenOrOdd { num: 2 }), None);
    ///
    /// assert_eq!(map.pop_equiv(&EvenOrOdd { num: 1 }), Some("foo"));
    /// assert_eq!(map.pop_equiv(&EvenOrOdd { num: 2 }), None);
    ///
    /// ```
    #[experimental]
    pub fn pop_equiv<Q:Hash<S> + Equiv<K>>(&mut self, k: &Q) -> Option<V> {
        if self.table.size() == 0 {
            return None
        }

        let potential_new_size = self.table.size() - 1;
        self.make_some_room(potential_new_size);

        match self.search_equiv_mut(k) {
            Some(bucket) => {
                Some(pop_internal(bucket))
            }
            _ => None
        }
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
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        self.iter().map(|(k, _v)| k)
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
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        self.iter().map(|(_k, v)| v)
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
    pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
        self.table.iter()
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
    /// for (_, val) in map.mut_iter() {
    ///     *val *= 2;
    /// }
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        self.table.mut_iter()
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
    /// let vec: Vec<(&str, int)> = map.move_iter().collect();
    /// ```
    pub fn move_iter(self) -> MoveEntries<K, V> {
        self.table.move_iter().map(|(_, k, v)| (k, v))
    }
}

impl<K: Eq + Hash<S>, V: Clone, S, H: Hasher<S>> HashMap<K, V, H> {
    /// Return a copy of the value corresponding to the key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map: HashMap<uint, String> = HashMap::new();
    /// map.insert(1u, "foo".to_string());
    /// let s: String = map.find_copy(&1).unwrap();
    /// ```
    pub fn find_copy(&self, k: &K) -> Option<V> {
        self.find(k).map(|v| (*v).clone())
    }

    /// Return a copy of the value corresponding to the key.
    ///
    /// # Failure
    ///
    /// Fails if the key is not present.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map: HashMap<uint, String> = HashMap::new();
    /// map.insert(1u, "foo".to_string());
    /// let s: String = map.get_copy(&1);
    /// ```
    pub fn get_copy(&self, k: &K) -> V {
        (*self.get(k)).clone()
    }
}

impl<K: Eq + Hash<S>, V: PartialEq, S, H: Hasher<S>> PartialEq for HashMap<K, V, H> {
    fn eq(&self, other: &HashMap<K, V, H>) -> bool {
        if self.len() != other.len() { return false; }

        self.iter()
          .all(|(key, value)| {
            match other.find(key) {
                None    => false,
                Some(v) => *value == *v
            }
        })
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

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> Default for HashMap<K, V, H> {
    fn default() -> HashMap<K, V, H> {
        HashMap::with_hasher(Default::default())
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> Index<K, V> for HashMap<K, V, H> {
    #[inline]
    fn index<'a>(&'a self, index: &K) -> &'a V {
        self.get(index)
    }
}

// FIXME(#12825) Indexing will always try IndexMut first and that causes issues.
/*impl<K: Eq + Hash<S>, V, S, H: Hasher<S>> ops::IndexMut<K, V> for HashMap<K, V, H> {
    #[inline]
    fn index_mut<'a>(&'a mut self, index: &K) -> &'a mut V {
        self.get_mut(index)
    }
}*/

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

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> FromIterator<(K, V)> for HashMap<K, V, H> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> HashMap<K, V, H> {
        let (lower, _) = iter.size_hint();
        let mut map = HashMap::with_capacity_and_hasher(lower, Default::default());
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash<S>, V, S, H: Hasher<S> + Default> Extendable<(K, V)> for HashMap<K, V, H> {
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
    use cmp::Equiv;
    use hash;
    use iter::{Iterator,range_inclusive,range_step_inclusive};
    use cell::RefCell;

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

        assert!(m.insert(1i, 1i));

        assert!(m.contains_key(&1));
        assert!(!m.contains_key(&0));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1i, 2i));
        assert_eq!(m.len(), 1);
        assert!(m.insert(2i, 4i));
        assert_eq!(m.len(), 2);
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert_eq!(*m.find(&2).unwrap(), 4);
    }

    local_data_key!(drop_vector: RefCell<Vec<int>>)

    #[deriving(Hash, PartialEq, Eq)]
    struct Dropable {
        k: uint
    }


    impl Dropable {
        fn new(k: uint) -> Dropable {
            let v = drop_vector.get().unwrap();
            v.borrow_mut().as_mut_slice()[k] += 1;

            Dropable { k: k }
        }
    }

    impl Drop for Dropable {
        fn drop(&mut self) {
            let v = drop_vector.get().unwrap();
            v.borrow_mut().as_mut_slice()[self.k] -= 1;
        }
    }

    impl Clone for Dropable {
        fn clone(&self) -> Dropable {
            Dropable::new(self.k)
        }
    }

    #[test]
    fn test_drops() {
        drop_vector.replace(Some(RefCell::new(Vec::from_elem(200, 0i))));

        {
            let mut m = HashMap::new();

            let v = drop_vector.get().unwrap();
            for i in range(0u, 200) {
                assert_eq!(v.borrow().as_slice()[i], 0);
            }
            drop(v);

            for i in range(0u, 100) {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                m.insert(d1, d2);
            }

            let v = drop_vector.get().unwrap();
            for i in range(0u, 200) {
                assert_eq!(v.borrow().as_slice()[i], 1);
            }
            drop(v);

            for i in range(0u, 50) {
                let k = Dropable::new(i);
                let v = m.pop(&k);

                assert!(v.is_some());

                let v = drop_vector.get().unwrap();
                assert_eq!(v.borrow().as_slice()[i], 1);
                assert_eq!(v.borrow().as_slice()[i+100], 1);
            }

            let v = drop_vector.get().unwrap();
            for i in range(0u, 50) {
                assert_eq!(v.borrow().as_slice()[i], 0);
                assert_eq!(v.borrow().as_slice()[i+100], 0);
            }

            for i in range(50u, 100) {
                assert_eq!(v.borrow().as_slice()[i], 1);
                assert_eq!(v.borrow().as_slice()[i+100], 1);
            }
        }

        let v = drop_vector.get().unwrap();
        for i in range(0u, 200) {
            assert_eq!(v.borrow().as_slice()[i], 0);
        }
    }

    #[test]
    fn test_move_iter_drops() {
        drop_vector.replace(Some(RefCell::new(Vec::from_elem(200, 0i))));

        let hm = {
            let mut hm = HashMap::new();

            let v = drop_vector.get().unwrap();
            for i in range(0u, 200) {
                assert_eq!(v.borrow().as_slice()[i], 0);
            }
            drop(v);

            for i in range(0u, 100) {
                let d1 = Dropable::new(i);
                let d2 = Dropable::new(i+100);
                hm.insert(d1, d2);
            }

            let v = drop_vector.get().unwrap();
            for i in range(0u, 200) {
                assert_eq!(v.borrow().as_slice()[i], 1);
            }
            drop(v);

            hm
        };

        drop(hm.clone());

        {
            let mut half = hm.move_iter().take(50);

            let v = drop_vector.get().unwrap();
            for i in range(0u, 200) {
                assert_eq!(v.borrow().as_slice()[i], 1);
            }
            drop(v);

            for _ in half {}

            let v = drop_vector.get().unwrap();
            let nk = range(0u, 100).filter(|&i| {
                v.borrow().as_slice()[i] == 1
            }).count();

            let nv = range(0u, 100).filter(|&i| {
                v.borrow().as_slice()[i+100] == 1
            }).count();

            assert_eq!(nk, 50);
            assert_eq!(nv, 50);
        };

        let v = drop_vector.get().unwrap();
        for i in range(0u, 200) {
            assert_eq!(v.borrow().as_slice()[i], 0);
        }
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
        for _ in range(0i, 10) {
            assert!(m.is_empty());

            for i in range_inclusive(1i, 1000) {
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

            for i in range_inclusive(1001i, 2000) {
                assert!(!m.contains_key(&i));
            }

            // remove forwards
            for i in range_inclusive(1i, 1000) {
                assert!(m.remove(&i));

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
                assert!(m.insert(i, i));
            }

            // remove backwards
            for i in range_step_inclusive(1000i, 1, -1) {
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
        assert!(m.insert(1i, 12i));
        assert!(m.insert(2i, 8i));
        assert!(m.insert(5i, 14i));
        let new = 100;
        match m.find_mut(&5) {
            None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1i, 2i));
        assert_eq!(*m.find(&1).unwrap(), 2);
        assert!(!m.insert(1i, 3i));
        assert_eq!(*m.find(&1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1i, 2i));
        assert!(m.insert(5i, 3i));
        assert!(m.insert(9i, 4i));
        assert_eq!(*m.find(&9).unwrap(), 4);
        assert_eq!(*m.find(&5).unwrap(), 3);
        assert_eq!(*m.find(&1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1i, 2i));
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
        assert!(m.insert(1i, 2i));
        assert!(!m.is_empty());
        assert!(m.remove(&1));
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = HashMap::new();
        m.insert(1i, 2i);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
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
    fn test_swap() {
        let mut m = HashMap::new();
        assert_eq!(m.swap(1i, 2i), None);
        assert_eq!(m.swap(1i, 3i), Some(2));
        assert_eq!(m.swap(1i, 4i), Some(3));
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in range(0u, 32) {
            assert!(m.insert(i, i*2));
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
        let map = vec.move_iter().collect::<HashMap<int, char>>();
        let keys = map.keys().map(|&k| k).collect::<Vec<int>>();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1i, 'a'), (2i, 'b'), (3i, 'c')];
        let map = vec.move_iter().collect::<HashMap<int, char>>();
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.find(&1i).is_none());
        m.insert(1i, 2i);
        match m.find(&1) {
            None => fail!(),
            Some(v) => assert_eq!(*v, 2)
        }
    }

    #[test]
    fn test_find_copy() {
        let mut m = HashMap::new();
        assert!(m.find(&1i).is_none());

        for i in range(1i, 10000) {
            m.insert(i, i + 7);
            match m.find_copy(&i) {
                None => fail!(),
                Some(v) => assert_eq!(v, i + 7)
            }
            for j in range(1i, i/100) {
                match m.find_copy(&j) {
                    None => fail!(),
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

        assert!(map_str == "{1: 2, 3: 4}".to_string() || map_str == "{3: 4, 1: 2}".to_string());
        assert_eq!(format!("{}", empty), "{}".to_string());
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
    fn test_resize_policy() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert_eq!(m.table.capacity(), 0);
        assert!(m.is_empty());

        m.insert(0, 0);
        m.remove(&0);
        assert!(m.is_empty());
        let initial_cap = m.table.capacity();
        m.reserve(initial_cap * 2);
        let cap = m.table.capacity();

        assert_eq!(cap, initial_cap * 2);

        let mut i = 0u;
        for _ in range(0, cap * 3 / 4) {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert_eq!(m.table.capacity(), cap);

        for _ in range(0, cap / 4) {
            m.insert(i, i);
            i += 1;
        }

        let new_cap = m.table.capacity();
        assert_eq!(new_cap, cap * 2);

        for _ in range(0, cap / 2) {
            i -= 1;
            m.remove(&i);
            assert_eq!(m.table.capacity(), new_cap);
        }

        for _ in range(0, cap / 2 - 1) {
            i -= 1;
            m.remove(&i);
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
        assert_eq!(m.table.capacity(), cap);
    }

    #[test]
    fn test_find_equiv() {
        let mut m = HashMap::new();

        let (foo, bar, baz) = (1i,2i,3i);
        m.insert("foo".to_string(), foo);
        m.insert("bar".to_string(), bar);
        m.insert("baz".to_string(), baz);


        assert_eq!(m.find_equiv(&("foo")), Some(&foo));
        assert_eq!(m.find_equiv(&("bar")), Some(&bar));
        assert_eq!(m.find_equiv(&("baz")), Some(&baz));

        assert_eq!(m.find_equiv(&("qux")), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1i, 1i), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
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

        let mut iter = map.mut_iter();

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
}
