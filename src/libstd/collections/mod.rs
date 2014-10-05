// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Collection types.
//!
//! Rust's standard collection library provides efficient implementations of the most common
//! general purpose programming data structures. By using the standard implementations,
//! it should be possible for two libraries to communicate without significant data conversion.
//!
//! To get this out of the way: you should probably just use `Vec` or `HashMap`. These two
//! collections cover most use cases for generic data storage and processing. They are
//! exceptionally good at doing what they do. All the other collections in the standard
//! library have specific use cases where they are the optimal choice, but these cases are
//! borderline *niche* in comparison. Even when `Vec` and `HashMap` are technically suboptimal,
//! they're probably a good enough choice to get started.
//!
//! Rust's collections can be grouped into four major categories:
//!
//! * Sequences: `Vec`, `RingBuf`, `DList`, `BitV`
//! * Maps: `HashMap`, `BTreeMap`, `TreeMap`, `TrieMap`, `SmallIntMap`, `LruCache`
//! * Sets: `HashSet`, `BTreeSet`, `TreeSet`, `TrieSet`, `BitVSet`, `EnumSet`
//! * Misc: `PriorityQueue`
//!
//! # When Should You Use Which Collection?
//!
//! These are fairly high-level and quick break-downs of when each collection should be
//! considered. Detailed discussions of strengths and weaknesses of individual collections
//! can be found on their own documentation pages.
//!
//! ### Use a `Vec` when:
//! * You want to collect items up to be processed or sent elsewhere later, and don't care about
//! any properties of the actual values being stored.
//! * You want a sequence of elements in a particular order, and will only be appending to
//! (or near) the end.
//! * You want a stack.
//! * You want a resizable array.
//! * You want a heap-allocated array.
//!
//! ### Use a `RingBuf` when:
//! * You want a `Vec` that supports efficient insertion at both ends of the sequence.
//! * You want a queue.
//! * You want a double-ended queue (deque).
//!
//! ### Use a `DList` when:
//! * You want a `Vec` or `RingBuf` of unknown size, and can't tolerate inconsistent
//! performance during insertions.
//! * You are *absolutely* certain you *really*, *truly*, want a doubly linked list.
//!
//! ### Use a `HashMap` when:
//! * You want to associate arbitrary keys with an arbitrary value.
//! * You want a cache.
//! * You want a map, with no extra functionality.
//!
//! ### Use a `BTreeMap` when:
//! * You're interested in what the smallest or largest key-value pair is.
//! * You want to find the largest or smallest key that is smaller or larger than something
//! * You want to be able to get all of the entries in order on-demand.
//! * You want a sorted map.
//!
//! ### Use a `TreeMap` when:
//! * You want a `BTreeMap`, but can't tolerate inconsistent performance.
//! * You want a `BTreeMap`, but have *very large* keys or values.
//! * You want a `BTreeMap`, but have keys that are expensive to compare.
//! * You want a `BTreeMap`, but you accept arbitrary untrusted inputs.
//!
//! ### Use a `TrieMap` when:
//! * You want a `HashMap`, but with many potentially large `uint` keys.
//! * You want a `BTreeMap`, but with potentially large `uint` keys.
//!
//! ### Use a `SmallIntMap` when:
//! * You want a `HashMap` but with known to be small `uint` keys.
//! * You want a `BTreeMap`, but with known to be small `uint` keys.
//!
//! ### Use the `Set` variant of any of these `Map`s when:
//! * You just want to remember which keys you've seen.
//! * There is no meaningful value to associate with your keys.
//! * You just want a set.
//!
//! ### Use a `BitV` when:
//! * You want to store an unbounded number of booleans in a small space.
//! * You want a bitvector.
//!
//! ### Use a `BitVSet` when:
//! * You want a `SmallIntSet`.
//!
//! ### Use an `EnumSet` when:
//! * You want a C-like enum, stored in a single `uint`.
//!
//! ### Use a `PriorityQueue` when:
//! * You want to store a bunch of elements, but only ever want to process the "biggest"
//! or "most important" one at any given time.
//! * You want a priority queue.
//!
//! ### Use an `LruCache` when:
//! * You want a cache that discards infrequently used items when it becomes full.
//! * You want a least-recently-used cache.
//!
//! # Correct and Efficient Usage of Collections
//!
//! Of course, knowing which collection is the right one for the job doesn't instantly
//! permit you to use it correctly. Here are some quick tips for efficient and correct
//! usage of the standard collections in general. If you're interested in how to use a
//! specific collection in particular, consult its documentation for detailed discussion
//! and code examples.
//!
//! ## Capacity Management
//!
//! Many collections provide several constructors and methods that refer to "capacity".
//! These collections are generally built on top of an array. Optimally, this array would be
//! exactly the right size to fit only the elements stored in the collection, but for the
//! collection to do this would be very inefficient. If the backing array was exactly the
//! right size at all times, then every time an element is inserted, the collection would
//! have to grow the array to fit it. Due to the way memory is allocated and managed on most
//! computers, this would almost surely require allocating an entirely new array and
//! copying every single element from the old one into the new one. Hopefully you can
//! see that this wouldn't be very efficient to do on every operation.
//!
//! Most collections therefore use an *amortized* allocation strategy. They generally let
//! themselves have a fair amount of unoccupied space so that they only have to grow
//! on occasion. When they do grow, they allocate a substantially larger array to move
//! the elements into so that it will take a while for another grow to be required. While
//! this strategy is great in general, it would be even better if the collection *never*
//! had to resize its backing array. Unfortunately, the collection itself doesn't have
//! enough information to do this itself. Therefore, it is up to us programmers to give it
//! hints.
//!
//! Any `with_capacity` constructor will instruct the collection to allocate enough space
//! for the specified number of elements. Ideally this will be for exactly that many
//! elements, but some implementation details may prevent this. `Vec` and `RingBuf` can
//! be relied on to allocate exactly the requested amount, though. Use `with_capacity`
//! when you know exactly how many elements will be inserted, or at least have a
//! reasonable upper-bound on that number.
//!
//! When anticipating a large influx of elements, the `reserve` family of methods can
//! be used to hint to the collection how much room it should make for the coming items.
//! As with `with_capacity`, the precise behavior of these methods will be specific to
//! the collection of interest.
//!
//! For optimal performance, collections will generally avoid shrinking themselves.
//! If you believe that a collection will not soon contain any more elements, or
//! just really need the memory, the `shrink_to_fit` method prompts the collection
//! to shrink the backing array to the minimum size capable of holding its elements.
//!
//! Finally, if ever you're interested in what the actual capacity of the collection is,
//! most collections provide a `capacity` method to query this information on demand.
//! This can be useful for debugging purposes, or for use with the `reserve` methods.
//!
//! ## Iterators
//!
//! Iterators are a powerful and robust mechanism used throughout Rust's standard
//! libraries. Iterators provide a sequence of values in a generic, safe, efficient
//! and convenient way. The contents of an iterator are usually *lazily* evaluated,
//! so that only the values that are actually needed are ever actually produced, and
//! no allocation need be done to temporarily store them. Iterators are primarily
//! consumed using a `for` loop, although many functions also take iterators where
//! a collection or sequence of values is desired.
//!
//! All of the standard collections provide several iterators for performing bulk
//! manipulation of their contents. The three primary iterators almost every collection
//! should provide are `iter`, `iter_mut`, and `into_iter`. Some of these are not
//! provided on collections where it would be unsound or unreasonable to provide them.
//!
//! `iter` provides an iterator of immutable references to all the contents of a
//! collection in the most "natural" order. For sequence collections like `Vec`, this
//! means the items will be yielded in increasing order of index starting at 0. For ordered
//! collections like `BTreeMap`, this means that the items will be yielded in sorted order.
//! For unordered collections like `HashMap`, the items will be yielded in whatever order
//! the internal representation made most convenient. This is great for reading through
//! all the contents of the collection.
//!
//! ```
//! let vec = vec![1u, 2, 3, 4];
//! for x in vec.iter() {
//!    println!("vec contained {}", x);
//! }
//! ```
//!
//! `iter_mut` provides an iterator of *mutable* references in the same order as `iter`.
//! This is great for mutating all the contents of the collection.
//!
//! ```
//! let mut vec = vec![1u, 2, 3, 4];
//! for x in vec.iter_mut() {
//!    *x += 1;
//! }
//! ```
//!
//! `into_iter` transforms the actual collection into an iterator over its contents
//! by-value. This is great when the collection itself is no longer needed, and the
//! values are needed elsewhere. Using `extend` with `into_iter` is the main way that
//! contents of one collection are moved into another. Calling `collect` on an iterator
//! itself is also a great way to convert one collection into another. Both of these
//! methods should internally use the capacity management tools discussed in the
//! previous section to do this as efficiently as possible.
//!
//! ```
//! let mut vec1 = vec![1u, 2, 3, 4];
//! let vec2 = vec![10u, 20, 30, 40];
//! vec1.extend(vec2.into_iter());
//! ```
//!
//! ```
//! use std::collections::RingBuf;
//!
//! let vec = vec![1u, 2, 3, 4];
//! let buf: RingBuf<uint> = vec.into_iter().collect();
//! ```
//!
//! Iterators also provide a series of *adapter* methods for performing common tasks to
//! sequences. Among the adapters are functional favorites like `map`, `fold`, `skip`,
//! and `take`. Of particular interest to collections is the `rev` adapter, that
//! reverses any iterator that supports this operation. Most collections provide reversible
//! iterators as the way to iterate over them in reverse order.
//!
//! ```
//! let vec = vec![1u, 2, 3, 4];
//! for x in vec.iter().rev() {
//!    println!("vec contained {}", x);
//! }
//! ```
//!
//! Several other collection methods also return iterators to yield a sequence of results
//! but avoid allocating an entire collection to store the result in. This provides maximum
//! flexibility as `collect` or `extend` can be called to "pipe" the sequence into any
//! collection if desired. Otherwise, the sequence can be looped over with a `for` loop. The
//! iterator can also be discarded after partial use, preventing the computation of the unused
//! items.
//!
//! ## Entries
//!
//! The `entry` API is intended to provide an efficient mechanism for manipulating
//! the contents of a map conditionally on the presence of a key or not. The primary
//! motivating use case for this is to provide efficient accumulator maps. For instance,
//! if one wishes to maintain a count of the number of times each key has been seen,
//! they will have to perform some conditional logic on whether this is the first time
//! the key has been seen or not. Normally, this would require a `find` followed by an
//! `insert`, effectively duplicating the search effort on each insertion.
//!
//! When a user calls `map.entry(key)`, the map will search for the key and then yield
//! a variant of the `Entry` enum.
//!
//! If a `Vacant(entry)` is yielded, then the key *was not* found. In this case the
//! only valid operation is to `set` the value of the entry. When this is done,
//! the vacant entry is consumed and converted into a mutable reference to the
//! the value that was inserted. This allows for further manipulation of the value
//! beyond the lifetime of the search itself. This is useful if complex logic needs to
//! be performed on the value regardless of whether the value was just inserted.
//!
//! If an `Occupied(entry)` is yielded, then the key *was* found. In this case, the user
//! has several options: they can `get`, `set`, or `take` the value of the occupied
//! entry. Additionally, they can convert the occupied entry into a mutable reference
//! to its value, providing symmetry to the vacant `set` case.
//!
//! ### Examples
//!
//! Here are the two primary ways in which `entry` is used. First, a simple example
//! where the logic performed on the values is trivial.
//!
//! #### Counting the number of times each character in a string occurs
//!
//! ```
//! use std::collections::btree::{BTreeMap, Occupied, Vacant};
//!
//! let mut count = BTreeMap::new();
//! let message = "she sells sea shells by the sea shore";
//!
//! for c in message.chars() {
//!     match count.entry(c) {
//!         Vacant(entry) => { entry.set(1u); },
//!         Occupied(mut entry) => *entry.get_mut() += 1,
//!     }
//! }
//!
//! assert_eq!(count.find(&'s'), Some(&8));
//!
//! println!("Number of occurences of each character");
//! for (char, count) in count.iter() {
//!     println!("{}: {}", char, count);
//! }
//! ```
//!
//! When the logic to be performed on the value is more complex, we may simply use
//! the `entry` API to ensure that the value is initialized, and perform the logic
//! afterwards.
//!
//! #### Tracking the inebriation of customers at a bar
//!
//! ```
//! use std::collections::btree::{BTreeMap, Occupied, Vacant};
//!
//! // A client of the bar. They have an id and a blood alcohol level.
//! struct Person { id: u32, blood_alcohol: f32 };
//!
//! // All the orders made to the bar, by client id.
//! let orders = vec![1,2,1,2,3,4,1,2,2,3,4,1,1,1];
//!
//! // Our clients.
//! let mut blood_alcohol = BTreeMap::new();
//!
//! for id in orders.into_iter() {
//!     // If this is the first time we've seen this customer, initialize them
//!     // with no blood alcohol. Otherwise, just retrieve them.
//!     let person = match blood_alcohol.entry(id) {
//!         Vacant(entry) => entry.set(Person{id: id, blood_alcohol: 0.0}),
//!         Occupied(entry) => entry.into_mut(),
//!     };
//!
//!     // Reduce their blood alcohol level. It takes time to order and drink a beer!
//!     person.blood_alcohol *= 0.9;
//!
//!     // Check if they're sober enough to have another beer.
//!     if person.blood_alcohol > 0.3 {
//!         // Too drunk... for now.
//!         println!("Sorry {}, I have to cut you off", person.id);
//!     } else {
//!         // Have another!
//!         person.blood_alcohol += 0.1;
//!     }
//! }
//! ```

#![experimental]

pub use core_collections::{Collection, Mutable, Map, MutableMap};
pub use core_collections::{Set, MutableSet, Deque, MutableSeq};
pub use core_collections::{Bitv, BitvSet, BTreeMap, BTreeSet, DList, EnumSet};
pub use core_collections::{PriorityQueue, RingBuf, SmallIntMap};
pub use core_collections::{TreeMap, TreeSet, TrieMap, TrieSet};
pub use core_collections::{bitv, btree, dlist, enum_set};
pub use core_collections::{priority_queue, ringbuf, smallintmap, treemap, trie};

pub use self::hashmap::{HashMap, HashSet};
pub use self::lru_cache::LruCache;

pub mod hashmap;
pub mod lru_cache;
