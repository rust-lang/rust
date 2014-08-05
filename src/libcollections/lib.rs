// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Rust's standard collections library provides several structures for organizing
//! and querying data. Choosing the right collection for the right job is a non-
//! trivial and important part of writing any good program. While Rust strives to
//! provide efficient and easy to use collections for common use-cases, given *only*
//! a list of the functions a collection provides it can be difficult to determine
//! the best choice. When in doubt, running tests on your actual code with your
//! actual data will always be the best way to identify the best collection for the
//! job. However, in practice this can be time-consuming or otherwise impractical to
//! do. As such, we strive to provide quality documentation on the absolute and
//! relative strengths and weaknesses of each collection.
//!
//! When in doubt, we recommend first considering [`Vec`](../vec/struct.Vec.html),
//! [`RingBuf`](struct.RingBuf.html), [`HashMap`](hashmap/struct.HashMap.html), and
//! [`HashSet`](hashmap/struct.HashSet.html) for the task, as their performance is
//! excellent both in theoretical and practical terms.
//! These collections are easily the most commonly used ones by
//! imperative programmers, and can often be acceptable even when they aren't the
//! *best* choice. Other collections fill important but potentially subtle niches,
//! and the importance of knowing when they are more or less appropriate cannot be
//! understated.
//!
//! # Measuring Performance
//!
//! The performance of a collection is a difficult thing to precisely capture. One
//! cannot simply perform an operation and measure how long it takes or how much
//! space is used, as the results will depend on details such as how it was
//! compiled, the hardware it's running on, the software managing its execution, and
//! the current state of the program. These precise details are independent of the
//! collection's implementation itself, and are far too diverse to exhaustively test
//! against.
//!
//! To avoid this issue, we use *asymptotic analysis* to measure how performance
//! *scales* with the size of the input (commonly denoted by `n`), or other more
//! exotic properties. This is an excellent first-order approximation of
//! performance, but has some drawbacks that we discuss below.
//!
//! ## Big-Oh Notation
//!
//! The most common tool in performing asymptotic analysis is *Big-Oh notation*.
//! Big-Oh notation is a way of expressing the relation between the growth rate of
//! two functions. Given two functions `f` and `g`, when we say `f(x)` *is*
//! `O(g(x))`, you can generally read this as "`f(x)` is *on the order of* `g(x)`".
//! Informally, we say `f(x)` is `O(g(x))` if `g(x)` grows at least as fast as
//! `f(x)`.  In effect, `g(x)` is an upper-bound on how `f(x)` scales with `x`. This
//! scaling ignores constant factors, so `2x` is `O(x)`, even though `2x` grows
//! faster.
//!
//! This ignoring of constants is exactly what we want when discussing the
//! performance of collections, because the the precise compilation and execution
//! details will generally only provide constant-factor speed ups. *In practice*,
//! these constant factors can be large and important, and should be part of the
//! collection selection process. However, Big-Oh notation provides a useful way to
//! quickly identify what a collection does well, and what a collection does poorly,
//! particularly in comparison to other collections. Note also that Big-Oh notation
//! is only interested in the *asymptotic* performance of the functions. For small
//! values of `x` the relationship between these two functions may be arbitrary.
//!
//! While the functions in Big-Oh notation can have arbitrary complexity, by
//! convention the function `g(x)` in `O(g(x))` should be written as simply as
//! possible, and is expected to be as tight as possible. For instance, `2x` is
//! `O(3x^2 + 5x - 2)`, but we would generally simplify the expression to only the
//! dominant factor, with constants stripped away. In this case, `x^2` grows the
//! fastest, and so we would simply say `2x` is `O(x^2)`. Similarly, although `2x`
//! *is* `O(x^2)`, this is needlessly weak. We would instead prefer to provide the
//! stronger bound `O(x)`.
//!
//! Several functions occur very often in Big-Oh notation, and so we note them here
//! for convenience:
//!
//! * `O(1)` - *Constant*: The performance of the operation is effectively
//! independent of context. This is usually *very* cheap.
//!
//! * `O(logn)` - *Logarithmic*: Performance scales with the logarithm of `n`.
//! This is usually cheap.
//!
//! * `O(n)` - *Linear*: Performance scales proportionally to `n`.
//! This is considered expensive, but tractable.
//!
//! * `O(nlogn)`: Performance scales a bit worse than linear.
//! Not to be done frequently if possible.
//!
//! * `O(n^2)` - *Quadratic*: Performance scales with the square of `n`.
//! This is considered very expensive, and is potentially catastrophic for large inputs.
//!
//! * `O(2^n)` - *Exponential*: Performance scales exponentially with `n`.
//! This is considered intractable for anything but very small inputs.
//!
//! ## Time Complexity
//!
//! The most common measure of performance is how long something takes. However,
//! even at the abstraction level of Big-Oh notation, this is not necessarily
//! straight forward. Time complexity is separated into several different
//! categories, to capture important distinctions. In the simplest case, an
//! operation *always* takes `O(g(x))` time to execute. However, we may also be
//! interested in the following measures of time:
//!
//! ### Worst-Case Time
//!
//! The amount of time an operation may take can vary greatly from input to input.
//! However, it is often possible to determine how much time is taken *in the worst-
//! case*. For some operations, the worst-case may be rare and very large. For other
//! operations, it may be the most common, with rare "fast" events.
//!
//! For instance, if an operation sometimes takes `O(1)`, `O(logn)`, or `O(n)` time,
//! we simply say it takes `O(n)` worst-case time, since `O(n)` is the largest.
//!
//! Worst-case analysis is often the easiest to perform, and is always applicable to
//! any operation. As such, it is the standard default measure of time complexity.
//! If time complexity is not qualified, it can be assumed to be worst-case. Having
//! a good worst-case time complexity is the most desirable, as it provides a strong
//! guarantee of reliable performance. However, sometimes the most efficient
//! operations in practice have poor worst-case times, due to rare degenerate
//! behaviors.
//!
//! Vec's push operation usually takes `O(1)` time, but occasionally takes `O(n)` time,
//! and so takes `O(n)` worst-case time.
//!
//! ### Expected Time
//!
//! The running time of some operations may depend on a random or pseudo-random
//! process. In this case, expected time is used to capture how long the operation
//! takes *on average*. The operation may take much more or less time on any given
//! input, or even on different calls on the same input.
//!
//! For instance, if an operation takes `O(nlogn)` time *with high probability*, but
//! very rarely takes `O(n^2)` time, then the operation takes `O(nlogn)` expected
//! time, even though it has a worst-case time of `O(n^2)`. `QuickSort` is the
//! canonical randomized operation, with exactly this performance analysis.
//!
//! ### Amortized Time
//!
//! Some operations can have a very high worst-case cost, but over a *sequence* of
//! `m` operations the total cost can sometimes be guaranteed to not exceed some
//! smaller bound than `m` times the worst-case.
//!
//! For instance, `Vec`'s push operation almost always takes `O(1)` time, but after
//! (approximately) `n`operations, a single push may take `O(n)` time. By worst-case
//! analysis, all we can say of this situation is that a sequence of `m` pushes will
//! take `O(mn)` time. However, in reality we know that the sequence will only take
//! `O(m)` time, since the expensive `O(n)` operation can be *amortized* across the
//! many cheap operations that are *guaranteed* to occur before an expensive
//! operation. Therefore, we say that `Vec.push()` takes `O(1)` amortized time.
//!
//! ## Space Complexity
//!
//! Space complexity is less commonly discussed, but still an important
//! consideration. It can be used to measure either how much space a structure
//! consumes with respect to its contents, or how much additional space an operation
//! temporarily uses. Generally, a fast operation cannot use much space, because
//! time complexity is bounded below by space complexity. That is, it takes `O(n)`
//! time to even *allocate* `O(n)` memory, let alone use it productively.
//!
//! However, space consumption can be important in resource constrained
//! environments, or just when working on large datasets. An operation that takes
//! `O(n^2)` time on a large data set might be unfortunate, but consuming `O(n^2)`
//! extra space to do it, even if only temporary, might prove catastrophic. If the
//! extra space consumed is greater than `O(1)`, it is also likely allocated on the
//! heap, which is generally an expensive operation. Knowing this can help give
//! context to otherwise abstract time complexities.
//!
//! Like time complexity, space complexity can be expressed in worst-case, expected,
//! or amortized terms. Unless otherwise stated, an operation can be assumed to use
//! only `O(1)` additional worst-case space, and a structure containing `n` items
//! can be assumed to have worst-case size `O(n)`.
//!
//! ## Problems with Big-Oh Notation
//!
//! Big-Oh notation is great for broad-strokes analysis of collections and
//! operations, but it can sometimes be misleading in practice.
//!
//! For instance, from a pure asymptotic analysis perspective, `RingBuf` appears to
//! be a strictly superior collection to `Vec`. `RingBuf` supports every operation
//! that `Vec` does in the "same" amount of time, while improving the performance of
//! some operations. However, in practice `Vec` will outperform `RingBuf` on many of
//! the operations they appear to be equally good at. This is because `RingBuf`
//! takes a small constant performance penalty to speed up its other operations.
//! This penalty is not reflected in asymptotic analysis, precisely *because* it is
//! a constant.
//!
//! Similarly, [`DList`](struct.DList.html) appears to be better than `Vec`
//! at many operations, and even
//! provides strong *worst-case* guarantees on operations like `push`, where `Vec`
//! only provides strong *amortized* guarantees. However, in practice `Vec` is
//! expected to *substantially* outperform DList over any large sequence of
//! `push`es.
//!
//! Worse yet, it can sometimes be the case that for all practically sized inputs,
//! an operation that appears to be asymptotically slower than another may be faster
//! in practice, because the "hidden" constant of the theoretically fast operation
//! can be catastrophically large.
//!
//! For these reasons, we will generally strive to discuss practical performance
//! considerations *in addition to* providing the much more convenient and simple
//! asymptotic bounds for high level comparisons. If an operation on a collection
//! does not provide any asymptotic performance information, it should be considered
//! a bug.

#![crate_name = "collections"]
#![experimental]
#![crate_type = "rlib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/master/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(macro_rules, managed_boxes, default_type_params, phase, globs)]
#![feature(unsafe_destructor)]
#![no_std]

#[phase(plugin, link)] extern crate core;
extern crate unicode;
extern crate alloc;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;
#[cfg(test)] extern crate debug;

#[cfg(test)] #[phase(plugin, link)] extern crate std;
#[cfg(test)] #[phase(plugin, link)] extern crate log;

use core::prelude::*;

pub use core::collections::Collection;
pub use bitv::{Bitv, BitvSet};
pub use btree::BTree;
pub use dlist::DList;
pub use enum_set::EnumSet;
pub use priority_queue::PriorityQueue;
pub use ringbuf::RingBuf;
pub use smallintmap::SmallIntMap;
pub use string::String;
pub use treemap::{TreeMap, TreeSet};
pub use trie::{TrieMap, TrieSet};
pub use vec::Vec;

mod macros;

pub mod bitv;
pub mod btree;
pub mod dlist;
pub mod enum_set;
pub mod priority_queue;
pub mod ringbuf;
pub mod smallintmap;
pub mod treemap;
pub mod trie;
pub mod slice;
pub mod str;
pub mod string;
pub mod vec;
pub mod hash;

mod deque;

/// A trait to represent mutable containers
pub trait Mutable: Collection {
    /// Clear the container, removing all values.
    ///
    /// # Example
    ///
    /// ```
    /// let mut v = vec![1i, 2, 3];
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    fn clear(&mut self);
}

/// A map is a key-value store where values may be looked up by their keys. This
/// trait provides basic operations to operate on these stores.
pub trait Map<K, V>: Collection {
    /// Return a reference to the value corresponding to the key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.find(&"a"), Some(&1i));
    /// assert_eq!(map.find(&"b"), None);
    /// ```
    fn find<'a>(&'a self, key: &K) -> Option<&'a V>;

    /// Return true if the map contains a value for the specified key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.contains_key(&"a"), true);
    /// assert_eq!(map.contains_key(&"b"), false);
    /// ```
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }
}

/// This trait provides basic operations to modify the contents of a map.
pub trait MutableMap<K, V>: Map<K, V> + Mutable {
    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.insert("key", 2i), true);
    /// assert_eq!(map.insert("key", 9i), false);
    /// assert_eq!(map.get(&"key"), &9i);
    /// ```
    #[inline]
    fn insert(&mut self, key: K, value: V) -> bool {
        self.swap(key, value).is_none()
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.remove(&"key"), false);
    /// map.insert("key", 2i);
    /// assert_eq!(map.remove(&"key"), true);
    /// ```
    #[inline]
    fn remove(&mut self, key: &K) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.swap("a", 37i), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert("a", 1i);
    /// assert_eq!(map.swap("a", 37i), Some(1i));
    /// assert_eq!(map.get(&"a"), &37i);
    /// ```
    fn swap(&mut self, k: K, v: V) -> Option<V>;

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map: HashMap<&str, int> = HashMap::new();
    /// map.insert("a", 1i);
    /// assert_eq!(map.pop(&"a"), Some(1i));
    /// assert_eq!(map.pop(&"a"), None);
    /// ```
    fn pop(&mut self, k: &K) -> Option<V>;

    /// Return a mutable reference to the value corresponding to the key.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.insert("a", 1i);
    /// match map.find_mut(&"a") {
    ///     Some(x) => *x = 7i,
    ///     None => (),
    /// }
    /// assert_eq!(map.get(&"a"), &7i);
    /// ```
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>;
}

/// A set is a group of objects which are each distinct from one another. This
/// trait represents actions which can be performed on sets to iterate over
/// them.
pub trait Set<T>: Collection {
    /// Return true if the set contains a value.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let set: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// assert_eq!(set.contains(&1), true);
    /// assert_eq!(set.contains(&4), false);
    /// ```
    fn contains(&self, value: &T) -> bool;

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let a: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let mut b: HashSet<int> = HashSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    fn is_disjoint(&self, other: &Self) -> bool;

    /// Return true if the set is a subset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sup: HashSet<int> = [1i, 2, 3].iter().map(|&x| x).collect();
    /// let mut set: HashSet<int> = HashSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    fn is_subset(&self, other: &Self) -> bool;

    /// Return true if the set is a superset of another.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let sub: HashSet<int> = [1i, 2].iter().map(|&x| x).collect();
    /// let mut set: HashSet<int> = HashSet::new();
    ///
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sub), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sub), true);
    /// ```
    fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    // FIXME #8154: Add difference, sym. difference, intersection and union iterators
}

/// This trait represents actions which can be performed on sets to mutate
/// them.
pub trait MutableSet<T>: Set<T> + Mutable {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    ///
    /// assert_eq!(set.insert(2i), true);
    /// assert_eq!(set.insert(2i), false);
    /// assert_eq!(set.len(), 1);
    /// ```
    fn insert(&mut self, value: T) -> bool;

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// let mut set = HashSet::new();
    ///
    /// set.insert(2i);
    /// assert_eq!(set.remove(&2), true);
    /// assert_eq!(set.remove(&2), false);
    /// ```
    fn remove(&mut self, value: &T) -> bool;
}

pub trait MutableSeq<T>: Mutable {
    /// Append an element to the back of a collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1i, 2);
    /// vec.push(3);
    /// assert_eq!(vec, vec!(1, 2, 3));
    /// ```
    fn push(&mut self, t: T);
    /// Remove the last element from a collection and return it, or `None` if it is
    /// empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut vec = vec!(1i, 2, 3);
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, vec!(1, 2));
    /// ```
    fn pop(&mut self) -> Option<T>;
}

/// A double-ended sequence that allows querying, insertion and deletion at both
/// ends.
///
/// # Example
///
/// With a `Deque` we can simulate a queue efficiently:
///
/// ```
/// use std::collections::{RingBuf, Deque};
///
/// let mut queue = RingBuf::new();
/// queue.push(1i);
/// queue.push(2i);
/// queue.push(3i);
///
/// // Will print 1, 2, 3
/// while !queue.is_empty() {
///     let x = queue.pop_front().unwrap();
///     println!("{}", x);
/// }
/// ```
///
/// We can also simulate a stack:
///
/// ```
/// use std::collections::{RingBuf, Deque};
///
/// let mut stack = RingBuf::new();
/// stack.push_front(1i);
/// stack.push_front(2i);
/// stack.push_front(3i);
///
/// // Will print 3, 2, 1
/// while !stack.is_empty() {
///     let x = stack.pop_front().unwrap();
///     println!("{}", x);
/// }
/// ```
///
/// And of course we can mix and match:
///
/// ```
/// use std::collections::{DList, Deque};
///
/// let mut deque = DList::new();
///
/// // Init deque with 1, 2, 3, 4
/// deque.push_front(2i);
/// deque.push_front(1i);
/// deque.push(3i);
/// deque.push(4i);
///
/// // Will print (1, 4) and (2, 3)
/// while !deque.is_empty() {
///     let f = deque.pop_front().unwrap();
///     let b = deque.pop().unwrap();
///     println!("{}", (f, b));
/// }
/// ```
pub trait Deque<T> : MutableSeq<T> {
    /// Provide a reference to the front element, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.front(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// assert_eq!(d.front(), Some(&1i));
    /// ```
    fn front<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the front element, or `None` if the
    /// sequence is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// assert_eq!(d.front_mut(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// match d.front_mut() {
    ///     Some(x) => *x = 9i,
    ///     None => (),
    /// }
    /// assert_eq!(d.front(), Some(&9i));
    /// ```
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Provide a reference to the back element, or `None` if the sequence is
    /// empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// assert_eq!(d.back(), Some(&2i));
    /// ```
    fn back<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the back element, or `None` if the sequence
    /// is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// assert_eq!(d.back(), None);
    ///
    /// d.push(1i);
    /// d.push(2i);
    /// match d.back_mut() {
    ///     Some(x) => *x = 9i,
    ///     None => (),
    /// }
    /// assert_eq!(d.back(), Some(&9i));
    /// ```
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Insert an element first in the sequence.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// d.push_front(1i);
    /// d.push_front(2i);
    /// assert_eq!(d.front(), Some(&2i));
    /// ```
    fn push_front(&mut self, elt: T);

    /// Insert an element last in the sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::{DList, Deque};
    ///
    /// let mut d = DList::new();
    /// d.push_back(1i);
    /// d.push_back(2i);
    /// assert_eq!(d.front(), Some(&1i));
    /// ```
    #[deprecated = "use the `push` method"]
    fn push_back(&mut self, elt: T) { self.push(elt) }

    /// Remove the last element and return it, or `None` if the sequence is empty.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// d.push_back(1i);
    /// d.push_back(2i);
    ///
    /// assert_eq!(d.pop_back(), Some(2i));
    /// assert_eq!(d.pop_back(), Some(1i));
    /// assert_eq!(d.pop_back(), None);
    /// ```
    #[deprecated = "use the `pop` method"]
    fn pop_back(&mut self) -> Option<T> { self.pop() }

    /// Remove the first element and return it, or `None` if the sequence is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::{RingBuf, Deque};
    ///
    /// let mut d = RingBuf::new();
    /// d.push(1i);
    /// d.push(2i);
    ///
    /// assert_eq!(d.pop_front(), Some(1i));
    /// assert_eq!(d.pop_front(), Some(2i));
    /// assert_eq!(d.pop_front(), None);
    /// ```
    fn pop_front(&mut self) -> Option<T>;
}

// FIXME(#14344) this shouldn't be necessary
#[doc(hidden)]
pub fn fixme_14344_be_sure_to_link_to_collections() {}

#[cfg(not(test))]
mod std {
    pub use core::fmt;      // necessary for fail!()
    pub use core::option;   // necessary for fail!()
    pub use core::clone;    // deriving(Clone)
    pub use core::cmp;      // deriving(Eq, Ord, etc.)
    pub use hash;           // deriving(Hash)

    pub mod collections {
        pub use MutableSeq;
    }
}
