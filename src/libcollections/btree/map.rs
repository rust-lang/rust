// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This implementation is largely based on the high-level description and analysis of B-Trees
// found in *Open Data Structures* (ODS). Although our implementation does not use any of
// the source found in ODS, if one wishes to review the high-level design of this structure, it
// can be freely downloaded at http://opendatastructures.org/. Its contents are as of this
// writing (August 2014) freely licensed under the following Creative Commons Attribution
// License: [CC BY 2.5 CA](http://creativecommons.org/licenses/by/2.5/ca/).

use self::Entry::*;

use core::prelude::*;

use core::cmp::Ordering;
use core::default::Default;
use core::fmt::Debug;
use core::hash::{Hash, Hasher};
use core::iter::{Map, FromIterator, IntoIterator};
use core::ops::{Index, IndexMut};
use core::{iter, fmt, mem};
use Bound::{self, Included, Excluded, Unbounded};

use borrow::Borrow;
use vec_deque::VecDeque;

use self::Continuation::{Continue, Finished};
use self::StackOp::*;
use super::node::ForceResult::{Leaf, Internal};
use super::node::TraversalItem::{self, Elem, Edge};
use super::node::{Traversal, MutTraversal, MoveTraversal};
use super::node::{self, Node, Found, GoDown};

/// A map based on a B-Tree.
///
/// B-Trees represent a fundamental compromise between cache-efficiency and actually minimizing
/// the amount of work performed in a search. In theory, a binary search tree (BST) is the optimal
/// choice for a sorted map, as a perfectly balanced BST performs the theoretical minimum amount of
/// comparisons necessary to find an element (log<sub>2</sub>n). However, in practice the way this
/// is done is *very* inefficient for modern computer architectures. In particular, every element
/// is stored in its own individually heap-allocated node. This means that every single insertion
/// triggers a heap-allocation, and every single comparison should be a cache-miss. Since these
/// are both notably expensive things to do in practice, we are forced to at very least reconsider
/// the BST strategy.
///
/// A B-Tree instead makes each node contain B-1 to 2B-1 elements in a contiguous array. By doing
/// this, we reduce the number of allocations by a factor of B, and improve cache efficiency in
/// searches. However, this does mean that searches will have to do *more* comparisons on average.
/// The precise number of comparisons depends on the node search strategy used. For optimal cache
/// efficiency, one could search the nodes linearly. For optimal comparisons, one could search
/// the node using binary search. As a compromise, one could also perform a linear search
/// that initially only checks every i<sup>th</sup> element for some choice of i.
///
/// Currently, our implementation simply performs naive linear search. This provides excellent
/// performance on *small* nodes of elements which are cheap to compare. However in the future we
/// would like to further explore choosing the optimal search strategy based on the choice of B,
/// and possibly other factors. Using linear search, searching for a random element is expected
/// to take O(B log<sub>B</sub>n) comparisons, which is generally worse than a BST. In practice,
/// however, performance is excellent.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BTreeMap<K, V> {
    root: Node<K, V>,
    length: usize,
    depth: usize,
    b: usize,
}

/// An abstract base over-which all other BTree iterators are built.
struct AbsIter<T> {
    traversals: VecDeque<T>,
    size: usize,
}

/// An iterator over a BTreeMap's entries.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, K: 'a, V: 'a> {
    inner: AbsIter<Traversal<'a, K, V>>
}

/// A mutable iterator over a BTreeMap's entries.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, K: 'a, V: 'a> {
    inner: AbsIter<MutTraversal<'a, K, V>>
}

/// An owning iterator over a BTreeMap's entries.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<K, V> {
    inner: AbsIter<MoveTraversal<K, V>>
}

/// An iterator over a BTreeMap's keys.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Keys<'a, K: 'a, V: 'a> {
    inner: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>
}

/// An iterator over a BTreeMap's values.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Values<'a, K: 'a, V: 'a> {
    inner: Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>
}

/// An iterator over a sub-range of BTreeMap's entries.
pub struct Range<'a, K: 'a, V: 'a> {
    inner: AbsIter<Traversal<'a, K, V>>
}

/// A mutable iterator over a sub-range of BTreeMap's entries.
pub struct RangeMut<'a, K: 'a, V: 'a> {
    inner: AbsIter<MutTraversal<'a, K, V>>
}

/// A view into a single entry in a map, which may either be vacant or occupied.
#[unstable(feature = "collections",
           reason = "precise API still under development")]
pub enum Entry<'a, K:'a, V:'a> {
    /// A vacant Entry
    Vacant(VacantEntry<'a, K, V>),
    /// An occupied Entry
    Occupied(OccupiedEntry<'a, K, V>),
}

/// A vacant Entry.
#[unstable(feature = "collections",
           reason = "precise API still under development")]
pub struct VacantEntry<'a, K:'a, V:'a> {
    key: K,
    stack: stack::SearchStack<'a, K, V, node::handle::Edge, node::handle::Leaf>,
}

/// An occupied Entry.
#[unstable(feature = "collections",
           reason = "precise API still under development")]
pub struct OccupiedEntry<'a, K:'a, V:'a> {
    stack: stack::SearchStack<'a, K, V, node::handle::KV, node::handle::LeafOrInternal>,
}

impl<K: Ord, V> BTreeMap<K, V> {
    /// Makes a new empty BTreeMap with a reasonable choice for B.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> BTreeMap<K, V> {
        //FIXME(Gankro): Tune this as a function of size_of<K/V>?
        BTreeMap::with_b(6)
    }

    /// Makes a new empty BTreeMap with the given B.
    ///
    /// B cannot be less than 2.
    pub fn with_b(b: usize) -> BTreeMap<K, V> {
        assert!(b > 1, "B must be greater than 1");
        BTreeMap {
            length: 0,
            depth: 1,
            root: Node::make_leaf_root(b),
            b: b,
        }
    }

    /// Clears the map, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        let b = self.b;
        // avoid recursive destructors by manually traversing the tree
        for _ in mem::replace(self, BTreeMap::with_b(b)) {};
    }

    // Searching in a B-Tree is pretty straightforward.
    //
    // Start at the root. Try to find the key in the current node. If we find it, return it.
    // If it's not in there, follow the edge *before* the smallest key larger than
    // the search key. If no such key exists (they're *all* smaller), then just take the last
    // edge in the node. If we're in a leaf and we don't find our key, then it's not
    // in the tree.

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V> where K: Borrow<Q>, Q: Ord {
        let mut cur_node = &self.root;
        loop {
            match Node::search(cur_node, key) {
                Found(handle) => return Some(handle.into_kv().1),
                GoDown(handle) => match handle.force() {
                    Leaf(_) => return None,
                    Internal(internal_handle) => {
                        cur_node = internal_handle.into_edge();
                        continue;
                    }
                }
            }
        }
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool where K: Borrow<Q>, Q: Ord {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[1], "b");
    /// ```
    // See `get` for implementation notes, this is basically a copy-paste with mut's added
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V> where K: Borrow<Q>, Q: Ord {
        // temp_node is a Borrowck hack for having a mutable value outlive a loop iteration
        let mut temp_node = &mut self.root;
        loop {
            let cur_node = temp_node;
            match Node::search(cur_node, key) {
                Found(handle) => return Some(handle.into_kv_mut().1),
                GoDown(handle) => match handle.force() {
                    Leaf(_) => return None,
                    Internal(internal_handle) => {
                        temp_node = internal_handle.into_edge_mut();
                        continue;
                    }
                }
            }
        }
    }

    // Insertion in a B-Tree is a bit complicated.
    //
    // First we do the same kind of search described in `find`. But we need to maintain a stack of
    // all the nodes/edges in our search path. If we find a match for the key we're trying to
    // insert, just swap the vals and return the old ones. However, when we bottom out in a leaf,
    // we attempt to insert our key-value pair at the same location we would want to follow another
    // edge.
    //
    // If the node has room, then this is done in the obvious way by shifting elements. However,
    // if the node itself is full, we split node into two, and give its median key-value
    // pair to its parent to insert the new node with. Of course, the parent may also be
    // full, and insertion can propagate until we reach the root. If we reach the root, and
    // it is *also* full, then we split the root and place the two nodes under a newly made root.
    //
    // Note that we subtly deviate from Open Data Structures in our implementation of split.
    // ODS describes inserting into the node *regardless* of its capacity, and then
    // splitting *afterwards* if it happens to be overfull. However, this is inefficient.
    // Instead, we split beforehand, and then insert the key-value pair into the appropriate
    // result node. This has two consequences:
    //
    // 1) While ODS produces a left node of size B-1, and a right node of size B,
    // we may potentially reverse this. However, this shouldn't effect the analysis.
    //
    // 2) While ODS may potentially return the pair we *just* inserted after
    // the split, we will never do this. Again, this shouldn't effect the analysis.

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[37], "c");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, mut key: K, mut value: V) -> Option<V> {
        // This is a stack of rawptrs to nodes paired with indices, respectively
        // representing the nodes and edges of our search path. We have to store rawptrs
        // because as far as Rust is concerned, we can mutate aliased data with such a
        // stack. It is of course correct, but what it doesn't know is that we will only
        // be popping and using these ptrs one at a time in child-to-parent order. The alternative
        // to doing this is to take the Nodes from their parents. This actually makes
        // borrowck *really* happy and everything is pretty smooth. However, this creates
        // *tons* of pointless writes, and requires us to always walk all the way back to
        // the root after an insertion, even if we only needed to change a leaf. Therefore,
        // we accept this potential unsafety and complexity in the name of performance.
        //
        // Regardless, the actual dangerous logic is completely abstracted away from BTreeMap
        // by the stack module. All it can do is immutably read nodes, and ask the search stack
        // to proceed down some edge by index. This makes the search logic we'll be reusing in a
        // few different methods much neater, and of course drastically improves safety.
        let mut stack = stack::PartialSearchStack::new(self);

        loop {
            let result = stack.with(move |pusher, node| {
                // Same basic logic as found in `find`, but with PartialSearchStack mediating the
                // actual nodes for us
                return match Node::search(node, &key) {
                    Found(mut handle) => {
                        // Perfect match, swap the values and return the old one
                        mem::swap(handle.val_mut(), &mut value);
                        Finished(Some(value))
                    },
                    GoDown(handle) => {
                        // We need to keep searching, try to get the search stack
                        // to go down further
                        match handle.force() {
                            Leaf(leaf_handle) => {
                                // We've reached a leaf, perform the insertion here
                                pusher.seal(leaf_handle).insert(key, value);
                                Finished(None)
                            }
                            Internal(internal_handle) => {
                                // We've found the subtree to insert this key/value pair in,
                                // keep searching
                                Continue((pusher.push(internal_handle), key, value))
                            }
                        }
                    }
                }
            });
            match result {
                Finished(ret) => { return ret; },
                Continue((new_stack, renewed_key, renewed_val)) => {
                    stack = new_stack;
                    key = renewed_key;
                    value = renewed_val;
                }
            }
        }
    }

    // Deletion is the most complicated operation for a B-Tree.
    //
    // First we do the same kind of search described in
    // `find`. But we need to maintain a stack of all the nodes/edges in our search path.
    // If we don't find the key, then we just return `None` and do nothing. If we do find the
    // key, we perform two operations: remove the item, and then possibly handle underflow.
    //
    // # removing the item
    //      If the node is a leaf, we just remove the item, and shift
    //      any items after it back to fill the hole.
    //
    //      If the node is an internal node, we *swap* the item with the smallest item in
    //      in its right subtree (which must reside in a leaf), and then revert to the leaf
    //      case
    //
    // # handling underflow
    //      After removing an item, there may be too few items in the node. We want nodes
    //      to be mostly full for efficiency, although we make an exception for the root, which
    //      may have as few as one item. If this is the case, we may first try to steal
    //      an item from our left or right neighbour.
    //
    //      To steal from the left (right) neighbour,
    //      we take the largest (smallest) item and child from it. We then swap the taken item
    //      with the item in their mutual parent that separates them, and then insert the
    //      parent's item and the taken child into the first (last) index of the underflowed node.
    //
    //      However, stealing has the possibility of underflowing our neighbour. If this is the
    //      case, we instead *merge* with our neighbour. This of course reduces the number of
    //      children in the parent. Therefore, we also steal the item that separates the now
    //      merged nodes, and insert it into the merged node.
    //
    //      Merging may cause the parent to underflow. If this is the case, then we must repeat
    //      the underflow handling process on the parent. If merging merges the last two children
    //      of the root, then we replace the root with the merged node.

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V> where K: Borrow<Q>, Q: Ord {
        // See `swap` for a more thorough description of the stuff going on in here
        let mut stack = stack::PartialSearchStack::new(self);
        loop {
            let result = stack.with(move |pusher, node| {
                return match Node::search(node, key) {
                    Found(handle) => {
                        // Perfect match. Terminate the stack here, and remove the entry
                        Finished(Some(pusher.seal(handle).remove()))
                    },
                    GoDown(handle) => {
                        // We need to keep searching, try to go down the next edge
                        match handle.force() {
                            // We're at a leaf; the key isn't in here
                            Leaf(_) => Finished(None),
                            Internal(internal_handle) => Continue(pusher.push(internal_handle))
                        }
                    }
                }
            });
            match result {
                Finished(ret) => return ret,
                Continue(new_stack) => stack = new_stack
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> IntoIterator for BTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        self.into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> IntoIterator for &'a BTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> IntoIterator for &'a mut BTreeMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(mut self) -> IterMut<'a, K, V> {
        self.iter_mut()
    }
}

/// A helper enum useful for deciding whether to continue a loop since we can't
/// return from a closure
enum Continuation<A, B> {
    Continue(A),
    Finished(B)
}

/// The stack module provides a safe interface for constructing and manipulating a stack of ptrs
/// to nodes. By using this module much better safety guarantees can be made, and more search
/// boilerplate gets cut out.
mod stack {
    use core::prelude::*;
    use core::marker;
    use core::mem;
    use core::ops::{Deref, DerefMut};
    use super::BTreeMap;
    use super::super::node::{self, Node, Fit, Split, Internal, Leaf};
    use super::super::node::handle;
    use vec::Vec;

    struct InvariantLifetime<'id>(
        marker::PhantomData<::core::cell::Cell<&'id ()>>);

    impl<'id> InvariantLifetime<'id> {
        fn new() -> InvariantLifetime<'id> {
            InvariantLifetime(marker::PhantomData)
        }
    }

    /// A generic mutable reference, identical to `&mut` except for the fact that its lifetime
    /// parameter is invariant. This means that wherever an `IdRef` is expected, only an `IdRef`
    /// with the exact requested lifetime can be used. This is in contrast to normal references,
    /// where `&'static` can be used in any function expecting any lifetime reference.
    pub struct IdRef<'id, T: 'id> {
        inner: &'id mut T,
        _marker: InvariantLifetime<'id>,
    }

    impl<'id, T> Deref for IdRef<'id, T> {
        type Target = T;

        fn deref(&self) -> &T {
            &*self.inner
        }
    }

    impl<'id, T> DerefMut for IdRef<'id, T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut *self.inner
        }
    }

    type StackItem<K, V> = node::Handle<*mut Node<K, V>, handle::Edge, handle::Internal>;
    type Stack<K, V> = Vec<StackItem<K, V>>;

    /// A `PartialSearchStack` handles the construction of a search stack.
    pub struct PartialSearchStack<'a, K:'a, V:'a> {
        map: &'a mut BTreeMap<K, V>,
        stack: Stack<K, V>,
        next: *mut Node<K, V>,
    }

    /// A `SearchStack` represents a full path to an element or an edge of interest. It provides
    /// methods depending on the type of what the path points to for removing an element, inserting
    /// a new element, and manipulating to element at the top of the stack.
    pub struct SearchStack<'a, K:'a, V:'a, Type, NodeType> {
        map: &'a mut BTreeMap<K, V>,
        stack: Stack<K, V>,
        top: node::Handle<*mut Node<K, V>, Type, NodeType>,
    }

    /// A `PartialSearchStack` that doesn't hold a a reference to the next node, and is just
    /// just waiting for a `Handle` to that next node to be pushed. See `PartialSearchStack::with`
    /// for more details.
    pub struct Pusher<'id, 'a, K:'a, V:'a> {
        map: &'a mut BTreeMap<K, V>,
        stack: Stack<K, V>,
        _marker: InvariantLifetime<'id>,
    }

    impl<'a, K, V> PartialSearchStack<'a, K, V> {
        /// Creates a new PartialSearchStack from a BTreeMap by initializing the stack with the
        /// root of the tree.
        pub fn new(map: &'a mut BTreeMap<K, V>) -> PartialSearchStack<'a, K, V> {
            let depth = map.depth;

            PartialSearchStack {
                next: &mut map.root as *mut _,
                map: map,
                stack: Vec::with_capacity(depth),
            }
        }

        /// Breaks up the stack into a `Pusher` and the next `Node`, allowing the given closure
        /// to interact with, search, and finally push the `Node` onto the stack. The passed in
        /// closure must be polymorphic on the `'id` lifetime parameter, as this statically
        /// ensures that only `Handle`s from the correct `Node` can be pushed.
        ///
        /// The reason this works is that the `Pusher` has an `'id` parameter, and will only accept
        /// handles with the same `'id`. The closure could only get references with that lifetime
        /// through its arguments or through some other `IdRef` that it has lying around. However,
        /// no other `IdRef` could possibly work - because the `'id` is held in an invariant
        /// parameter, it would need to have precisely the correct lifetime, which would mean that
        /// at least one of the calls to `with` wouldn't be properly polymorphic, wanting a
        /// specific lifetime instead of the one that `with` chooses to give it.
        ///
        /// See also Haskell's `ST` monad, which uses a similar trick.
        pub fn with<T, F: for<'id> FnOnce(Pusher<'id, 'a, K, V>,
                                          IdRef<'id, Node<K, V>>) -> T>(self, closure: F) -> T {
            let pusher = Pusher {
                map: self.map,
                stack: self.stack,
                _marker: InvariantLifetime::new(),
            };
            let node = IdRef {
                inner: unsafe { &mut *self.next },
                _marker: InvariantLifetime::new(),
            };

            closure(pusher, node)
        }
    }

    impl<'id, 'a, K, V> Pusher<'id, 'a, K, V> {
        /// Pushes the requested child of the stack's current top on top of the stack. If the child
        /// exists, then a new PartialSearchStack is yielded. Otherwise, a VacantSearchStack is
        /// yielded.
        pub fn push(mut self, mut edge: node::Handle<IdRef<'id, Node<K, V>>,
                                                     handle::Edge,
                                                     handle::Internal>)
                    -> PartialSearchStack<'a, K, V> {
            self.stack.push(edge.as_raw());
            PartialSearchStack {
                map: self.map,
                stack: self.stack,
                next: edge.edge_mut() as *mut _,
            }
        }

        /// Converts the PartialSearchStack into a SearchStack.
        pub fn seal<Type, NodeType>
                   (self, mut handle: node::Handle<IdRef<'id, Node<K, V>>, Type, NodeType>)
                    -> SearchStack<'a, K, V, Type, NodeType> {
            SearchStack {
                map: self.map,
                stack: self.stack,
                top: handle.as_raw(),
            }
        }
    }

    impl<'a, K, V, NodeType> SearchStack<'a, K, V, handle::KV, NodeType> {
        /// Gets a reference to the value the stack points to.
        pub fn peek(&self) -> &V {
            unsafe { self.top.from_raw().into_kv().1 }
        }

        /// Gets a mutable reference to the value the stack points to.
        pub fn peek_mut(&mut self) -> &mut V {
            unsafe { self.top.from_raw_mut().into_kv_mut().1 }
        }

        /// Converts the stack into a mutable reference to the value it points to, with a lifetime
        /// tied to the original tree.
        pub fn into_top(mut self) -> &'a mut V {
            unsafe {
                mem::copy_mut_lifetime(
                    self.map,
                    self.top.from_raw_mut().val_mut()
                )
            }
        }
    }

    impl<'a, K, V> SearchStack<'a, K, V, handle::KV, handle::Leaf> {
        /// Removes the key and value in the top element of the stack, then handles underflows as
        /// described in BTree's pop function.
        fn remove_leaf(mut self) -> V {
            self.map.length -= 1;

            // Remove the key-value pair from the leaf that this search stack points to.
            // Then, note if the leaf is underfull, and promptly forget the leaf and its ptr
            // to avoid ownership issues.
            let (value, mut underflow) = unsafe {
                let (_, value) = self.top.from_raw_mut().remove_as_leaf();
                let underflow = self.top.from_raw().node().is_underfull();
                (value, underflow)
            };

            loop {
                match self.stack.pop() {
                    None => {
                        // We've reached the root, so no matter what, we're done. We manually
                        // access the root via the tree itself to avoid creating any dangling
                        // pointers.
                        if self.map.root.len() == 0 && !self.map.root.is_leaf() {
                            // We've emptied out the root, so make its only child the new root.
                            // If it's a leaf, we just let it become empty.
                            self.map.depth -= 1;
                            self.map.root.hoist_lone_child();
                        }
                        return value;
                    }
                    Some(mut handle) => {
                        if underflow {
                            // Underflow! Handle it!
                            unsafe {
                                handle.from_raw_mut().handle_underflow();
                                underflow = handle.from_raw().node().is_underfull();
                            }
                        } else {
                            // All done!
                            return value;
                        }
                    }
                }
            }
        }
    }

    impl<'a, K, V> SearchStack<'a, K, V, handle::KV, handle::LeafOrInternal> {
        /// Removes the key and value in the top element of the stack, then handles underflows as
        /// described in BTree's pop function.
        pub fn remove(self) -> V {
            // Ensure that the search stack goes to a leaf. This is necessary to perform deletion
            // in a BTree. Note that this may put the tree in an inconsistent state (further
            // described in into_leaf's comments), but this is immediately fixed by the
            // removing the value we want to remove
            self.into_leaf().remove_leaf()
        }

        /// Subroutine for removal. Takes a search stack for a key that might terminate at an
        /// internal node, and mutates the tree and search stack to *make* it a search stack
        /// for that same key that *does* terminates at a leaf. If the mutation occurs, then this
        /// leaves the tree in an inconsistent state that must be repaired by the caller by
        /// removing the entry in question. Specifically the key-value pair and its successor will
        /// become swapped.
        fn into_leaf(mut self) -> SearchStack<'a, K, V, handle::KV, handle::Leaf> {
            unsafe {
                let mut top_raw = self.top;
                let mut top = top_raw.from_raw_mut();

                let key_ptr = top.key_mut() as *mut _;
                let val_ptr = top.val_mut() as *mut _;

                // Try to go into the right subtree of the found key to find its successor
                match top.force() {
                    Leaf(mut leaf_handle) => {
                        // We're a proper leaf stack, nothing to do
                        return SearchStack {
                            map: self.map,
                            stack: self.stack,
                            top: leaf_handle.as_raw()
                        }
                    }
                    Internal(mut internal_handle) => {
                        let mut right_handle = internal_handle.right_edge();

                        //We're not a proper leaf stack, let's get to work.
                        self.stack.push(right_handle.as_raw());

                        let mut temp_node = right_handle.edge_mut();
                        loop {
                            // Walk into the smallest subtree of this node
                            let node = temp_node;

                            match node.kv_handle(0).force() {
                                Leaf(mut handle) => {
                                    // This node is a leaf, do the swap and return
                                    mem::swap(handle.key_mut(), &mut *key_ptr);
                                    mem::swap(handle.val_mut(), &mut *val_ptr);
                                    return SearchStack {
                                        map: self.map,
                                        stack: self.stack,
                                        top: handle.as_raw()
                                    }
                                },
                                Internal(kv_handle) => {
                                    // This node is internal, go deeper
                                    let mut handle = kv_handle.into_left_edge();
                                    self.stack.push(handle.as_raw());
                                    temp_node = handle.into_edge_mut();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    impl<'a, K, V> SearchStack<'a, K, V, handle::Edge, handle::Leaf> {
        /// Inserts the key and value into the top element in the stack, and if that node has to
        /// split recursively inserts the split contents into the next element stack until
        /// splits stop.
        ///
        /// Assumes that the stack represents a search path from the root to a leaf.
        ///
        /// An &mut V is returned to the inserted value, for callers that want a reference to this.
        pub fn insert(mut self, key: K, val: V) -> &'a mut V {
            unsafe {
                self.map.length += 1;

                // Insert the key and value into the leaf at the top of the stack
                let (mut insertion, inserted_ptr) = self.top.from_raw_mut()
                                                        .insert_as_leaf(key, val);

                loop {
                    match insertion {
                        Fit => {
                            // The last insertion went off without a hitch, no splits! We can stop
                            // inserting now.
                            return &mut *inserted_ptr;
                        }
                        Split(key, val, right) => match self.stack.pop() {
                            // The last insertion triggered a split, so get the next element on the
                            // stack to recursively insert the split node into.
                            None => {
                                // The stack was empty; we've split the root, and need to make a
                                // a new one. This is done in-place because we can't move the
                                // root out of a reference to the tree.
                                Node::make_internal_root(&mut self.map.root, self.map.b,
                                                         key, val, right);

                                self.map.depth += 1;
                                return &mut *inserted_ptr;
                            }
                            Some(mut handle) => {
                                // The stack wasn't empty, do the insertion and recurse
                                insertion = handle.from_raw_mut()
                                                  .insert_as_internal(key, val, right);
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    fn from_iter<T: IntoIterator<Item=(K, V)>>(iter: T) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();
        map.extend(iter);
        map
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V> Extend<(K, V)> for BTreeMap<K, V> {
    #[inline]
    fn extend<T: IntoIterator<Item=(K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

#[cfg(stage0)]
#[stable(feature = "rust1", since = "1.0.0")]
impl<S: Hasher, K: Hash<S>, V: Hash<S>> Hash<S> for BTreeMap<K, V> {
    fn hash(&self, state: &mut S) {
        for elt in self {
            elt.hash(state);
        }
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Hash, V: Hash> Hash for BTreeMap<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elt in self {
            elt.hash(state);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V> Default for BTreeMap<K, V> {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> BTreeMap<K, V> {
        BTreeMap::new()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: PartialEq, V: PartialEq> PartialEq for BTreeMap<K, V> {
    fn eq(&self, other: &BTreeMap<K, V>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Eq, V: Eq> Eq for BTreeMap<K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: PartialOrd, V: PartialOrd> PartialOrd for BTreeMap<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &BTreeMap<K, V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, V: Ord> Ord for BTreeMap<K, V> {
    #[inline]
    fn cmp(&self, other: &BTreeMap<K, V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Debug, V: Debug> Debug for BTreeMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "BTreeMap {{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{:?}: {:?}", *k, *v));
        }

        write!(f, "}}")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, Q: ?Sized, V> Index<Q> for BTreeMap<K, V>
    where K: Borrow<Q>, Q: Ord
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K: Ord, Q: ?Sized, V> IndexMut<Q> for BTreeMap<K, V>
    where K: Borrow<Q>, Q: Ord
{
    fn index_mut(&mut self, key: &Q) -> &mut V {
        self.get_mut(key).expect("no entry found for key")
    }
}

/// Genericises over how to get the correct type of iterator from the correct type
/// of Node ownership.
trait Traverse<N> {
    fn traverse(node: N) -> Self;
}

impl<'a, K, V> Traverse<&'a Node<K, V>> for Traversal<'a, K, V> {
    fn traverse(node: &'a Node<K, V>) -> Traversal<'a, K, V> {
        node.iter()
    }
}

impl<'a, K, V> Traverse<&'a mut Node<K, V>> for MutTraversal<'a, K, V> {
    fn traverse(node: &'a mut Node<K, V>) -> MutTraversal<'a, K, V> {
        node.iter_mut()
    }
}

impl<K, V> Traverse<Node<K, V>> for MoveTraversal<K, V> {
    fn traverse(node: Node<K, V>) -> MoveTraversal<K, V> {
        node.into_iter()
    }
}

/// Represents an operation to perform inside the following iterator methods.
/// This is necessary to use in `next` because we want to modify `self.traversals` inside
/// a match that borrows it. Similarly in `next_back`. Instead, we use this enum to note
/// what we want to do, and do it after the match.
enum StackOp<T> {
    Push(T),
    Pop,
}
impl<K, V, E, T> Iterator for AbsIter<T> where
    T: DoubleEndedIterator<Item=TraversalItem<K, V, E>> + Traverse<E>,
{
    type Item = (K, V);

    // Our iterator represents a queue of all ancestors of elements we have
    // yet to yield, from smallest to largest.  Note that the design of these
    // iterators permits an *arbitrary* initial pair of min and max, making
    // these arbitrary sub-range iterators.
    fn next(&mut self) -> Option<(K, V)> {
        loop {
            // We want the smallest element, so try to get the back of the queue
            let op = match self.traversals.back_mut() {
                None => return None,
                // The queue wasn't empty, so continue along the node in its head
                Some(iter) => match iter.next() {
                    // The head is empty, so Pop it off and continue the process
                    None => Pop,
                    // The head yielded an edge, so make that the new head
                    Some(Edge(next)) => Push(Traverse::traverse(next)),
                    // The head yielded an entry, so yield that
                    Some(Elem(kv)) => {
                        self.size -= 1;
                        return Some(kv)
                    }
                }
            };

            // Handle any operation as necessary, without a conflicting borrow of the queue
            match op {
                Push(item) => { self.traversals.push_back(item); },
                Pop => { self.traversals.pop_back(); },
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<K, V, E, T> DoubleEndedIterator for AbsIter<T> where
    T: DoubleEndedIterator<Item=TraversalItem<K, V, E>> + Traverse<E>,
{
    // next_back is totally symmetric to next
    #[inline]
    fn next_back(&mut self) -> Option<(K, V)> {
        loop {
            let op = match self.traversals.front_mut() {
                None => return None,
                Some(iter) => match iter.next_back() {
                    None => Pop,
                    Some(Edge(next)) => Push(Traverse::traverse(next)),
                    Some(Elem(kv)) => {
                        self.size -= 1;
                        return Some(kv)
                    }
                }
            };

            match op {
                Push(item) => { self.traversals.push_front(item); },
                Pop => { self.traversals.pop_front(); }
            }
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<(K, V)> { self.inner.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<K, V> ExactSizeIterator for IntoIter<K, V> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<(&'a K)> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for Keys<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K)> { self.inner.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}


#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<(&'a V)> { self.inner.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> DoubleEndedIterator for Values<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a V)> { self.inner.next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next() }
}
impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next_back() }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next() }
}
impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next_back() }
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    #[unstable(feature = "collections",
               reason = "matches collection reform v2 specification, waiting for dust to settle")]
    /// Returns a mutable reference to the entry if occupied, or the VacantEntry if vacant
    pub fn get(self) -> Result<&'a mut V, VacantEntry<'a, K, V>> {
        match self {
            Occupied(entry) => Ok(entry.into_mut()),
            Vacant(entry) => Err(entry),
        }
    }
}

impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(self, value: V) -> &'a mut V {
        self.stack.insert(self.key, value)
    }
}

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get(&self) -> &V {
        self.stack.peek()
    }

    /// Gets a mutable reference to the value in the entry.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn get_mut(&mut self) -> &mut V {
        self.stack.peek_mut()
    }

    /// Converts the entry into a mutable reference to its value.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_mut(self) -> &'a mut V {
        self.stack.into_top()
    }

    /// Sets the value of the entry with the OccupiedEntry's key,
    /// and returns the entry's old value.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn insert(&mut self, mut value: V) -> V {
        mem::swap(self.stack.peek_mut(), &mut value);
        value
    }

    /// Takes the value of the entry out of the map, and returns it.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn remove(self) -> V {
        self.stack.remove()
    }
}

impl<K, V> BTreeMap<K, V> {
    /// Gets an iterator over the entries of the map.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    ///
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    ///
    /// let (first_key, first_value) = map.iter().next().unwrap();
    /// assert_eq!((*first_key, *first_value), (1, "a"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<K, V> {
        let len = self.len();
        // NB. The initial capacity for ringbuf is large enough to avoid reallocs in many cases.
        let mut lca = VecDeque::new();
        lca.push_back(Traverse::traverse(&self.root));
        Iter {
            inner: AbsIter {
                traversals: lca,
                size: len,
            }
        }
    }

    /// Gets a mutable iterator over the entries of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // add 10 to the value if the key isn't "a"
    /// for (key, value) in map.iter_mut() {
    ///     if key != &"a" {
    ///         *value += 10;
    ///     }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        let len = self.len();
        let mut lca = VecDeque::new();
        lca.push_back(Traverse::traverse(&mut self.root));
        IterMut {
            inner: AbsIter {
                traversals: lca,
                size: len,
            }
        }
    }

    /// Gets an owning iterator over the entries of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    ///
    /// for (key, value) in map.into_iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_iter(self) -> IntoIter<K, V> {
        let len = self.len();
        let mut lca = VecDeque::new();
        lca.push_back(Traverse::traverse(self.root));
        IntoIter {
            inner: AbsIter {
                traversals: lca,
                size: len,
            }
        }
    }

    /// Gets an iterator over the keys of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let keys: Vec<usize> = a.keys().cloned().collect();
    /// assert_eq!(keys, vec![1,2,]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        fn first<A, B>((a, _): (A, B)) -> A { a }
        let first: fn((&'a K, &'a V)) -> &'a K = first; // coerce to fn pointer

        Keys { inner: self.iter().map(first) }
    }

    /// Gets an iterator over the values of the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// a.insert(1, "a");
    /// a.insert(2, "b");
    ///
    /// let values: Vec<&str> = a.values().cloned().collect();
    /// assert_eq!(values, vec!["a","b"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        fn second<A, B>((_, b): (A, B)) -> B { b }
        let second: fn((&'a K, &'a V)) -> &'a V = second; // coerce to fn pointer

        Values { inner: self.iter().map(second) }
    }

    /// Return the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize { self.length }

    /// Return true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    ///
    /// let mut a = BTreeMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

macro_rules! range_impl {
    ($root:expr, $min:expr, $max:expr, $as_slices_internal:ident, $iter:ident, $Range:ident,
                                       $edges:ident, [$($mutability:ident)*]) => (
        {
            // A deque that encodes two search paths containing (left-to-right):
            // a series of truncated-from-the-left iterators, the LCA's doubly-truncated iterator,
            // and a series of truncated-from-the-right iterators.
            let mut traversals = VecDeque::new();
            let (root, min, max) = ($root, $min, $max);

            let mut leftmost = None;
            let mut rightmost = None;

            match (&min, &max) {
                (&Unbounded, &Unbounded) => {
                    traversals.push_back(Traverse::traverse(root))
                }
                (&Unbounded, &Included(_)) | (&Unbounded, &Excluded(_)) => {
                    rightmost = Some(root);
                }
                (&Included(_), &Unbounded) | (&Excluded(_), &Unbounded) => {
                    leftmost = Some(root);
                }
                  (&Included(min_key), &Included(max_key))
                | (&Included(min_key), &Excluded(max_key))
                | (&Excluded(min_key), &Included(max_key))
                | (&Excluded(min_key), &Excluded(max_key)) => {
                    // lca represents the Lowest Common Ancestor, above which we never
                    // walk, since everything else is outside the range to iterate.
                    //       ___________________
                    //      |__0_|_80_|_85_|_90_|  (root)
                    //      |    |    |    |    |
                    //           |
                    //           v
                    //  ___________________
                    // |__5_|_15_|_30_|_73_|
                    // |    |    |    |    |
                    //                |
                    //                v
                    //       ___________________
                    //      |_33_|_58_|_63_|_68_|  lca for the range [41, 65]
                    //      |    |\___|___/|    |  iterator at traversals[2]
                    //           |         |
                    //           |         v
                    //           v         rightmost
                    //           leftmost
                    let mut is_leaf = root.is_leaf();
                    let mut lca = root.$as_slices_internal();
                    loop {
                        let slice = lca.slice_from(min_key).slice_to(max_key);
                        if let [ref $($mutability)* edge] = slice.edges {
                            // Follow the only edge that leads the node that covers the range.
                            is_leaf = edge.is_leaf();
                            lca = edge.$as_slices_internal();
                        } else {
                            let mut iter = slice.$iter();
                            if is_leaf {
                                leftmost = None;
                                rightmost = None;
                            } else {
                                // Only change the state of nodes with edges.
                                leftmost = iter.next_edge_item();
                                rightmost = iter.next_edge_item_back();
                            }
                            traversals.push_back(iter);
                            break;
                        }
                    }
                }
            }
            // Keep narrowing the range by going down.
            //               ___________________
            //              |_38_|_43_|_48_|_53_|
            //              |    |____|____|____/ iterator at traversals[1]
            //                   |
            //                   v
            //  ___________________
            // |_39_|_40_|_41_|_42_|  (leaf, the last leftmost)
            //           \_________|  iterator at traversals[0]
            match min {
                Included(key) | Excluded(key) =>
                    while let Some(left) = leftmost {
                        let is_leaf = left.is_leaf();
                        let mut iter = left.$as_slices_internal().slice_from(key).$iter();
                        leftmost = if is_leaf {
                            None
                        } else {
                            // Only change the state of nodes with edges.
                            iter.next_edge_item()
                        };
                        traversals.push_back(iter);
                    },
                _ => {}
            }
            // If the leftmost iterator starts with an element, then it was an exact match.
            if let (Excluded(_), Some(leftmost_iter)) = (min, traversals.back_mut()) {
                // Drop this excluded element. `next_kv_item` has no effect when
                // the next item is an edge.
                leftmost_iter.next_kv_item();
            }

            // The code for the right side is similar.
            match max {
                Included(key) | Excluded(key) =>
                    while let Some(right) = rightmost {
                        let is_leaf = right.is_leaf();
                        let mut iter = right.$as_slices_internal().slice_to(key).$iter();
                        rightmost = if is_leaf {
                            None
                        } else {
                            iter.next_edge_item_back()
                        };
                        traversals.push_front(iter);
                    },
                _ => {}
            }
            if let (Excluded(_), Some(rightmost_iter)) = (max, traversals.front_mut()) {
                rightmost_iter.next_kv_item_back();
            }

            $Range {
                inner: AbsIter {
                    traversals: traversals,
                    size: 0, // unused
                }
            }
        }
    )
}

impl<K: Ord, V> BTreeMap<K, V> {
    /// Constructs a double-ended iterator over a sub-range of elements in the map, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use std::collections::Bound::{Included, Unbounded};
    ///
    /// let mut map = BTreeMap::new();
    /// map.insert(3, "a");
    /// map.insert(5, "b");
    /// map.insert(8, "c");
    /// for (&key, &value) in map.range(Included(&4), Included(&8)) {
    ///     println!("{}: {}", key, value);
    /// }
    /// assert_eq!(Some((&5, &"b")), map.range(Included(&4), Unbounded).next());
    /// ```
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn range<'a>(&'a self, min: Bound<&K>, max: Bound<&K>) -> Range<'a, K, V> {
        range_impl!(&self.root, min, max, as_slices_internal, iter, Range, edges, [])
    }

    /// Constructs a mutable double-ended iterator over a sub-range of elements in the map, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use std::collections::Bound::{Included, Excluded};
    ///
    /// let mut map: BTreeMap<&str, i32> = ["Alice", "Bob", "Carol", "Cheryl"].iter()
    ///                                                                       .map(|&s| (s, 0))
    ///                                                                       .collect();
    /// for (_, balance) in map.range_mut(Included(&"B"), Excluded(&"Cheryl")) {
    ///     *balance += 100;
    /// }
    /// for (name, balance) in map.iter() {
    ///     println!("{} => {}", name, balance);
    /// }
    /// ```
    #[unstable(feature = "collections",
               reason = "matches collection reform specification, waiting for dust to settle")]
    pub fn range_mut<'a>(&'a mut self, min: Bound<&K>, max: Bound<&K>) -> RangeMut<'a, K, V> {
        range_impl!(&mut self.root, min, max, as_slices_internal_mut, iter_mut, RangeMut,
                                                                      edges_mut, [mut])
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use std::collections::btree_map::Entry;
    ///
    /// let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    ///
    /// // count the number of occurrences of letters in the vec
    /// for x in vec!["a","b","a","c","a","b"].iter() {
    ///     match count.entry(*x) {
    ///         Entry::Vacant(view) => {
    ///             view.insert(1);
    ///         },
    ///         Entry::Occupied(mut view) => {
    ///             let v = view.get_mut();
    ///             *v += 1;
    ///         },
    ///     }
    /// }
    ///
    /// assert_eq!(count["a"], 3);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn entry(&mut self, mut key: K) -> Entry<K, V> {
        // same basic logic of `swap` and `pop`, blended together
        let mut stack = stack::PartialSearchStack::new(self);
        loop {
            let result = stack.with(move |pusher, node| {
                return match Node::search(node, &key) {
                    Found(handle) => {
                        // Perfect match
                        Finished(Occupied(OccupiedEntry {
                            stack: pusher.seal(handle)
                        }))
                    },
                    GoDown(handle) => {
                        match handle.force() {
                            Leaf(leaf_handle) => {
                                Finished(Vacant(VacantEntry {
                                    stack: pusher.seal(leaf_handle),
                                    key: key,
                                }))
                            },
                            Internal(internal_handle) => {
                                Continue((
                                    pusher.push(internal_handle),
                                    key
                                ))
                            }
                        }
                    }
                }
            });
            match result {
                Finished(finished) => return finished,
                Continue((new_stack, renewed_key)) => {
                    stack = new_stack;
                    key = renewed_key;
                }
            }
        }
    }
}





#[cfg(test)]
mod test {
    use prelude::*;
    use std::iter::range_inclusive;

    use super::BTreeMap;
    use super::Entry::{Occupied, Vacant};
    use Bound::{self, Included, Excluded, Unbounded};

    #[test]
    fn test_basic_large() {
        let mut map = BTreeMap::new();
        let size = 10000;
        assert_eq!(map.len(), 0);

        for i in 0..size {
            assert_eq!(map.insert(i, 10*i), None);
            assert_eq!(map.len(), i + 1);
        }

        for i in 0..size {
            assert_eq!(map.get(&i).unwrap(), &(i*10));
        }

        for i in size..size*2 {
            assert_eq!(map.get(&i), None);
        }

        for i in 0..size {
            assert_eq!(map.insert(i, 100*i), Some(10*i));
            assert_eq!(map.len(), size);
        }

        for i in 0..size {
            assert_eq!(map.get(&i).unwrap(), &(i*100));
        }

        for i in 0..size/2 {
            assert_eq!(map.remove(&(i*2)), Some(i*200));
            assert_eq!(map.len(), size - i - 1);
        }

        for i in 0..size/2 {
            assert_eq!(map.get(&(2*i)), None);
            assert_eq!(map.get(&(2*i+1)).unwrap(), &(i*200 + 100));
        }

        for i in 0..size/2 {
            assert_eq!(map.remove(&(2*i)), None);
            assert_eq!(map.remove(&(2*i+1)), Some(i*200 + 100));
            assert_eq!(map.len(), size/2 - i - 1);
        }
    }

    #[test]
    fn test_basic_small() {
        let mut map = BTreeMap::new();
        assert_eq!(map.remove(&1), None);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.insert(1, 1), None);
        assert_eq!(map.get(&1), Some(&1));
        assert_eq!(map.insert(1, 2), Some(1));
        assert_eq!(map.get(&1), Some(&2));
        assert_eq!(map.insert(2, 4), None);
        assert_eq!(map.get(&2), Some(&4));
        assert_eq!(map.remove(&1), Some(2));
        assert_eq!(map.remove(&2), Some(4));
        assert_eq!(map.remove(&1), None);
    }

    #[test]
    fn test_iter() {
        let size = 10000;

        // Forwards
        let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test<T>(size: usize, mut iter: T) where T: Iterator<Item=(usize, usize)> {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }
        test(size, map.iter().map(|(&k, &v)| (k, v)));
        test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
        test(size, map.into_iter());
    }

    #[test]
    fn test_iter_rev() {
        let size = 10000;

        // Forwards
        let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test<T>(size: usize, mut iter: T) where T: Iterator<Item=(usize, usize)> {
            for i in 0..size {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (size - i - 1, size - i - 1));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }
        test(size, map.iter().rev().map(|(&k, &v)| (k, v)));
        test(size, map.iter_mut().rev().map(|(&k, &mut v)| (k, v)));
        test(size, map.into_iter().rev());
    }

    #[test]
    fn test_iter_mixed() {
        let size = 10000;

        // Forwards
        let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test<T>(size: usize, mut iter: T)
                where T: Iterator<Item=(usize, usize)> + DoubleEndedIterator {
            for i in 0..size / 4 {
                assert_eq!(iter.size_hint(), (size - i * 2, Some(size - i * 2)));
                assert_eq!(iter.next().unwrap(), (i, i));
                assert_eq!(iter.next_back().unwrap(), (size - i - 1, size - i - 1));
            }
            for i in size / 4..size * 3 / 4 {
                assert_eq!(iter.size_hint(), (size * 3 / 4 - i, Some(size * 3 / 4 - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }
        test(size, map.iter().map(|(&k, &v)| (k, v)));
        test(size, map.iter_mut().map(|(&k, &mut v)| (k, v)));
        test(size, map.into_iter());
    }

    #[test]
    fn test_range_small() {
        let size = 5;

        // Forwards
        let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

        let mut j = 0;
        for ((&k, &v), i) in map.range(Included(&2), Unbounded).zip(2..size) {
            assert_eq!(k, i);
            assert_eq!(v, i);
            j += 1;
        }
        assert_eq!(j, size - 2);
    }

    #[test]
    fn test_range_1000() {
        let size = 1000;
        let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

        fn test(map: &BTreeMap<u32, u32>, size: u32, min: Bound<&u32>, max: Bound<&u32>) {
            let mut kvs = map.range(min, max).map(|(&k, &v)| (k, v));
            let mut pairs = (0..size).map(|i| (i, i));

            for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                assert_eq!(kv, pair);
            }
            assert_eq!(kvs.next(), None);
            assert_eq!(pairs.next(), None);
        }
        test(&map, size, Included(&0), Excluded(&size));
        test(&map, size, Unbounded, Excluded(&size));
        test(&map, size, Included(&0), Included(&(size - 1)));
        test(&map, size, Unbounded, Included(&(size - 1)));
        test(&map, size, Included(&0), Unbounded);
        test(&map, size, Unbounded, Unbounded);
    }

    #[test]
    fn test_range() {
        let size = 200;
        let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();

        for i in 0..size {
            for j in i..size {
                let mut kvs = map.range(Included(&i), Included(&j)).map(|(&k, &v)| (k, v));
                let mut pairs = range_inclusive(i, j).map(|i| (i, i));

                for (kv, pair) in kvs.by_ref().zip(pairs.by_ref()) {
                    assert_eq!(kv, pair);
                }
                assert_eq!(kvs.next(), None);
                assert_eq!(pairs.next(), None);
            }
        }
    }

    #[test]
    fn test_entry(){
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: BTreeMap<_, _> = xs.iter().cloned().collect();

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
                *v *= 10;
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
}






#[cfg(test)]
mod bench {
    use prelude::*;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    use super::BTreeMap;

    map_insert_rand_bench!{insert_rand_100,    100,    BTreeMap}
    map_insert_rand_bench!{insert_rand_10_000, 10_000, BTreeMap}

    map_insert_seq_bench!{insert_seq_100,    100,    BTreeMap}
    map_insert_seq_bench!{insert_seq_10_000, 10_000, BTreeMap}

    map_find_rand_bench!{find_rand_100,    100,    BTreeMap}
    map_find_rand_bench!{find_rand_10_000, 10_000, BTreeMap}

    map_find_seq_bench!{find_seq_100,    100,    BTreeMap}
    map_find_seq_bench!{find_seq_10_000, 10_000, BTreeMap}

    fn bench_iter(b: &mut Bencher, size: i32) {
        let mut map = BTreeMap::<i32, i32>::new();
        let mut rng = weak_rng();

        for _ in 0..size {
            map.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            for entry in &map {
                black_box(entry);
            }
        });
    }

    #[bench]
    pub fn iter_20(b: &mut Bencher) {
        bench_iter(b, 20);
    }

    #[bench]
    pub fn iter_1000(b: &mut Bencher) {
        bench_iter(b, 1000);
    }

    #[bench]
    pub fn iter_100000(b: &mut Bencher) {
        bench_iter(b, 100000);
    }
}
