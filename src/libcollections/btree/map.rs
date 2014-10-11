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

use core::prelude::*;

use super::node::*;
use std::hash::{Writer, Hash};
use core::default::Default;
use core::{iter, fmt, mem};
use core::fmt::Show;

use {Deque, Map, MutableMap, Mutable, MutableSeq};
use ringbuf::RingBuf;



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
/// this, we reduce the number of allocations by a factor of B, and improve cache effeciency in
/// searches. However, this does mean that searches will have to do *more* comparisons on average.
/// The precise number of comparisons depends on the node search strategy used. For optimal cache
/// effeciency, one could search the nodes linearly. For optimal comparisons, one could search
/// the node using binary search. As a compromise, one could also perform a linear search
/// that initially only checks every i<sup>th</sup> element for some choice of i.
///
/// Currently, our implementation simply performs naive linear search. This provides excellent
/// performance on *small* nodes of elements which are cheap to compare. However in the future we
/// would like to further explore choosing the optimal search strategy based on the choice of B,
/// and possibly other factors. Using linear search, searching for a random element is expected
/// to take O(B log<sub>B</sub>n) comparisons, which is generally worse than a BST. In practice,
/// however, performance is excellent. `BTreeMap` is able to readily outperform `TreeMap` under
/// many workloads, and is competetive where it doesn't. BTreeMap also generally *scales* better
/// than TreeMap, making it more appropriate for large datasets.
///
/// However, `TreeMap` may still be more appropriate to use in many contexts. If elements are very
/// large or expensive to compare, `TreeMap` may be more appropriate. It won't allocate any
/// more space than is needed, and will perform the minimal number of comparisons necessary.
/// `TreeMap` also provides much better performance stability guarantees. Generally, very few
/// changes need to be made to update a BST, and two updates are expected to take about the same
/// amount of time on roughly equal sized BSTs. However a B-Tree's performance is much more
/// amortized. If a node is overfull, it must be split into two nodes. If a node is underfull, it
/// may be merged with another. Both of these operations are relatively expensive to perform, and
/// it's possible to force one to occur at every single level of the tree in a single insertion or
/// deletion. In fact, a malicious or otherwise unlucky sequence of insertions and deletions can
/// force this degenerate behaviour to occur on every operation. While the total amount of work
/// done on each operation isn't *catastrophic*, and *is* still bounded by O(B log<sub>B</sub>n),
/// it is certainly much slower when it does.
#[deriving(Clone)]
pub struct BTreeMap<K, V> {
    root: Node<K, V>,
    length: uint,
    depth: uint,
    b: uint,
}

/// An abstract base over-which all other BTree iterators are built.
struct AbsEntries<T> {
    lca: T,
    left: RingBuf<T>,
    right: RingBuf<T>,
    size: uint,
}

/// An iterator over a BTreeMap's entries.
pub struct Entries<'a, K: 'a, V: 'a> {
    inner: AbsEntries<Traversal<'a, K, V>>
}

/// A mutable iterator over a BTreeMap's entries.
pub struct MutEntries<'a, K: 'a, V: 'a> {
    inner: AbsEntries<MutTraversal<'a, K, V>>
}

/// An owning iterator over a BTreeMap's entries.
pub struct MoveEntries<K, V> {
    inner: AbsEntries<MoveTraversal<K, V>>
}

/// An iterator over a BTreeMap's keys.
pub type Keys<'a, K, V> = iter::Map<'static, (&'a K, &'a V), &'a K, Entries<'a, K, V>>;

/// An iterator over a BTreeMap's values.
pub type Values<'a, K, V> = iter::Map<'static, (&'a K, &'a V), &'a V, Entries<'a, K, V>>;

/// A view into a single entry in a map, which may either be vacant or occupied.
pub enum Entry<'a, K:'a, V:'a> {
    /// A vacant Entry
    Vacant(VacantEntry<'a, K, V>),
    /// An occupied Entry
    Occupied(OccupiedEntry<'a, K, V>),
}

/// A vacant Entry.
pub struct VacantEntry<'a, K:'a, V:'a> {
    key: K,
    stack: stack::SearchStack<'a, K, V>,
}

/// An occupied Entry.
pub struct OccupiedEntry<'a, K:'a, V:'a> {
    stack: stack::SearchStack<'a, K, V>,
}

impl<K: Ord, V> BTreeMap<K, V> {
    /// Makes a new empty BTreeMap with a reasonable choice for B.
    pub fn new() -> BTreeMap<K, V> {
        //FIXME(Gankro): Tune this as a function of size_of<K/V>?
        BTreeMap::with_b(6)
    }

    /// Makes a new empty BTreeMap with the given B.
    ///
    /// B cannot be less than 2.
    pub fn with_b(b: uint) -> BTreeMap<K, V> {
        assert!(b > 1, "B must be greater than 1");
        BTreeMap {
            length: 0,
            depth: 1,
            root: Node::make_leaf_root(b),
            b: b,
        }
    }
}

impl<K: Ord, V> Map<K, V> for BTreeMap<K, V> {
    // Searching in a B-Tree is pretty straightforward.
    //
    // Start at the root. Try to find the key in the current node. If we find it, return it.
    // If it's not in there, follow the edge *before* the smallest key larger than
    // the search key. If no such key exists (they're *all* smaller), then just take the last
    // edge in the node. If we're in a leaf and we don't find our key, then it's not
    // in the tree.
    fn find(&self, key: &K) -> Option<&V> {
        let mut cur_node = &self.root;
        loop {
            match cur_node.search(key) {
                Found(i) => return cur_node.val(i),
                GoDown(i) => match cur_node.edge(i) {
                    None => return None,
                    Some(next_node) => {
                        cur_node = next_node;
                        continue;
                    }
                }
            }
        }
    }
}

impl<K: Ord, V> MutableMap<K, V> for BTreeMap<K, V> {
    // See `find` for implementation notes, this is basically a copy-paste with mut's added
    fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        // temp_node is a Borrowck hack for having a mutable value outlive a loop iteration
        let mut temp_node = &mut self.root;
        loop {
            let cur_node = temp_node;
            match cur_node.search(key) {
                Found(i) => return cur_node.val_mut(i),
                GoDown(i) => match cur_node.edge_mut(i) {
                    None => return None,
                    Some(next_node) => {
                        temp_node = next_node;
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

    fn swap(&mut self, key: K, mut value: V) -> Option<V> {
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
            // Same basic logic as found in `find`, but with PartialSearchStack mediating the
            // actual nodes for us
            match stack.next().search(&key) {
                Found(i) => unsafe {
                    // Perfect match, swap the values and return the old one
                    let next = stack.into_next();
                    mem::swap(next.unsafe_val_mut(i), &mut value);
                    return Some(value);
                },
                GoDown(i) => {
                    // We need to keep searching, try to get the search stack
                    // to go down further
                    stack = match stack.push(i) {
                        stack::Done(new_stack) => {
                            // We've reached a leaf, perform the insertion here
                            new_stack.insert(key, value);
                            return None;
                        }
                        stack::Grew(new_stack) => {
                            // We've found the subtree to insert this key/value pair in,
                            // keep searching
                            new_stack
                        }
                    };
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

    fn pop(&mut self, key: &K) -> Option<V> {
        // See `swap` for a more thorough description of the stuff going on in here
        let mut stack = stack::PartialSearchStack::new(self);
        loop {
            match stack.next().search(key) {
                Found(i) => {
                    // Perfect match. Terminate the stack here, and remove the entry
                    return Some(stack.seal(i).remove());
                },
                GoDown(i) => {
                    // We need to keep searching, try to go down the next edge
                    stack = match stack.push(i) {
                        stack::Done(_) => return None, // We're at a leaf; the key isn't in here
                        stack::Grew(new_stack) => {
                            new_stack
                        }
                    };
                }
            }
        }
    }
}

/// The stack module provides a safe interface for constructing and manipulating a stack of ptrs
/// to nodes. By using this module much better safety guarantees can be made, and more search
/// boilerplate gets cut out.
mod stack {
    use core::prelude::*;
    use super::BTreeMap;
    use super::super::node::*;
    use {MutableMap, MutableSeq};
    use vec::Vec;

    type StackItem<K, V> = (*mut Node<K, V>, uint);
    type Stack<K, V> = Vec<StackItem<K, V>>;

    /// A PartialSearchStack handles the construction of a search stack.
    pub struct PartialSearchStack<'a, K:'a, V:'a> {
        map: &'a mut BTreeMap<K, V>,
        stack: Stack<K, V>,
        next: *mut Node<K, V>,
    }

    /// A SearchStack represents a full path to an element of interest. It provides methods
    /// for manipulating the element at the top of its stack.
    pub struct SearchStack<'a, K:'a, V:'a> {
        map: &'a mut BTreeMap<K, V>,
        stack: Stack<K, V>,
        top: StackItem<K, V>,
    }

    /// The result of asking a PartialSearchStack to push another node onto itself. Either it
    /// Grew, in which case it's still Partial, or it found its last node was actually a leaf, in
    /// which case it seals itself and yields a complete SearchStack.
    pub enum PushResult<'a, K:'a, V:'a> {
        Grew(PartialSearchStack<'a, K, V>),
        Done(SearchStack<'a, K, V>),
    }

    impl<'a, K, V> PartialSearchStack<'a, K, V> {
        /// Creates a new PartialSearchStack from a BTreeMap by initializing the stack with the
        /// root of the tree.
        pub fn new<'a>(map: &'a mut BTreeMap<K, V>) -> PartialSearchStack<'a, K, V> {
            let depth = map.depth;

            PartialSearchStack {
                next: &mut map.root as *mut _,
                map: map,
                stack: Vec::with_capacity(depth),
            }
        }

        /// Pushes the requested child of the stack's current top on top of the stack. If the child
        /// exists, then a new PartialSearchStack is yielded. Otherwise, a full SearchStack is
        /// yielded.
        pub fn push(self, edge: uint) -> PushResult<'a, K, V> {
            let map = self.map;
            let mut stack = self.stack;
            let next_ptr = self.next;
            let next_node = unsafe {
                &mut *next_ptr
            };
            let to_insert = (next_ptr, edge);
            match next_node.edge_mut(edge) {
                None => Done(SearchStack {
                    map: map,
                    stack: stack,
                    top: to_insert,
                }),
                Some(node) => {
                    stack.push(to_insert);
                    Grew(PartialSearchStack {
                        map: map,
                        stack: stack,
                        next: node as *mut _,
                    })
                },
            }
        }

        /// Converts the stack into a mutable reference to its top.
        pub fn into_next(self) -> &'a mut Node<K, V> {
            unsafe {
                &mut *self.next
            }
        }

        /// Gets the top of the stack.
        pub fn next(&self) -> &Node<K, V> {
            unsafe {
                &*self.next
            }
        }

        /// Converts the PartialSearchStack into a SearchStack.
        pub fn seal(self, index: uint) -> SearchStack<'a, K, V> {
            SearchStack {
                map: self.map,
                stack: self.stack,
                top: (self.next as *mut _, index),
            }
        }
    }

    impl<'a, K, V> SearchStack<'a, K, V> {
        /// Gets a reference to the value the stack points to.
        pub fn peek(&self) -> &V {
            let (node_ptr, index) = self.top;
            unsafe {
                (*node_ptr).val(index).unwrap()
            }
        }

        /// Gets a mutable reference to the value the stack points to.
        pub fn peek_mut(&mut self) -> &mut V {
            let (node_ptr, index) = self.top;
            unsafe {
                (*node_ptr).val_mut(index).unwrap()
            }
        }

        /// Converts the stack into a mutable reference to the value it points to, with a lifetime
        /// tied to the original tree.
        pub fn into_top(self) -> &'a mut V {
            let (node_ptr, index) = self.top;
            unsafe {
                (*node_ptr).val_mut(index).unwrap()
            }
        }

        /// Inserts the key and value into the top element in the stack, and if that node has to
        /// split recursively inserts the split contents into the next element stack until
        /// splits stop.
        ///
        /// Assumes that the stack represents a search path from the root to a leaf.
        ///
        /// An &mut V is returned to the inserted value, for callers that want a reference to this.
        pub fn insert(self, key: K, val: V) -> &'a mut V {
            unsafe {
                let map = self.map;
                map.length += 1;

                let mut stack = self.stack;
                // Insert the key and value into the leaf at the top of the stack
                let (node, index) = self.top;
                let (mut insertion, inserted_ptr) = {
                    (*node).insert_as_leaf(index, key, val)
                };

                loop {
                    match insertion {
                        Fit => {
                            // The last insertion went off without a hitch, no splits! We can stop
                            // inserting now.
                            return &mut *inserted_ptr;
                        }
                        Split(key, val, right) => match stack.pop() {
                            // The last insertion triggered a split, so get the next element on the
                            // stack to recursively insert the split node into.
                            None => {
                                // The stack was empty; we've split the root, and need to make a
                                // a new one. This is done in-place because we can't move the
                                // root out of a reference to the tree.
                                Node::make_internal_root(&mut map.root, map.b, key, val, right);

                                map.depth += 1;
                                return &mut *inserted_ptr;
                            }
                            Some((node, index)) => {
                                // The stack wasn't empty, do the insertion and recurse
                                insertion = (*node).insert_as_internal(index, key, val, right);
                                continue;
                            }
                        }
                    }
                }
            }
        }

        /// Removes the key and value in the top element of the stack, then handles underflows as
        /// described in BTree's pop function.
        pub fn remove(mut self) -> V {
            // Ensure that the search stack goes to a leaf. This is necessary to perform deletion
            // in a BTree. Note that this may put the tree in an inconsistent state (further
            // described in leafify's comments), but this is immediately fixed by the
            // removing the value we want to remove
            self.leafify();

            let map = self.map;
            map.length -= 1;

            let mut stack = self.stack;

            // Remove the key-value pair from the leaf that this search stack points to.
            // Then, note if the leaf is underfull, and promptly forget the leaf and its ptr
            // to avoid ownership issues.
            let (value, mut underflow) = unsafe {
                let (leaf_ptr, index) = self.top;
                let leaf = &mut *leaf_ptr;
                let (_key, value) = leaf.remove_as_leaf(index);
                let underflow = leaf.is_underfull();
                (value, underflow)
            };

            loop {
                match stack.pop() {
                    None => {
                        // We've reached the root, so no matter what, we're done. We manually
                        // access the root via the tree itself to avoid creating any dangling
                        // pointers.
                        if map.root.len() == 0 && !map.root.is_leaf() {
                            // We've emptied out the root, so make its only child the new root.
                            // If it's a leaf, we just let it become empty.
                            map.depth -= 1;
                            map.root = map.root.pop_edge().unwrap();
                        }
                        return value;
                    }
                    Some((parent_ptr, index)) => {
                        if underflow {
                            // Underflow! Handle it!
                            unsafe {
                                let parent = &mut *parent_ptr;
                                parent.handle_underflow(index);
                                underflow = parent.is_underfull();
                            }
                        } else {
                            // All done!
                            return value;
                        }
                    }
                }
            }
        }

        /// Subroutine for removal. Takes a search stack for a key that might terminate at an
        /// internal node, and mutates the tree and search stack to *make* it a search stack
        /// for that same key that *does* terminates at a leaf. If the mutation occurs, then this
        /// leaves the tree in an inconsistent state that must be repaired by the caller by
        /// removing the entry in question. Specifically the key-value pair and its successor will
        /// become swapped.
        fn leafify(&mut self) {
            unsafe {
                let (node_ptr, index) = self.top;
                // First, get ptrs to the found key-value pair
                let node = &mut *node_ptr;
                let (key_ptr, val_ptr) = {
                    (node.unsafe_key_mut(index) as *mut _,
                     node.unsafe_val_mut(index) as *mut _)
                };

                // Try to go into the right subtree of the found key to find its successor
                match node.edge_mut(index + 1) {
                    None => {
                        // We're a proper leaf stack, nothing to do
                    }
                    Some(mut temp_node) => {
                        //We're not a proper leaf stack, let's get to work.
                        self.stack.push((node_ptr, index + 1));
                        loop {
                            // Walk into the smallest subtree of this node
                            let node = temp_node;
                            let node_ptr = node as *mut _;

                            if node.is_leaf() {
                                // This node is a leaf, do the swap and return
                                self.top = (node_ptr, 0);
                                node.unsafe_swap(0, &mut *key_ptr, &mut *val_ptr);
                                break;
                            } else {
                                // This node is internal, go deeper
                                self.stack.push((node_ptr, 0));
                                temp_node = node.unsafe_edge_mut(0);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<K, V> Collection for BTreeMap<K, V> {
    fn len(&self) -> uint {
        self.length
    }
}

impl<K: Ord, V> Mutable for BTreeMap<K, V> {
    fn clear(&mut self) {
        let b = self.b;
        // avoid recursive destructors by manually traversing the tree
        for _ in mem::replace(self, BTreeMap::with_b(b)).into_iter() {};
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for BTreeMap<K, V> {
    fn from_iter<T: Iterator<(K, V)>>(iter: T) -> BTreeMap<K, V> {
        let mut map = BTreeMap::new();
        map.extend(iter);
        map
    }
}

impl<K: Ord, V> Extendable<(K, V)> for BTreeMap<K, V> {
    #[inline]
    fn extend<T: Iterator<(K, V)>>(&mut self, mut iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<S: Writer, K: Hash<S>, V: Hash<S>> Hash<S> for BTreeMap<K, V> {
    fn hash(&self, state: &mut S) {
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<K: Ord, V> Default for BTreeMap<K, V> {
    fn default() -> BTreeMap<K, V> {
        BTreeMap::new()
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for BTreeMap<K, V> {
    fn eq(&self, other: &BTreeMap<K, V>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<K: Eq, V: Eq> Eq for BTreeMap<K, V> {}

impl<K: PartialOrd, V: PartialOrd> PartialOrd for BTreeMap<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &BTreeMap<K, V>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

impl<K: Ord, V: Ord> Ord for BTreeMap<K, V> {
    #[inline]
    fn cmp(&self, other: &BTreeMap<K, V>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

impl<K: Show, V: Show> Show for BTreeMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{}: {}", *k, *v));
        }

        write!(f, "}}")
    }
}

impl<K: Ord, V> Index<K, V> for BTreeMap<K, V> {
    fn index(&self, key: &K) -> &V {
        self.find(key).expect("no entry found for key")
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
/// This is necessary to use in `next` because we want to modify self.left inside
/// a match that borrows it. Similarly, in `next_back` for self.right. Instead, we use this
/// enum to note what we want to do, and do it after the match.
enum StackOp<T> {
    Push(T),
    Pop,
}

impl<K, V, E, T: Traverse<E> + DoubleEndedIterator<TraversalItem<K, V, E>>>
        Iterator<(K, V)> for AbsEntries<T> {
    // This function is pretty long, but only because there's a lot of cases to consider.
    // Our iterator represents two search paths, left and right, to the smallest and largest
    // elements we have yet to yield. lca represents the least common ancestor of these two paths,
    // above-which we never walk, since everything outside it has already been consumed (or was
    // never in the range to iterate).
    //
    // Note that the design of these iterators permits an *arbitrary* initial pair of min and max,
    // making these arbitrary sub-range iterators. However the logic to construct these paths
    // efficiently is fairly involved, so this is a FIXME. The sub-range iterators also wouldn't be
    // able to accurately predict size, so those iterators can't implement ExactSize.
    fn next(&mut self) -> Option<(K, V)> {
        loop {
            // We want the smallest element, so try to get the top of the left stack
            let op = match self.left.back_mut() {
                // The left stack is empty, so try to get the next element of the two paths
                // LCAs (the left search path is currently a subpath of the right one)
                None => match self.lca.next() {
                    // The lca has been exhausted, walk further down the right path
                    None => match self.right.pop_front() {
                        // The right path is exhausted, so we're done
                        None => return None,
                        // The right path had something, make that the new LCA
                        // and restart the whole process
                        Some(right) => {
                            self.lca = right;
                            continue;
                        }
                    },
                    // The lca yielded an edge, make that the new head of the left path
                    Some(Edge(next)) => Push(Traverse::traverse(next)),
                    // The lca yielded an entry, so yield that
                    Some(Elem(k, v)) => {
                        self.size -= 1;
                        return Some((k, v))
                    }
                },
                // The left stack wasn't empty, so continue along the node in its head
                Some(iter) => match iter.next() {
                    // The head of the left path is empty, so Pop it off and restart the process
                    None => Pop,
                    // The head of the left path yielded an edge, so make that the new head
                    // of the left path
                    Some(Edge(next)) => Push(Traverse::traverse(next)),
                    // The head of the left path yielded entry, so yield that
                    Some(Elem(k, v)) => {
                        self.size -= 1;
                        return Some((k, v))
                    }
                }
            };

            // Handle any operation on the left stack as necessary
            match op {
                Push(item) => { self.left.push(item); },
                Pop => { self.left.pop(); },
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.size, Some(self.size))
    }
}

impl<K, V, E, T: Traverse<E> + DoubleEndedIterator<TraversalItem<K, V, E>>>
        DoubleEndedIterator<(K, V)> for AbsEntries<T> {
    // next_back is totally symmetric to next
    fn next_back(&mut self) -> Option<(K, V)> {
        loop {
            let op = match self.right.back_mut() {
                None => match self.lca.next_back() {
                    None => match self.left.pop_front() {
                        None => return None,
                        Some(left) => {
                            self.lca = left;
                            continue;
                        }
                    },
                    Some(Edge(next)) => Push(Traverse::traverse(next)),
                    Some(Elem(k, v)) => {
                        self.size -= 1;
                        return Some((k, v))
                    }
                },
                Some(iter) => match iter.next_back() {
                    None => Pop,
                    Some(Edge(next)) => Push(Traverse::traverse(next)),
                    Some(Elem(k, v)) => {
                        self.size -= 1;
                        return Some((k, v))
                    }
                }
            };

            match op {
                Push(item) => { self.right.push(item); },
                Pop => { self.right.pop(); }
            }
        }
    }
}

impl<'a, K, V> Iterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next() }
    fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}
impl<'a, K, V> DoubleEndedIterator<(&'a K, &'a V)> for Entries<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> { self.inner.next_back() }
}
impl<'a, K, V> ExactSize<(&'a K, &'a V)> for Entries<'a, K, V> {}


impl<'a, K, V> Iterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next() }
    fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}
impl<'a, K, V> DoubleEndedIterator<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> { self.inner.next_back() }
}
impl<'a, K, V> ExactSize<(&'a K, &'a mut V)> for MutEntries<'a, K, V> {}


impl<K, V> Iterator<(K, V)> for MoveEntries<K, V> {
    fn next(&mut self) -> Option<(K, V)> { self.inner.next() }
    fn size_hint(&self) -> (uint, Option<uint>) { self.inner.size_hint() }
}
impl<K, V> DoubleEndedIterator<(K, V)> for MoveEntries<K, V> {
    fn next_back(&mut self) -> Option<(K, V)> { self.inner.next_back() }
}
impl<K, V> ExactSize<(K, V)> for MoveEntries<K, V> {}



impl<'a, K: Ord, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry with the VacantEntry's key,
    /// and returns a mutable reference to it.
    pub fn set(self, value: V) -> &'a mut V {
        self.stack.insert(self.key, value)
    }
}

impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        self.stack.peek()
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        self.stack.peek_mut()
    }

    /// Converts the entry into a mutable reference to its value.
    pub fn into_mut(self) -> &'a mut V {
        self.stack.into_top()
    }

    /// Sets the value of the entry with the OccupiedEntry's key,
    /// and returns the entry's old value.
    pub fn set(&mut self, mut value: V) -> V {
        mem::swap(self.stack.peek_mut(), &mut value);
        value
    }

    /// Takes the value of the entry out of the map, and returns it.
    pub fn take(self) -> V {
        self.stack.remove()
    }
}

impl<K, V> BTreeMap<K, V> {
    /// Gets an iterator over the entries of the map.
    pub fn iter<'a>(&'a self) -> Entries<'a, K, V> {
        let len = self.len();
        Entries {
            inner: AbsEntries {
                lca: Traverse::traverse(&self.root),
                left: RingBuf::new(),
                right: RingBuf::new(),
                size: len,
            }
        }
    }

    /// Gets a mutable iterator over the entries of the map.
    pub fn iter_mut<'a>(&'a mut self) -> MutEntries<'a, K, V> {
        let len = self.len();
        MutEntries {
            inner: AbsEntries {
                lca: Traverse::traverse(&mut self.root),
                left: RingBuf::new(),
                right: RingBuf::new(),
                size: len,
            }
        }
    }

    /// Gets an owning iterator over the entries of the map.
    pub fn into_iter(self) -> MoveEntries<K, V> {
        let len = self.len();
        MoveEntries {
            inner: AbsEntries {
                lca: Traverse::traverse(self.root),
                left: RingBuf::new(),
                right: RingBuf::new(),
                size: len,
            }
        }
    }

    /// Gets an iterator over the keys of the map.
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        self.iter().map(|(k, _)| k)
    }

    /// Gets an iterator over the values of the map.
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        self.iter().map(|(_, v)| v)
    }
}

impl<K: Ord, V> BTreeMap<K, V> {
    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    pub fn entry<'a>(&'a mut self, key: K) -> Entry<'a, K, V> {
        // same basic logic of `swap` and `pop`, blended together
        let mut stack = stack::PartialSearchStack::new(self);
        loop {
            match stack.next().search(&key) {
                Found(i) => {
                    // Perfect match
                    return Occupied(OccupiedEntry {
                        stack: stack.seal(i)
                    });
                },
                GoDown(i) => {
                    stack = match stack.push(i) {
                        stack::Done(new_stack) => {
                            // Not in the tree, but we've found where it goes
                            return Vacant(VacantEntry {
                                stack: new_stack,
                                key: key,
                            });
                        }
                        stack::Grew(new_stack) => {
                            // We've found the subtree this key must go in
                            new_stack
                        }
                    };
                }
            }
        }
    }
}





#[cfg(test)]
mod test {
    use std::prelude::*;

    use {Map, MutableMap};
    use super::{BTreeMap, Occupied, Vacant};

    #[test]
    fn test_basic_large() {
        let mut map = BTreeMap::new();
        let size = 10000u;
        assert_eq!(map.len(), 0);

        for i in range(0, size) {
            assert_eq!(map.swap(i, 10*i), None);
            assert_eq!(map.len(), i + 1);
        }

        for i in range(0, size) {
            assert_eq!(map.find(&i).unwrap(), &(i*10));
        }

        for i in range(size, size*2) {
            assert_eq!(map.find(&i), None);
        }

        for i in range(0, size) {
            assert_eq!(map.swap(i, 100*i), Some(10*i));
            assert_eq!(map.len(), size);
        }

        for i in range(0, size) {
            assert_eq!(map.find(&i).unwrap(), &(i*100));
        }

        for i in range(0, size/2) {
            assert_eq!(map.pop(&(i*2)), Some(i*200));
            assert_eq!(map.len(), size - i - 1);
        }

        for i in range(0, size/2) {
            assert_eq!(map.find(&(2*i)), None);
            assert_eq!(map.find(&(2*i+1)).unwrap(), &(i*200 + 100));
        }

        for i in range(0, size/2) {
            assert_eq!(map.pop(&(2*i)), None);
            assert_eq!(map.pop(&(2*i+1)), Some(i*200 + 100));
            assert_eq!(map.len(), size/2 - i - 1);
        }
    }

    #[test]
    fn test_basic_small() {
        let mut map = BTreeMap::new();
        assert_eq!(map.pop(&1), None);
        assert_eq!(map.find(&1), None);
        assert_eq!(map.swap(1u, 1u), None);
        assert_eq!(map.find(&1), Some(&1));
        assert_eq!(map.swap(1, 2), Some(1));
        assert_eq!(map.find(&1), Some(&2));
        assert_eq!(map.swap(2, 4), None);
        assert_eq!(map.find(&2), Some(&4));
        assert_eq!(map.pop(&1), Some(2));
        assert_eq!(map.pop(&2), Some(4));
        assert_eq!(map.pop(&1), None);
    }

    #[test]
    fn test_iter() {
        let size = 10000u;

        // Forwards
        let mut map: BTreeMap<uint, uint> = Vec::from_fn(size, |i| (i, i)).into_iter().collect();

        {
            let mut iter = map.iter();
            for i in range(0, size) {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (&i, &i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }

        {
            let mut iter = map.iter_mut();
            for i in range(0, size) {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (&i, &mut (i + 0)));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }

        {
            let mut iter = map.into_iter();
            for i in range(0, size) {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (i, i));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }

    }

    #[test]
    fn test_iter_rev() {
        let size = 10000u;

        // Forwards
        let mut map: BTreeMap<uint, uint> = Vec::from_fn(size, |i| (i, i)).into_iter().collect();

        {
            let mut iter = map.iter().rev();
            for i in range(0, size) {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (&(size - i - 1), &(size - i - 1)));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }

        {
            let mut iter = map.iter_mut().rev();
            for i in range(0, size) {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (&(size - i - 1), &mut(size - i - 1)));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }

        {
            let mut iter = map.into_iter().rev();
            for i in range(0, size) {
                assert_eq!(iter.size_hint(), (size - i, Some(size - i)));
                assert_eq!(iter.next().unwrap(), (size - i - 1, size - i - 1));
            }
            assert_eq!(iter.size_hint(), (0, Some(0)));
            assert_eq!(iter.next(), None);
        }

    }

    #[test]
    fn test_entry(){
        let xs = [(1i, 10i), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: BTreeMap<int, int> = xs.iter().map(|&x| x).collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.set(100), 10);
            }
        }
        assert_eq!(map.find(&1).unwrap(), &100);
        assert_eq!(map.len(), 6);


        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                *v *= 10;
            }
        }
        assert_eq!(map.find(&2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(view) => {
                assert_eq!(view.take(), 30);
            }
        }
        assert_eq!(map.find(&3), None);
        assert_eq!(map.len(), 5);


        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(view) => {
                assert_eq!(*view.set(1000), 1000);
            }
        }
        assert_eq!(map.find(&10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }
}






#[cfg(test)]
mod bench {
    use std::prelude::*;
    use std::rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    use super::BTreeMap;
    use MutableMap;
    use deque::bench::{insert_rand_n, insert_seq_n, find_rand_n, find_seq_n};

    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        insert_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        insert_rand_n(10_000, &mut m, b);
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        insert_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        insert_seq_n(10_000, &mut m, b);
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        find_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_rand_10_000(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        find_rand_n(10_000, &mut m, b);
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        find_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_seq_10_000(b: &mut Bencher) {
        let mut m : BTreeMap<uint,uint> = BTreeMap::new();
        find_seq_n(10_000, &mut m, b);
    }

    fn bench_iter(b: &mut Bencher, size: uint) {
        let mut map = BTreeMap::<uint, uint>::new();
        let mut rng = weak_rng();

        for _ in range(0, size) {
            map.swap(rng.gen(), rng.gen());
        }

        b.iter(|| {
            for entry in map.iter() {
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
