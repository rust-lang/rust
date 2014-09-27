// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This module represents all the internal representation and logic for a B-Tree's node
// with a safe interface, so that BTreeMap itself does not depend on any of these details.

use core::prelude::*;

use core::{slice, mem, ptr};
use core::iter::Zip;
use MutableSeq;

use vec;
use vec::Vec;

/// Represents the result of an Insertion: either the item fit, or the node had to split
pub enum InsertionResult<K, V> {
    /// The inserted element fit
    Fit,
    /// The inserted element did not fit, so the node was split
    Split(K, V, Node<K, V>),
}

/// Represents the result of a search for a key in a single node
pub enum SearchResult {
    /// The element was found at the given index
    Found(uint),
    /// The element wasn't found, but if it's anywhere, it must be beyond this edge
    GoDown(uint),
}

/// A B-Tree Node. We keep keys/edges/values separate to optimize searching for keys.
#[deriving(Clone)]
pub struct Node<K, V> {
    // FIXME(Gankro): This representation is super safe and easy to reason about, but painfully
    // inefficient. As three Vecs, each node consists of *9* words: (ptr, cap, size) * 3. In
    // theory, if we take full control of allocation like HashMap's RawTable does,
    // and restrict leaves to max size 256 (not unreasonable for a btree node) we can cut
    // this down to just (ptr, cap: u8, size: u8, is_leaf: bool). With generic
    // integer arguments, cap can even move into the the type, reducing this just to
    // (ptr, size, is_leaf). This could also have cache benefits for very small nodes, as keys
    // could bleed into edges and vals.
    //
    // However doing this would require a fair amount of code to reimplement all
    // the Vec logic and iterators. It would also use *way* more unsafe code, which sucks and is
    // hard. For now, we accept this cost in the name of correctness and simplicity.
    //
    // As a compromise, keys and vals could be merged into one Vec<(K, V)>, which would shave
    // off 3 words, but possibly hurt our cache effeciency during search, which only cares about
    // keys. This would also avoid the Zip we use in our iterator implementations. This is
    // probably worth investigating.
    //
    // Note that this space waste is especially tragic since we store the Nodes by value in their
    // parent's edges Vec, so unoccupied spaces in the edges Vec are quite large, and we have
    // to shift around a lot more bits during insertion/removal.

    keys: Vec<K>,
    edges: Vec<Node<K, V>>,
    vals: Vec<V>,
}

impl<K: Ord, V> Node<K, V> {
    /// Searches for the given key in the node. If it finds an exact match,
    /// `Found` will be yielded with the matching index. If it fails to find an exact match,
    /// `GoDown` will be yielded with the index of the subtree the key must lie in.
    pub fn search(&self, key: &K) -> SearchResult {
        // FIXME(Gankro): Tune when to search linear or binary based on B (and maybe K/V).
        // For the B configured as of this writing (B = 6), binary search was *singnificantly*
        // worse for uints.
        self.search_linear(key)
    }

    fn search_linear(&self, key: &K) -> SearchResult {
        for (i, k) in self.keys.iter().enumerate() {
            match k.cmp(key) {
                Less => {},
                Equal => return Found(i),
                Greater => return GoDown(i),
            }
        }
        GoDown(self.len())
    }
}

// Public interface
impl <K, V> Node<K, V> {
    /// Make a new internal node
    pub fn new_internal(capacity: uint) -> Node<K, V> {
        Node {
            keys: Vec::with_capacity(capacity),
            vals: Vec::with_capacity(capacity),
            edges: Vec::with_capacity(capacity + 1),
        }
    }

    /// Make a new leaf node
    pub fn new_leaf(capacity: uint) -> Node<K, V> {
        Node {
            keys: Vec::with_capacity(capacity),
            vals: Vec::with_capacity(capacity),
            edges: Vec::new(),
        }
    }

    /// Make a leaf root from scratch
    pub fn make_leaf_root(b: uint) -> Node<K, V> {
        Node::new_leaf(capacity_from_b(b))
    }

    /// Make an internal root and swap it with an old root
    pub fn make_internal_root(left_and_out: &mut Node<K,V>, b: uint, key: K, value: V,
            right: Node<K,V>) {
        let mut node = Node::new_internal(capacity_from_b(b));
        mem::swap(left_and_out, &mut node);
        left_and_out.keys.push(key);
        left_and_out.vals.push(value);
        left_and_out.edges.push(node);
        left_and_out.edges.push(right);
    }


    /// How many key-value pairs the node contains
    pub fn len(&self) -> uint {
        self.keys.len()
    }

    /// How many key-value pairs the node can fit
    pub fn capacity(&self) -> uint {
        self.keys.capacity()
    }

    /// If the node has any children
    pub fn is_leaf(&self) -> bool {
        self.edges.is_empty()
    }

    /// if the node has too few elements
    pub fn is_underfull(&self) -> bool {
        self.len() < min_load_from_capacity(self.capacity())
    }

    /// if the node cannot fit any more elements
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    /// Swap the given key-value pair with the key-value pair stored in the node's index,
    /// without checking bounds.
    pub unsafe fn unsafe_swap(&mut self, index: uint, key: &mut K, val: &mut V) {
        mem::swap(self.keys.as_mut_slice().unsafe_mut(index), key);
        mem::swap(self.vals.as_mut_slice().unsafe_mut(index), val);
    }

    /// Get the node's key mutably without any bounds checks.
    pub unsafe fn unsafe_key_mut(&mut self, index: uint) -> &mut K {
        self.keys.as_mut_slice().unsafe_mut(index)
    }

    /// Get the node's value at the given index
    pub fn val(&self, index: uint) -> Option<&V> {
        self.vals.as_slice().get(index)
    }

    /// Get the node's value at the given index
    pub fn val_mut(&mut self, index: uint) -> Option<&mut V> {
        self.vals.as_mut_slice().get_mut(index)
    }

    /// Get the node's value mutably without any bounds checks.
    pub unsafe fn unsafe_val_mut(&mut self, index: uint) -> &mut V {
        self.vals.as_mut_slice().unsafe_mut(index)
    }

    /// Get the node's edge at the given index
    pub fn edge(&self, index: uint) -> Option<&Node<K,V>> {
        self.edges.as_slice().get(index)
    }

    /// Get the node's edge mutably at the given index
    pub fn edge_mut(&mut self, index: uint) -> Option<&mut Node<K,V>> {
        self.edges.as_mut_slice().get_mut(index)
    }

    /// Get the node's edge mutably without any bounds checks.
    pub unsafe fn unsafe_edge_mut(&mut self, index: uint) -> &mut Node<K,V> {
        self.edges.as_mut_slice().unsafe_mut(index)
    }

    /// Pop an edge off the end of the node
    pub fn pop_edge(&mut self) -> Option<Node<K,V>> {
        self.edges.pop()
    }

    /// Try to insert this key-value pair at the given index in this internal node
    /// If the node is full, we have to split it.
    ///
    /// Returns a *mut V to the inserted value, because the caller may want this when
    /// they're done mutating the tree, but we don't want to borrow anything for now.
    pub fn insert_as_leaf(&mut self, index: uint, key: K, value: V) ->
            (InsertionResult<K, V>, *mut V) {
        if !self.is_full() {
            // The element can fit, just insert it
            self.insert_fit_as_leaf(index, key, value);
            (Fit, unsafe { self.unsafe_val_mut(index) as *mut _ })
        } else {
            // The element can't fit, this node is full. Split it into two nodes.
            let (new_key, new_val, mut new_right) = self.split();
            let left_len = self.len();

            let ptr = if index <= left_len {
                self.insert_fit_as_leaf(index, key, value);
                unsafe { self.unsafe_val_mut(index) as *mut _ }
            } else {
                new_right.insert_fit_as_leaf(index - left_len - 1, key, value);
                unsafe { new_right.unsafe_val_mut(index - left_len - 1) as *mut _ }
            };

            (Split(new_key, new_val, new_right), ptr)
        }
    }

    /// Try to insert this key-value pair at the given index in this internal node
    /// If the node is full, we have to split it.
    pub fn insert_as_internal(&mut self, index: uint, key: K, value: V, right: Node<K, V>)
            -> InsertionResult<K, V> {
        if !self.is_full() {
            // The element can fit, just insert it
            self.insert_fit_as_internal(index, key, value, right);
            Fit
        } else {
            // The element can't fit, this node is full. Split it into two nodes.
            let (new_key, new_val, mut new_right) = self.split();
            let left_len = self.len();

            if index <= left_len {
                self.insert_fit_as_internal(index, key, value, right);
            } else {
                new_right.insert_fit_as_internal(index - left_len - 1, key, value, right);
            }

            Split(new_key, new_val, new_right)
        }
    }

    /// Remove the key-value pair at the given index
    pub fn remove_as_leaf(&mut self, index: uint) -> (K, V) {
        match (self.keys.remove(index), self.vals.remove(index)) {
            (Some(k), Some(v)) => (k, v),
            _ => unreachable!(),
        }
    }

    /// Handle an underflow in this node's child. We favour handling "to the left" because we know
    /// we're empty, but our neighbour can be full. Handling to the left means when we choose to
    /// steal, we pop off the end of our neighbour (always fast) and "unshift" ourselves
    /// (always slow, but at least faster since we know we're half-empty).
    /// Handling "to the right" reverses these roles. Of course, we merge whenever possible
    /// because we want dense nodes, and merging is about equal work regardless of direction.
    pub fn handle_underflow(&mut self, underflowed_child_index: uint) {
        assert!(underflowed_child_index <= self.len());
        unsafe {
            if underflowed_child_index > 0 {
                self.handle_underflow_to_left(underflowed_child_index);
            } else {
                self.handle_underflow_to_right(underflowed_child_index);
            }
        }
    }

    pub fn iter<'a>(&'a self) -> Traversal<'a, K, V> {
        let is_leaf = self.is_leaf();
        Traversal {
            elems: self.keys.as_slice().iter().zip(self.vals.as_slice().iter()),
            edges: self.edges.as_slice().iter(),
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> MutTraversal<'a, K, V> {
        let is_leaf = self.is_leaf();
        MutTraversal {
            elems: self.keys.as_slice().iter().zip(self.vals.as_mut_slice().iter_mut()),
            edges: self.edges.as_mut_slice().iter_mut(),
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
    }

    pub fn into_iter(self) -> MoveTraversal<K, V> {
        let is_leaf = self.is_leaf();
        MoveTraversal {
            elems: self.keys.into_iter().zip(self.vals.into_iter()),
            edges: self.edges.into_iter(),
            head_is_edge: true,
            tail_is_edge: true,
            has_edges: !is_leaf,
        }
    }
}

// Private implementation details
impl<K, V> Node<K, V> {
    /// Make a node from its raw components
    fn from_vecs(keys: Vec<K>, vals: Vec<V>, edges: Vec<Node<K, V>>) -> Node<K, V> {
        Node {
            keys: keys,
            vals: vals,
            edges: edges,
        }
    }

    /// We have somehow verified that this key-value pair will fit in this internal node,
    /// so insert under that assumption.
    fn insert_fit_as_leaf(&mut self, index: uint, key: K, val: V) {
        self.keys.insert(index, key);
        self.vals.insert(index, val);
    }

    /// We have somehow verified that this key-value pair will fit in this internal node,
    /// so insert under that assumption
    fn insert_fit_as_internal(&mut self, index: uint, key: K, val: V, right: Node<K, V>) {
        self.keys.insert(index, key);
        self.vals.insert(index, val);
        self.edges.insert(index + 1, right);
    }

    /// Node is full, so split it into two nodes, and yield the middle-most key-value pair
    /// because we have one too many, and our parent now has one too few
    fn split(&mut self) -> (K, V, Node<K, V>) {
        let r_keys = split(&mut self.keys);
        let r_vals = split(&mut self.vals);
        let r_edges = if self.edges.is_empty() {
            Vec::new()
        } else {
            split(&mut self.edges)
        };

        let right = Node::from_vecs(r_keys, r_vals, r_edges);
        // Pop it
        let key = self.keys.pop().unwrap();
        let val = self.vals.pop().unwrap();

        (key, val, right)
    }

    /// Right is underflowed. Try to steal from left,
    /// but merge left and right if left is low too.
    unsafe fn handle_underflow_to_left(&mut self, underflowed_child_index: uint) {
        let left_len = self.edges[underflowed_child_index - 1].len();
        if left_len > min_load_from_capacity(self.capacity()) {
            self.steal_to_left(underflowed_child_index);
        } else {
            self.merge_children(underflowed_child_index - 1);
        }
    }

    /// Left is underflowed. Try to steal from the right,
    /// but merge left and right if right is low too.
    unsafe fn handle_underflow_to_right(&mut self, underflowed_child_index: uint) {
        let right_len = self.edges[underflowed_child_index + 1].len();
        if right_len > min_load_from_capacity(self.capacity()) {
            self.steal_to_right(underflowed_child_index);
        } else {
            self.merge_children(underflowed_child_index);
        }
    }

    /// Steal! Stealing is roughly analagous to a binary tree rotation.
    /// In this case, we're "rotating" right.
    unsafe fn steal_to_left(&mut self, underflowed_child_index: uint) {
        // Take the biggest stuff off left
        let (mut key, mut val, edge) = {
            let left = self.unsafe_edge_mut(underflowed_child_index - 1);
            match (left.keys.pop(), left.vals.pop(), left.edges.pop()) {
                (Some(k), Some(v), e) => (k, v, e),
                _ => unreachable!(),
            }
        };

        // Swap the parent's seperating key-value pair with left's
        self.unsafe_swap(underflowed_child_index - 1, &mut key, &mut val);

        // Put them at the start of right
        {
            let right = self.unsafe_edge_mut(underflowed_child_index);
            right.keys.insert(0, key);
            right.vals.insert(0, val);
            match edge {
                None => {}
                Some(e) => right.edges.insert(0, e)
            }
        }
    }

    /// Steal! Stealing is roughly analagous to a binary tree rotation.
    /// In this case, we're "rotating" left.
    unsafe fn steal_to_right(&mut self, underflowed_child_index: uint) {
        // Take the smallest stuff off right
        let (mut key, mut val, edge) = {
            let right = self.unsafe_edge_mut(underflowed_child_index + 1);
            match (right.keys.remove(0), right.vals.remove(0), right.edges.remove(0)) {
                (Some(k), Some(v), e) => (k, v, e),
                _ => unreachable!(),
            }
        };

        // Swap the parent's seperating key-value pair with right's
        self.unsafe_swap(underflowed_child_index, &mut key, &mut val);

        // Put them at the end of left
        {
            let left = self.unsafe_edge_mut(underflowed_child_index);
            left.keys.push(key);
            left.vals.push(val);
            match edge {
                None => {}
                Some(e) => left.edges.push(e)
            }
        }
    }

    /// Merge! Left and right will be smooshed into one node, along with the key-value
    /// pair that seperated them in their parent.
    unsafe fn merge_children(&mut self, left_index: uint) {
        // Permanently remove right's index, and the key-value pair that seperates
        // left and right
        let (key, val, right) = {
            match (self.keys.remove(left_index),
                self.vals.remove(left_index),
                self.edges.remove(left_index + 1)) {
                (Some(k), Some(v), Some(e)) => (k, v, e),
                _ => unreachable!(),
            }
        };

        // Give left right's stuff.
        let left = self.unsafe_edge_mut(left_index);
        left.absorb(key, val, right);
    }

    /// Take all the values from right, seperated by the given key and value
    fn absorb(&mut self, key: K, val: V, right: Node<K, V>) {
        // Just as a sanity check, make sure we can fit this guy in
        debug_assert!(self.len() + right.len() <= self.capacity())

        self.keys.push(key);
        self.vals.push(val);
        self.keys.extend(right.keys.into_iter());
        self.vals.extend(right.vals.into_iter());
        self.edges.extend(right.edges.into_iter());
    }
}

/// Takes a Vec, and splits half the elements into a new one.
fn split<T>(left: &mut Vec<T>) -> Vec<T> {
    // This function is intended to be called on a full Vec of size 2B - 1 (keys, values),
    // or 2B (edges). In the former case, left should get B elements, and right should get
    // B - 1. In the latter case, both should get B. Therefore, we can just always take the last
    // size / 2 elements from left, and put them on right. This also ensures this method is
    // safe, even if the Vec isn't full. Just uninteresting for our purposes.
    let len = left.len();
    let right_len = len / 2;
    let left_len = len - right_len;
    let mut right = Vec::with_capacity(left.capacity());
    unsafe {
        let left_ptr = left.as_slice().unsafe_get(left_len) as *const _;
        let right_ptr = right.as_mut_slice().as_mut_ptr();
        ptr::copy_nonoverlapping_memory(right_ptr, left_ptr, right_len);
        left.set_len(left_len);
        right.set_len(right_len);
    }
    right
}

/// Get the capacity of a node from the order of the parent B-Tree
fn capacity_from_b(b: uint) -> uint {
    2 * b - 1
}

/// Get the minimum load of a node from its capacity
fn min_load_from_capacity(cap: uint) -> uint {
    // B - 1
    cap / 2
}

/// An abstraction over all the different kinds of traversals a node supports
struct AbsTraversal<Elems, Edges> {
    elems: Elems,
    edges: Edges,
    head_is_edge: bool,
    tail_is_edge: bool,
    has_edges: bool,
}

/// A single atomic step in a traversal. Either an element is visited, or an edge is followed
pub enum TraversalItem<K, V, E> {
    Elem(K, V),
    Edge(E),
}

/// A traversal over a node's entries and edges
pub type Traversal<'a, K, V> = AbsTraversal<Zip<slice::Items<'a, K>, slice::Items<'a, V>>,
                                            slice::Items<'a, Node<K, V>>>;

/// A mutable traversal over a node's entries and edges
pub type MutTraversal<'a, K, V> = AbsTraversal<Zip<slice::Items<'a, K>, slice::MutItems<'a, V>>,
                                               slice::MutItems<'a, Node<K, V>>>;

/// An owning traversal over a node's entries and edges
pub type MoveTraversal<K, V> = AbsTraversal<Zip<vec::MoveItems<K>, vec::MoveItems<V>>,
                                                vec::MoveItems<Node<K, V>>>;


impl<K, V, E, Elems: Iterator<(K, V)>, Edges: Iterator<E>>
        Iterator<TraversalItem<K, V, E>> for AbsTraversal<Elems, Edges> {

    fn next(&mut self) -> Option<TraversalItem<K, V, E>> {
        let head_is_edge = self.head_is_edge;
        self.head_is_edge = !head_is_edge;

        if head_is_edge && self.has_edges {
            self.edges.next().map(|node| Edge(node))
        } else {
            self.elems.next().map(|(k, v)| Elem(k, v))
        }
    }
}

impl<K, V, E, Elems: DoubleEndedIterator<(K, V)>, Edges: DoubleEndedIterator<E>>
        DoubleEndedIterator<TraversalItem<K, V, E>> for AbsTraversal<Elems, Edges> {

    fn next_back(&mut self) -> Option<TraversalItem<K, V, E>> {
        let tail_is_edge = self.tail_is_edge;
        self.tail_is_edge = !tail_is_edge;

        if tail_is_edge && self.has_edges {
            self.edges.next_back().map(|node| Edge(node))
        } else {
            self.elems.next_back().map(|(k, v)| Elem(k, v))
        }
    }
}
