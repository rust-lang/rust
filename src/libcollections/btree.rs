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

use alloc::boxed::Box;
use vec::Vec;
use core::mem;
use core::iter::range_inclusive;
use {Mutable, MutableMap, Map, MutableSeq};

/// "Order" of the B-tree, from which all other properties are derived
static B: uint = 6;
/// Maximum number of elements in a node
static CAPACITY: uint = 2 * B - 1;
/// Minimum number of elements in a node
static MIN_LOAD: uint = B - 1;
/// Maximum number of children in a node
static EDGE_CAPACITY: uint = CAPACITY + 1;
/// Amount to take off the tail of a node being split
static SPLIT_LEN: uint = B - 1;

/// Represents a search path for mutating. The rawptrs here should never be
/// null or dangling, and should be accessed one-at-a-time via pops.
type SearchStack<K,V> = Vec<(*mut Node<K,V>, uint)>;

/// Represents the result of an Insertion: either the item fit, or the node had to split
enum InsertionResult<K,V>{
    Fit,
    Split(K, V, Box<Node<K,V>>),
}

/// Represents the result of a search for a key in a single node
enum SearchResult {
    Found(uint), Bound(uint),
}

/// A B-Tree Node
struct Node<K,V> {
    length: uint,
    keys: [Option<K>, ..CAPACITY],
    edges: [Option<Box<Node<K,V>>>, ..EDGE_CAPACITY],
    vals: [Option<V>, ..CAPACITY],
}


/// A B-Tree of Order 6
pub struct BTree<K,V>{
    root: Option<Box<Node<K,V>>>,
    length: uint,
    depth: uint,
}

impl<K,V> BTree<K,V> {
    /// Make a new empty BTree
    pub fn new() -> BTree<K,V> {
        BTree {
            length: 0,
            depth: 0,
            root: None,
        }
    }
}

impl<K: Ord, V> Map<K,V> for BTree<K,V> {
    // Searching in a B-Tree is pretty straightforward.
    //
    // Start at the root. Try to find the key in the current node. If we find it, return it.
    // If it's not in there, follow the edge *before* the smallest key larger than
    // the search key. If no such key exists (they're *all* smaller), then just take the last
    // edge in the node. If we're in a leaf and we don't find our key, then it's not
    // in the tree.
    fn find(&self, key: &K) -> Option<&V> {
        match self.root.as_ref() {
            None => None,
            Some(root) => {
                let mut cur_node = &**root;
                loop {
                    match cur_node.search(key) {
                        Found(i) => return cur_node.vals[i].as_ref(), // Found the key
                        Bound(i) => match cur_node.edges[i].as_ref() { // Didn't find the key
                            None => return None, // We're a leaf, it's not in here
                            Some(next_node) => { // We're an internal node, search the subtree
                                cur_node = &**next_node;
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<K: Ord, V> MutableMap<K,V> for BTree<K,V> {
    // See `find` for implementation notes, this is basically a copy-paste with mut's added
    fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.root.as_mut() {
            None => None,
            Some(root) => {
                // temp_node is a Borrowck hack for having a mutable value outlive a loop iteration
                let mut temp_node = &mut **root;
                loop {
                    let cur_node = temp_node;
                    match cur_node.search(key) {
                        Found(i) => return cur_node.vals[i].as_mut(),
                        Bound(i) => match cur_node.edges[i].as_mut() {
                            None => return None,
                            Some(next_node) => {
                                temp_node = &mut **next_node;
                                continue;
                            }
                        }
                    }
                }
            }
        }
    }

    // Insertion in a B-Tree is a bit complicated.
    //
    // First we do the same kind of search described in
    // `find`, but we need to maintain a stack of all the nodes/edges in our search path.
    // If we find a match for the key we're trying to insert, just swap the.vals and return the
    // old ones. However, when we bottom out in a leaf, we attempt to insert our key-value pair
    // at the same location we would want to follow another edge.
    //
    // If the node has room, then this is done in the obvious way by shifting elements. However,
    // if the node itself is full, we split node into two, and give its median
    // key-value pair to its parent to insert the new node with. Of course, the parent may also be
    // full, and insertion can propogate until we reach the root. If we reach the root, and
    // it is *also* full, then we split the root and place the two nodes under a newly made root.
    //
    // Note that we subtly deviate from Open Data Structures in our implementation of split.
    // ODS describes inserting into the node *regardless* of its capacity, and then
    // splitting *afterwards* if it happens to be overfull. However, this is inefficient
    // (or downright impossible, depending on the design).
    // Instead, we split beforehand, and then insert the key-value pair into the appropriate
    // result node. This has two consequences:
    //
    // 1) While ODS produces a left node of size B-1, and a right node of size B,
    // we may potentially reverse this. However, this shouldn't effect the analysis.
    //
    // 2) While ODS may potentially return the pair we *just* inserted after
    // the split, we will never do this. Again, this shouldn't effect the analysis.

    fn swap(&mut self, mut key: K, mut value: V) -> Option<V> {
        // FIXME(Gankro): this is gross because of lexical borrows.
        // If pcwalton's work pans out, this can be made much better!
        // See `find` for a more idealized structure
        if self.root.is_none() {
            self.root = Some(Node::make_leaf_root(key, value));
            self.length += 1;
            self.depth += 1;
            None
        } else {
            let visit_stack = {
                // Borrowck hack, see `find_mut`
                let mut temp_node = &mut **self.root.as_mut().unwrap();
                // visit_stack is a stack of rawptrs to nodes paired with indices, respectively
                // representing the nodes and edges of our search path. We have to store rawptrs
                // because as far as Rust is concerned, we can mutate aliased data with such a
                // stack. It is of course correct, but what it doesn't know is the following:
                //
                // * The nodes in the visit_stack don't move in memory (at least, don't move
                // in memory between now and when we've finished handling the raw pointer to it)
                //
                // * We don't mutate anything through a given ptr until we've popped and forgotten
                // all the ptrs after it, at which point we don't have any pointers to children of
                // that node
                //
                // An alternative is to take the Node boxes from their parents. This actually makes
                // borrowck *really* happy and everything is pretty smooth. However, this creates
                // *tons* of pointless writes, and requires us to always walk all the way back to
                // the root after an insertion, even if we only needed to change a leaf. Therefore,
                // we accept this potential unsafety and complexity in the name of performance.
                let mut visit_stack = Vec::with_capacity(self.depth);

                loop {
                    let cur_node = temp_node;
                    let cur_node_ptr = cur_node as *mut _;

                    // See `find` for a description of this search
                    match cur_node.search(&key) {
                        Found(i) => {
                            // Perfect match, swap the contents and return the old ones
                            mem::swap(cur_node.vals[i].as_mut().unwrap(), &mut value);
                            mem::swap(cur_node.keys[i].as_mut().unwrap(), &mut key);
                            return Some(value);
                        },
                        Bound(i) => {
                            visit_stack.push((cur_node_ptr, i));
                            match cur_node.edges[i].as_mut() {
                                None => {
                                    // We've found where to insert this key/value pair
                                    break;
                                }
                                Some(next_node) => {
                                    // We've found the subtree to insert this key/value pair in
                                    temp_node = &mut **next_node;
                                    continue;
                                }
                            }
                        }
                    }
                }
                visit_stack
            };

            // If we get here then we need to insert a new element
            self.insert_stack(visit_stack, key, value);
            None
        }
    }

    // Deletion is the most complicated operation for a B-Tree.
    //
    // First we do the same kind of search described in `find`, but we need to maintain a stack
    // of all the nodes/edges in our search path. If we don't find the key, then we just return
    // `None` and do nothing. If we do find the key, we perform two operations: remove the item,
    // and then possibly handle underflow.
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
    //      with the item in their mutual parent that seperates them, and then insert the
    //      parent's item and the taken child into the first (last) index of the underflowed node.
    //
    //      However, stealing has the possibility of underflowing our neighbour. If this is the
    //      case, we instead *merge* with our neighbour. This of course reduces the number of
    //      children in the parent. Therefore, we also steal the item that seperates the now
    //      merged nodes, and insert it into the merged node.
    //
    //      Merging may cause the parent to underflow. If this is the case, then we must repeat
    //      the underflow handling process on the parent. If merging merges the last two children
    //      of the root, then we replace the root with the merged node.

    fn pop(&mut self, key: &K) -> Option<V> {
        // See `pop` for a discussion of why this is gross
        if self.root.is_none() {
            // We're empty, get lost!
            None
        } else {
            let visit_stack = {
                // Borrowck hack, see `find_mut`
                let mut temp_node = &mut **self.root.as_mut().unwrap();
                // See `pop` for a description of this variable
                let mut visit_stack = Vec::with_capacity(self.depth);

                loop {
                    let cur_node = temp_node;
                    let cur_node_ptr = cur_node as *mut _;

                    // See `find` for a description of this search
                    match cur_node.search(key) {
                        Found(i) => {
                            // Perfect match. Terminate the stack here, and move to the
                            // next phase (remove_stack).
                            visit_stack.push((cur_node_ptr, i));

                            if cur_node.edges[i].is_some() {
                                // We found the key in an internal node, but that's annoying,
                                // so let's swap it with a leaf key and pretend we *did* find
                                // it in a leaf. Note that after calling this, the tree is in an
                                // inconsistent state, but will be consistent after we remove the
                                // swapped value in `remove_stack`
                                leafify_stack(&mut visit_stack);
                            }
                            break;
                        },
                        Bound(i) => match cur_node.edges[i].as_mut() {
                            None => return None, // We're at a leaf; the key isn't in this tree
                            Some(next_node) => {
                                // We've found the subtree the key must be in
                                visit_stack.push((cur_node_ptr, i));
                                temp_node = &mut **next_node;
                                continue;
                            }
                        }
                    }
                }
                visit_stack
            };

            // If we get here then we found the key, let's remove it
            Some(self.remove_stack(visit_stack))
        }
    }
}

impl<K: Ord, V> BTree<K,V> {
    /// insert the key and value into the top element in the stack, and if that node has to split
    /// recursively insert the split contents into the stack until splits stop. Then replace the
    /// stack back into the tree.
    ///
    /// Assumes that the stack represents a search path from the root to a leaf, and that the
    /// search path is non-empty
    fn insert_stack(&mut self, mut stack: SearchStack<K,V>, key: K, value: V) {
        self.length += 1;

        // Insert the key and value into the leaf at the top of the stack
        let (node, index) = stack.pop().unwrap();
        let mut insertion = unsafe {
            (*node).insert_as_leaf(index, key, value)
        };

        loop {
            match insertion {
                Fit => {
                    // The last insertion went off without a hitch, no splits! We can stop
                    // inserting now.
                    return;
                }
                Split(key, value, right) => match stack.pop() {
                    // The last insertion triggered a split, so get the next element on the
                    // stack to recursively insert the split node into.
                    None => {
                        // The stack was empty; we've split the root, and need to make a new one.
                        let left = self.root.take().unwrap();
                        self.root = Some(Node::make_internal_root(key, value, left, right));
                        self.depth += 1;
                        return;
                    }
                    Some((node, index)) => {
                        // The stack wasn't empty, do the insertion and recurse
                        unsafe {
                            insertion = (*node).insert_as_internal(index, key, value, right);
                        }
                        continue;
                    }
                }
            }
        }
    }

    /// Remove the key and value in the top element of the stack, then handle underflows.
    /// Assumes the stack represents a search path from the root to a leaf.
    fn remove_stack(&mut self, mut stack: SearchStack<K,V>) -> V {
        self.length -= 1;

        // Remove the key-value pair from the leaf, check if the node is underfull, and then
        // promptly forget the leaf and ptr to avoid ownership issues
        let (value, mut underflow) = unsafe {
            let (node_ptr, index) = stack.pop().unwrap();
            let node = &mut *node_ptr;
            let (_key, value) = node.remove_as_leaf(index);
            let underflow = node.length < MIN_LOAD;
            (value, underflow)
        };

        loop {
            match stack.pop() {
                None => {
                    // We've reached the root, so no matter what, we're done. We manually access
                    // the root via the tree itself to avoid creating any dangling pointers.
                    if self.root.as_ref().unwrap().length == 0 {
                        // We've emptied out the root, so make its only child the new root.
                        // If the root is a leaf, this will set the root to `None`
                        self.depth -= 1;
                        self.root = self.root.take().unwrap().edges[0].take();
                    }
                    return value;
                }
                Some((parent_ptr, index)) => {
                    if underflow {
                        // Underflow! Handle it!
                        unsafe {
                            let parent = &mut *parent_ptr;
                            parent.handle_underflow(index);
                            underflow = parent.length < MIN_LOAD;
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

impl<K: Ord, V> Node<K,V> {
    /// Make a new node
    fn new() -> Node<K,V> {
        Node {
            length: 0,
            // FIXME(Gankro): this is gross, I guess you need a macro? [None, ..capacity] uses copy
            keys:   [None, None, None, None, None, None, None, None, None, None, None],
            vals: [None, None, None, None, None, None, None, None, None, None, None],
            edges:  [None, None, None, None, None, None, None, None, None, None, None, None],
        }
    }

    /// An iterator for the keys of this node
    fn keys<'a>(&'a self) -> Keys<'a, K> {
        Keys{ it: self.keys.iter() }
    }

    /// Searches for the given key in the node. If it finds an exact match,
    /// `Found` will be yielded with the matching index. If it fails to find an exact match,
    /// `Bound` will be yielded with the index of the subtree the key must lie in.
    fn search(&self, key: &K) -> SearchResult {
        // linear search the node's keys because we're small
        // FIXME(Gankro): if we ever get generic integer arguments
        // to support variable choices of `B`, then this should be
        // tuned to fall into binary search at some arbitrary level
        for (i, k) in self.keys().enumerate() {
            match k.cmp(key) {
                Less => {}, // keep walkin' son, she's too small
                Equal => return Found(i),
                Greater => return Bound(i),
            }
        }
        Bound(self.length)
    }

    /// Make a leaf root from scratch
    fn make_leaf_root(key: K, value: V) -> Box<Node<K,V>> {
        let mut node = box Node::new();
        node.insert_fit_as_leaf(0, key, value);
        node
    }

    /// Make an internal root from scratch
    fn make_internal_root(key: K, value: V, left: Box<Node<K,V>>, right: Box<Node<K,V>>)
            -> Box<Node<K,V>> {
        let mut node = box Node::new();
        node.keys[0] = Some(key);
        node.vals[0] = Some(value);
        node.edges[0] = Some(left);
        node.edges[1] = Some(right);
        node.length = 1;
        node
    }

    /// Try to insert this key-value pair at the given index in this internal node
    /// If the node is full, we have to split it.
    fn insert_as_leaf(&mut self, index: uint, key: K, value: V) -> InsertionResult<K,V> {
        let len = self.length;
        if len < CAPACITY {
            // The element can fit, just insert it
            self.insert_fit_as_leaf(index, key, value);
            Fit
        } else {
            // The element can't fit, this node is full. Split it into two nodes.
            let (new_key, new_val, mut new_right) = self.split();
            let left_len = self.length;

            if index <= left_len {
                self.insert_fit_as_leaf(index, key, value);
            } else {
                new_right.insert_fit_as_leaf(index - left_len - 1, key, value);
            }

            Split(new_key, new_val, new_right)
        }
    }

    /// Try to insert this key-value pair at the given index in this internal node
    /// If the node is full, we have to split it.
    fn insert_as_internal(&mut self, index: uint, key: K, value: V, right: Box<Node<K,V>>)
            -> InsertionResult<K,V> {
        let len = self.length;
        if len < CAPACITY {
            // The element can fit, just insert it
            self.insert_fit_as_internal(index, key, value, right);
            Fit
        } else {
            // The element can't fit, this node is full. Split it into two nodes.
            let (new_key, new_val, mut new_right) = self.split();
            let left_len = self.length;

            if index <= left_len {
                self.insert_fit_as_internal(index, key, value, right);
            } else {
                new_right.insert_fit_as_internal(index - left_len - 1, key, value, right);
            }

            Split(new_key, new_val, new_right)
        }
    }

    /// We have somehow verified that this key-value pair will fit in this internal node,
    /// so insert under that assumption.
    fn insert_fit_as_leaf(&mut self, index: uint, key: K, value: V) {
        let len = self.length;
        shift_and_insert(self.keys.mut_slice_to(len + 1), index, Some(key));
        shift_and_insert(self.vals.mut_slice_to(len + 1), index, Some(value));
        self.length += 1;
    }

    /// We have somehow verified that this key-value pair will fit in this internal node,
    /// so insert under that assumption
    fn insert_fit_as_internal(&mut self, index: uint, key: K, value: V, right: Box<Node<K,V>>) {
        let len = self.length;
        shift_and_insert(self.keys.mut_slice_to(len + 1), index, Some(key));
        shift_and_insert(self.vals.mut_slice_to(len + 1), index, Some(value));
        shift_and_insert(self.edges.mut_slice_to(len + 2), index + 1, Some(right));
        self.length += 1;
    }

    /// node is full, so split it into two nodes, and yield the middle-most key-value pair
    /// because we have one too many, and our parent now has one too few
    fn split(&mut self) -> (K, V, Box<Node<K, V>>) {
        let mut right = box Node::new();

        steal_last(self.vals.as_mut_slice(), right.vals.as_mut_slice(), SPLIT_LEN);
        steal_last(self.keys.as_mut_slice(), right.keys.as_mut_slice(), SPLIT_LEN);
        // FIXME(Gankro): This isn't necessary for leaf nodes
        steal_last(self.edges.as_mut_slice(), right.edges.as_mut_slice(), SPLIT_LEN + 1);

        // How much each node got
        let left_len = CAPACITY - SPLIT_LEN;
        let right_len = SPLIT_LEN;

        // But we're gonna pop one off the end of the left one, so subtract one
        self.length = left_len - 1;
        right.length = right_len;

        // Pop it
        let key = self.keys[left_len - 1].take().unwrap();
        let val = self.vals[left_len - 1].take().unwrap();

        (key, val, right)
    }


    /// Remove the key-value pair at the given index
    fn remove_as_leaf(&mut self, index: uint) -> (K, V) {
        let len = self.length;
        let key = remove_and_shift(self.keys.mut_slice_to(len), index).unwrap();
        let value = remove_and_shift(self.vals.mut_slice_to(len), index).unwrap();
        self.length -= 1;
        (key, value)
    }

    /// Handle an underflow in this node's child. We favour handling "to the left" because we know
    /// we're empty, but our neighbour can be full. Handling to the left means when we choose to
    /// steal, we pop off the end of our neighbour (always fast) and "unshift" ourselves
    /// (always slow, but at least faster since we know we're half-empty).
    /// Handling "to the right" reverses these roles. Of course, we merge whenever possible
    /// because we want dense nodes, and merging is about equal work regardless of direction.
    fn handle_underflow(&mut self, underflowed_child_index: uint) {
        if underflowed_child_index > 0 {
            self.handle_underflow_to_left(underflowed_child_index);
        } else {
            self.handle_underflow_to_right(underflowed_child_index);
        }
    }

    fn handle_underflow_to_left(&mut self, underflowed_child_index: uint) {
        // Right is underflowed. Try to steal from left,
        // but merge left and right if left is low too.
        let mut left = self.edges[underflowed_child_index - 1].take().unwrap();
        let left_len = left.length;
        if left_len > MIN_LOAD {
            // Steal! Stealing is roughly analagous to a binary tree rotation.
            // In this case, we're "rotating" right.

            // Take the biggest stuff off left
            let mut key = remove_and_shift(left.keys.mut_slice_to(left_len), left_len - 1);
            let mut val = remove_and_shift(left.vals.mut_slice_to(left_len), left_len - 1);
            let edge = remove_and_shift(left.edges.mut_slice_to(left_len + 1), left_len);
            left.length -= 1;

            // Swap the parent's seperating key-value pair with left's
            mem::swap(&mut self.keys[underflowed_child_index - 1], &mut key);
            mem::swap(&mut self.vals[underflowed_child_index - 1], &mut val);

            // Put them at the start of right
            {
                let right = self.edges[underflowed_child_index].as_mut().unwrap();
                let right_len = right.length;
                shift_and_insert(right.keys.mut_slice_to(right_len + 1), 0, key);
                shift_and_insert(right.vals.mut_slice_to(right_len + 1), 0, val);
                shift_and_insert(right.edges.mut_slice_to(right_len + 2), 0, edge);
                right.length += 1;
            }

            // Put left back where we found it
            self.edges[underflowed_child_index - 1] = Some(left);
        } else {
            // Merge! Left and right will be smooshed into one node, along with the key-value
            // pair that seperated them in their parent.
            let len = self.length;

            // Permanently remove left's index, and the key-value pair that seperates
            // left and right
            let key = remove_and_shift(self.keys.mut_slice_to(len), underflowed_child_index - 1);
            let val = remove_and_shift(self.vals.mut_slice_to(len), underflowed_child_index - 1);
            remove_and_shift(self.edges.mut_slice_to(len + 1), underflowed_child_index - 1);

            self.length -= 1;

            // Give left right's stuff, and put left where right was. Note that all the indices
            // in the parent have been shifted left at this point.
            let right = self.edges[underflowed_child_index - 1].take().unwrap();
            left.absorb(key, val, right);
            self.edges[underflowed_child_index - 1] = Some(left);
        }
    }

    fn handle_underflow_to_right(&mut self, underflowed_child_index: uint) {
        // Left is underflowed. Try to steal from the right,
        // but merge left and right if right is low too.
        let mut right = self.edges[underflowed_child_index + 1].take().unwrap();
        let right_len = right.length;
        if right_len > MIN_LOAD {
            // Steal! Stealing is roughly analagous to a binary tree rotation.
            // In this case, we're "rotating" left.

            // Take the smallest stuff off right
            let mut key = remove_and_shift(right.keys.mut_slice_to(right_len), 0);
            let mut val = remove_and_shift(right.vals.mut_slice_to(right_len), 0);
            let edge = remove_and_shift(right.edges.mut_slice_to(right_len + 1), 0);
            right.length -= 1;

            // Swap the parent's seperating key-value pair with right's
            mem::swap(&mut self.keys[underflowed_child_index], &mut key);
            mem::swap(&mut self.vals[underflowed_child_index], &mut val);

            // Put them at the end of left
            {
                let left = self.edges[underflowed_child_index].as_mut().unwrap();
                let left_len = left.length;
                shift_and_insert(left.keys.mut_slice_to(left_len + 1), left_len, key);
                shift_and_insert(left.vals.mut_slice_to(left_len + 1), left_len, val);
                shift_and_insert(left.edges.mut_slice_to(left_len + 2), left_len + 1, edge);
                left.length += 1;
            }

            // Put right back where we found it
            self.edges[underflowed_child_index + 1] = Some(right);
        } else {
            // Merge! Left and right will be smooshed into one node, along with the key-value
            // pair that seperated them in their parent.
            let len = self.length;

            // Permanently remove right's index, and the key-value pair that seperates
            // left and right
            let key = remove_and_shift(self.keys.mut_slice_to(len), underflowed_child_index);
            let val = remove_and_shift(self.vals.mut_slice_to(len), underflowed_child_index);
            remove_and_shift(self.edges.mut_slice_to(len + 1), underflowed_child_index + 1);

            self.length -= 1;

            // Give left right's stuff. Note that unlike handle_underflow_to_left, we don't need
            // to compensate indices, and we don't need to put left "back".
            let left = self.edges[underflowed_child_index].as_mut().unwrap();
            left.absorb(key, val, right);
        }
    }

    /// Take all the values from right, seperated by the given key and value
    fn absorb(&mut self, key: Option<K>, value: Option<V>, mut right: Box<Node<K,V>>) {
        let len = self.length;
        let r_len = right.length;

        self.keys[len] = key;
        self.vals[len] = value;

        merge(self.keys.mut_slice_to(len + r_len + 1), right.keys.mut_slice_to(r_len));
        merge(self.vals.mut_slice_to(len + r_len + 1), right.vals.mut_slice_to(r_len));
        merge(self.edges.mut_slice_to(len + r_len +  2), right.edges.mut_slice_to(r_len + 1));

        self.length += r_len + 1;
    }
}

struct Keys<'a, K>{
    it: Items<'a, Option<K>>
}

impl<'a, K> Iterator<&'a K> for Keys<'a, K> {
    fn next(&mut self) -> Option<&'a K> {
        self.it.next().and_then(|x| x.as_ref())
    }
}

/// Subroutine for removal. Takes a search stack for a key that terminates at an
/// internal node, and mutates the tree and search stack to make it a search
/// stack for that key that terminates at a leaf. This leaves the tree in an inconsistent
/// state that must be repaired by the caller by removing the key in question.
fn leafify_stack<K,V>(stack: &mut SearchStack<K,V>) {
    let (node_ptr, index) = stack.pop().unwrap();
    unsafe {
        // First, get ptrs to the found key-value pair
        let node = &mut *node_ptr;
        let (key_ptr, val_ptr) = {
            (&mut node.keys[index] as *mut _, &mut node.vals[index] as *mut _)
        };

        // Go into the right subtree of the found key
        stack.push((node_ptr, index + 1));
        let mut temp_node = &mut **node.edges[index + 1].as_mut().unwrap();

        loop {
            // Walk into the smallest subtree of this
            let node = temp_node;
            let node_ptr = node as *mut _;
            stack.push((node_ptr, 0));
            let next = node.edges[0].as_mut();
            if next.is_some() {
                // This node is internal, go deeper
                temp_node = &mut **next.unwrap();
            } else {
                // This node is a leaf, do the swap and return
                mem::swap(&mut *key_ptr, &mut node.keys[0]);
                mem::swap(&mut *val_ptr, &mut node.vals[0]);
                break;
            }
        }
    }
}

/// Basically `Vec.insert(index)`. Assumes that the last element in the slice is
/// Somehow "empty" and can be overwritten.
fn shift_and_insert<T>(slice: &mut [T], index: uint, elem: T) {
    // FIXME(Gankro): This should probably be a copy_memory and a write?
    for i in range(index, slice.len() - 1).rev() {
        slice.swap(i, i + 1);
    }
    slice[index] = elem;
}

/// Basically `Vec.remove(index)`.
fn remove_and_shift<T>(slice: &mut [Option<T>], index: uint) -> Option<T> {
    let result = slice[index].take();
    // FIXME(Gankro): This should probably be a copy_memory and write?
    for i in range(index, slice.len() - 1) {
        slice.swap(i, i + 1);
    }
    result
}

/// Subroutine for splitting a node. Put the `SPLIT_LEN` last elements from left,
/// (which should be full) and put them at the start of right (which should be empty)
fn steal_last<T>(left: &mut[T], right: &mut[T], amount: uint) {
    // Is there a better way to do this?
    // Maybe copy_nonoverlapping_memory and then bulk None out the old Location?
    let offset = left.len() - amount;
    for (a,b) in left.mut_slice_from(offset).mut_iter()
            .zip(right.mut_slice_to(amount).mut_iter()) {
        mem::swap(a, b);
    }
}

/// Subroutine for merging the contents of right into left
/// Assumes left has space for all of right
fn merge<T>(left: &mut[Option<T>], right: &mut[Option<T>]) {
    let left_len = left.len();
    let right_len = right.len();
    for i in range(0, right_len) {
        left[left_len - right_len + i] = right[i].take();
    }
}

impl<K,V> Collection for BTree<K,V>{
    fn len(&self) -> uint {
        self.length
    }
}

impl<K,V> Mutable for BTree<K,V> {
    fn clear(&mut self) {
        // Note that this will trigger a lot of recursive destructors, but BTrees can't get
        // very deep, so we won't worry about it for now.
        self.root = None;
        self.length = 0;
        self.depth = 0;
    }
}







#[cfg(test)]
mod test {
    use std::prelude::*;

    use super::BTree;
    use {Map, MutableMap, Mutable, MutableSeq};

    #[test]
    fn test_basic() {
        let mut map = BTree::new();
        assert_eq!(map.len(), 0);

        for i in range(0u, 10000) {
            assert_eq!(map.swap(i, 10*i), None);
            assert_eq!(map.len(), i + 1);
        }

        for i in range(0u, 10000) {
            assert_eq!(map.find(&i).unwrap(), &(i*10));
        }

        for i in range(10000, 20000) {
            assert_eq!(map.find(&i), None);
        }

        for i in range(0u, 10000) {
            assert_eq!(map.swap(i, 100*i), Some(10*i));
            assert_eq!(map.len(), 10000);
        }

        for i in range(0u, 10000) {
            assert_eq!(map.find(&i).unwrap(), &(i*100));
        }

        for i in range(0u, 5000) {
            assert_eq!(map.pop(&(i*2)), Some(i*200));
            assert_eq!(map.len(), 10000 - i - 1);
        }

        for i in range(0u, 5000) {
            assert_eq!(map.find(&(2*i)), None);
            assert_eq!(map.find(&(2*i+1)).unwrap(), &(i*200 + 100));
        }

        for i in range(0u, 5000) {
            assert_eq!(map.pop(&(2*i)), None);
            assert_eq!(map.pop(&(2*i+1)), Some(i*200 + 100));
            assert_eq!(map.len(), 5000 - i - 1);
        }
    }
}




#[cfg(test)]
mod bench {
    use test::Bencher;

    use super::BTree;
    use deque::bench::{insert_rand_n, insert_seq_n, find_rand_n, find_seq_n};

    // Find seq
    #[bench]
    pub fn insert_rand_100(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        insert_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_rand_10_000(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        insert_rand_n(10_000, &mut m, b);
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        insert_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn insert_seq_10_000(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        insert_seq_n(10_000, &mut m, b);
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        find_rand_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_rand_10_000(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        find_rand_n(10_000, &mut m, b);
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        find_seq_n(100, &mut m, b);
    }

    #[bench]
    pub fn find_seq_10_000(b: &mut Bencher) {
        let mut m : BTree<uint,uint> = BTree::new();
        find_seq_n(10_000, &mut m, b);
    }
}
