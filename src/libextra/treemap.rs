// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An ordered map and set implemented as self-balancing binary search
//! trees. The only requirement for the types is that the key implements
//! `TotalOrd`.

use std::uint;

pub use self::own::{TreeMap, TreeSet,
    TreeMapIterator, TreeMapRevIterator, TreeMapMoveIterator,
    TreeSetIterator, TreeSetRevIterator, TreeSetMoveIterator,
    TreeMapHandleIterator,
    Difference, SymDifference, Intersection, Union};

// This is implemented as an AA tree, which is a simplified variation of
// a red-black tree where red (horizontal) nodes can only be added
// as a right child. The time complexity is the same, and re-balancing
// operations are more frequent but also cheaper.

// Future improvements:

// range search - O(log n) retrieval of an iterator from some key

// (possibly) implement the overloads Python does for sets:
//   * intersection: &
//   * difference: -
//   * symmetric difference: ^
//   * union: |
// These would be convenient since the methods work like `each`

/// structure representing a path into a balanced binary tree
/// this supports up to 62/126 elements which is enough given the balance assumption

trait TreePathBuilder {
     /// push a descent direction into the path
    fn push(&mut self, b: bool);

    /// pop a descent direction from the path
    fn pop(&mut self);
}

#[deriving(Clone, Default, Eq)]
struct ForgetTreePath;

impl TreePathBuilder for ForgetTreePath {
    fn push(&mut self, _b: bool) {}
    fn pop(&mut self) {}
}

#[deriving(Clone, Eq)]
struct TreePath
{
    v: uint, // 0001hijk, or 000001ab
    r: uint, // 1abcdefg, or 0
}

impl TreePath
{
    /// create an empty TreePath
    pub fn new() -> TreePath
    {
        TreePath {v: 1, r: 0}
    }

    fn push_out_of_space(&mut self)
    {
        if self.r != 0 {
            fail2!("TreePath too long for a balanced binary tree in RAM");
        }
        self.r = self.v;
        self.v = 1;
    }

    fn pop_out_of_space(&mut self)
    {
        if self.r != 0 {
            fail2!("popping from empty TreePath");
        }
        self.v = self.r;
        self.r = 0;
    }

    /// return an iterator
    pub fn iter<'a>(&'a self) -> TreePathIterator
    {
        let z = self.v.leading_zeros();
        let v = (self.v + self.v + 1) << z;
        if self.r == 0 {
            TreePathIterator {v: v, r: 0}
        } else {
            TreePathIterator {v: self.r + self.r + 1, r: v}
        }
    }
}

impl Default for TreePath {
    fn default() -> TreePath {TreePath::new()}
}

impl TreePathBuilder for TreePath {
    #[inline]
    fn push(&mut self, b: bool)
    {
        if((self.v as int) < 0) {
            self.push_out_of_space();
        }
        self.v = self.v + self.v + (b as uint);
    }
    
    #[inline]
    fn pop(&mut self)
    {
        self.v >>= 1;
        if(self.v == 1) {
            self.pop_out_of_space();
        }
    }
}

struct TreePathIterator
{
    v: uint, // abcde100,, or 0
    r: uint, // fghi1000, or 0
}

impl Iterator<bool> for TreePathIterator {
    #[inline]
    fn next(&mut self) -> Option<bool> {
        let p = self.v;
        self.v = p + p;
        if self.v != 0 {
            Some((p as int) < 0)
        } else if p != 0 && self.r != 0 {
            let p = self.r;
            self.r = 0;
            self.v = p + p;
            Some((p as int) < 0)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some((uint::bits - 1) * 2))
    }
}

#[deriving(Clone)]
enum TreeSubset<T>
{
    EmptyTreeSubset,
    SubTree(T),
    InOrderFrom(T)
}

macro_rules! iterator {
    ($mod_name:ident, $Name:ident, $RM:ty, $ROP:ty, $RP:ty, $RK:ty, $RV:ty, $get:ident, $as_ref:ident, $destructure: ident, $ctor:ident, $Clone:ident) => {
        mod $mod_name {
            use std::cmp::Ordering;
            use std::rc::Rc;
            use std::owned::Own;
            use arc::Arc;

            use super::super::{TreePath, TreePathIterator, TreePathBuilder, ForgetTreePath, TreeSubset, EmptyTreeSubset, SubTree, InOrderFrom, Dummy};
            use super::{TreeMap, TreeNode, TreeDir, TreeLeft, TreeRight};

            /// Lazy forward iterator over a map
            /// iterates over the subtree pointed to by node, then over stack in pop order,
            /// visiting the elements of stack itself and their right (forward)
            /// or left (for backward) subtrees
            pub struct $Name<'self, K, V, D, TP> {
                priv stack: ~[($ROP, $RK, $RV, TP)],
                priv node: Option<($RP, TP)>,
                priv remaining_min: uint,
                priv remaining_max: uint,
                priv dir: D
            }

            impl<'self, K, V, D: TreeDir, TP: Default> $Name<'self, K, V, D, TP> {
                /// create a new iterator
               pub fn new(root: $ROP, len: uint) -> $Name<'self, K, V, D, TP> {
                    $Name {
                        stack: ~[],
                        node: match root.$as_ref() {Some(root) => Some((root, Default::default())), None => None},
                        remaining_min: len,
                        remaining_max: len,
                        dir: TreeDir::new()
                    }
                }
            }
            
            impl<'self, K: TotalOrd + $Clone, V: $Clone, D: TreeDir, TP: TreePathBuilder + Default + Clone> $Name<'self, K, V, D, TP> {
                /// create a new iterator starting at a specific key
                pub fn new_at(root: $ROP, len: uint, k: &K, which: Ordering) -> $Name<'self, K, V, D, TP> {
                    let mut iter: $Name<'self, K, V, D, TP> = $Name::new(root, len);
                    loop {
                        let dir = match iter.node {
                          Some((ref mut rr, _)) => {
                            let r = rr.get();
                            match (*k).cmp(&r.key) {
                                Equal => which,
                                x => x
                            }
                          }
                          None => {
                            Equal
                          }
                        };
                        match dir {
                          Less => iter.traverse::<TreeLeft>(),
                          Greater => iter.traverse::<TreeRight>(),
                          Equal => {
                            iter.traverse_complete();
                            return iter;
                          }
                        }
                    }
                }                    
            }
            
            impl<'self, K: $Clone, V: $Clone, D: TreeDir, TP: TreePathBuilder + Clone> $Name<'self, K, V, D, TP> {
                /// internal: return next element and tree path to it
                pub fn next_with_path(&mut self) -> Option<($RK, $RV, TP)> {
                    while !self.stack.is_empty() || self.node.is_some() {
                        match self.node.take() {
                          Some((x, path_)) => {
                            let mut path = path_;
                            let (key, value, left, right) = x.$get().$destructure();
                            let (to_stack, to_node) = if self.dir.is_right() {(right, left)} else {(left, right)};
                            self.stack.push((to_stack, key, value, path.clone()));
                            self.node = match to_node.$as_ref() {
                                Some(node) => {
                                    path.push(self.dir.is_left());
                                    Some((node, path))
                                },
                                None => None
                            };
                          }
                          None => {
                            let (next, key, value, path) = self.stack.pop();
                            self.node = match next.$as_ref() {
                                Some(node) => {
                                    let mut node_path = path.clone();
                                    node_path.push(self.dir.is_right());
                                    Some((node, node_path))
                                }
                                None => None
                            };
                            self.remaining_max -= 1;
                            if self.remaining_min > 0 {
                                self.remaining_min -= 1;
                            }
                            return Some((key, value, path));
                          }
                        }
                    }
                    None
                }
                
                /// traverse_left, traverse_right and traverse_complete are used to
                /// initialize TreeMapIterator pointing to element inside tree structure.
                ///
                /// They should be used in following manner:
                ///   - create iterator using TreeMap::iter_for_traversal
                ///   - find required node using `traverse_left`/`traverse_right`
                ///     (current node is `TreeMapIterator::node` field)
                ///   - complete initialization with `traverse_complete`
                #[inline]
                fn traverse<TD: TreeDir>(&mut self) {
                    let dir: TD = TreeDir::new();
                    match self.node.take() {
                        None => fail2!(),
                        Some((node, path_)) => {
                            let mut path = path_;
                            let (key, value, left, right) = node.$get().$destructure();
                            let to_node = if dir.is_right() != self.dir.is_right() {
                                if dir.is_right() {
                                    self.stack.push((left, key, value, path.clone()));
                                    right
                                } else {
                                    self.stack.push((right, key, value, path.clone()));
                                    left
                                }
                            } else {
                                self.remaining_min = 0;
                                if dir.is_right() {right} else {left}
                            };
                            
                            self.node = match to_node.$as_ref() {
                                None => None,
                                Some(node) => {
                                    path.push(dir.is_right());
                                    Some((node, path))
                                }
                            }
                        }
                    }
                }

                /// traverse_left, traverse_right and traverse_complete are used to
                /// initialize TreeMapIterator pointing to element inside tree structure.
                ///
                /// Completes traversal. Should be called before using iterator.
                /// Iteration will start from `self.node`.
                /// If `self.node` is None iteration will start from last node from which we
                /// traversed left.
                #[inline]
                fn traverse_complete(&mut self) {
                    match self.node.take() {
                        Some((node, path)) => {
                            let (key, value, left, right) = node.$get().$destructure();
                            self.stack.push((if self.dir.is_right() {right} else {left}, key, value, path));
                            self.node = None;
                            self.remaining_min = 0;
                        }
                        None => ()
                    }
                }
            }

            impl<'self, K: $Clone, V: $Clone, D: TreeDir, TP: TreePathBuilder + Clone> Iterator<($RK, $RV)> for $Name<'self, K, V, D, TP> {
                /// Advance the iterator to the next node (in order) and return a
                /// tuple with a reference to the key and value. If there are no
                /// more nodes, return `None`.
                fn next(&mut self) -> Option<($RK, $RV)> {
                    match self.next_with_path() {
                        None => None,
                        Some((k, v, _)) => Some((k, v))
                    }
                }

                #[inline]
                fn size_hint(&self) -> (uint, Option<uint>) {
                    (self.remaining_min, Some(self.remaining_max))
                }
            }

            impl<'self, K, V, D: TreeDir, TP: TreePathBuilder + Default + Clone> $Name<'self, K, V, D, TP> {
                /// return the subset that identifies the items in this iterator
                pub fn get_subset(&self) -> TreeSubset<TP> {
                    match self.node {
                        Some((_, ref path)) => SubTree(path.clone()),
                        None => match self.stack.len() {
                            0 => EmptyTreeSubset,
                            len => match self.stack[len - 1] {
                                (_, _, _, ref path) => InOrderFrom(path.clone())
                            }
                        }
                    }
                }
            }

            impl<'self, K: TotalOrd + $Clone, V: $Clone, D: TreeDir> $Name<'self, K, V, D, TreePath> {
                /// create an iterator based on a subset
                pub fn from_subset<'a>(map: $RM, subset: TreeSubset<TreePath>)
                    -> $Name<'a, K, V, D, TreePath> {
                    let mut iter = map.$ctor();
                    match subset {
                        EmptyTreeSubset => iter.node = None,
                        SubTree(ref path) | InOrderFrom(ref path) => {
                            for dir in path.iter() {
                                if !dir {
                                    iter.traverse::<TreeLeft>();
                                } else {
                                    iter.traverse::<TreeRight>();
                                }
                            }
                        }
                    }
                    
                    match subset {
                        InOrderFrom(_) => iter.traverse_complete(),
                        _ => {}
                    };
                    
                    iter
                }
            }
        }
    }
}

macro_rules! treemap {
    // we can't use multiplicity for New because the separator token would be +,
    // which conflicts with the Kleene star plus, and cannot be escaped
    ($mod_name:ident, $P:ty, $new:expr, $Clone:ident, $New1:ident + $New2:ident) => {
        mod $mod_name {
            #[allow(unused_imports)];

            use std::util::{swap, replace};
            use std::iter::{Peekable};
            use std::cmp::Ordering;
            use std::rc::Rc;
            use std::owned::Own;
            use arc::Arc;
            use std::cast::transmute;

            use super::{TreePath, TreePathIterator, TreePathBuilder, ForgetTreePath, TreeSubset, EmptyTreeSubset, SubTree, InOrderFrom, Dummy};
            
            pub use self::base_iter::TreeMapBaseIterator;
            pub use self::mut_iter::TreeMapMutBaseIterator;
            
            #[allow(missing_doc)]
            #[deriving(Clone)]
            pub struct TreeMap<K, V> {
                priv root: Option<$P>,
                priv length: uint
            }

            impl<K: Eq + TotalOrd, V: Eq> Eq for TreeMap<K, V> {
                fn eq(&self, other: &TreeMap<K, V>) -> bool {
                    self.len() == other.len() &&
                        self.iter().zip(other.iter()).all(|(a, b)| a == b)
                }
            }

            // Lexicographical comparison
            fn lt<K: Ord + TotalOrd, V: Ord>(a: &TreeMap<K, V>,
                                             b: &TreeMap<K, V>) -> bool {
                // the Zip iterator is as long as the shortest of a and b.
                for ((key_a, value_a), (key_b, value_b)) in a.iter().zip(b.iter()) {
                    if *key_a < *key_b { return true; }
                    if *key_a > *key_b { return false; }
                    if *value_a < *value_b { return true; }
                    if *value_a > *value_b { return false; }
                }

                a.len() < b.len()
            }

            impl<K: Ord + TotalOrd, V: Ord> Ord for TreeMap<K, V> {
                #[inline]
                fn lt(&self, other: &TreeMap<K, V>) -> bool { lt(self, other) }
            }

            impl<K: TotalOrd, V> Container for TreeMap<K, V> {
                /// Return the number of elements in the map
                fn len(&self) -> uint { self.length }

                /// Return true if the map contains no elements
                fn is_empty(&self) -> bool { self.root.is_none() }
            }

            impl<K: TotalOrd, V> Mutable for TreeMap<K, V> {
                /// Clear the map, removing all key-value pairs.
                fn clear(&mut self) {
                    self.root = None;
                    self.length = 0
                }
            }

            impl<K: TotalOrd, V> Map<K, V> for TreeMap<K, V> {
                /// Return a reference to the value corresponding to the key
                fn find<'a>(&'a self, key: &K) -> Option<&'a V> {
                    let mut current: &'a Option<$P> = &self.root;
                    loop {
                        match *current {
                          Some(ref rr) => {
                            let r = rr.get();
                            match key.cmp(&r.key) {
                              Less => current = &r.left,
                              Greater => current = &r.right,
                              Equal => return Some(&r.value)
                            }
                          }
                          None => return None
                        }
                    }
                }
            }

            impl<K: TotalOrd + $Clone + $New1 + $New2, V: $Clone + $New1 + $New2>
                MutableMap<K, V> for TreeMap<K, V> {
                /// Return a mutable reference to the value corresponding to the key
                #[inline]
                fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
                    find_mut(&mut self.root, key)
                }

                /// Insert a key-value pair from the map. If the key already had a value
                /// present in the map, that value is returned. Otherwise None is returned.
                fn swap(&mut self, key: K, value: V) -> Option<V> {
                    let ret = insert(&mut self.root, key, value);
                    if ret.is_none() { self.length += 1 }
                    ret
                }

                /// Removes a key from the map, returning the value at the key if the key
                /// was previously in the map.
                fn pop(&mut self, key: &K) -> Option<V> {
                    let ret = remove(&mut self.root, key);
                    if ret.is_some() { self.length -= 1 }
                    ret
                }
            }

            impl<K: TotalOrd, V> TreeMap<K, V> {
                /// Create an empty TreeMap
                pub fn new() -> TreeMap<K, V> { TreeMap{root: None, length: 0} }

                fn iter_<'a, D: TreeDir, TP: Default>(&'a self) -> TreeMapBaseIterator<'a, K, V, D, TP> {
                    TreeMapBaseIterator::new(&self.root, self.length)
                }

                fn iter_at_<'a, D: TreeDir, TP: TreePathBuilder + Clone + Default>(&'a self, k: &K, which: Ordering)
                    -> TreeMapBaseIterator<'a, K, V, D, TP> {
                    TreeMapBaseIterator::new_at(&self.root, self.length, k, which)
                }

                fn handle_iter_<'a, D: TreeDir>(&'a mut self) -> TreeMapHandleIterator<'a, K, V, D> {
                    TreeMapHandleIterator {map: self, state: Left(self.iter_()), cache: None}
                }

                fn handle_iter_at_<'a, D: TreeDir>(&'a mut self, k: &K, which: Ordering)
                    -> TreeMapHandleIterator<'a, K, V, D> {
                    TreeMapHandleIterator {map: self, state: Left(self.iter_at_(k, which)), cache: None}
                }

                /// Get a lazy iterator over the key-value pairs in the map.
                /// Requires that it be frozen (immutable).
                pub fn iter<'a>(&'a self) -> TreeMapIterator<'a, K, V> {
                    self.iter_()
                }

                /// Get a lazy reverse iterator over the key-value pairs in the map.
                /// Requires that it be frozen (immutable).
                pub fn rev_iter<'a>(&'a self) -> TreeMapRevIterator<'a, K, V> {
                    self.iter_()
                }

                /// Lazy iterator to the first key-value pair whose key is not less than `k`
                /// If all keys in map are less than `k` an empty iterator is returned.
                pub fn lower_bound_iter<'a>(&'a self, k: &K) -> TreeMapIterator<'a, K, V> {
                    self.iter_at_(k, Equal)
                }

                /// Lazy iterator to the first key-value pair whose key is greater than `k`
                /// If all keys in map are not greater than `k` an empty iterator is returned.
                pub fn upper_bound_iter<'a>(&'a self, k: &K) -> TreeMapIterator<'a, K, V> {
                    self.iter_at_(k, Greater)
                }

                /// Lazy rev iterator to the last key-value pair whose key is not greater than `k`
                /// If all keys in map are greater than `k` an empty iterator is returned.
                pub fn rev_lower_bound_iter<'a>(&'a self, k: &K) -> TreeMapRevIterator<'a, K, V> {
                    self.iter_at_(k, Equal)
                }

                /// Lazy rev iterator to the first key-value pair whose key is less than `k`
                /// If all keys in map are not less than `k` an empty iterator is returned.
                pub fn rev_upper_bound_iter<'a>(&'a self, k: &K) -> TreeMapRevIterator<'a, K, V> {
                    self.iter_at_(k, Less)
                }

                /// Get a lazy iterator over the key-value pairs in the map.
                pub fn handle_iter<'a>(&'a mut self) -> TreeMapHandleIterator<'a, K, V, TreeRight> {
                    self.handle_iter_()
                }

                /// Get a lazy reverse iterator over the key-value pairs in the map.
                pub fn handle_rev_iter<'a>(&'a mut self) -> TreeMapHandleIterator<'a, K, V, TreeLeft> {
                    self.handle_iter_()
                }

                /// Lazy iterator to the first key-value pair whose key is not less than `k`
                /// If all keys in map are less than `k` an empty iterator is returned.
                pub fn handle_lower_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapHandleIterator<'a, K, V, TreeRight> {
                    self.handle_iter_at_(k, Equal)
                }

                /// Lazy iterator to the first key-value pair whose key is greater than `k`
                /// If all keys in map are not greater than `k` an empty iterator is returned.
                pub fn handle_upper_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapHandleIterator<'a, K, V, TreeRight> {
                    self.handle_iter_at_(k, Greater)
                }

                /// Lazy rev iterator to the last key-value pair whose key is not greater than `k`
                /// If all keys in map are greater than `k` an empty iterator is returned.
                pub fn handle_rev_lower_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapHandleIterator<'a, K, V, TreeLeft> {
                    self.handle_iter_at_(k, Equal)
                }

                /// Lazy rev iterator to the first key-value pair whose key is less than `k`
                /// If all keys in map are not less than `k` an empty iterator is returned.
                pub fn handle_rev_upper_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapHandleIterator<'a, K, V, TreeLeft> {
                    self.handle_iter_at_(k, Less)
                }
            }

            impl<K: TotalOrd + $Clone, V: $Clone> TreeMap<K, V> {
                fn mut_iter_<'a, D: TreeDir, TP: Default>(&'a mut self) -> TreeMapMutBaseIterator<'a, K, V, D, TP> {
                    TreeMapMutBaseIterator::new(&mut self.root, self.length)
                }

                fn mut_iter_at_<'a, D: TreeDir, TP: TreePathBuilder + Clone + Default>(&'a mut self, k: &K, which: Ordering)
                    -> TreeMapMutBaseIterator<'a, K, V, D, TP> {
                    TreeMapMutBaseIterator::new_at(&mut self.root, self.length, k, which)
                }
                
                /// Get a lazy iterator over the key-value pairs in the map.
                pub fn mut_iter<'a>(&'a mut self) -> TreeMapMutIterator<'a, K, V> {
                    self.mut_iter_()
                }

                /// Get a lazy reverse iterator over the key-value pairs in the map.
                pub fn mut_rev_iter<'a>(&'a mut self) -> TreeMapMutRevIterator<'a, K, V> {
                    self.mut_iter_()
                }

                /// Lazy iterator to the first key-value pair whose key is not less than `k`
                /// If all keys in map are less than `k` an empty iterator is returned.
                pub fn mut_lower_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapMutIterator<'a, K, V> {
                    self.mut_iter_at_(k, Equal)
                }

                /// Lazy iterator to the first key-value pair whose key is greater than `k`
                /// If all keys in map are not greater than `k` an empty iterator is returned.
                pub fn mut_upper_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapMutIterator<'a, K, V> {
                    self.mut_iter_at_(k, Greater)
                }

                /// Lazy rev iterator to the last key-value pair whose key is not greater than `k`
                /// If all keys in map are greater than `k` an empty iterator is returned.
                pub fn mut_rev_lower_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapMutRevIterator<'a, K, V> {
                    self.mut_iter_at_(k, Equal)
                }

                /// Lazy rev iterator to the first key-value pair whose key is less than `k`
                /// If all keys in map are not less than `k` an empty iterator is returned.
                pub fn mut_rev_upper_bound_iter<'a>(&'a mut self, k: &K) -> TreeMapMutRevIterator<'a, K, V> {
                    self.mut_iter_at_(k, Less)
                }

                /// Get a lazy iterator that consumes the TreeMap.
                pub fn move_iter(self) -> TreeMapMoveIterator<K, V> {
                    let TreeMap { root: root, length: length } = self;
                    let stk = match root {
                        None => ~[],
                        Some(tn) => ~[tn.value()]
                    };
                    TreeMapMoveIterator {
                        stack: stk,
                        remaining: length
                    }
                }
            }

            iterator!(base_iter, TreeMapBaseIterator, &'a TreeMap<K, V>, &'self Option<$P>, &'self $P, &'self K, &'self V, get, as_ref, destructure, iter_, Dummy)
            iterator!(mut_iter, TreeMapMutBaseIterator, &'a mut TreeMap<K, V>, &'self mut Option<$P>, &'self mut $P, &'self K, &'self mut V, cow, as_mut, destructure_mut, mut_iter_, $Clone)

            pub type TreeMapIterator<'self, K, V> = TreeMapBaseIterator<'self, K, V, TreeRight, ForgetTreePath>;
            pub type TreeMapRevIterator<'self, K, V> = TreeMapBaseIterator<'self, K, V, TreeLeft, ForgetTreePath>;
            pub type TreeMapMutIterator<'self, K, V> = TreeMapMutBaseIterator<'self, K, V, TreeRight, ForgetTreePath>;
            pub type TreeMapMutRevIterator<'self, K, V> = TreeMapMutBaseIterator<'self, K, V, TreeLeft, ForgetTreePath>;

            /// mutable iterator for TreeMap
            pub struct TreeMapHandleIterator<'self, K, V, D> {
                priv map: &'self TreeMap<K, V>,
                priv state: Either<TreeMapBaseIterator<'self, K, V, D, TreePath>, TreeSubset<TreePath>>,
                priv cache: Option<(&'self K, &'self V, TreePath)>,
            }
            
            /// handle to a key-value pair in the treemap
            pub struct TreeMapHandle<'self, K, V, D> {
                priv iter: *TreeMapHandleIterator<'self, K, V, D>,
                priv path: TreePath
            }

            impl<'self, K: TotalOrd + $Clone, V: $Clone, D: TreeDir> Iterator<TreeMapHandle<'self, K, V, D>> for TreeMapHandleIterator<'self, K, V, D> {
                // unfortunately we can't return the pointers, because we can't put a lifetime parameter here
                fn next(&mut self) -> Option<TreeMapHandle<'self, K, V, D>> {
                    match self.next() {
                        None => None,
                        Some((_, _, handle)) => Some(handle),
                    }
                }
            }
            
            impl<'self, K: TotalOrd + $Clone, V: $Clone, D: TreeDir> TreeMapHandleIterator<'self, K, V, D> {                
                /// returns the next item
                pub fn next<'a>(&'a mut self) -> Option<(&'a K, &'a V, TreeMapHandle<'self, K, V, D>)> {
                    let res = self.get_iter().next_with_path();
                    if res.is_none() {
                        return None
                    }
                    self.cache = res;
                    let (key, value, ref path) = self.cache.unwrap();
                    return Some((key, value, TreeMapHandle {iter: self as *mut TreeMapHandleIterator<'self, K, V, D> as *TreeMapHandleIterator<'self, K, V, D>, path: path.clone()}))
                }

                fn get_iter<'a>(&'a mut self) -> &'a mut TreeMapBaseIterator<'self, K, V, D, TreePath> {
                    let subset = match self.state {
                        Left(ref mut iter) => return iter,
                        Right(subset) => subset,
                    };
                    self.state = Left(TreeMapBaseIterator::from_subset(self.map, subset));
                    match self.state {
                        Left(ref mut iter) => iter,
                        Right(_) => unreachable!(),
                    }
                }

                fn destroy_iter(&mut self) {
                    let subset = match self.state {
                        Left(ref iter) => iter.get_subset(),
                        Right(_) => return,
                    };
                    self.state = Right(subset);
                    self.cache = None;
                }

                fn resolve_handle<'a>(&self, handle: &'a TreeMapHandle<'self, K, V, D>) -> &'a TreePath {
                    if handle.iter != (self as *TreeMapHandleIterator<'self, K, V, D>) {
                        fail2!("the TreeMapHandle belongs to another iterator")
                    }
                    &handle.path
                }

                /// get a reference to the element pointed to by the handle
                pub fn get<'a>(&'a self, handle: &TreeMapHandle<'self, K, V, D>) -> (&'a K, &'a V) {
                    let path = self.resolve_handle(handle);

                    match self.cache {
                        Some((key, value, cache_path)) => if *path == cache_path {
                            return (key, value)
                        },
                        None => {}
                    }
                     
                    let node = follow_path(&self.map.root, *path).get_ref().get();
                    //self.cache = Some((&node.key, &node.value, handle.path.clone()));
                    (&node.key, &node.value)
                }
            }

            impl<'self, K: TotalOrd + $Clone, V: $Clone, D: TreeDir> TreeMapHandleIterator<'self, K, V, D> {
                /// get a mutable reference to the element pointed to by the handle
                pub fn get_mut<'a>(&'a mut self, handle: &TreeMapHandle<'self, K, V, D>) -> (&'a K, &'a mut V) {
                    let path = self.resolve_handle(handle);

                    self.destroy_iter();
                    unsafe {
                        let mut_map: &mut TreeMap<K, V> = transmute(self.map);
                        let node = follow_path_mut(&mut mut_map.root, *path).get_mut_ref().cow();
                        self.cache = Some((transmute(&node.key), transmute(&node.value), handle.path.clone()));
                        (&node.key, &mut node.value)
                    }
                }

                /// remove an item (and destroy the iterator)
                pub fn remove(self, handle: TreeMapHandle<'self, K, V, D>) {
                    let mut this = self;
                    let path = this.resolve_handle(&handle);

                    this.destroy_iter();
                    unsafe {
                        let mut_map: &mut TreeMap<K, V> = transmute(this.map);
                        remove_path(&mut mut_map.root, path.iter());
                    }
                }
            }

            /// Lazy forward iterator over a map that consumes the map while iterating
            pub struct TreeMapMoveIterator<K, V> {
                priv stack: ~[TreeNode<K, V>],
                priv remaining: uint
            }

            impl<K: $Clone, V: $Clone> Iterator<(K, V)> for TreeMapMoveIterator<K,V> {
                #[inline]
                fn next(&mut self) -> Option<(K, V)> {
                    while !self.stack.is_empty() {
                        let TreeNode {
                            key: key,
                            value: value,
                            left: left,
                            right: right,
                            level: level
                        } = self.stack.pop();

                        match left {
                            Some(left) => {
                                let n = TreeNode {
                                    key: key,
                                    value: value,
                                    left: None,
                                    right: right,
                                    level: level
                                };
                                self.stack.push(n);
                                self.stack.push(left.value());
                            }
                            None => {
                                match right {
                                    Some(right) => self.stack.push(right.value()),
                                    None => ()
                                }
                                self.remaining -= 1;
                                return Some((key, value))
                            }
                        }
                    }
                    None
                }

                #[inline]
                fn size_hint(&self) -> (uint, Option<uint>) {
                    (self.remaining, Some(self.remaining))
                }

            }

            impl<'self, T> Iterator<&'self T> for TreeSetIterator<'self, T> {
                /// Advance the iterator to the next node (in order).
                /// If there are no more nodes, return `None`.
                #[inline]
                fn next(&mut self) -> Option<&'self T> {
                    do self.iter.next().map |(value, _)| { value }
                }
            }

            impl<'self, T> Iterator<&'self T> for TreeSetRevIterator<'self, T> {
                /// Advance the iterator to the next node (in order).
                /// If there are no more nodes, return `None`.
                #[inline]
                fn next(&mut self) -> Option<&'self T> {
                    do self.iter.next().map |(value, _)| { value }
                }
            }

            impl<T: $Clone> Iterator<T> for TreeSetMoveIterator<T> {
                /// Advance the iterator to the next node (in order).
                /// If there are no more nodes, return `None`.
                #[inline]
                fn next(&mut self) -> Option<T> {
                    do self.iter.next().map |(value, _)| { value }
                }
            }

            /// A implementation of the `Set` trait on top of the `TreeMap` container. The
            /// only requirement is that the type of the elements contained ascribes to the
            /// `TotalOrd` trait.
            pub struct TreeSet<T> {
                priv map: TreeMap<T, ()>
            }

            impl<T: Eq + TotalOrd> Eq for TreeSet<T> {
                #[inline]
                fn eq(&self, other: &TreeSet<T>) -> bool { self.map == other.map }
            }

            impl<T: Ord + TotalOrd> Ord for TreeSet<T> {
                #[inline]
                fn lt(&self, other: &TreeSet<T>) -> bool { self.map < other.map }
            }

            impl<T: TotalOrd> Container for TreeSet<T> {
                /// Return the number of elements in the set
                #[inline]
                fn len(&self) -> uint { self.map.len() }

                /// Return true if the set contains no elements
                #[inline]
                fn is_empty(&self) -> bool { self.map.is_empty() }
            }

            impl<T: TotalOrd> Mutable for TreeSet<T> {
                /// Clear the set, removing all values.
                #[inline]
                fn clear(&mut self) { self.map.clear() }
            }

            impl<T: TotalOrd> Set<T> for TreeSet<T> {
                /// Return true if the set contains a value
                #[inline]
                fn contains(&self, value: &T) -> bool {
                    self.map.contains_key(value)
                }

                /// Return true if the set has no elements in common with `other`.
                /// This is equivalent to checking for an empty intersection.
                fn is_disjoint(&self, other: &TreeSet<T>) -> bool {
                    self.intersection(other).next().is_none()
                }

                /// Return true if the set is a subset of another
                #[inline]
                fn is_subset(&self, other: &TreeSet<T>) -> bool {
                    other.is_superset(self)
                }

                /// Return true if the set is a superset of another
                fn is_superset(&self, other: &TreeSet<T>) -> bool {
                    let mut x = self.iter();
                    let mut y = other.iter();
                    let mut a = x.next();
                    let mut b = y.next();
                    while b.is_some() {
                        if a.is_none() {
                            return false
                        }

                        let a1 = a.unwrap();
                        let b1 = b.unwrap();

                        match a1.cmp(b1) {
                          Less => (),
                          Greater => return false,
                          Equal => b = y.next(),
                        }

                        a = x.next();
                    }
                    true
                }
            }

            impl<T: TotalOrd + $Clone + $New1 + $New2> MutableSet<T> for TreeSet<T> {
                /// Add a value to the set. Return true if the value was not already
                /// present in the set.
                #[inline]
                fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

                /// Remove a value from the set. Return true if the value was
                /// present in the set.
                #[inline]
                fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
            }

            impl<T: TotalOrd> TreeSet<T> {
                /// Create an empty TreeSet
                #[inline]
                pub fn new() -> TreeSet<T> { TreeSet{map: TreeMap::new()} }

                /// Get a lazy iterator over the values in the set.
                /// Requires that it be frozen (immutable).
                #[inline]
                pub fn iter<'a>(&'a self) -> TreeSetIterator<'a, T> {
                    TreeSetIterator{iter: self.map.iter()}
                }

                /// Get a lazy iterator over the values in the set.
                /// Requires that it be frozen (immutable).
                #[inline]
                pub fn rev_iter<'a>(&'a self) -> TreeSetRevIterator<'a, T> {
                    TreeSetRevIterator{iter: self.map.rev_iter()}
                }

                /// Lazy iterator pointing to the first value not less than `v` (greater or equal).
                /// If all elements in the set are less than `v` empty iterator is returned.
                #[inline]
                pub fn lower_bound_iter<'a>(&'a self, v: &T) -> TreeSetIterator<'a, T> {
                    TreeSetIterator{iter: self.map.lower_bound_iter(v)}
                }

                /// Lazy iterator pointing to the first value greater than `v`.
                /// If all elements in the set are not greater than `v` empty iterator is returned.
                #[inline]
                pub fn upper_bound_iter<'a>(&'a self, v: &T) -> TreeSetIterator<'a, T> {
                    TreeSetIterator{iter: self.map.upper_bound_iter(v)}
                }

                /// Lazy iterator pointing to the last value not greater than `v` (less or equal).
                /// If all elements in the set are greater than `v` empty iterator is returned.
                #[inline]
                pub fn rev_lower_bound_iter<'a>(&'a self, v: &T) -> TreeSetRevIterator<'a, T> {
                    TreeSetRevIterator{iter: self.map.rev_lower_bound_iter(v)}
                }

                /// Lazy iterator pointing to the last value less than `v`.
                /// If all elements in the set are not less than `v` empty iterator is returned.
                #[inline]
                pub fn rev_upper_bound_iter<'a>(&'a self, v: &T) -> TreeSetRevIterator<'a, T> {
                    TreeSetRevIterator{iter: self.map.rev_upper_bound_iter(v)}
                }

                /// Visit the values (in-order) representing the difference
                pub fn difference<'a>(&'a self, other: &'a TreeSet<T>) -> Difference<'a, T> {
                    Difference{a: self.iter().peekable(), b: other.iter().peekable()}
                }

                /// Visit the values (in-order) representing the symmetric difference
                pub fn symmetric_difference<'a>(&'a self, other: &'a TreeSet<T>)
                    -> SymDifference<'a, T> {
                    SymDifference{a: self.iter().peekable(), b: other.iter().peekable()}
                }

                /// Visit the values (in-order) representing the intersection
                pub fn intersection<'a>(&'a self, other: &'a TreeSet<T>)
                    -> Intersection<'a, T> {
                    Intersection{a: self.iter().peekable(), b: other.iter().peekable()}
                }

                /// Visit the values (in-order) representing the union
                pub fn union<'a>(&'a self, other: &'a TreeSet<T>) -> Union<'a, T> {
                    Union{a: self.iter().peekable(), b: other.iter().peekable()}
                }
            }

            impl<T: TotalOrd + $Clone> TreeSet<T> {
                 /// Get a lazy iterator that consumes the TreeMap.
                #[inline]
                pub fn move_iter(self) -> TreeSetMoveIterator<T> {
                    TreeSetMoveIterator{iter: self.map.move_iter()}
                }
            }

            /// Lazy forward iterator over a set
            pub struct TreeSetIterator<'self, T> {
                priv iter: TreeMapIterator<'self, T, ()>
            }

            /// Lazy backward iterator over a set
            pub struct TreeSetRevIterator<'self, T> {
                priv iter: TreeMapRevIterator<'self, T, ()>
            }

            /// Move iterator over a set
            pub struct TreeSetMoveIterator<T> {
                priv iter: TreeMapMoveIterator<T, ()>
            }

            /// Lazy iterator producing elements in the set difference (in-order)
            pub struct Difference<'self, T> {
                priv a: Peekable<&'self T, TreeSetIterator<'self, T>>,
                priv b: Peekable<&'self T, TreeSetIterator<'self, T>>,
            }

            /// Lazy iterator producing elements in the set symmetric difference (in-order)
            pub struct SymDifference<'self, T> {
                priv a: Peekable<&'self T, TreeSetIterator<'self, T>>,
                priv b: Peekable<&'self T, TreeSetIterator<'self, T>>,
            }

            /// Lazy iterator producing elements in the set intersection (in-order)
            pub struct Intersection<'self, T> {
                priv a: Peekable<&'self T, TreeSetIterator<'self, T>>,
                priv b: Peekable<&'self T, TreeSetIterator<'self, T>>,
            }

            /// Lazy iterator producing elements in the set intersection (in-order)
            pub struct Union<'self, T> {
                priv a: Peekable<&'self T, TreeSetIterator<'self, T>>,
                priv b: Peekable<&'self T, TreeSetIterator<'self, T>>,
            }

            /// Compare `x` and `y`, but return `short` if x is None and `long` if y is None
            fn cmp_opt<T: TotalOrd>(x: Option<&T>, y: Option<&T>,
                                    short: Ordering, long: Ordering) -> Ordering {
                match (x, y) {
                    (None    , _       ) => short,
                    (_       , None    ) => long,
                    (Some(x1), Some(y1)) => x1.cmp(y1),
                }
            }

            impl<'self, T: TotalOrd> Iterator<&'self T> for Difference<'self, T> {
                fn next(&mut self) -> Option<&'self T> {
                    loop {
                        match cmp_opt(self.a.peek(), self.b.peek(), Less, Less) {
                            Less    => return self.a.next(),
                            Equal   => { self.a.next(); self.b.next(); }
                            Greater => { self.b.next(); }
                        }
                    }
                }
            }

            impl<'self, T: TotalOrd> Iterator<&'self T> for SymDifference<'self, T> {
                fn next(&mut self) -> Option<&'self T> {
                    loop {
                        match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                            Less    => return self.a.next(),
                            Equal   => { self.a.next(); self.b.next(); }
                            Greater => return self.b.next(),
                        }
                    }
                }
            }

            impl<'self, T: TotalOrd> Iterator<&'self T> for Intersection<'self, T> {
                fn next(&mut self) -> Option<&'self T> {
                    loop {
                        let o_cmp = match (self.a.peek(), self.b.peek()) {
                            (None    , _       ) => None,
                            (_       , None    ) => None,
                            (Some(a1), Some(b1)) => Some(a1.cmp(b1)),
                        };
                        match o_cmp {
                            None          => return None,
                            Some(Less)    => { self.a.next(); }
                            Some(Equal)   => { self.b.next(); return self.a.next() }
                            Some(Greater) => { self.b.next(); }
                        }
                    }
                }
            }

            impl<'self, T: TotalOrd> Iterator<&'self T> for Union<'self, T> {
                fn next(&mut self) -> Option<&'self T> {
                    loop {
                        match cmp_opt(self.a.peek(), self.b.peek(), Greater, Less) {
                            Less    => return self.a.next(),
                            Equal   => { self.b.next(); return self.a.next() }
                            Greater => return self.b.next(),
                        }
                    }
                }
            }


            // Nodes keep track of their level in the tree, starting at 1 in the
            // leaves and with a red child sharing the level of the parent.
            #[deriving(Clone)]
            struct TreeNode<K, V> {
                key: K,
                value: V,
                left: Option<$P>,
                right: Option<$P>,
                level: uint
            }
            
            impl<K, V> TreeNode<K, V> {
                fn destructure<'a>(&'a self) -> (&'a K, &'a V, &'a Option<$P>,  &'a Option<$P>) {
                    let TreeNode {key: ref key, value: ref value, left: ref left, right: ref right, _} = *self;
                    (key, value, left, right)
                }

                fn destructure_mut<'a>(&'a mut self) -> (&'a K, &'a mut V, &'a mut Option<$P>,  &'a mut Option<$P>) {
                    let TreeNode {key: ref key, value: ref mut value, left: ref mut left, right: ref mut right, _} = *self;
                    (key, value, left, right)
                }
            }

            impl<K: TotalOrd, V> TreeNode<K, V> {
                /// Creates a new tree node.
                #[inline]
                pub fn new(key: K, value: V) -> TreeNode<K, V> {
                    TreeNode{key: key, value: value, left: None, right: None, level: 1}
                }
            }
            
            #[allow(missing_doc)]
            trait TreeDir
            {
                fn new() -> Self;

                fn is_right(&self) -> bool;
                fn is_left(&self) -> bool;
            }
            
            struct TreeLeft;
            
            impl TreeDir for TreeLeft {
                fn new() -> TreeLeft {TreeLeft}

                fn is_right(&self) -> bool {false}
                fn is_left(&self) -> bool {true}
            }

            struct TreeRight;
            
            impl TreeDir for TreeRight {
                fn new() -> TreeRight {TreeRight}

                fn is_right(&self) -> bool {true}
                fn is_left(&self) -> bool {false}
            }

            // Remove left horizontal link by rotating right
            fn skew<K: TotalOrd + $Clone, V: $Clone>(mut node: $P) ->$P {
                if node.get().left.as_ref().map_default(false,
                    |x| x.get().level == node.get().level) {
                    let mut left: $P;
                    {
                        let mut_left = {
                            let mut_node = node.cow();
                            left = mut_node.left.take_unwrap();
                            let mut_left = left.cow();
                            swap(&mut mut_node.left, &mut mut_left.right); // left.right now None
                            mut_left
                        };
                        mut_left.right = Some(node);
                    }
                    left
                } else {
                    node
                }
            }

            // Remove dual horizontal link by rotating left and increasing level of
            // the parent
            fn split<K: TotalOrd + $Clone, V: $Clone>(mut node: $P) -> $P {
                if node.get().right.as_ref().map_default(false,
                  |x| x.get().right.as_ref().map_default(false,
                    |y| y.get().level == node.get().level)) {
                    let mut right: $P;
                    {
                        let mut_right = {
                            let mut_node = node.cow();
                            right = mut_node.right.take_unwrap();
                            let mut_right = right.cow();
                            mut_right.level += 1;
                            swap(&mut mut_node.right, &mut mut_right.left); // right.left now None
                            mut_right
                        };
                        mut_right.left = Some(node);
                    }
                    right
                } else {
                    node
                }
            }

            fn rebalance<K: TotalOrd + $Clone, V: $Clone>(
                node_opt: &mut Option<$P> /* always Some in input and output */) {
                let (level, left_level, right_level) = {
                    let ref_node = node_opt.get_ref().get();
                    (ref_node.level,
                    ref_node.left.as_ref().map_default(0, |x| x.get().level),
                    ref_node.right.as_ref().map_default(0, |x| x.get().level))
                };

                // re-balance, if necessary
                if left_level < level - 1 || right_level < level - 1 {
                    let mut node = node_opt.take_unwrap();
                    {
                        let mut_node = node.cow();
                        mut_node.level -= 1;

                        if right_level > mut_node.level {
                            for x in mut_node.right.mut_iter() { x.cow().level = mut_node.level }
                        }
                    }

                    node = skew(node);

                    node.cow().right.mutate(|mut right| {
                        right = skew(right);
                        right.cow().right.mutate(skew);
                        right
                    });

                    node = split(node);
                    node.cow().right.mutate(split);
                    *node_opt = Some(node);
                }
            }

            fn iterate<A, B>(init: A, f: &fn(A) -> Either<A, B>) -> B {
                let mut accum = init;
                loop {
                    match f(accum) {
                        Left(x) => { accum = x; }
                        Right(y) => { return y; }
                    }
                }
            }

            fn find_mut<'r, K: TotalOrd + $Clone, V: $Clone>(
                node: &'r mut Option<$P>, key: &K) -> Option<&'r mut V> {
                iterate(node, |node| {
                    match *node {
                      Some(ref mut x) => {
                        match x.try_get_mut() {
                            Left(x) => {
                                match find_path(x, key) {
                                    Some(path) => Right(Some(&mut follow_path_mut_inner(x, path).cow().value)),
                                    None => Right(None)
                                }
                            },
                            Right(mut_x) => {
                                match key.cmp(&mut_x.key) {
                                  Less => Left(&mut mut_x.left),
                                  Greater => Left(&mut mut_x.right),
                                  Equal => Right(Some(&mut mut_x.value)),
                                }
                            }
                         }
                      }
                      None => Right(None)
                    }
                })
            }

            fn follow_path<'r, K: TotalOrd + $Clone, V: $Clone>(
                node: &'r Option<$P>, path: TreePath) -> &'r Option<$P> {
                path.iter().fold(node, |node, dir| {
                    if(!dir) {
                        &node.get_ref().get().left
                    } else {
                        &node.get_ref().get().right
                    }
                })
            }

            fn follow_path_mut<'r, K: TotalOrd + $Clone, V: $Clone>(
                node: &'r mut Option<$P>, path: TreePath) -> &'r mut Option<$P> {
                path.iter().fold(node, |node, dir| {
                    if(!dir) {
                        &mut node.get_mut_ref().cow().left
                    } else {
                        &mut node.get_mut_ref().cow().right
                    }
                })
            }

            fn follow_path_mut_inner<'r, K: TotalOrd + $Clone, V: $Clone>(
                node: &'r mut $P, path: TreePath) -> &'r mut $P {
                path.iter().fold(node, |node, dir| {
                    if(!dir) {
                        node.cow().left.get_mut_ref()
                    } else {
                        node.cow().right.get_mut_ref()
                    }
                })
            }

            fn find_path<'r, K: TotalOrd, V>(
                mut node: &'r $P, key: &K) -> Option<TreePath> {
                let mut path = TreePath::new();
                loop {
                    let r = node.get();
                    let next = match key.cmp(&r.key) {
                      Less => {path.push(false); &r.left},
                      Greater => {path.push(true); &r.right},
                      Equal => return Some(path)
                    };

                    node = match *next {
                        Some(ref r) => r,
                        None => return None
                    }
                }
            }

            fn insert<K: TotalOrd + $Clone + $New1 + $New2, V: $Clone + $New1 + $New2>(
                node_opt: &mut Option<$P>, key: K, value: V) -> Option<V> {
                match node_opt.take() {
                  Some(node_) => {
                    let mut node = node_;
                    let (node, old) = match key.cmp(&node.get().key) {
                      Less => {
                        let old = insert(&mut node.cow().left, key, value);
                        (split(skew(node)), old)
                      }
                      Greater => {
                        let old = insert(&mut node.cow().right, key, value);
                        (split(skew(node)), old)
                      }
                      Equal => {
                        let old = {
                          let mut_node = node.cow();
                          mut_node.key = key;
                          Some(replace(&mut mut_node.value, value))
                        };
                        (node, old)
                      }
                    };
                    *node_opt = Some(node);
                    old
                  }
                  None => {
                   *node_opt = Some($new(TreeNode::new(key, value)));
                    None
                  }
                }
            }

            fn remove<K: TotalOrd + $Clone, V: $Clone>(node_opt: &mut Option<$P>,
                                      key: &K) -> Option<V> {
                enum RemoveResult<V> {
                    Shared,
                    ThisNode,
                    Removed(V)
                }

                let res = match *node_opt {
                    None => return None,
                    Some(ref mut node) => {
                        match node.try_get_mut() {
                            Left(_) => Shared,
                            Right(mut_node) => {
                                match key.cmp(&mut_node.key) {
                                    Less => Removed(remove(&mut mut_node.left, key)),
                                    Greater => Removed(remove(&mut mut_node.right, key)),
                                    Equal => ThisNode
                                }
                            }
                        }
                    }
                };

                let value_opt = match res {
                  Shared => {
                    match find_path(node_opt.get_ref(), key) {
                      Some(path) => return Some(remove_path(node_opt, path.iter())),
                      None => return None
                    }
                  },
                  ThisNode => return Some(remove_node(node_opt)),
                  Removed(value_opt) => value_opt
                };

                rebalance(node_opt);
                value_opt
            }

            fn remove_node<K: TotalOrd + $Clone, V: $Clone>(
                node_opt: &mut Option<$P> /* always Some in input */) -> V {
                fn swap_max<K: TotalOrd + $Clone, V: $Clone>(node: &mut $P,
                    mut_target: &mut TreeNode<K, V>) {
                    let mut_node = node.cow();
                    if mut_node.right.is_some() {
                        swap_max(mut_node.right.as_mut().unwrap(), mut_target)
                    } else {
                        swap(&mut mut_node.key, &mut mut_target.key);
                        swap(&mut mut_node.value, &mut mut_target.value);
                    }
                }

                let value = if node_opt.get_ref().get().left.is_some() {
                    if node_opt.get_ref().get().right.is_some() {
                        let mut_node = node_opt.get_mut_ref().cow();
                        let mut left = mut_node.left.take();
                        swap_max(left.as_mut().unwrap(), mut_node);
                        let value = remove_max(&mut left);
                        mut_node.left = left;
                        value
                    } else {
                        let TreeNode {value: value, left: left, _} =
                            node_opt.take_unwrap().value();
                        *node_opt = left;
                        value
                    }
                } else if node_opt.get_ref().get().right.is_some() {
                        let TreeNode {value: value, right: right, _} =
                            node_opt.take_unwrap().value();
                        *node_opt = right;
                        value
                } else {
                    let TreeNode {value: value, _} = node_opt.take_unwrap().value();
                    *node_opt = None;
                    return value
                };

                rebalance(node_opt);
                value
            }

            fn remove_max<K: TotalOrd + $Clone, V: $Clone>(
                node_opt: &mut Option<$P> /* always Some in input */) -> V {
                let value = if node_opt.get_mut_ref().get().right.is_some() {
                    remove_max(&mut node_opt.get_mut_ref().cow().right)
                } else {
                    return remove_node(node_opt)
                };

                rebalance(node_opt);
                value
            }

            fn remove_path<K: TotalOrd + $Clone, V: $Clone>(
                node_opt: &mut Option<$P> /* always Some in input */,
                mut path_iter: TreePathIterator) -> V {
                let value = match path_iter.next() {
                    None => return remove_node(node_opt),
                    Some(false) => remove_path(&mut node_opt.get_mut_ref().cow().left, path_iter),
                    Some(true) => remove_path(&mut node_opt.get_mut_ref().cow().right, path_iter)
                };
                rebalance(node_opt);
                value
            }

            impl<K: TotalOrd + $Clone + $New1 + $New2, V: $Clone + $New1 + $New2>
                FromIterator<(K, V)> for TreeMap<K, V> {
                fn from_iterator<T: Iterator<(K, V)>>(iter: &mut T) -> TreeMap<K, V> {
                    let mut map = TreeMap::new();
                    map.extend(iter);
                    map
                }
            }

            impl<K: TotalOrd + $Clone + $New1 + $New2, V: $Clone + $New1 + $New2>
                Extendable<(K, V)> for TreeMap<K, V> {
                #[inline]
                fn extend<T: Iterator<(K, V)>>(&mut self, iter: &mut T) {
                    for (k, v) in *iter {
                        self.insert(k, v);
                    }
                }
            }

            impl<T: TotalOrd + $Clone + $New1 + $New2> FromIterator<T> for TreeSet<T> {
                fn from_iterator<Iter: Iterator<T>>(iter: &mut Iter) -> TreeSet<T> {
                    let mut set = TreeSet::new();
                    set.extend(iter);
                    set
                }
            }

            impl<T: TotalOrd + $Clone + $New1 + $New2> Extendable<T> for TreeSet<T> {
                #[inline]
                fn extend<Iter: Iterator<T>>(&mut self, iter: &mut Iter) {
                    for elem in *iter {
                        self.insert(elem);
                    }
                }
            }

            #[cfg(test)]
            mod test_TreeMap {

                use super::*;

                use std::rand::Rng;
                use std::rand;

                #[test]
                fn find_empty() {
                    let m: TreeMap<int,int> = TreeMap::new();
                    assert!(m.find(&5) == None);
                }

                #[test]
                fn find_not_found() {
                    let mut m = TreeMap::new();
                    assert!(m.insert(1, 2));
                    assert!(m.insert(5, 3));
                    assert!(m.insert(9, 3));
                    assert_eq!(m.find(&2), None);
                }

                #[test]
                fn test_find_mut() {
                    let mut m = TreeMap::new();
                    assert!(m.insert(1, 12));
                    assert!(m.insert(2, 8));
                    assert!(m.insert(5, 14));
                    let new = 100;
                    match m.find_mut(&5) {
                      None => fail2!(), Some(x) => *x = new
                    }
                    assert_eq!(m.find(&5), Some(&new));
                }

                #[test]
                fn insert_replace() {
                    let mut m = TreeMap::new();
                    assert!(m.insert(5, 2));
                    assert!(m.insert(2, 9));
                    assert!(!m.insert(2, 11));
                    assert_eq!(m.find(&2).unwrap(), &11);
                }

                #[test]
                fn test_clear() {
                    let mut m = TreeMap::new();
                    m.clear();
                    assert!(m.insert(5, 11));
                    assert!(m.insert(12, -3));
                    assert!(m.insert(19, 2));
                    m.clear();
                    assert!(m.find(&5).is_none());
                    assert!(m.find(&12).is_none());
                    assert!(m.find(&19).is_none());
                    assert!(m.is_empty());
                }

                #[test]
                fn u8_map() {
                    let mut m = TreeMap::new();

                    let k1 = "foo".as_bytes();
                    let k2 = "bar".as_bytes();
                    let v1 = "baz".as_bytes();
                    let v2 = "foobar".as_bytes();

                    m.insert(k1.clone(), v1.clone());
                    m.insert(k2.clone(), v2.clone());

                    assert_eq!(m.find(&k2), Some(&v2));
                    assert_eq!(m.find(&k1), Some(&v1));
                }

                fn check_equal<K: Eq + TotalOrd, V: Eq>(ctrl: &[(K, V)],
                                                        map: &TreeMap<K, V>) {
                    assert_eq!(ctrl.is_empty(), map.is_empty());
                    for x in ctrl.iter() {
                        let &(ref k, ref v) = x;
                        assert!(map.find(k).unwrap() == v)
                    }
                    for (map_k, map_v) in map.iter() {
                        let mut found = false;
                        for x in ctrl.iter() {
                            let &(ref ctrl_k, ref ctrl_v) = x;
                            if *map_k == *ctrl_k {
                                assert!(*map_v == *ctrl_v);
                                found = true;
                                break;
                            }
                        }
                        assert!(found);
                    }
                }

                fn check_left<K: TotalOrd, V>(
                    node: &Option<$P>, parent: &$P) {
                    match *node {
                      Some(ref r) => {
                        assert_eq!(r.key.cmp(&parent.key), Less);
                        assert!(r.level == parent.level - 1); // left is black
                        check_left(&r.left, r);
                        check_right(&r.right, r, false);
                      }
                      None => assert!(parent.level == 1) // parent is leaf
                    }
                }

                fn check_right<K: TotalOrd, V>(
                    node: &Option<$P>, parent: &$P, parent_red: bool) {
                    match *node {
                      Some(ref r) => {
                        assert_eq!(r.key.cmp(&parent.key), Greater);
                        let red = r.level == parent.level;
                        if parent_red { assert!(!red) } // no dual horizontal links
                        // Right red or black
                        assert!(red || r.level == parent.level - 1);
                        check_left(&r.left, r);
                        check_right(&r.right, r, red);
                      }
                      None => assert!(parent.level == 1) // parent is leaf
                    }
                }

                fn check_structure<K: TotalOrd, V>(map: &TreeMap<K, V>) {
                    match map.root {
                      Some(ref r) => {
                        check_left(&r.left, r);
                        check_right(&r.right, r, false);
                      }
                      None => ()
                    }
                }

                #[test]
                fn test_rand_int() {
                    let mut map: TreeMap<int,int> = TreeMap::new();
                    let mut ctrl = ~[];

                    check_equal(ctrl, &map);
                    assert!(map.find(&5).is_none());

                    let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(&[42]);

                    do 3.times {
                        do 90.times {
                            let k = rng.gen();
                            let v = rng.gen();
                            if !ctrl.iter().any(|x| x == &(k, v)) {
                                assert!(map.insert(k, v));
                                ctrl.push((k, v));
                                check_structure(&map);
                                check_equal(ctrl, &map);
                            }
                        }

                        do 30.times {
                            let r = rng.gen_integer_range(0, ctrl.len());
                            let (key, _) = ctrl.remove(r);
                            assert!(map.remove(&key));
                            check_structure(&map);
                            check_equal(ctrl, &map);
                        }
                    }
                }

                #[test]
                fn test_len() {
                    let mut m = TreeMap::new();
                    assert!(m.insert(3, 6));
                    assert_eq!(m.len(), 1);
                    assert!(m.insert(0, 0));
                    assert_eq!(m.len(), 2);
                    assert!(m.insert(4, 8));
                    assert_eq!(m.len(), 3);
                    assert!(m.remove(&3));
                    assert_eq!(m.len(), 2);
                    assert!(!m.remove(&5));
                    assert_eq!(m.len(), 2);
                    assert!(m.insert(2, 4));
                    assert_eq!(m.len(), 3);
                    assert!(m.insert(1, 2));
                    assert_eq!(m.len(), 4);
                }

                #[test]
                fn test_iterator() {
                    let mut m = TreeMap::new();

                    assert!(m.insert(3, 6));
                    assert!(m.insert(0, 0));
                    assert!(m.insert(4, 8));
                    assert!(m.insert(2, 4));
                    assert!(m.insert(1, 2));

                    let mut n = 0;
                    for (k, v) in m.iter() {
                        assert_eq!(*k, n);
                        assert_eq!(*v, n * 2);
                        n += 1;
                    }
                    assert_eq!(n, 5);
                }

                #[test]
                fn test_interval_iteration() {
                    let mut m = TreeMap::new();
                    for i in range(1, 100) {
                        assert!(m.insert(i * 2, i * 4));
                    }

                    for i in range(1, 198) {
                        let mut lb_it = m.lower_bound_iter(&i);
                        let (&k, &v) = lb_it.next().unwrap();
                        let lb = i + i % 2;
                        assert_eq!(lb, k);
                        assert_eq!(lb * 2, v);

                        let mut ub_it = m.upper_bound_iter(&i);
                        let (&k, &v) = ub_it.next().unwrap();
                        let ub = i + 2 - i % 2;
                        assert_eq!(ub, k);
                        assert_eq!(ub * 2, v);
                    }
                    let mut end_it = m.lower_bound_iter(&199);
                    assert_eq!(end_it.next(), None);
                }

                #[test]
                fn test_rev_iter() {
                    let mut m = TreeMap::new();

                    assert!(m.insert(3, 6));
                    assert!(m.insert(0, 0));
                    assert!(m.insert(4, 8));
                    assert!(m.insert(2, 4));
                    assert!(m.insert(1, 2));

                    let mut n = 4;
                    for (k, v) in m.rev_iter() {
                        assert_eq!(*k, n);
                        assert_eq!(*v, n * 2);
                        n -= 1;
                    }
                }

                #[test]
                fn test_eq() {
                    let mut a = TreeMap::new();
                    let mut b = TreeMap::new();

                    assert!(a == b);
                    assert!(a.insert(0, 5));
                    assert!(a != b);
                    assert!(b.insert(0, 4));
                    assert!(a != b);
                    assert!(a.insert(5, 19));
                    assert!(a != b);
                    assert!(!b.insert(0, 5));
                    assert!(a != b);
                    assert!(b.insert(5, 19));
                    assert!(a == b);
                }

                #[test]
                fn test_lt() {
                    let mut a = TreeMap::new();
                    let mut b = TreeMap::new();

                    assert!(!(a < b) && !(b < a));
                    assert!(b.insert(0, 5));
                    assert!(a < b);
                    assert!(a.insert(0, 7));
                    assert!(!(a < b) && b < a);
                    assert!(b.insert(-2, 0));
                    assert!(b < a);
                    assert!(a.insert(-5, 2));
                    assert!(a < b);
                    assert!(a.insert(6, 2));
                    assert!(a < b && !(b < a));
                }

                #[test]
                fn test_ord() {
                    let mut a = TreeMap::new();
                    let mut b = TreeMap::new();

                    assert!(a <= b && a >= b);
                    assert!(a.insert(1, 1));
                    assert!(a > b && a >= b);
                    assert!(b < a && b <= a);
                    assert!(b.insert(2, 2));
                    assert!(b > a && b >= a);
                    assert!(a < b && a <= b);
                }

                #[test]
                fn test_lazy_iterator() {
                    let mut m = TreeMap::new();
                    let (x1, y1) = (2, 5);
                    let (x2, y2) = (9, 12);
                    let (x3, y3) = (20, -3);
                    let (x4, y4) = (29, 5);
                    let (x5, y5) = (103, 3);

                    assert!(m.insert(x1, y1));
                    assert!(m.insert(x2, y2));
                    assert!(m.insert(x3, y3));
                    assert!(m.insert(x4, y4));
                    assert!(m.insert(x5, y5));

                    let m = m;
                    let mut a = m.iter();

                    assert_eq!(a.next().unwrap(), (&x1, &y1));
                    assert_eq!(a.next().unwrap(), (&x2, &y2));
                    assert_eq!(a.next().unwrap(), (&x3, &y3));
                    assert_eq!(a.next().unwrap(), (&x4, &y4));
                    assert_eq!(a.next().unwrap(), (&x5, &y5));

                    assert!(a.next().is_none());

                    let mut b = m.iter();

                    let expected = [(&x1, &y1), (&x2, &y2), (&x3, &y3), (&x4, &y4),
                                    (&x5, &y5)];
                    let mut i = 0;

                    for x in b {
                        assert_eq!(expected[i], x);
                        i += 1;

                        if i == 2 {
                            break
                        }
                    }

                    for x in b {
                        assert_eq!(expected[i], x);
                        i += 1;
                    }
                }

                #[test]
                fn test_from_iter() {
                    let xs = ~[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

                    let map: TreeMap<int, int> = xs.iter().map(|&x| x).collect();

                    for &(k, v) in xs.iter() {
                        assert_eq!(map.find(&k), Some(&v));
                    }
                }

            }

            #[cfg(test)]
            mod bench {

                use super::*;
                use test::BenchHarness;
                use container::bench::*;

                // Find seq
                #[bench]
                pub fn insert_rand_100(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    insert_rand_n(100, &mut m, bh);
                }

                #[bench]
                pub fn insert_rand_10_000(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    insert_rand_n(10_000, &mut m, bh);
                }

                // Insert seq
                #[bench]
                pub fn insert_seq_100(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    insert_seq_n(100, &mut m, bh);
                }

                #[bench]
                pub fn insert_seq_10_000(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    insert_seq_n(10_000, &mut m, bh);
                }

                // Find rand
                #[bench]
                pub fn find_rand_100(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    find_rand_n(100, &mut m, bh);
                }

                #[bench]
                pub fn find_rand_10_000(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    find_rand_n(10_000, &mut m, bh);
                }

                // Find seq
                #[bench]
                pub fn find_seq_100(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    find_seq_n(100, &mut m, bh);
                }

                #[bench]
                pub fn find_seq_10_000(bh: &mut BenchHarness) {
                    let mut m : TreeMap<uint,uint> = TreeMap::new();
                    find_seq_n(10_000, &mut m, bh);
                }
            }

            #[cfg(test)]
            mod test_set {

                use super::*;

                #[test]
                fn test_clear() {
                    let mut s = TreeSet::new();
                    s.clear();
                    assert!(s.insert(5));
                    assert!(s.insert(12));
                    assert!(s.insert(19));
                    s.clear();
                    assert!(!s.contains(&5));
                    assert!(!s.contains(&12));
                    assert!(!s.contains(&19));
                    assert!(s.is_empty());
                }

                #[test]
                fn test_disjoint() {
                    let mut xs = TreeSet::new();
                    let mut ys = TreeSet::new();
                    assert!(xs.is_disjoint(&ys));
                    assert!(ys.is_disjoint(&xs));
                    assert!(xs.insert(5));
                    assert!(ys.insert(11));
                    assert!(xs.is_disjoint(&ys));
                    assert!(ys.is_disjoint(&xs));
                    assert!(xs.insert(7));
                    assert!(xs.insert(19));
                    assert!(xs.insert(4));
                    assert!(ys.insert(2));
                    assert!(ys.insert(-11));
                    assert!(xs.is_disjoint(&ys));
                    assert!(ys.is_disjoint(&xs));
                    assert!(ys.insert(7));
                    assert!(!xs.is_disjoint(&ys));
                    assert!(!ys.is_disjoint(&xs));
                }

                #[test]
                fn test_subset_and_superset() {
                    let mut a = TreeSet::new();
                    assert!(a.insert(0));
                    assert!(a.insert(5));
                    assert!(a.insert(11));
                    assert!(a.insert(7));

                    let mut b = TreeSet::new();
                    assert!(b.insert(0));
                    assert!(b.insert(7));
                    assert!(b.insert(19));
                    assert!(b.insert(250));
                    assert!(b.insert(11));
                    assert!(b.insert(200));

                    assert!(!a.is_subset(&b));
                    assert!(!a.is_superset(&b));
                    assert!(!b.is_subset(&a));
                    assert!(!b.is_superset(&a));

                    assert!(b.insert(5));

                    assert!(a.is_subset(&b));
                    assert!(!a.is_superset(&b));
                    assert!(!b.is_subset(&a));
                    assert!(b.is_superset(&a));
                }

                #[test]
                fn test_iterator() {
                    let mut m = TreeSet::new();

                    assert!(m.insert(3));
                    assert!(m.insert(0));
                    assert!(m.insert(4));
                    assert!(m.insert(2));
                    assert!(m.insert(1));

                    let mut n = 0;
                    for x in m.iter() {
                        assert_eq!(*x, n);
                        n += 1
                    }
                }

                #[test]
                fn test_rev_iter() {
                    let mut m = TreeSet::new();

                    assert!(m.insert(3));
                    assert!(m.insert(0));
                    assert!(m.insert(4));
                    assert!(m.insert(2));
                    assert!(m.insert(1));

                    let mut n = 4;
                    for x in m.rev_iter() {
                        assert_eq!(*x, n);
                        n -= 1;
                    }
                }

                fn check(a: &[int], b: &[int], expected: &[int],
                         f: &fn(&TreeSet<int>, &TreeSet<int>, f: &fn(&int) -> bool) -> bool) {
                    let mut set_a = TreeSet::new();
                    let mut set_b = TreeSet::new();

                    for x in a.iter() { assert!(set_a.insert(*x)) }
                    for y in b.iter() { assert!(set_b.insert(*y)) }

                    let mut i = 0;
                    do f(&set_a, &set_b) |x| {
                        assert_eq!(*x, expected[i]);
                        i += 1;
                        true
                    };
                    assert_eq!(i, expected.len());
                }

                #[test]
                fn test_intersection() {
                    fn check_intersection(a: &[int], b: &[int], expected: &[int]) {
                        check(a, b, expected, |x, y, f| x.intersection(y).advance(f))
                    }

                    check_intersection([], [], []);
                    check_intersection([1, 2, 3], [], []);
                    check_intersection([], [1, 2, 3], []);
                    check_intersection([2], [1, 2, 3], [2]);
                    check_intersection([1, 2, 3], [2], [2]);
                    check_intersection([11, 1, 3, 77, 103, 5, -5],
                                       [2, 11, 77, -9, -42, 5, 3],
                                       [3, 5, 11, 77]);
                }

                #[test]
                fn test_difference() {
                    fn check_difference(a: &[int], b: &[int], expected: &[int]) {
                        check(a, b, expected, |x, y, f| x.difference(y).advance(f))
                    }

                    check_difference([], [], []);
                    check_difference([1, 12], [], [1, 12]);
                    check_difference([], [1, 2, 3, 9], []);
                    check_difference([1, 3, 5, 9, 11],
                                     [3, 9],
                                     [1, 5, 11]);
                    check_difference([-5, 11, 22, 33, 40, 42],
                                     [-12, -5, 14, 23, 34, 38, 39, 50],
                                     [11, 22, 33, 40, 42]);
                }

                #[test]
                fn test_symmetric_difference() {
                    fn check_symmetric_difference(a: &[int], b: &[int],
                                                  expected: &[int]) {
                        check(a, b, expected, |x, y, f| x.symmetric_difference(y).advance(f))
                    }

                    check_symmetric_difference([], [], []);
                    check_symmetric_difference([1, 2, 3], [2], [1, 3]);
                    check_symmetric_difference([2], [1, 2, 3], [1, 3]);
                    check_symmetric_difference([1, 3, 5, 9, 11],
                                               [-2, 3, 9, 14, 22],
                                               [-2, 1, 5, 11, 14, 22]);
                }

                #[test]
                fn test_union() {
                    fn check_union(a: &[int], b: &[int],
                                                  expected: &[int]) {
                        check(a, b, expected, |x, y, f| x.union(y).advance(f))
                    }

                    check_union([], [], []);
                    check_union([1, 2, 3], [2], [1, 2, 3]);
                    check_union([2], [1, 2, 3], [1, 2, 3]);
                    check_union([1, 3, 5, 9, 11, 16, 19, 24],
                                [-2, 1, 5, 9, 13, 19],
                                [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24]);
                }

                #[test]
                fn test_zip() {
                    let mut x = TreeSet::new();
                    x.insert(5u);
                    x.insert(12u);
                    x.insert(11u);

                    let mut y = TreeSet::new();
                    y.insert("foo");
                    y.insert("bar");

                    let x = x;
                    let y = y;
                    let mut z = x.iter().zip(y.iter());

                    // FIXME: #5801: this needs a type hint to compile...
                    let result: Option<(&uint, & &'static str)> = z.next();
                    assert_eq!(result.unwrap(), (&5u, & &"bar"));

                    let result: Option<(&uint, & &'static str)> = z.next();
                    assert_eq!(result.unwrap(), (&11u, & &"foo"));

                    let result: Option<(&uint, & &'static str)> = z.next();
                    assert!(result.is_none());
                }

                #[test]
                fn test_swap() {
                    let mut m = TreeMap::new();
                    assert_eq!(m.swap(1, 2), None);
                    assert_eq!(m.swap(1, 3), Some(2));
                    assert_eq!(m.swap(1, 4), Some(3));
                }

                #[test]
                fn test_pop() {
                    let mut m = TreeMap::new();
                    m.insert(1, 2);
                    assert_eq!(m.pop(&1), Some(2));
                    assert_eq!(m.pop(&1), None);
                }

                #[test]
                fn test_from_iter() {
                    let xs = ~[1, 2, 3, 4, 5, 6, 7, 8, 9];

                    let set: TreeSet<int> = xs.iter().map(|&x| x).collect();

                    for x in xs.iter() {
                        assert!(set.contains(x));
                    }
                }
            }
        }
    }
}

trait Dummy {}
impl<T> Dummy for T {}

treemap!(rc, Rc<TreeNode<K, V>>, Rc::new, Clone, Freeze + Freeze)
treemap!(arc, Arc<TreeNode<K, V>>, Arc::new, Clone, Send + Freeze)
treemap!(own, Own<TreeNode<K, V>>, Own::new, Dummy, Dummy + Dummy)

#[cfg(test)]
mod test_TreePath {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let p = TreePath::new();
        assert!(p.iter().next().is_none())
    }

    fn test(even: bool, odd: bool, n: uint) {
        let p = TreePath::new();
        for i in range(0, n) {
            p.push(if (i & 1) == 0 {even} else {odd})
        }
        let iter = p.iter();
        for i in range(0, uint::bits * 2 - 2) {
            assert!(iter.next() == Some(if (i & 1) == 0 {even} else {odd}))
        }
        assert!(iter.next().is_none())
    }

    #[test] fn test_all_zeroes_long() {test(false, false, uint::bits * 2 - 2)}
    #[test] fn test_all_ones_long() {test(true, true, uint::bits * 2 - 2)}
    #[test] fn test_zero_one_long() { test(false, true, uint::bits * 2 - 2)}

    #[test] fn test_all_zeroes_mid() {test(false, false, uint::bits + 8)}
    #[test] fn test_all_ones_mid() {test(true, true, uint::bits + 8)}
    #[test] fn test_zero_one_mid() {test(false, true, uint::bits + 8)}

    #[test] fn test_all_zeroes_short() {test(false, false, 8)}
    #[test] fn test_all_ones_short() {test(true, true, 8)}
    #[test] fn test_zero_one_short() {test(false, true, 8)}
}

