// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// btree.rs
//

// NB. this is not deprecated for removal, just deprecating the
// current implementation. If the major pain-points are addressed
// (overuse of by-value self and .clone), this can be removed.
#![deprecated = "the current implementation is extremely inefficient, \
                 prefer a HashMap, TreeMap or TrieMap"]
#![allow(deprecated)]

//! Starting implementation of a btree for rust.
//! Structure inspired by github user davidhalperin's gist.

///A B-tree contains a root node (which contains a vector of elements),
///a length (the height of the tree), and lower and upper bounds on the
///number of elements that a given node can contain.

use core::prelude::*;

use alloc::boxed::Box;
use core::fmt;
use core::fmt::Show;

use MutableSeq;
use vec::Vec;

#[allow(missing_doc)]
pub struct BTree<K, V> {
    root: Node<K, V>,
    len: uint,
    lower_bound: uint,
    upper_bound: uint
}

impl<K: Ord, V> BTree<K, V> {

    ///Returns new BTree with root node (leaf) and user-supplied lower bound
    ///The lower bound applies to every node except the root node.
    pub fn new(k: K, v: V, lb: uint) -> BTree<K, V> {
        BTree {
            root: Node::new_leaf(vec!(LeafElt::new(k, v))),
            len: 1,
            lower_bound: lb,
            upper_bound: 2 * lb
        }
    }

    ///Helper function for clone: returns new BTree with supplied root node,
    ///length, and lower bound.  For use when the length is known already.
    fn new_with_node_len(n: Node<K, V>,
                         length: uint,
                         lb: uint) -> BTree<K, V> {
        BTree {
            root: n,
            len: length,
            lower_bound: lb,
            upper_bound: 2 * lb
        }
    }
}

//We would probably want to remove the dependence on the Clone trait in the future.
//It is here as a crutch to ensure values can be passed around through the tree's nodes
//especially during insertions and deletions.
impl<K: Clone + Ord, V: Clone> BTree<K, V> {
    ///Returns the value of a given key, which may not exist in the tree.
    ///Calls the root node's get method.
    pub fn get(self, k: K) -> Option<V> {
        return self.root.get(k);
    }

    ///An insert method that uses the clone() feature for support.
    pub fn insert(mut self, k: K, v: V) -> BTree<K, V> {
        let (a, b) = self.root.clone().insert(k, v, self.upper_bound.clone());
        if b {
            match a.clone() {
                LeafNode(leaf) => {
                    self.root = Node::new_leaf(leaf.clone().elts);
                }
                BranchNode(branch) => {
                    self.root = Node::new_branch(branch.clone().elts,
                                                 branch.clone().rightmost_child);
                }
            }
        }
        self
    }
}

impl<K: Clone + Ord, V: Clone> Clone for BTree<K, V> {
    ///Implements the Clone trait for the BTree.
    ///Uses a helper function/constructor to produce a new BTree.
    fn clone(&self) -> BTree<K, V> {
        BTree::new_with_node_len(self.root.clone(), self.len, self.lower_bound)
    }
}

impl<K: Ord, V: Eq> PartialEq for BTree<K, V> {
    fn eq(&self, other: &BTree<K, V>) -> bool {
        self.root.cmp(&other.root) == Equal
    }
}

impl<K: Ord, V: Eq> Eq for BTree<K, V> {}

impl<K: Ord, V: Eq> PartialOrd for BTree<K, V> {
    fn partial_cmp(&self, other: &BTree<K, V>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V: Eq> Ord for BTree<K, V> {
    ///Returns an ordering based on the root nodes of each BTree.
    fn cmp(&self, other: &BTree<K, V>) -> Ordering {
        self.root.cmp(&other.root)
    }
}

impl<K: fmt::Show + Ord, V: fmt::Show> fmt::Show for BTree<K, V> {
    ///Returns a string representation of the BTree
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.root.fmt(f)
    }
}


//Node types
//A node is either a LeafNode or a BranchNode, which contain either a Leaf or a Branch.
//Branches contain BranchElts, which contain a left child (another node) and a key-value
//pair.  Branches also contain the rightmost child of the elements in the array.
//Leaves contain LeafElts, which do not have children.
enum Node<K, V> {
    LeafNode(Leaf<K, V>),
    BranchNode(Branch<K, V>)
}


//Node functions/methods
impl<K: Ord, V> Node<K, V> {
    ///Creates a new leaf node given a vector of elements.
    fn new_leaf(vec: Vec<LeafElt<K, V>>) -> Node<K,V> {
        LeafNode(Leaf::new(vec))
    }

    ///Creates a new branch node given a vector of an elements and a pointer to a rightmost child.
    fn new_branch(vec: Vec<BranchElt<K, V>>, right: Box<Node<K, V>>)
                  -> Node<K, V> {
        BranchNode(Branch::new(vec, right))
    }

    ///Determines whether the given Node contains a Branch or a Leaf.
    ///Used in testing.
    fn is_leaf(&self) -> bool {
        match self {
            &LeafNode(..) => true,
            &BranchNode(..) => false
        }
    }

    ///A binary search function for Nodes.
    ///Calls either the Branch's or the Leaf's bsearch function.
    fn bsearch_node(&self, k: K) -> Option<uint> {
         match self {
             &LeafNode(ref leaf) => leaf.bsearch_leaf(k),
             &BranchNode(ref branch) => branch.bsearch_branch(k)
         }
     }
}

impl<K: Clone + Ord, V: Clone> Node<K, V> {
    ///Returns the corresponding value to the provided key.
    ///get() is called in different ways on a branch or a leaf.
    fn get(&self, k: K) -> Option<V> {
        match *self {
            LeafNode(ref leaf) => return leaf.get(k),
            BranchNode(ref branch) => return branch.get(k)
        }
    }

    ///Matches on the Node, then performs and returns the appropriate insert method.
    fn insert(self, k: K, v: V, ub: uint) -> (Node<K, V>, bool) {
        match self {
            LeafNode(leaf) => leaf.insert(k, v, ub),
            BranchNode(branch) => branch.insert(k, v, ub)
        }
    }
}

impl<K: Clone + Ord, V: Clone> Clone for Node<K, V> {
    ///Returns a new node based on whether or not it is a branch or a leaf.
    fn clone(&self) -> Node<K, V> {
        match *self {
            LeafNode(ref leaf) => {
                Node::new_leaf(leaf.elts.clone())
            }
            BranchNode(ref branch) => {
                Node::new_branch(branch.elts.clone(),
                                 branch.rightmost_child.clone())
            }
        }
    }
}

impl<K: Ord, V: Eq> PartialEq for Node<K, V> {
    fn eq(&self, other: &Node<K, V>) -> bool {
        match *self{
            BranchNode(ref branch) => {
                if other.is_leaf() {
                    return false;
                }
                match *other {
                    BranchNode(ref branch2) => branch.cmp(branch2) == Equal,
                    LeafNode(..) => false
                }
            }
            LeafNode(ref leaf) => {
                match *other {
                    LeafNode(ref leaf2) => leaf.cmp(leaf2) == Equal,
                    BranchNode(..) => false
                }
            }
        }
    }
}

impl<K: Ord, V: Eq> Eq for Node<K, V> {}

impl<K: Ord, V: Eq> PartialOrd for Node<K, V> {
    fn partial_cmp(&self, other: &Node<K, V>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V: Eq> Ord for Node<K, V> {
    ///Implementation of Ord for Nodes.
    fn cmp(&self, other: &Node<K, V>) -> Ordering {
        match *self {
            LeafNode(ref leaf) => {
                match *other {
                    LeafNode(ref leaf2) => leaf.cmp(leaf2),
                    BranchNode(_) => Less
                }
            }
            BranchNode(ref branch) => {
                match *other {
                    BranchNode(ref branch2) => branch.cmp(branch2),
                    LeafNode(_) => Greater
                }
            }
        }
    }
}

impl<K: fmt::Show + Ord, V: fmt::Show> fmt::Show for Node<K, V> {
    ///Returns a string representation of a Node.
    ///Will iterate over the Node and show "Key: x, value: y, child: () // "
    ///for all elements in the Node. "Child" only exists if the Node contains
    ///a branch.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LeafNode(ref leaf) => leaf.fmt(f),
            BranchNode(ref branch) => branch.fmt(f),
        }
    }
}


//A leaf is a vector with elements that contain no children.  A leaf also
//does not contain a rightmost child.
struct Leaf<K, V> {
    elts: Vec<LeafElt<K, V>>
}

//Vector of values with children, plus a rightmost child (greater than all)
struct Branch<K, V> {
    elts: Vec<BranchElt<K,V>>,
    rightmost_child: Box<Node<K, V>>,
}


impl<K: Ord, V> Leaf<K, V> {
    ///Creates a new Leaf from a vector of LeafElts.
    fn new(vec: Vec<LeafElt<K, V>>) -> Leaf<K, V> {
        Leaf {
            elts: vec
        }
    }

    ///Searches a leaf for a spot for a new element using a binary search.
    ///Returns None if the element is already in the vector.
    fn bsearch_leaf(&self, k: K) -> Option<uint> {
        let mut high: uint = self.elts.len();
        let mut low: uint = 0;
        let mut midpoint: uint = (high - low) / 2 ;
        if midpoint == high {
            midpoint = 0;
        }
        loop {
            let order = self.elts[midpoint].key.cmp(&k);
            match order {
                Equal => {
                    return None;
                }
                Greater => {
                    if midpoint > 0 {
                        if self.elts[midpoint - 1].key.cmp(&k) == Less {
                            return Some(midpoint);
                        }
                        else {
                            let tmp = midpoint;
                            midpoint = midpoint / 2;
                            high = tmp;
                            continue;
                        }
                    }
                    else {
                        return Some(0);
                    }
                }
                Less => {
                    if midpoint + 1 < self.elts.len() {
                        if self.elts[midpoint + 1].key.cmp(&k) == Greater {
                            return Some(midpoint);
                        }
                        else {
                            let tmp = midpoint;
                            midpoint = (high + low) / 2;
                            low = tmp;
                        }
                    }
                    else {
                        return Some(self.elts.len());
                    }
                }
            }
        }
    }
}


impl<K: Clone + Ord, V: Clone> Leaf<K, V> {
    ///Returns the corresponding value to the supplied key.
    fn get(&self, k: K) -> Option<V> {
        for s in self.elts.iter() {
            let order = s.key.cmp(&k);
            match order {
                Equal => return Some(s.value.clone()),
                _ => {}
            }
        }
        return None;
    }

    ///Uses clone() to facilitate inserting new elements into a tree.
    fn insert(mut self, k: K, v: V, ub: uint) -> (Node<K, V>, bool) {
        let to_insert = LeafElt::new(k, v);
        let index: Option<uint> = self.bsearch_leaf(to_insert.clone().key);
        //Check index to see whether we actually inserted the element into the vector.
        match index {
            //If the index is None, the new element already exists in the vector.
            None => {
                return (Node::new_leaf(self.clone().elts), false);
            }
            //If there is an index, insert at that index.
            Some(i) => {
                if i >= self.elts.len() {
                    self.elts.push(to_insert.clone());
                }
                else {
                    self.elts.insert(i, to_insert.clone());
                }
            }
        }
        //If we have overfilled the vector (by making its size greater than the
        //upper bound), we return a new Branch with one element and two children.
        if self.elts.len() > ub {
            let midpoint_opt = self.elts.remove(ub / 2);
            let midpoint = midpoint_opt.unwrap();
            let (left_leaf, right_leaf) = self.elts.partition(|le|
                                                              le.key.cmp(&midpoint.key.clone())
                                                              == Less);
            let branch_return = Node::new_branch(vec!(BranchElt::new(midpoint.key.clone(),
                                                                  midpoint.value.clone(),
                                                             box Node::new_leaf(left_leaf))),
                                            box Node::new_leaf(right_leaf));
            return (branch_return, true);
        }
        (Node::new_leaf(self.elts.clone()), true)
    }
}

impl<K: Clone + Ord, V: Clone> Clone for Leaf<K, V> {
    ///Returns a new Leaf with the same elts.
    fn clone(&self) -> Leaf<K, V> {
        Leaf::new(self.elts.clone())
    }
}

impl<K: Ord, V: Eq> PartialEq for Leaf<K, V> {
    fn eq(&self, other: &Leaf<K, V>) -> bool {
        self.elts == other.elts
    }
}

impl<K: Ord, V: Eq> Eq for Leaf<K, V> {}

impl<K: Ord, V: Eq> PartialOrd for Leaf<K, V> {
    fn partial_cmp(&self, other: &Leaf<K, V>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V: Eq> Ord for Leaf<K, V> {
    ///Returns an ordering based on the first element of each Leaf.
    fn cmp(&self, other: &Leaf<K, V>) -> Ordering {
        if self.elts.len() > other.elts.len() {
            return Greater;
        }
        if self.elts.len() < other.elts.len() {
            return Less;
        }
        self.elts[0].cmp(&other.elts[0])
    }
}


impl<K: fmt::Show + Ord, V: fmt::Show> fmt::Show for Leaf<K, V> {
    ///Returns a string representation of a Leaf.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, s) in self.elts.iter().enumerate() {
            if i != 0 { try!(write!(f, " // ")) }
            try!(write!(f, "{}", *s))
        }
        Ok(())
    }
}


impl<K: Ord, V> Branch<K, V> {
    ///Creates a new Branch from a vector of BranchElts and a rightmost child (a node).
    fn new(vec: Vec<BranchElt<K, V>>, right: Box<Node<K, V>>)
           -> Branch<K, V> {
        Branch {
            elts: vec,
            rightmost_child: right
        }
    }

    fn bsearch_branch(&self, k: K) -> Option<uint> {
        let mut midpoint: uint = self.elts.len() / 2;
        let mut high: uint = self.elts.len();
        let mut low: uint = 0u;
        if midpoint == high {
            midpoint = 0u;
        }
        loop {
            let order = self.elts[midpoint].key.cmp(&k);
            match order {
                Equal => {
                    return None;
                }
                Greater => {
                    if midpoint > 0 {
                        if self.elts[midpoint - 1].key.cmp(&k) == Less {
                            return Some(midpoint);
                        }
                        else {
                            let tmp = midpoint;
                            midpoint = (midpoint - low) / 2;
                            high = tmp;
                            continue;
                        }
                    }
                    else {
                        return Some(0);
                    }
                }
                Less => {
                    if midpoint + 1 < self.elts.len() {
                        if self.elts[midpoint + 1].key.cmp(&k) == Greater {
                            return Some(midpoint);
                        }
                        else {
                            let tmp = midpoint;
                            midpoint = (high - midpoint) / 2;
                            low = tmp;
                        }
                    }
                    else {
                        return Some(self.elts.len());
                    }
                }
            }
        }
    }
}

impl<K: Clone + Ord, V: Clone> Branch<K, V> {
    ///Returns the corresponding value to the supplied key.
    ///If the key is not there, find the child that might hold it.
    fn get(&self, k: K) -> Option<V> {
        for s in self.elts.iter() {
            let order = s.key.cmp(&k);
            match order {
                Less => return s.left.get(k),
                Equal => return Some(s.value.clone()),
                _ => {}
            }
        }
        self.rightmost_child.get(k)
    }

    ///An insert method that uses .clone() for support.
    fn insert(mut self, k: K, v: V, ub: uint) -> (Node<K, V>, bool) {
        let mut new_branch = Node::new_branch(self.clone().elts, self.clone().rightmost_child);
        let mut outcome = false;
        let index: Option<uint> = new_branch.bsearch_node(k.clone());
        //First, find which path down the tree will lead to the appropriate leaf
        //for the key-value pair.
        match index.clone() {
            None => {
                return (Node::new_branch(self.clone().elts,
                                         self.clone().rightmost_child),
                        outcome);
            }
            Some(i) => {
                if i == self.elts.len() {
                    let new_outcome = self.clone().rightmost_child.insert(k.clone(),
                                                                          v.clone(),
                                                                          ub.clone());
                    new_branch = new_outcome.clone().val0();
                    outcome = new_outcome.val1();
                }
                else {
                    let new_outcome = self.elts[i].left.clone().insert(k.clone(),
                                                                       v.clone(),
                                                                       ub.clone());
                    new_branch = new_outcome.clone().val0();
                    outcome = new_outcome.val1();
                }
                //Check to see whether a branch or a leaf was returned from the
                //tree traversal.
                match new_branch.clone() {
                    //If we have a leaf, we do not need to resize the tree,
                    //so we can return false.
                    LeafNode(..) => {
                        if i == self.elts.len() {
                            self.rightmost_child = box new_branch.clone();
                        }
                        else {
                            self.elts.get_mut(i).left = box new_branch.clone();
                        }
                        return (Node::new_branch(self.clone().elts,
                                                 self.clone().rightmost_child),
                                true);
                    }
                    //If we have a branch, we might need to refactor the tree.
                    BranchNode(..) => {}
                }
            }
        }
        //If we inserted something into the tree, do the following:
        if outcome {
            match new_branch.clone() {
                //If we have a new leaf node, integrate it into the current branch
                //and return it, saying we have inserted a new element.
                LeafNode(..) => {
                    if index.unwrap() == self.elts.len() {
                        self.rightmost_child = box new_branch;
                    }
                    else {
                        self.elts.get_mut(index.unwrap()).left = box new_branch;
                    }
                    return (Node::new_branch(self.clone().elts,
                                             self.clone().rightmost_child),
                            true);
                }
                //If we have a new branch node, attempt to insert it into the tree
                //as with the key-value pair, then check to see if the node is overfull.
                BranchNode(branch) => {
                    let new_elt = branch.elts[0].clone();
                    let new_elt_index = self.bsearch_branch(new_elt.clone().key);
                    match new_elt_index {
                        None => {
                            return (Node::new_branch(self.clone().elts,
                                                     self.clone().rightmost_child),
                                    false);
                            }
                        Some(i) => {
                            self.elts.insert(i, new_elt);
                            if i + 1 >= self.elts.len() {
                                self.rightmost_child = branch.clone().rightmost_child;
                            }
                            else {
                                self.elts.get_mut(i + 1).left =
                                    branch.clone().rightmost_child;
                            }
                        }
                    }
                }
            }
            //If the current node is overfilled, create a new branch with one element
            //and two children.
            if self.elts.len() > ub {
                let midpoint = self.elts.remove(ub / 2).unwrap();
                let (new_left, new_right) = self.clone().elts.partition(|le|
                                                                midpoint.key.cmp(&le.key)
                                                                        == Greater);
                new_branch = Node::new_branch(
                    vec!(BranchElt::new(midpoint.clone().key,
                                     midpoint.clone().value,
                                     box Node::new_branch(new_left,
                                                       midpoint.clone().left))),
                    box Node::new_branch(new_right, self.clone().rightmost_child));
                return (new_branch, true);
            }
        }
        (Node::new_branch(self.elts.clone(), self.rightmost_child.clone()), outcome)
    }
}

impl<K: Clone + Ord, V: Clone> Clone for Branch<K, V> {
    ///Returns a new branch using the clone methods of the Branch's internal variables.
    fn clone(&self) -> Branch<K, V> {
        Branch::new(self.elts.clone(), self.rightmost_child.clone())
    }
}

impl<K: Ord, V: Eq> PartialEq for Branch<K, V> {
    fn eq(&self, other: &Branch<K, V>) -> bool {
        self.elts == other.elts
    }
}

impl<K: Ord, V: Eq> Eq for Branch<K, V> {}

impl<K: Ord, V: Eq> PartialOrd for Branch<K, V> {
    fn partial_cmp(&self, other: &Branch<K, V>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V: Eq> Ord for Branch<K, V> {
    ///Compares the first elements of two branches to determine an ordering
    fn cmp(&self, other: &Branch<K, V>) -> Ordering {
        if self.elts.len() > other.elts.len() {
            return Greater;
        }
        if self.elts.len() < other.elts.len() {
            return Less;
        }
        self.elts[0].cmp(&other.elts[0])
    }
}

impl<K: fmt::Show + Ord, V: fmt::Show> fmt::Show for Branch<K, V> {
    ///Returns a string representation of a Branch.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, s) in self.elts.iter().enumerate() {
            if i != 0 { try!(write!(f, " // ")) }
            try!(write!(f, "{}", *s))
        }
        write!(f, " // rightmost child: ({}) ", *self.rightmost_child)
    }
}

//A LeafElt contains no left child, but a key-value pair.
struct LeafElt<K, V> {
    key: K,
    value: V
}

//A BranchElt has a left child in insertion to a key-value pair.
struct BranchElt<K, V> {
    left: Box<Node<K, V>>,
    key: K,
    value: V
}

impl<K: Ord, V> LeafElt<K, V> {
    ///Creates a new LeafElt from a supplied key-value pair.
    fn new(k: K, v: V) -> LeafElt<K, V> {
        LeafElt {
            key: k,
            value: v
        }
    }
}

impl<K: Clone + Ord, V: Clone> Clone for LeafElt<K, V> {
    ///Returns a new LeafElt by cloning the key and value.
    fn clone(&self) -> LeafElt<K, V> {
        LeafElt::new(self.key.clone(), self.value.clone())
    }
}

impl<K: Ord, V: Eq> PartialEq for LeafElt<K, V> {
    fn eq(&self, other: &LeafElt<K, V>) -> bool {
        self.key == other.key && self.value == other.value
    }
}

impl<K: Ord, V: Eq> Eq for LeafElt<K, V> {}

impl<K: Ord, V: Eq> PartialOrd for LeafElt<K, V> {
    fn partial_cmp(&self, other: &LeafElt<K, V>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V: Eq> Ord for LeafElt<K, V> {
    ///Returns an ordering based on the keys of the LeafElts.
    fn cmp(&self, other: &LeafElt<K, V>) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl<K: fmt::Show + Ord, V: fmt::Show> fmt::Show for LeafElt<K, V> {
    ///Returns a string representation of a LeafElt.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Key: {}, value: {};", self.key, self.value)
    }
}

impl<K: Ord, V> BranchElt<K, V> {
    ///Creates a new BranchElt from a supplied key, value, and left child.
    fn new(k: K, v: V, n: Box<Node<K, V>>) -> BranchElt<K, V> {
        BranchElt {
            left: n,
            key: k,
            value: v
        }
    }
}


impl<K: Clone + Ord, V: Clone> Clone for BranchElt<K, V> {
    ///Returns a new BranchElt by cloning the key, value, and left child.
    fn clone(&self) -> BranchElt<K, V> {
        BranchElt::new(self.key.clone(),
                       self.value.clone(),
                       self.left.clone())
    }
}

impl<K: Ord, V: Eq> PartialEq for BranchElt<K, V>{
    fn eq(&self, other: &BranchElt<K, V>) -> bool {
        self.key == other.key && self.value == other.value
    }
}

impl<K: Ord, V: Eq> Eq for BranchElt<K, V>{}

impl<K: Ord, V: Eq> PartialOrd for BranchElt<K, V> {
    fn partial_cmp(&self, other: &BranchElt<K, V>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Ord, V: Eq> Ord for BranchElt<K, V> {
    ///Fulfills Ord for BranchElts
    fn cmp(&self, other: &BranchElt<K, V>) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl<K: fmt::Show + Ord, V: fmt::Show> fmt::Show for BranchElt<K, V> {
    /// Returns string containing key, value, and child (which should recur to a
    /// leaf) Consider changing in future to be more readable.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Key: {}, value: {}, (child: {})",
               self.key, self.value, *self.left)
    }
}

#[cfg(test)]
mod test_btree {
    use std::prelude::*;

    use super::{BTree, Node, LeafElt};

    use MutableSeq;

    //Tests the functionality of the insert methods (which are unfinished).
    #[test]
    fn insert_test_one() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        let is_insert = b.insert(2i, "xyz".to_string());
        assert!(is_insert.root.is_leaf());
    }

    #[test]
    fn insert_test_two() {
        let leaf_elt_1 = LeafElt::new(1i, "aaa".to_string());
        let leaf_elt_2 = LeafElt::new(2i, "bbb".to_string());
        let leaf_elt_3 = LeafElt::new(3i, "ccc".to_string());
        let n = Node::new_leaf(vec!(leaf_elt_1, leaf_elt_2, leaf_elt_3));
        let b = BTree::new_with_node_len(n, 3, 2);
        //println!("{}", b.clone().insert(4, "ddd".to_string()).to_string());
        assert!(b.insert(4, "ddd".to_string()).root.is_leaf());
    }

    #[test]
    fn insert_test_three() {
        let leaf_elt_1 = LeafElt::new(1i, "aaa".to_string());
        let leaf_elt_2 = LeafElt::new(2i, "bbb".to_string());
        let leaf_elt_3 = LeafElt::new(3i, "ccc".to_string());
        let leaf_elt_4 = LeafElt::new(4i, "ddd".to_string());
        let n = Node::new_leaf(vec!(leaf_elt_1, leaf_elt_2, leaf_elt_3, leaf_elt_4));
        let b = BTree::new_with_node_len(n, 3, 2);
        //println!("{}", b.clone().insert(5, "eee".to_string()).to_string());
        assert!(!b.insert(5, "eee".to_string()).root.is_leaf());
    }

    #[test]
    fn insert_test_four() {
        let leaf_elt_1 = LeafElt::new(1i, "aaa".to_string());
        let leaf_elt_2 = LeafElt::new(2i, "bbb".to_string());
        let leaf_elt_3 = LeafElt::new(3i, "ccc".to_string());
        let leaf_elt_4 = LeafElt::new(4i, "ddd".to_string());
        let n = Node::new_leaf(vec!(leaf_elt_1, leaf_elt_2, leaf_elt_3, leaf_elt_4));
        let mut b = BTree::new_with_node_len(n, 3, 2);
        b = b.clone().insert(5, "eee".to_string());
        b = b.clone().insert(6, "fff".to_string());
        b = b.clone().insert(7, "ggg".to_string());
        b = b.clone().insert(8, "hhh".to_string());
        b = b.clone().insert(0, "omg".to_string());
        //println!("{}", b.clone().to_string());
        assert!(!b.root.is_leaf());
    }

    #[test]
    fn bsearch_test_one() {
        let b = BTree::new(1i, "abc".to_string(), 2u);
        assert_eq!(Some(1), b.root.bsearch_node(2));
    }

    #[test]
    fn bsearch_test_two() {
        let b = BTree::new(1i, "abc".to_string(), 2u);
        assert_eq!(Some(0), b.root.bsearch_node(0));
    }

    #[test]
    fn bsearch_test_three() {
        let leaf_elt_1 = LeafElt::new(1i, "aaa".to_string());
        let leaf_elt_2 = LeafElt::new(2i, "bbb".to_string());
        let leaf_elt_3 = LeafElt::new(4i, "ccc".to_string());
        let leaf_elt_4 = LeafElt::new(5i, "ddd".to_string());
        let n = Node::new_leaf(vec!(leaf_elt_1, leaf_elt_2, leaf_elt_3, leaf_elt_4));
        let b = BTree::new_with_node_len(n, 3, 2);
        assert_eq!(Some(2), b.root.bsearch_node(3));
    }

    #[test]
    fn bsearch_test_four() {
        let leaf_elt_1 = LeafElt::new(1i, "aaa".to_string());
        let leaf_elt_2 = LeafElt::new(2i, "bbb".to_string());
        let leaf_elt_3 = LeafElt::new(4i, "ccc".to_string());
        let leaf_elt_4 = LeafElt::new(5i, "ddd".to_string());
        let n = Node::new_leaf(vec!(leaf_elt_1, leaf_elt_2, leaf_elt_3, leaf_elt_4));
        let b = BTree::new_with_node_len(n, 3, 2);
        assert_eq!(Some(4), b.root.bsearch_node(800));
    }

    //Tests the functionality of the get method.
    #[test]
    fn get_test() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        let val = b.get(1);
        assert_eq!(val, Some("abc".to_string()));
    }

    //Tests the BTree's clone() method.
    #[test]
    fn btree_clone_test() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        let b2 = b.clone();
        assert!(b.root == b2.root)
    }

    //Tests the BTree's cmp() method when one node is "less than" another.
    #[test]
    fn btree_cmp_test_less() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        let b2 = BTree::new(2i, "bcd".to_string(), 2);
        assert!(&b.cmp(&b2) == &Less)
    }

    //Tests the BTree's cmp() method when two nodes are equal.
    #[test]
    fn btree_cmp_test_eq() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        let b2 = BTree::new(1i, "bcd".to_string(), 2);
        assert!(&b.cmp(&b2) == &Equal)
    }

    //Tests the BTree's cmp() method when one node is "greater than" another.
    #[test]
    fn btree_cmp_test_greater() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        let b2 = BTree::new(2i, "bcd".to_string(), 2);
        assert!(&b2.cmp(&b) == &Greater)
    }

    //Tests the BTree's to_string() method.
    #[test]
    fn btree_tostr_test() {
        let b = BTree::new(1i, "abc".to_string(), 2);
        assert_eq!(b.to_string(), "Key: 1, value: abc;".to_string())
    }

}
