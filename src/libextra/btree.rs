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

//! Starting implementation of a btree for rust.
//! Structure inspired by github user davidhalperin's gist.

#[allow(dead_code)];
use std::util::replace;

///A B-tree contains a root node (which contains a vector of elements),
///a length (the height of the tree), and lower and upper bounds on the
///number of elements that a given node can contain.
#[allow(missing_doc)]
pub struct BTree<K, V> {
    root: Node<K, V>,
    len: uint,
    lower_bound: uint,
    upper_bound: uint
}

//We would probably want to remove the dependence on the Clone trait in the future.
//It is here as a crutch to ensure values can be passed around through the tree's nodes
//especially during insertions and deletions.
//Using the swap or replace methods is one option for replacing dependence on Clone, or
//changing the way in which the BTree is stored could also potentially work.
impl<K: Clone + TotalOrd, V: Clone> BTree<K, V> {

    ///Returns new BTree with root node (leaf) and user-supplied lower bound
    pub fn new(k: K, v: V, lb: uint) -> BTree<K, V> {
        BTree {
            root: Node::new_leaf(~[LeafElt::new(k, v)]),
            len: 1,
            lower_bound: lb,
            upper_bound: 2 * lb
        }
    }

    ///Helper function for clone: returns new BTree with supplied root node,
    ///length, and lower bound.  For use when the length is known already.
    pub fn new_with_node_len(n: Node<K, V>,
                             length: uint,
                             lb: uint) -> BTree<K, V> {
        BTree {
            root: n,
            len: length,
            lower_bound: lb,
            upper_bound: 2 * lb
        }
    }

    ///Implements the Clone trait for the BTree.
    ///Uses a helper function/constructor to produce a new BTree.
    pub fn clone(&self) -> BTree<K, V> {
        return BTree::new_with_node_len(self.root.clone(), self.len, self.lower_bound);
    }

    ///Returns the value of a given key, which may not exist in the tree.
    ///Calls the root node's get method.
    pub fn get(self, k: K) -> Option<V> {
        return self.root.get(k);
    }

    ///Checks to see if the key already exists in the tree, and if it is not,
    ///the key-value pair is added to the tree by calling add on the root node.
    pub fn add(self, k: K, v: V) -> bool {
        let is_get = &self.clone().get(k.clone());
        if is_get.is_some(){ return false; }
        else {
            replace(&mut self.root.clone(),self.root.add(k.clone(), v));
            return true;
        }
    }
}

impl<K: ToStr + TotalOrd, V: ToStr> ToStr for BTree<K, V> {
    ///Returns a string representation of the BTree
    fn to_str(&self) -> ~str {
        let ret = self.root.to_str();
        ret
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
impl<K: Clone + TotalOrd, V: Clone> Node<K, V> {

    ///Differentiates between leaf and branch nodes.
    fn is_leaf(&self) -> bool{
        match self{
            &LeafNode(..) => true,
            &BranchNode(..) => false
        }
    }

    ///Creates a new leaf node given a vector of elements.
    fn new_leaf(vec: ~[LeafElt<K, V>]) -> Node<K,V> {
        LeafNode(Leaf::new(vec))
    }

    ///Creates a new branch node given a vector of an elements and a pointer to a rightmost child.
    fn new_branch(vec: ~[BranchElt<K, V>], right: ~Node<K, V>) -> Node<K, V> {
        BranchNode(Branch::new(vec, right))
    }


    ///Returns the corresponding value to the provided key.
    ///get() is called in different ways on a branch or a leaf.
    fn get(&self, k: K) -> Option<V> {
        match *self {
            LeafNode(ref leaf) => return leaf.get(k),
            BranchNode(ref branch) => return branch.get(k)
        }
    }

    ///A placeholder for add
    ///Currently returns a leaf node with a single value (the added one)
    fn add(self, k: K, v: V) -> Node<K, V> {
        return Node::new_leaf(~[LeafElt::new(k, v)]);
    }
}

//Again, this might not be necessary in the future.
impl<K: Clone + TotalOrd, V: Clone> Clone for Node<K, V> {

    ///Returns a new node based on whether or not it is a branch or a leaf.
    fn clone(&self) -> Node<K, V> {
        match *self {
            LeafNode(ref leaf) => {
                return Node::new_leaf(leaf.elts.clone());
            }
            BranchNode(ref branch) => {
                return Node::new_branch(branch.elts.clone(),
                                        branch.rightmost_child.clone());
            }
        }
    }
}

//The following impl is unfinished.  Old iterations of code are left in for
//future reference when implementing this trait (commented-out).
impl<K: Clone + TotalOrd, V: Clone> TotalOrd for Node<K, V> {

    ///Placeholder for an implementation of TotalOrd for Nodes.
    #[allow(unused_variable)]
    fn cmp(&self, other: &Node<K, V>) -> Ordering {
        //Requires a match statement--defer these procs to branch and leaf.
        /* if self.elts[0].less_than(other.elts[0]) { return Less}
        if self.elts[0].greater_than(other.elts[0]) {return Greater}
            else {return Equal}
         */
        return Equal;
    }
}

//The following impl is unfinished.  Old iterations of code are left in for
//future reference when implementing this trait (commented-out).
impl<K: Clone + TotalOrd, V: Clone> TotalEq for Node<K, V> {

    ///Placeholder for an implementation of TotalEq for Nodes.
    #[allow(unused_variable)]
    fn equals(&self, other: &Node<K, V>) -> bool {
        /* put in a match and defer this stuff to branch and leaf

        let mut shorter = 0;
        if self.elts.len() <= other.elts.len(){
        shorter = self.elts.len();
    }
            else{
        shorter = other.elts.len();
    }
        let mut i = 0;
        while i < shorter{
        if !self.elts[i].has_key(other.elts[i].key){
        return false;
    }
        i +=1;
    }
        return true;
         */
        return true;
    }
}


impl<K: ToStr + TotalOrd, V: ToStr> ToStr for Node<K, V> {
    ///Returns a string representation of a Node.
    ///The Branch's to_str() is not implemented yet.
    fn to_str(&self) -> ~str {
        match *self {
            LeafNode(ref leaf) => leaf.to_str(),
            BranchNode(..) => ~""
        }
    }
}


//A leaf is a vector with elements that contain no children.  A leaf also
//does not contain a rightmost child.
struct Leaf<K, V> {
    elts: ~[LeafElt<K, V>]
}

//Vector of values with children, plus a rightmost child (greater than all)
struct Branch<K, V> {
    elts: ~[BranchElt<K,V>],
    rightmost_child: ~Node<K, V>
}


impl<K: Clone + TotalOrd, V: Clone> Leaf<K, V> {

    ///Creates a new Leaf from a vector of LeafElts.
    fn new(vec: ~[LeafElt<K, V>]) -> Leaf<K, V> {
        Leaf {
            elts: vec
        }
    }

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

    ///Placeholder for add method in progress.
    ///Currently returns a new Leaf containing a single LeafElt.
    fn add(&self, k: K, v: V) -> Node<K, V> {
        return Node::new_leaf(~[LeafElt::new(k, v)]);
    }

}

impl<K: ToStr + TotalOrd, V: ToStr> ToStr for Leaf<K, V> {

    ///Returns a string representation of a Leaf.
    fn to_str(&self) -> ~str {
        let mut ret = ~"";
        for s in self.elts.iter() {
            ret = ret + " // " + s.to_str();
        }
        ret
    }

}


impl<K: Clone + TotalOrd, V: Clone> Branch<K, V> {

    ///Creates a new Branch from a vector of BranchElts and a rightmost child (a node).
    fn new(vec: ~[BranchElt<K, V>], right: ~Node<K, V>) -> Branch<K, V> {
        Branch {
            elts: vec,
            rightmost_child: right
        }
    }

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
        return self.rightmost_child.get(k);
    }


    ///Placeholder for add method in progress
    fn add(&self, k: K, v: V) -> Node<K, V> {
        return Node::new_leaf(~[LeafElt::new(k, v)]);
    }
}

//A LeafElt containts no left child, but a key-value pair.
struct LeafElt<K, V> {
    key: K,
    value: V
}

//A BranchElt has a left child in addition to a key-value pair.
struct BranchElt<K, V> {
    left: Node<K, V>,
    key: K,
    value: V
}

impl<K: Clone + TotalOrd, V> LeafElt<K, V> {

    ///Creates a new LeafElt from a supplied key-value pair.
    fn new(k: K, v: V) -> LeafElt<K, V> {
        LeafElt {
            key: k,
            value: v
        }
    }

    ///Compares another LeafElt against itself and determines whether
    ///the original LeafElt's key is less than the other one's key.
    fn less_than(&self, other: LeafElt<K, V>) -> bool {
        let order = self.key.cmp(&other.key);
        match order {
            Less => true,
            _ => false
        }
    }

    ///Compares another LeafElt against itself and determines whether
    ///the original LeafElt's key is greater than the other one's key.
    fn greater_than(&self, other: LeafElt<K, V>) -> bool {
        let order = self.key.cmp(&other.key);
        match order {
            Greater => true,
            _ => false
        }
    }

    ///Takes a key and determines whether its own key and the supplied key
    ///are the same.
    fn has_key(&self, other: K) -> bool {
        let order = self.key.cmp(&other);
        match order {
            Equal => true,
            _ => false
        }
    }
}

//This may be eliminated in the future to perserve efficiency by adjusting the way
//the BTree as a whole is stored in memory.
impl<K: Clone + TotalOrd, V: Clone> Clone for LeafElt<K, V> {

    ///Returns a new LeafElt by cloning the key and value.
    fn clone(&self) -> LeafElt<K, V> {
        return LeafElt::new(self.key.clone(), self.value.clone());
    }
}

impl<K: ToStr + TotalOrd, V: ToStr> ToStr for LeafElt<K, V> {

    ///Returns a string representation of a LeafElt.
    fn to_str(&self) -> ~str {
        return "Key: " + self.key.to_str() + ", value: "
                       + self.value.to_str() + "; ";
    }

}

impl<K: Clone + TotalOrd, V: Clone> BranchElt<K, V> {

    ///Creates a new BranchElt from a supplied key, value, and left child.
    fn new(k: K, v: V, n: Node<K, V>) -> BranchElt<K, V> {
        BranchElt {
            left: n,
            key: k,
            value: v
        }
    }

    ///Placeholder for add method in progress.
    ///Overall implementation will determine the actual return value of this method.
    fn add(&self, k: K, v: V) -> LeafElt<K, V> {
        return LeafElt::new(k, v);
    }
}

impl<K: Clone + TotalOrd, V: Clone> Clone for BranchElt<K, V> {

    ///Returns a new BranchElt by cloning the key, value, and left child.
    fn clone(&self) -> BranchElt<K, V> {
        return BranchElt::new(self.key.clone(),
                              self.value.clone(),
                              self.left.clone());
    }
}

#[cfg(test)]
mod test_btree{

    use super::{BTree, LeafElt};

    ///Tests the functionality of the add methods (which are unfinished).
    #[test]
    fn add_test(){
        let b = BTree::new(1, ~"abc", 2);
        let is_add = b.add(2, ~"xyz");
        assert!(is_add);
    }

    ///Tests the functionality of the get method.
    #[test]
    fn get_test(){
        let b = BTree::new(1, ~"abc", 2);
        let val = b.get(1);
        assert_eq!(val, Some(~"abc"));
    }

    ///Tests the LeafElt's less_than() method.
    #[test]
    fn leaf_lt(){
        let l1 = LeafElt::new(1, ~"abc");
        let l2 = LeafElt::new(2, ~"xyz");
        assert!(l1.less_than(l2));
    }


    ///Tests the LeafElt's greater_than() method.
    #[test]
    fn leaf_gt(){
        let l1 = LeafElt::new(1, ~"abc");
        let l2 = LeafElt::new(2, ~"xyz");
        assert!(l2.greater_than(l1));
    }

    ///Tests the LeafElt's has_key() method.
    #[test]
    fn leaf_hk(){
        let l1 = LeafElt::new(1, ~"abc");
        assert!(l1.has_key(1));
    }
}

