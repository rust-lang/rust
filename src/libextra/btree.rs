//
// btree.rs
// Nif Ward
// 10/24/13
//
// starting implementation of a btree for rust
// inspired by github user davidhalperin's gist


//What's in a BTree?
pub struct BTree<K, V>{
    root: Node<K, V>,
    len: uint,
    lower_bound: uint,
    upper_bound: uint
}


impl<K: Clone + TotalOrd, V: Clone> BTree<K, V>{
    
    //Returns new BTree with root node (leaf) and user-supplied lower bound
    fn new(k: K, v: V, lb: uint) -> BTree<K, V>{
        BTree{
	    root: Node::new_leaf(~[LeafElt::new(k, v)]),
	    len: 1,
	    lower_bound: lb,
	    upper_bound: 2 * lb
        }
    }

    //Helper function for clone
    fn new_with_node_len(n: Node<K, V>, length: uint, lb: uint) -> BTree<K, V>{
        BTree{
	    root: n,
	    len: length,
	    lower_bound: lb,
	    upper_bound: 2 * lb
	}
    }


    fn clone(&self) -> BTree<K, V>{
        return BTree::new_with_node_len(self.root.clone(), self.len, self.lower_bound);
    }

    fn get(self, k: K) -> Option<V>{
        return self.root.get(k);
    }


    fn add(self, k: K, v: V) -> bool{
        let is_get = &self.clone().get(k.clone());
	if is_get.is_some(){ return false; }
	else{
	    std::util::replace(&mut self.root.clone(),self.root.add(k.clone(), v));
	    return true;
	}

    }



}

impl<K: ToStr + TotalOrd, V: ToStr> ToStr for BTree<K, V>{
    //Returns a string representation of the BTree
    fn to_str(&self) -> ~str{
        let ret=self.root.to_str();
	return ret;
    }
}


//Node types
enum Node<K, V>{
    LeafNode(Leaf<K, V>),
    BranchNode(Branch<K, V>)
}


//Node functions/methods
impl<K: Clone + TotalOrd, V: Clone> Node<K, V>{
    //differentiates between leaf and branch nodes
    fn is_leaf(&self) -> bool{
        match self{
	    &LeafNode(*) => true,
	    &BranchNode(*) => false
        }
    }
    
    //Creates a new leaf or branch node
    fn new_leaf(vec: ~[LeafElt<K, V>]) -> Node<K,V>{
         LeafNode(Leaf::new(vec))
    }
    fn new_branch(vec: ~[BranchElt<K, V>], right: ~Node<K, V>) -> Node<K, V>{
        BranchNode(Branch::new(vec, right))
    }

    fn get(&self, k: K) -> Option<V>{
        match *self{
	    LeafNode(ref leaf) => return leaf.get(k),
	    BranchNode(ref branch) => return branch.get(k)
        }
    }

    //A placeholder for add
    //Currently returns a leaf node with a single value (the added one)
    fn add(self, k: K, v: V) -> Node<K, V>{
        return Node::new_leaf(~[LeafElt::new(k, v)]);
    }
}


impl<K: Clone + TotalOrd, V: Clone> Clone for Node<K, V>{
    fn clone(&self) -> Node<K, V>{
        match *self{
	    LeafNode(ref leaf) => return Node::new_leaf(leaf.elts.clone()),
	    BranchNode(ref branch) => return Node::new_branch(branch.elts.clone(), branch.rightmost_child.clone())
	}
    }
}

impl<K: Clone + TotalOrd, V: Clone> TotalOrd for Node<K, V>{
    #[allow(unused_variable)]
    fn cmp(&self, other: &Node<K, V>) -> Ordering{
        //Requires a match statement--defer these procs to branch and leaf.
        /* if self.elts[0].less_than(other.elts[0]) { return Less}
	if self.elts[0].greater_than(other.elts[0]) {return Greater}
	else {return Equal}
	*/
	return Equal;
    }
}

impl<K: Clone + TotalOrd, V: Clone> TotalEq for Node<K, V>{
    //Making sure Nodes have TotalEq
    #[allow(unused_variable)]
    fn equals(&self, other: &Node<K, V>) -> bool{
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


impl<K: ToStr + TotalOrd, V: ToStr> ToStr for Node<K, V>{
    fn to_str(&self) -> ~str{
       match *self{
           LeafNode(ref leaf) => leaf.to_str(),
	   BranchNode(*) => ~""
       }
    }
}


//Array with no children
struct Leaf<K, V>{
    elts: ~[LeafElt<K, V>]
}

//Array of values with children, plus a rightmost child (greater than all)
struct Branch<K, V>{
    elts: ~[BranchElt<K,V>],
    rightmost_child: ~Node<K, V>
}


impl<K: Clone + TotalOrd, V: Clone> Leaf<K, V>{
    //Constructor takes in a vector of leaves
    fn new(vec: ~[LeafElt<K, V>]) -> Leaf<K, V>{
        Leaf{
            elts: vec
        }
    }


    fn get(&self, k: K) -> Option<V>{
        for s in self.elts.iter(){
	    let order=s.key.cmp(&k);
	    match order{
	        Equal => return Some(s.value.clone()),
		_ => {}
	    }
	}
	return None;
    }

    //Add method in progress
    fn add(&self, k: K, v: V) -> Node<K, V>{
        return Node::new_leaf(~[LeafElt::new(k, v)]);
    }

}

impl<K: ToStr + TotalOrd, V: ToStr> ToStr for Leaf<K, V>{
    fn to_str(&self) -> ~str{
       let mut ret=~"";
       for s in self.elts.iter(){
           ret = ret+" // "+ s.to_str();
       }
       return ret;
    }

}


impl<K: Clone + TotalOrd, V: Clone> Branch<K, V>{
    //constructor takes a branch vector and a rightmost child
    fn new(vec: ~[BranchElt<K, V>], right: ~Node<K, V>) -> Branch<K, V>{
        Branch{
	    elts: vec,
	    rightmost_child: right
        }
    }

    fn get(&self, k: K) -> Option<V>{
        for s in self.elts.iter(){
	    let order = s.key.cmp(&k);
	    match order{
	        Less => return s.left.get(k),
		Equal => return Some(s.value.clone()),
		_ => {}
	    }
	}
	return self.rightmost_child.get(k);
    }


    //Add method in progress
    fn add(&self, k: K, v: V) -> Node<K, V>{
        return Node::new_leaf(~[LeafElt::new(k, v)]);
    }
}

//No left child
struct LeafElt<K, V>{
    key: K,
    value: V
}

//Has a left child
struct BranchElt<K, V>{
    left: Node<K, V>,
    key: K,
    value: V
}

impl<K: Clone + TotalOrd, V> LeafElt<K, V>{
    fn new(k: K, v: V) -> LeafElt<K, V>{
        LeafElt{
            key: k,
	    value: v
	}
    }

    fn less_than(&self, other: LeafElt<K, V>) -> bool{
        let order = self.key.cmp(&other.key);
	match order{
	    Less => true,
	    _ => false
	}
    }

    fn greater_than(&self, other: LeafElt<K, V>) -> bool{
        let order = self.key.cmp(&other.key);
	match order{
	    Greater => true,
	    _ => false
	}
    }


    fn has_key(&self, other: K) -> bool{
        let order = self.key.cmp(&other);
	match order{
	    Equal => true,
	    _ => false
	}
    }

}

impl<K: Clone + TotalOrd, V: Clone> Clone for LeafElt<K, V>{
    fn clone(&self) -> LeafElt<K, V>{
        return LeafElt::new(self.key.clone(), self.value.clone());
    }
}

impl<K: ToStr + TotalOrd, V: ToStr> ToStr for LeafElt<K, V>{
    fn to_str(&self) -> ~str{
        return "Key: "+self.key.to_str()+", value: "+self.value.to_str()+"; ";
    }

}

impl<K: Clone + TotalOrd, V: Clone> BranchElt<K, V>{
    fn new(k: K, v: V, n: Node<K, V>) -> BranchElt<K, V>{
        BranchElt{
            left: n,
            key: k,
            value: v
        }
    }

    //Add method in progress.  Should it return a branch or a leaf elt?  It will depend on implementation.
    fn add(&self, k: K, v: V) -> LeafElt<K, V>{
        return LeafElt::new(k, v);
    }
}

impl<K: Clone + TotalOrd, V: Clone> Clone for BranchElt<K, V>{
    fn clone(&self) -> BranchElt<K, V>{
        return BranchElt::new(self.key.clone(), self.value.clone(), self.left.clone());
    }
}

#[test]
fn add_test(){
    let b = BTree::new(1, ~"abc", 2);
    let is_add = b.add(2, ~"xyz");
    assert!(is_add);

}

#[test]
fn get_test(){
    let b = BTree::new(1, ~"abc", 2);
    let val = b.get(1);
    assert_eq!(val, Some(~"abc"));
}

//Testing LeafElt<K, V> functions (less_than, greater_than, and has_key)
#[test]
fn leaf_lt(){
    let l1 = LeafElt::new(1, ~"abc");
    let l2 = LeafElt::new(2, ~"xyz");
    assert!(l1.less_than(l2));
}

#[test]
fn leaf_gt(){
    let l1 = LeafElt::new(1, ~"abc");
    let l2 = LeafElt::new(2, ~"xyz");
    assert!(l2.greater_than(l1));
}

#[test]
fn leaf_hk(){
    let l1 = LeafElt::new(1, ~"abc");
    assert!(l1.has_key(1));
}

fn main(){


}
