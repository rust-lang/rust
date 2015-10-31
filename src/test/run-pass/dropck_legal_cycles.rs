// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test exercises cases where cyclic structure is legal,
// including when the cycles go through data-structures such
// as `Vec` or `TypedArena`.
//
// The intent is to cover as many such cases as possible, ensuring
// that if the compiler did not complain circa Rust 1.x (1.2 as of
// this writing), then it will continue to not complain in the future.
//
// Note that while some of the tests are only exercising using the
// given collection as a "backing store" for a set of nodes that hold
// the actual cycle (and thus the cycle does not go through the
// collection itself in such cases), in general we *do* want to make
// sure to have at least one example exercising a cycle that goes
// through the collection, for every collection type that supports
// this.

// HIGH LEVEL DESCRIPTION OF THE TEST ARCHITECTURE
// -----------------------------------------------
//
// We pick a data structure and want to make a cyclic construction
// from it. Each test of interest is labelled starting with "Cycle N:
// { ... }" where N is the test number and the "..."`is filled in with
// a graphviz-style description of the graph structure that the
// author believes is being made. So "{ a -> b, b -> (c,d), (c,d) -> e }"
// describes a line connected to a diamond:
//
//                           c
//                          / \
//                     a - b   e
//                          \ /
//                           d
//
// (Note that the above directed graph is actually acyclic.)
//
// The different graph structures are often composed of different data
// types. Some may be built atop `Vec`, others atop `HashMap`, etc.
//
// For each graph structure, we actually *confirm* that a cycle exists
// (as a safe-guard against a test author accidentally leaving it out)
// by traversing each graph and "proving" that a cycle exists within it.
//
// To do this, while trying to keep the code uniform (despite working
// with different underlying collection and smart-pointer types), we
// have a standard traversal API:
//
// 1. every node in the graph carries a `mark` (a u32, init'ed to 0).
//
// 2. every node provides a method to visit its children
//
// 3. a traversal attmepts to visit the nodes of the graph and prove that
//    it sees the same node twice. It does this by setting the mark of each
//    node to a fresh non-zero value, and if it sees the current mark, it
//    "knows" that it must have found a cycle, and stops attempting further
//    traversal.
//
// 4. each traversal is controlled by a bit-string that tells it which child
//    it visit when it can take different paths. As a simple example,
//    in a binary tree, 0 could mean "left" (and 1, "right"), so that
//    "00010" means "left, left, left, right, left". (In general it will
//    read as many bits as it needs to choose one child.)
//
//    The graphs in this test are all meant to be very small, and thus
//    short bitstrings of less than 64 bits should always suffice.
//
//    (An earlier version of this test infrastructure simply had any
//    given traversal visit all children it encountered, in a
//    depth-first manner; one problem with this approach is that an
//    acyclic graph can still have sharing, which would then be treated
//    as a repeat mark and reported as a detected cycle.)
//
// The travseral code is a little more complicated because it has been
// programmed in a somewhat defensive manner. For example it also has
// a max threshold for the number of nodes it will visit, to guard
// against scenarios where the nodes are not correctly setting their
// mark when asked. There are various other methods not discussed here
// that are for aiding debugging the test when it runs, such as the
// `name` method that all nodes provide.
//
// So each test:
//
// 1. allocates the nodes in the graph,
//
// 2. sets up the links in the graph,
//
// 3. clones the "ContextData"
//
// 4. chooses a new current mark value for this test
//
// 5. initiates a traversal, potentially from multiple starting points
//    (aka "roots"), with a given control-string (potentially a
//    different string for each root). if it does start from a
//    distinct root, then such a test should also increment the
//    current mark value, so that this traversal is considered
//    distinct from the prior one on this graph structure.
//
//    Note that most of the tests work with the default control string
//    of all-zeroes.
//
// 6. assert that the context confirms that it actually saw a cycle (since a traversal
//    might have terminated, e.g. on a tree structure that contained no cycles).

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::LinkedList;
use std::collections::VecDeque;
use std::collections::btree_map::BTreeMap;
use std::collections::btree_set::BTreeSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, RwLock, Mutex};

const PRINT: bool = false;

pub fn main() {
    let c_orig = ContextData {
        curr_depth: 0,
        max_depth: 3,
        visited: 0,
        max_visits: 1000,
        skipped: 0,
        curr_mark: 0,
        saw_prev_marked: false,
        control_bits: 0,
    };

    // SANITY CHECK FOR TEST SUITE (thus unnumbered)
    // Not a cycle: { v[0] -> (v[1], v[2]), v[1] -> v[3], v[2] -> v[3] };
    let v: Vec<S2> = vec![Named::new("s0"),
                          Named::new("s1"),
                          Named::new("s2"),
                          Named::new("s3")];
    v[0].next.set((Some(&v[1]), Some(&v[2])));
    v[1].next.set((Some(&v[3]), None));
    v[2].next.set((Some(&v[3]), None));
    v[3].next.set((None, None));

    let mut c = c_orig.clone();
    c.curr_mark = 10;
    assert!(!c.saw_prev_marked);
    v[0].descend_into_self(&mut c);
    assert!(!c.saw_prev_marked); // <-- different from below, b/c acyclic above

    if PRINT { println!(""); }

    // Cycle 1: { v[0] -> v[1], v[1] -> v[0] };
    // does not exercise `v` itself
    let v: Vec<S> = vec![Named::new("s0"),
                         Named::new("s1")];
    v[0].next.set(Some(&v[1]));
    v[1].next.set(Some(&v[0]));

    let mut c = c_orig.clone();
    c.curr_mark = 10;
    assert!(!c.saw_prev_marked);
    v[0].descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 2: { v[0] -> v, v[1] -> v }
    let v: V = Named::new("v");
    v.contents[0].set(Some(&v));
    v.contents[1].set(Some(&v));

    let mut c = c_orig.clone();
    c.curr_mark = 20;
    assert!(!c.saw_prev_marked);
    v.descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 3: { hk0 -> hv0, hv0 -> hk0, hk1 -> hv1, hv1 -> hk1 };
    // does not exercise `h` itself

    let mut h: HashMap<H,H> = HashMap::new();
    h.insert(Named::new("hk0"), Named::new("hv0"));
    h.insert(Named::new("hk1"), Named::new("hv1"));
    for (key, val) in h.iter() {
        val.next.set(Some(key));
        key.next.set(Some(val));
    }

    let mut c = c_orig.clone();
    c.curr_mark = 30;
    for (key, _) in h.iter() {
        c.curr_mark += 1;
        c.saw_prev_marked = false;
        key.descend_into_self(&mut c);
        assert!(c.saw_prev_marked);
    }

    if PRINT { println!(""); }

    // Cycle 4: { h -> (hmk0,hmv0,hmk1,hmv1), {hmk0,hmv0,hmk1,hmv1} -> h }

    let mut h: HashMap<HM,HM> = HashMap::new();
    h.insert(Named::new("hmk0"), Named::new("hmv0"));
    h.insert(Named::new("hmk0"), Named::new("hmv0"));
    for (key, val) in h.iter() {
        val.contents.set(Some(&h));
        key.contents.set(Some(&h));
    }

    let mut c = c_orig.clone();
    c.max_depth = 2;
    c.curr_mark = 40;
    for (key, _) in h.iter() {
        c.curr_mark += 1;
        c.saw_prev_marked = false;
        key.descend_into_self(&mut c);
        assert!(c.saw_prev_marked);
        // break;
    }

    if PRINT { println!(""); }

    // Cycle 5: { vd[0] -> vd[1], vd[1] -> vd[0] };
    // does not exercise vd itself
    let mut vd: VecDeque<S> = VecDeque::new();
    vd.push_back(Named::new("d0"));
    vd.push_back(Named::new("d1"));
    vd[0].next.set(Some(&vd[1]));
    vd[1].next.set(Some(&vd[0]));

    let mut c = c_orig.clone();
    c.curr_mark = 50;
    assert!(!c.saw_prev_marked);
    vd[0].descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 6: { vd -> (vd0, vd1), {vd0, vd1} -> vd }
    let mut vd: VecDeque<VD> = VecDeque::new();
    vd.push_back(Named::new("vd0"));
    vd.push_back(Named::new("vd1"));
    vd[0].contents.set(Some(&vd));
    vd[1].contents.set(Some(&vd));

    let mut c = c_orig.clone();
    c.curr_mark = 60;
    assert!(!c.saw_prev_marked);
    vd[0].descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 7: { vm -> (vm0, vm1), {vm0, vm1} -> vm }
    let mut vm: HashMap<usize, VM> = HashMap::new();
    vm.insert(0, Named::new("vm0"));
    vm.insert(1, Named::new("vm1"));
    vm[&0].contents.set(Some(&vm));
    vm[&1].contents.set(Some(&vm));

    let mut c = c_orig.clone();
    c.curr_mark = 70;
    assert!(!c.saw_prev_marked);
    vm[&0].descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 8: { ll -> (ll0, ll1), {ll0, ll1} -> ll }
    let mut ll: LinkedList<LL> = LinkedList::new();
    ll.push_back(Named::new("ll0"));
    ll.push_back(Named::new("ll1"));
    for e in &ll {
        e.contents.set(Some(&ll));
    }

    let mut c = c_orig.clone();
    c.curr_mark = 80;
    for e in &ll {
        c.curr_mark += 1;
        c.saw_prev_marked = false;
        e.descend_into_self(&mut c);
        assert!(c.saw_prev_marked);
        // break;
    }

    if PRINT { println!(""); }

    // Cycle 9: { bh -> (bh0, bh1), {bh0, bh1} -> bh }
    let mut bh: BinaryHeap<BH> = BinaryHeap::new();
    bh.push(Named::new("bh0"));
    bh.push(Named::new("bh1"));
    for b in bh.iter() {
        b.contents.set(Some(&bh));
    }

    let mut c = c_orig.clone();
    c.curr_mark = 90;
    for b in &bh {
        c.curr_mark += 1;
        c.saw_prev_marked = false;
        b.descend_into_self(&mut c);
        assert!(c.saw_prev_marked);
        // break;
    }

    if PRINT { println!(""); }

    // Cycle 10: { btm -> (btk0, btv1), {bt0, bt1} -> btm }
    let mut btm: BTreeMap<BTM, BTM> = BTreeMap::new();
    btm.insert(Named::new("btk0"), Named::new("btv0"));
    btm.insert(Named::new("btk1"), Named::new("btv1"));
    for (k, v) in btm.iter() {
        k.contents.set(Some(&btm));
        v.contents.set(Some(&btm));
    }

    let mut c = c_orig.clone();
    c.curr_mark = 100;
    for (k, _) in &btm {
        c.curr_mark += 1;
        c.saw_prev_marked = false;
        k.descend_into_self(&mut c);
        assert!(c.saw_prev_marked);
        // break;
    }

    if PRINT { println!(""); }

    // Cycle 10: { bts -> (bts0, bts1), {bts0, bts1} -> btm }
    let mut bts: BTreeSet<BTS> = BTreeSet::new();
    bts.insert(Named::new("bts0"));
    bts.insert(Named::new("bts1"));
    for v in bts.iter() {
        v.contents.set(Some(&bts));
    }

    let mut c = c_orig.clone();
    c.curr_mark = 100;
    for b in &bts {
        c.curr_mark += 1;
        c.saw_prev_marked = false;
        b.descend_into_self(&mut c);
        assert!(c.saw_prev_marked);
        // break;
    }

    if PRINT { println!(""); }

    // Cycle 11: { rc0 -> (rc1, rc2), rc1 -> (), rc2 -> rc0 }
    let (rc0, rc1, rc2): (RCRC, RCRC, RCRC);
    rc0 = RCRC::new("rcrc0");
    rc1 = RCRC::new("rcrc1");
    rc2 = RCRC::new("rcrc2");
    rc0.0.borrow_mut().children.0 = Some(&rc1);
    rc0.0.borrow_mut().children.1 = Some(&rc2);
    rc2.0.borrow_mut().children.0 = Some(&rc0);

    let mut c = c_orig.clone();
    c.control_bits = 0b1;
    c.curr_mark = 110;
    assert!(!c.saw_prev_marked);
    rc0.descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // We want to take the previous Rc case and generalize it to Arc.
    //
    // We can use refcells if we're single-threaded (as this test is).
    // If one were to generalize these constructions to a
    // multi-threaded context, then it might seem like we could choose
    // between either a RwLock or a Mutex to hold the owned arcs on
    // each node.
    //
    // Part of the point of this test is to actually confirm that the
    // cycle exists by traversing it. We can do that just fine with an
    // RwLock (since we can grab the child pointers in read-only
    // mode), but we cannot lock a std::sync::Mutex to guard reading
    // from each node via the same pattern, since once you hit the
    // cycle, you'll be trying to acquring the same lock twice.
    // (We deal with this by exiting the traversal early if try_lock fails.)

    // Cycle 12: { arc0 -> (arc1, arc2), arc1 -> (), arc2 -> arc0 }, refcells
    let (arc0, arc1, arc2): (ARCRC, ARCRC, ARCRC);
    arc0 = ARCRC::new("arcrc0");
    arc1 = ARCRC::new("arcrc1");
    arc2 = ARCRC::new("arcrc2");
    arc0.0.borrow_mut().children.0 = Some(&arc1);
    arc0.0.borrow_mut().children.1 = Some(&arc2);
    arc2.0.borrow_mut().children.0 = Some(&arc0);

    let mut c = c_orig.clone();
    c.control_bits = 0b1;
    c.curr_mark = 110;
    assert!(!c.saw_prev_marked);
    arc0.descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 13: { arc0 -> (arc1, arc2), arc1 -> (), arc2 -> arc0 }, rwlocks
    let (arc0, arc1, arc2): (ARCRW, ARCRW, ARCRW);
    arc0 = ARCRW::new("arcrw0");
    arc1 = ARCRW::new("arcrw1");
    arc2 = ARCRW::new("arcrw2");
    arc0.0.write().unwrap().children.0 = Some(&arc1);
    arc0.0.write().unwrap().children.1 = Some(&arc2);
    arc2.0.write().unwrap().children.0 = Some(&arc0);

    let mut c = c_orig.clone();
    c.control_bits = 0b1;
    c.curr_mark = 110;
    assert!(!c.saw_prev_marked);
    arc0.descend_into_self(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 14: { arc0 -> (arc1, arc2), arc1 -> (), arc2 -> arc0 }, mutexs
    let (arc0, arc1, arc2): (ARCM, ARCM, ARCM);
    arc0 = ARCM::new("arcm0");
    arc1 = ARCM::new("arcm1");
    arc2 = ARCM::new("arcm2");
    arc0.1.lock().unwrap().children.0 = Some(&arc1);
    arc0.1.lock().unwrap().children.1 = Some(&arc2);
    arc2.1.lock().unwrap().children.0 = Some(&arc0);

    let mut c = c_orig.clone();
    c.control_bits = 0b1;
    c.curr_mark = 110;
    assert!(!c.saw_prev_marked);
    arc0.descend_into_self(&mut c);
    assert!(c.saw_prev_marked);
}

trait Named {
    fn new(&'static str) -> Self;
    fn name(&self) -> &str;
}

trait Marked<M> {
    fn mark(&self) -> M;
    fn set_mark(&self, mark: M);
}

struct S<'a> {
    name: &'static str,
    mark: Cell<u32>,
    next: Cell<Option<&'a S<'a>>>,
}

impl<'a> Named for S<'a> {
    fn new<'b>(name: &'static str) -> S<'b> {
        S { name: name, mark: Cell::new(0), next: Cell::new(None) }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for S<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

struct S2<'a> {
    name: &'static str,
    mark: Cell<u32>,
    next: Cell<(Option<&'a S2<'a>>, Option<&'a S2<'a>>)>,
}

impl<'a> Named for S2<'a> {
    fn new<'b>(name: &'static str) -> S2<'b> {
        S2 { name: name, mark: Cell::new(0), next: Cell::new((None, None)) }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for S2<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) {
        self.mark.set(mark);
    }
}

struct V<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Vec<Cell<Option<&'a V<'a>>>>,
}

impl<'a> Named for V<'a> {
    fn new<'b>(name: &'static str) -> V<'b> {
        V { name: name,
            mark: Cell::new(0),
            contents: vec![Cell::new(None), Cell::new(None)]
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for V<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

#[derive(Eq)]
struct H<'a> {
    name: &'static str,
    mark: Cell<u32>,
    next: Cell<Option<&'a H<'a>>>,
}

impl<'a> Named for H<'a> {
    fn new<'b>(name: &'static str) -> H<'b> {
        H { name: name, mark: Cell::new(0), next: Cell::new(None) }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for H<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

impl<'a> PartialEq for H<'a> {
    fn eq(&self, rhs: &H<'a>) -> bool {
        self.name == rhs.name
    }
}

impl<'a> Hash for H<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

#[derive(Eq)]
struct HM<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a HashMap<HM<'a>, HM<'a>>>>,
}

impl<'a> Named for HM<'a> {
    fn new<'b>(name: &'static str) -> HM<'b> {
        HM { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for HM<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

impl<'a> PartialEq for HM<'a> {
    fn eq(&self, rhs: &HM<'a>) -> bool {
        self.name == rhs.name
    }
}

impl<'a> Hash for HM<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}


struct VD<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a VecDeque<VD<'a>>>>,
}

impl<'a> Named for VD<'a> {
    fn new<'b>(name: &'static str) -> VD<'b> {
        VD { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for VD<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

struct VM<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a HashMap<usize, VM<'a>>>>,
}

impl<'a> Named for VM<'a> {
    fn new<'b>(name: &'static str) -> VM<'b> {
        VM { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for VM<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

struct LL<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a LinkedList<LL<'a>>>>,
}

impl<'a> Named for LL<'a> {
    fn new<'b>(name: &'static str) -> LL<'b> {
        LL { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for LL<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

struct BH<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a BinaryHeap<BH<'a>>>>,
}

impl<'a> Named for BH<'a> {
    fn new<'b>(name: &'static str) -> BH<'b> {
        BH { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for BH<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

impl<'a> Eq for BH<'a> { }

impl<'a> PartialEq for BH<'a> {
    fn eq(&self, rhs: &BH<'a>) -> bool {
        self.name == rhs.name
    }
}

impl<'a> PartialOrd for BH<'a> {
    fn partial_cmp(&self, rhs: &BH<'a>) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl<'a> Ord for BH<'a> {
    fn cmp(&self, rhs: &BH<'a>) -> Ordering {
        self.name.cmp(rhs.name)
    }
}

struct BTM<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a BTreeMap<BTM<'a>, BTM<'a>>>>,
}

impl<'a> Named for BTM<'a> {
    fn new<'b>(name: &'static str) -> BTM<'b> {
        BTM { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for BTM<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

impl<'a> Eq for BTM<'a> { }

impl<'a> PartialEq for BTM<'a> {
    fn eq(&self, rhs: &BTM<'a>) -> bool {
        self.name == rhs.name
    }
}

impl<'a> PartialOrd for BTM<'a> {
    fn partial_cmp(&self, rhs: &BTM<'a>) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl<'a> Ord for BTM<'a> {
    fn cmp(&self, rhs: &BTM<'a>) -> Ordering {
        self.name.cmp(rhs.name)
    }
}

struct BTS<'a> {
    name: &'static str,
    mark: Cell<u32>,
    contents: Cell<Option<&'a BTreeSet<BTS<'a>>>>,
}

impl<'a> Named for BTS<'a> {
    fn new<'b>(name: &'static str) -> BTS<'b> {
        BTS { name: name,
             mark: Cell::new(0),
             contents: Cell::new(None)
        }
    }
    fn name(&self) -> &str { self.name }
}

impl<'a> Marked<u32> for BTS<'a> {
    fn mark(&self) -> u32 { self.mark.get() }
    fn set_mark(&self, mark: u32) { self.mark.set(mark); }
}

impl<'a> Eq for BTS<'a> { }

impl<'a> PartialEq for BTS<'a> {
    fn eq(&self, rhs: &BTS<'a>) -> bool {
        self.name == rhs.name
    }
}

impl<'a> PartialOrd for BTS<'a> {
    fn partial_cmp(&self, rhs: &BTS<'a>) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

impl<'a> Ord for BTS<'a> {
    fn cmp(&self, rhs: &BTS<'a>) -> Ordering {
        self.name.cmp(rhs.name)
    }
}

#[derive(Clone)]
struct RCRCData<'a> {
    name: &'static str,
    mark: Cell<u32>,
    children: (Option<&'a RCRC<'a>>, Option<&'a RCRC<'a>>),
}
#[derive(Clone)]
struct RCRC<'a>(Rc<RefCell<RCRCData<'a>>>);

impl<'a> Named for RCRC<'a> {
    fn new(name: &'static str) -> Self {
        RCRC(Rc::new(RefCell::new(RCRCData {
            name: name, mark: Cell::new(0), children: (None, None), })))
    }
    fn name(&self) -> &str { self.0.borrow().name }
}

impl<'a> Marked<u32> for RCRC<'a> {
    fn mark(&self) -> u32 { self.0.borrow().mark.get() }
    fn set_mark(&self, mark: u32) { self.0.borrow().mark.set(mark); }
}

impl<'a> Children<'a> for RCRC<'a> {
    fn count_children(&self) -> usize { 2 }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        let children = &self.0.borrow().children;
        let child = match index {
            0 => if let Some(child) = children.0 { child } else { return; },
            1 => if let Some(child) = children.1 { child } else { return; },
            _ => panic!("bad children"),
        };
        // println!("S2 {} descending into child {} at index {}", self.name, child.name, index);
        child.descend_into_self(context);
    }
}
#[derive(Clone)]
struct ARCRCData<'a> {
    name: &'static str,
    mark: Cell<u32>,
    children: (Option<&'a ARCRC<'a>>, Option<&'a ARCRC<'a>>),
}
#[derive(Clone)]
struct ARCRC<'a>(Arc<RefCell<ARCRCData<'a>>>);

impl<'a> Named for ARCRC<'a> {
    fn new(name: &'static str) -> Self {
        ARCRC(Arc::new(RefCell::new(ARCRCData {
            name: name, mark: Cell::new(0), children: (None, None), })))
    }
    fn name(&self) -> &str { self.0.borrow().name }
}

impl<'a> Marked<u32> for ARCRC<'a> {
    fn mark(&self) -> u32 { self.0.borrow().mark.get() }
    fn set_mark(&self, mark: u32) { self.0.borrow().mark.set(mark); }
}

impl<'a> Children<'a> for ARCRC<'a> {
    fn count_children(&self) -> usize { 2 }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        let children = &self.0.borrow().children;
        match index {
            0 => if let Some(ref child) = children.0 {
                child.descend_into_self(context);
            },
            1 => if let Some(ref child) = children.1 {
                child.descend_into_self(context);
            },
            _ => panic!("bad children!"),
        }
    }
}

#[derive(Clone)]
struct ARCMData<'a> {
    mark: Cell<u32>,
    children: (Option<&'a ARCM<'a>>, Option<&'a ARCM<'a>>),
}

#[derive(Clone)]
struct ARCM<'a>(&'static str, Arc<Mutex<ARCMData<'a>>>);

impl<'a> Named for ARCM<'a> {
    fn new(name: &'static str) -> Self {
        ARCM(name, Arc::new(Mutex::new(ARCMData {
            mark: Cell::new(0), children: (None, None), })))
    }
    fn name(&self) -> &str { self.0 }
}

impl<'a> Marked<u32> for ARCM<'a> {
    fn mark(&self) -> u32 { self.1.lock().unwrap().mark.get() }
    fn set_mark(&self, mark: u32) { self.1.lock().unwrap().mark.set(mark); }
}

impl<'a> Children<'a> for ARCM<'a> {
    fn count_children(&self) -> usize { 2 }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        let ref children = if let Ok(data) = self.1.try_lock() {
            data.children
        } else { return; };
        match index {
            0 => if let Some(ref child) = children.0 {
                child.descend_into_self(context);
            },
            1 => if let Some(ref child) = children.1 {
                child.descend_into_self(context);
            },
            _ => panic!("bad children!"),
        }
    }
}

#[derive(Clone)]
struct ARCRWData<'a> {
    name: &'static str,
    mark: Cell<u32>,
    children: (Option<&'a ARCRW<'a>>, Option<&'a ARCRW<'a>>),
}

#[derive(Clone)]
struct ARCRW<'a>(Arc<RwLock<ARCRWData<'a>>>);

impl<'a> Named for ARCRW<'a> {
    fn new(name: &'static str) -> Self {
        ARCRW(Arc::new(RwLock::new(ARCRWData {
            name: name, mark: Cell::new(0), children: (None, None), })))
    }
    fn name(&self) -> &str { self.0.read().unwrap().name }
}

impl<'a> Marked<u32> for ARCRW<'a> {
    fn mark(&self) -> u32 { self.0.read().unwrap().mark.get() }
    fn set_mark(&self, mark: u32) { self.0.read().unwrap().mark.set(mark); }
}

impl<'a> Children<'a> for ARCRW<'a> {
    fn count_children(&self) -> usize { 2 }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        let children = &self.0.read().unwrap().children;
        match index {
            0 => if let Some(ref child) = children.0 {
                child.descend_into_self(context);
            },
            1 => if let Some(ref child) = children.1 {
                child.descend_into_self(context);
            },
            _ => panic!("bad children!"),
        }
    }
}

trait Context {
    fn next_index(&mut self, len: usize) -> usize;
    fn should_act(&self) -> bool;
    fn increase_visited(&mut self);
    fn increase_skipped(&mut self);
    fn increase_depth(&mut self);
    fn decrease_depth(&mut self);
}

trait PrePost<T> {
    fn pre(&mut self, &T);
    fn post(&mut self, &T);
    fn hit_limit(&mut self, &T);
}

trait Children<'a> {
    fn count_children(&self) -> usize;
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized;

    fn next_child<C>(&self, context: &mut C)
        where C: Context + PrePost<Self>, Self: Sized
    {
        let index = context.next_index(self.count_children());
        self.descend_one_child(context, index);
    }

    fn descend_into_self<C>(&self, context: &mut C)
        where C: Context + PrePost<Self>, Self: Sized
    {
        context.pre(self);
        if context.should_act() {
            context.increase_visited();
            context.increase_depth();
            self.next_child(context);
            context.decrease_depth();
        } else {
            context.hit_limit(self);
            context.increase_skipped();
        }
        context.post(self);
    }

    fn descend<'b, C>(&self, c: &Cell<Option<&'b Self>>, context: &mut C)
        where C: Context + PrePost<Self>, Self: Sized
    {
        if let Some(r) = c.get() {
            r.descend_into_self(context);
        }
    }
}

impl<'a> Children<'a> for S<'a> {
    fn count_children(&self) -> usize { 1 }
    fn descend_one_child<C>(&self, context: &mut C, _: usize)
        where C: Context + PrePost<Self>, Self: Sized {
            self.descend(&self.next, context);
        }
}

impl<'a> Children<'a> for S2<'a> {
    fn count_children(&self) -> usize { 2 }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        let children = self.next.get();
        let child = match index {
            0 => if let Some(child) = children.0 { child } else { return; },
            1 => if let Some(child) = children.1 { child } else { return; },
            _ => panic!("bad children"),
        };
        // println!("S2 {} descending into child {} at index {}", self.name, child.name, index);
        child.descend_into_self(context);
    }
}

impl<'a> Children<'a> for V<'a> {
    fn count_children(&self) -> usize { self.contents.len() }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        if let Some(child) = self.contents[index].get() {
            child.descend_into_self(context);
        }
    }
}

impl<'a> Children<'a> for H<'a> {
    fn count_children(&self) -> usize { 1 }
    fn descend_one_child<C>(&self, context: &mut C, _: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        self.descend(&self.next, context);
    }
}

impl<'a> Children<'a> for HM<'a> {
    fn count_children(&self) -> usize {
        if let Some(m) = self.contents.get() { 2 * m.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        if let Some(ref hm) = self.contents.get() {
            for (k, v) in hm.iter().nth(index / 2) {
                [k, v][index % 2].descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for VD<'a> {
    fn count_children(&self) -> usize {
        if let Some(d) = self.contents.get() { d.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<Self>, Self: Sized
    {
        if let Some(ref vd) = self.contents.get() {
            for r in vd.iter().nth(index) {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for VM<'a> {
    fn count_children(&self) -> usize {
        if let Some(m) = self.contents.get() { m.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<VM<'a>>
    {
        if let Some(ref vd) = self.contents.get() {
            for (_idx, r) in vd.iter().nth(index) {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for LL<'a> {
    fn count_children(&self) -> usize {
        if let Some(l) = self.contents.get() { l.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<LL<'a>>
    {
        if let Some(ref ll) = self.contents.get() {
            for r in ll.iter().nth(index) {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for BH<'a> {
    fn count_children(&self) -> usize {
        if let Some(h) = self.contents.get() { h.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<BH<'a>>
    {
        if let Some(ref bh) = self.contents.get() {
            for r in bh.iter().nth(index) {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for BTM<'a> {
    fn count_children(&self) -> usize {
        if let Some(m) = self.contents.get() { 2 * m.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<BTM<'a>>
    {
        if let Some(ref bh) = self.contents.get() {
            for (k, v) in bh.iter().nth(index / 2) {
                [k, v][index % 2].descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for BTS<'a> {
    fn count_children(&self) -> usize {
        if let Some(s) = self.contents.get() { s.iter().count() } else { 0 }
    }
    fn descend_one_child<C>(&self, context: &mut C, index: usize)
        where C: Context + PrePost<BTS<'a>>
    {
        if let Some(ref bh) = self.contents.get() {
            for r in bh.iter().nth(index) {
                r.descend_into_self(context);
            }
        }
    }
}

#[derive(Copy, Clone)]
struct ContextData {
    curr_depth: usize,
    max_depth: usize,
    visited: usize,
    max_visits: usize,
    skipped: usize,
    curr_mark: u32,
    saw_prev_marked: bool,
    control_bits: u64,
}

impl Context for ContextData {
    fn next_index(&mut self, len: usize) -> usize {
        if len < 2 { return 0; }
        let mut pow2 = len.next_power_of_two();
        let _pow2_orig = pow2;
        let mut idx = 0;
        let mut bits = self.control_bits;
        while pow2 > 1 {
            idx = (idx << 1) | (bits & 1) as usize;
            bits = bits >> 1;
            pow2 = pow2 >> 1;
        }
        idx = idx % len;
        // println!("next_index({} [{:b}]) says {}, pre(bits): {:b} post(bits): {:b}",
        //          len, _pow2_orig, idx, self.control_bits, bits);
        self.control_bits = bits;
        return idx;
    }
    fn should_act(&self) -> bool {
        self.curr_depth < self.max_depth && self.visited < self.max_visits
    }
    fn increase_visited(&mut self) { self.visited += 1; }
    fn increase_skipped(&mut self) { self.skipped += 1; }
    fn increase_depth(&mut self) {  self.curr_depth += 1; }
    fn decrease_depth(&mut self) {  self.curr_depth -= 1; }
}

impl<T:Named+Marked<u32>> PrePost<T> for ContextData {
    fn pre(&mut self, t: &T) {
        for _ in 0..self.curr_depth {
            if PRINT { print!(" "); }
        }
        if PRINT { println!("prev {}", t.name()); }
        if t.mark() == self.curr_mark {
            for _ in 0..self.curr_depth {
                if PRINT { print!(" "); }
            }
            if PRINT { println!("(probably previously marked)"); }
            self.saw_prev_marked = true;
        }
        t.set_mark(self.curr_mark);
    }
    fn post(&mut self, t: &T) {
        for _ in 0..self.curr_depth {
            if PRINT { print!(" "); }
        }
        if PRINT { println!("post {}", t.name()); }
    }
    fn hit_limit(&mut self, t: &T) {
        for _ in 0..self.curr_depth {
            if PRINT { print!(" "); }
        }
        if PRINT { println!("LIMIT {}", t.name()); }
    }
}
