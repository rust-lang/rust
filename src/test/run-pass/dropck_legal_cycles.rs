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

#![feature(vecmap)]

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::LinkedList;
use std::collections::VecDeque;
use std::collections::VecMap;
use std::collections::btree_map::BTreeMap;
use std::collections::btree_set::BTreeSet;
use std::hash::{Hash, Hasher};

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
    };

    // Cycle 1: { v[0] -> v[1], v[1] -> v[0] };
    // does not exercise `v` itself
    let v: Vec<S> = vec![Named::new("s0"),
                         Named::new("s1")];
    v[0].next.set(Some(&v[1]));
    v[1].next.set(Some(&v[0]));

    let mut c = c_orig.clone();
    c.curr_mark = 10;
    assert!(!c.saw_prev_marked);
    v[0].for_each_child(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 2: { v[0] -> v, v[1] -> v }
    let v: V = Named::new("v");
    v.contents[0].set(Some(&v));
    v.contents[1].set(Some(&v));

    let mut c = c_orig.clone();
    c.curr_mark = 20;
    assert!(!c.saw_prev_marked);
    v.for_each_child(&mut c);
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
        key.for_each_child(&mut c);
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
        key.for_each_child(&mut c);
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
    vd[0].for_each_child(&mut c);
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
    vd[0].for_each_child(&mut c);
    assert!(c.saw_prev_marked);

    if PRINT { println!(""); }

    // Cycle 7: { vm -> (vm0, vm1), {vm0, vm1} -> vm }
    let mut vm: VecMap<VM> = VecMap::new();
    vm.insert(0, Named::new("vm0"));
    vm.insert(1, Named::new("vm1"));
    vm[0].contents.set(Some(&vm));
    vm[1].contents.set(Some(&vm));

    let mut c = c_orig.clone();
    c.curr_mark = 70;
    assert!(!c.saw_prev_marked);
    vm[0].for_each_child(&mut c);
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
        e.for_each_child(&mut c);
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
        b.for_each_child(&mut c);
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
        k.for_each_child(&mut c);
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
        b.for_each_child(&mut c);
        assert!(c.saw_prev_marked);
        // break;
    }
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
    contents: Cell<Option<&'a VecMap<VM<'a>>>>,
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


trait Context {
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
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<Self>, Self: Sized;

    fn descend_into_self<C>(&self, context: &mut C)
        where C: Context + PrePost<Self>, Self: Sized
    {
        context.pre(self);
        if context.should_act() {
            context.increase_visited();
            context.increase_depth();
            self.for_each_child(context);
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
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<S<'a>>
    {
        self.descend(&self.next, context);
    }
}

impl<'a> Children<'a> for V<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<V<'a>>
    {
        for r in &self.contents {
            self.descend(r, context);
        }
    }
}

impl<'a> Children<'a> for H<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<H<'a>>
    {
        self.descend(&self.next, context);
    }
}

impl<'a> Children<'a> for HM<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<HM<'a>>
    {
        if let Some(ref hm) = self.contents.get() {
            for (k, v) in hm.iter() {
                for r in &[k, v] {
                    r.descend_into_self(context);
                }
            }
        }
    }
}

impl<'a> Children<'a> for VD<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<VD<'a>>
    {
        if let Some(ref vd) = self.contents.get() {
            for r in vd.iter() {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for VM<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<VM<'a>>
    {
        if let Some(ref vd) = self.contents.get() {
            for (_idx, r) in vd.iter() {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for LL<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<LL<'a>>
    {
        if let Some(ref ll) = self.contents.get() {
            for r in ll.iter() {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for BH<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<BH<'a>>
    {
        if let Some(ref bh) = self.contents.get() {
            for r in bh.iter() {
                r.descend_into_self(context);
            }
        }
    }
}

impl<'a> Children<'a> for BTM<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<BTM<'a>>
    {
        if let Some(ref bh) = self.contents.get() {
            for (k, v) in bh.iter() {
                for r in &[k, v] {
                    r.descend_into_self(context);
                }
            }
        }
    }
}

impl<'a> Children<'a> for BTS<'a> {
    fn for_each_child<C>(&self, context: &mut C)
        where C: Context + PrePost<BTS<'a>>
    {
        if let Some(ref bh) = self.contents.get() {
            for r in bh.iter() {
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
}

impl Context for ContextData {
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
