// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::SeekLocation::*;

use rustc_data_structures::indexed_set::{IdxSet, IdxSetBuf};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::bitvec::BitMatrix;
use rustc::mir::*;
use std::collections::{BTreeMap, VecDeque};
use std::iter;
use std::marker::PhantomData;
use analysis::locations::{FlatLocation, FlatLocations};

// FIXME(eddyb) move to rustc_data_structures.
#[derive(Clone)]
pub struct SparseBitSet<I: Idx> {
    map: BTreeMap<u32, u128>,
    _marker: PhantomData<I>,
}

fn key_and_mask<I: Idx>(index: I) -> (u32, u128) {
    let index = index.index();
    let key = index / 128;
    let key_u32 = key as u32;
    assert_eq!(key_u32 as usize, key);
    (key_u32, 1 << (index % 128))
}

impl<I: Idx> SparseBitSet<I> {
    pub fn new() -> Self {
        SparseBitSet {
            map: BTreeMap::new(),
            _marker: PhantomData
        }
    }

    pub fn capacity(&self) -> usize {
        self.map.len() * 128
    }

    pub fn contains(&self, index: I) -> bool {
        let (key, mask) = key_and_mask(index);
        self.map.get(&key).map_or(false, |bits| (bits & mask) != 0)
    }

    pub fn insert(&mut self, index: I) -> bool {
        let (key, mask) = key_and_mask(index);
        let bits = self.map.entry(key).or_insert(0);
        let old_bits = *bits;
        let new_bits = old_bits | mask;
        *bits = new_bits;
        new_bits != old_bits
    }

    pub fn remove(&mut self, index: I) -> bool {
        let (key, mask) = key_and_mask(index);
        if let Some(bits) = self.map.get_mut(&key) {
            let old_bits = *bits;
            let new_bits = old_bits & !mask;
            *bits = new_bits;
            // FIXME(eddyb) maybe remove entry if now `0`.
            new_bits != old_bits
        } else {
            false
        }
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = I> + 'a {
        self.map.iter().flat_map(|(&key, &bits)| {
            let base = key as usize * 128;
            (0..128).filter_map(move |i| {
                if (bits & (1 << i)) != 0 {
                    Some(I::new(base + i))
                } else {
                    None
                }
            })
        })
    }
}

/// A pair with named fields (`past` and `future`).
/// Used solely to avoid mixing the two up.
#[derive(Copy, Clone, Default)]
pub struct PastAndFuture<P, F> {
    pub past: P,
    pub future: F
}

/// Event (dataflow) propagation direction.
/// Past events are propagated forward, while
/// future events are propagated backward.
pub trait Direction: 'static {
    const FORWARD: bool;
    fn each_propagation_edge<F>(mir: &Mir, from: BasicBlock, f: F)
        where F: FnMut(BasicBlock);
    fn each_block_location<F>(flat_locations: &FlatLocations, block: BasicBlock, f: F)
        where F: FnMut(FlatLocation);
}

pub enum Forward {}
impl Direction for Forward {
    const FORWARD: bool = true;
    fn each_propagation_edge<F>(mir: &Mir, from: BasicBlock, mut f: F)
        where F: FnMut(BasicBlock)
    {
        for &to in mir.basic_blocks()[from].terminator().successors().iter() {
            f(to);
        }
    }
    fn each_block_location<F>(flat_locations: &FlatLocations, block: BasicBlock, mut f: F)
        where F: FnMut(FlatLocation)
    {
        let range = flat_locations.block_range(block);
        // FIXME(eddyb) implement `Step` on `FlatLocation`.
        for i in range.start.index()..range.end.index() {
            f(FlatLocation::new(i))
        }
    }
}

pub enum Backward {}
impl Direction for Backward {
    const FORWARD: bool = false;
    fn each_propagation_edge<F>(mir: &Mir, from: BasicBlock, mut f: F)
        where F: FnMut(BasicBlock)
    {
        for &to in mir.predecessors_for(from).iter() {
            f(to);
        }
    }
    fn each_block_location<F>(flat_locations: &FlatLocations, block: BasicBlock, mut f: F)
        where F: FnMut(FlatLocation)
    {
        let range = flat_locations.block_range(block);
        // FIXME(eddyb) implement `Step` on `FlatLocation`.
        for i in (range.start.index()..range.end.index()).rev() {
            f(FlatLocation::new(i))
        }
    }
}

pub struct Events<'a, 'b, 'tcx: 'a, I: Idx> {
    mir: &'a Mir<'tcx>,
    flat_locations: &'b FlatLocations,
    count: usize,
    at_location: IndexVec<FlatLocation, SparseBitSet<I>>,
    in_block: BitMatrix
}

impl<'a, 'b, 'tcx, I: Idx> Events<'a, 'b, 'tcx, I> {
    pub fn new(mir: &'a Mir<'tcx>,
               flat_locations: &'b FlatLocations,
               count: usize)
               -> Self {
        Events {
            mir,
            flat_locations,
            count,
            at_location: IndexVec::from_elem_n(SparseBitSet::new(),
                flat_locations.total_count),
            in_block: BitMatrix::new(mir.basic_blocks().len(), count)
        }
    }

    pub fn insert_at(&mut self, index: I, location: Location) {
        let flat_location = self.flat_locations.get(location);
        self.at_location[flat_location].insert(index);
        self.in_block.add(location.block.index(), index.index());
    }

    pub fn flow<P>(self, entry_past: P)
                   -> PastAndFuture<EventFlowResults<'b, Forward, I>,
                                    EventFlowResults<'b, Backward, I>>
        where P: Iterator<Item = I>
    {

        PastAndFuture {
            past: self.flow_in_direction(entry_past.map(|i| (START_BLOCK, i))),
            future: self.flow_in_direction(iter::empty())
        }
    }

    fn flow_in_direction<D: Direction, E>(&self, external: E)
                                          -> EventFlowResults<'b, D, I>
        where E: Iterator<Item = (BasicBlock, I)>
    {
        let mut queue = VecDeque::with_capacity(self.mir.basic_blocks().len());
        let mut enqueued = IdxSetBuf::new_filled(self.mir.basic_blocks().len());

        // 0. Add some external events in the past/future of certain blocks.
        let mut into_block = BitMatrix::new(self.mir.basic_blocks().len(), self.count);
        for (block, i) in external {
            into_block.add(block.index(), i.index());
        }

        // 1. Propagate `in_block` events to immediate successors/predecessors.
        for from in self.mir.basic_blocks().indices() {
            D::each_propagation_edge(&self.mir, from, |to| {
                // FIXME(eddyb) This could use a version of `BitMatrix::merge`
                // between two rows that are in diferent `BitMatrix`es.
                for i in self.in_block.iter(from.index()) {
                    into_block.add(to.index(), i);
                }
            });
            queue.push_back(from);
        }

        // 2. Propagate `into_block` events until saturation is achieved.
        while let Some(from) = queue.pop_front() {
            D::each_propagation_edge(&self.mir, from, |to| {
                if into_block.merge(from.index(), to.index()) {
                    if enqueued.add(&to) {
                        queue.push_back(to);
                    }
                }
            });
            enqueued.remove(&from);
        }

        // 3. Cache the difference between consecutive locations within each block.
        let mut out_of_block = into_block.clone();
        let mut diff_at_location = IndexVec::from_elem_n(SparseBitSet::new(),
            self.flat_locations.total_count);
        for block in self.mir.basic_blocks().indices() {
            D::each_block_location(&self.flat_locations, block, |flat_location| {
                let at_location = &self.at_location[flat_location];
                let diff = &mut diff_at_location[flat_location];
                // FIXME(eddyb) This could use per-"word" bitwise operations.
                for i in at_location.iter() {
                    if out_of_block.add(block.index(), i.index()) {
                        diff.insert(i);
                    }
                }
            });
        }

        let (block_entry, block_exit) = if D::FORWARD {
            (into_block, out_of_block)
        } else {
            (out_of_block, into_block)
        };

        EventFlowResults {
            flat_locations: self.flat_locations,
            count: self.count,
            block_entry,
            block_exit,
            diff_at_location,
            _marker: PhantomData
        }
    }
}

#[derive(Clone)]
pub struct EventFlowResults<'a, D: Direction, I: Idx> {
    flat_locations: &'a FlatLocations,
    count: usize,

    /// Bits propagated into the start of the block, from predecessors.
    block_entry: BitMatrix,

    /// Bits propagated out of the end of the block, into successors.
    block_exit: BitMatrix,

    /// Bits that change at each statement/terminator, because they're
    /// either the first occurence (in the past only after the location),
    /// or the last occurence (in the future only before the location).
    diff_at_location: IndexVec<FlatLocation, SparseBitSet<I>>,

    _marker: PhantomData<D>
}

impl<'a, D: Direction, I: Idx> EventFlowResults<'a, D, I> {
    pub fn observe(&'a self) -> Observer<'a, D, I> {
        Observer {
            results: self,
            location: Location {
                block: START_BLOCK,
                statement_index: !0
            },
            state_before: IdxSetBuf::new_empty(self.count),
        }
    }
}

impl<'a, I: Idx> PastAndFuture<EventFlowResults<'a, Forward, I>,
                               EventFlowResults<'a, Backward, I>> {
    pub fn observe(&'a self) -> PastAndFuture<Observer<'a, Forward, I>,
                                              Observer<'a, Backward, I>> {
        PastAndFuture {
            past: self.past.observe(),
            future: self.future.observe()
        }
    }
}

pub struct Observer<'a, D: Direction, I: Idx> {
    results: &'a EventFlowResults<'a, D, I>,
    location: Location,
    state_before: IdxSetBuf<I>,
}

#[derive(Copy, Clone)]
pub enum SeekLocation {
    Before(Location),
    After(Location)
}

impl<'a, D: Direction, I: Idx> Observer<'a, D, I> {
    pub fn seek(&mut self, to: SeekLocation) -> &IdxSet<I> {
        // Ensure the location is valid for a statement/terminator.
        match to {
            Before(location) | After(location) => {
                self.results.flat_locations.get(location);
            }
        }

        let to = match to {
            Before(location) => location,
            After(location) => location.successor_within_block()
        };

        // Seek to the start or end of the block if we were in a different one.
        if self.location.block != to.block || self.location.statement_index == !0 {
            self.state_before.clear();

            let block_range = self.results.flat_locations.block_range(to.block);
            let locations_in_block = block_range.end.index() - block_range.start.index();

            // FIXME(eddyb) These could use copies of whole rows.
            if to.statement_index < locations_in_block / 2 {
                for i in self.results.block_entry.iter(to.block.index()) {
                    self.state_before.add(&I::new(i));
                }
                self.location.statement_index = 0;
            } else {
                for i in self.results.block_exit.iter(to.block.index()) {
                    self.state_before.add(&I::new(i));
                }
                self.location.statement_index = locations_in_block;
            }
            self.location.block = to.block;
        }

        while self.location.statement_index < to.statement_index {
            let flat_location = self.results.flat_locations.get(self.location);
            // FIXME(eddyb) These could use per-"word" bitwise operations.
            for i in self.results.diff_at_location[flat_location].iter() {
                if D::FORWARD {
                    self.state_before.add(&i);
                } else {
                    self.state_before.remove(&i);
                }
            }
            self.location.statement_index += 1;
        }

        while self.location.statement_index > to.statement_index {
            self.location.statement_index -= 1;
            let flat_location = self.results.flat_locations.get(self.location);
            // FIXME(eddyb) These could use per-"word" bitwise operations.
            for i in self.results.diff_at_location[flat_location].iter() {
                if D::FORWARD {
                    self.state_before.remove(&i);
                } else {
                    self.state_before.add(&i);
                }
            }
        }

        &self.state_before
    }
}

impl<'a, I: Idx> PastAndFuture<Observer<'a, Forward, I>,
                               Observer<'a, Backward, I>> {
    pub fn seek(&mut self, to: SeekLocation)
                -> PastAndFuture<&IdxSet<I>, &IdxSet<I>> {
        PastAndFuture {
            past: self.past.seek(to),
            future: self.future.seek(to)
        }
    }
}
