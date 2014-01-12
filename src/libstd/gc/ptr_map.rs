// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use container::Container;
use container::MutableMap;
use iter::Iterator;
use option::{Option, Some, None};
use trie::{TrieMap, TrieMapIterator};
use vec::ImmutableVector;

use gc::collector::TracingFunc;

pub struct PtrMap {
    // a map from the start of each allocation to a descriptor
    // containing information about it.
    priv map: TrieMap<~PtrDescr>,
    // The state of the REACHABLE bit of `PtrDescr.flags` that
    // represents whether an allocations is reachable.
    priv reachable_state: Flag,
}

type Flag = u8;
static REACHABLE: Flag = 0b0000_0001;
static USED: Flag      = 0b0000_0010;

/// This representation could be optimised.
pub struct PtrDescr {
    // arbitrary data associated with this pointer.
    metadata: uint,
    // the function to use to perform tracing on this allocation
    tracer: Option<TracingFunc>,
    // the finaliser to run
    finaliser: Option<fn(*mut ())>,
    // tiny properties about this allocation.
    flags: Flag,
}

impl PtrDescr {
    fn is_used(&self) -> bool {
        self.flags & USED == USED
    }
    fn is_used_and_unreachable(&self, map_reachable_state: Flag) -> bool {
        let unreachable = !(self.flags & REACHABLE == map_reachable_state);
        self.is_used() & unreachable
    }

    fn set_reachable(&mut self, reachability: Flag) {
        // filter out the reachable bit and then set it explicitly
        self.flags = (self.flags & !REACHABLE) | (reachability & REACHABLE);
    }
}

impl PtrMap {
    /// Create a new PtrMap.
    pub fn new() -> PtrMap {
        PtrMap {
            map: TrieMap::new(),
            reachable_state: REACHABLE
        }
    }

    /// Register an allocation starting at `ptr`, with an arbitrary
    /// piece of information `metadata`, a function to trace `tracer`
    /// and the destructor `finaliser`.
    pub fn insert_alloc(&mut self,
                        ptr: uint,
                        metadata: uint,
                        tracer: Option<TracingFunc>,
                        finaliser: Option<fn(*mut ())>) {
        let descr = ~PtrDescr {
            flags: self.reachable_state | USED,
            tracer: tracer,
            finaliser: finaliser,
            metadata: metadata
        };
        self.map.insert(ptr, descr);
    }

    /// Attempt to reuse the allocation starting at `ptr`. Returns
    /// `true` if it was successfully registered, otherwise `false`
    /// (attempting to reuse an live allocation, or an allocation that
    /// wasn't found).
    pub fn reuse_alloc(&mut self,
                       ptr: uint,
                       metadata: uint,
                       tracer: Option<TracingFunc>,
                       finaliser: Option<fn(*mut ())>) -> bool {
        match self.map.find_mut(&ptr) {
            Some(descr) => {
                if descr.is_used() {
                    warn!("attempting to reuse a used allocation")
                    false // don't overwrite
                } else {
                    descr.finaliser = finaliser;
                    descr.metadata = metadata;
                    descr.tracer = tracer;
                    descr.flags = self.reachable_state | USED;
                    true
                }
            }
            None => false
        }
    }

    /// Look up the allocation starting at `ptr` and, if it is
    /// currently marked as unreachable, mark it as reachable and
    /// retrieve the high end & whether it requires scanning;
    /// otherwise, return None.
    pub fn mark_reachable_scan_info(&mut self, ptr: uint) -> Option<(uint, Option<TracingFunc>)> {
        match self.map.find_mut(&ptr) {
            Some(descr) => {
                if descr.is_used_and_unreachable(self.reachable_state) {
                    // mark it reachable
                    descr.set_reachable(self.reachable_state);
                    Some((descr.metadata, descr.tracer))
                } else {
                    None
                }
            }
            None => None
        }
    }

    /// Set the word of metadata associated with `ptr` to `metadata`.
    pub fn update_metadata<'a>(&'a mut self, ptr: uint, metadata: uint) -> bool {
        match self.map.find_mut(&ptr) {
            Some(ref mut descr) if descr.is_used() => { descr.metadata = metadata; true }
            _ => false
        }
    }

    /// Find the unreachable pointers in the map, iterating over
    /// `(low, descriptor)`. This marks each of these pointers as
    /// unused (and clears their destructors) after calling `f`.
    pub fn each_unreachable(&mut self, f: |uint, &PtrDescr| -> bool) -> bool {
        self.map.each_mut(|low, descr| {
                if descr.is_used_and_unreachable(self.reachable_state) {
                    let cont = f(*low, *descr);
                    // mark as unused
                    descr.finaliser = None;
                    descr.flags &= !USED;
                    cont
                } else {
                    // continue
                    true
                }
            })
    }

    /// After a collection this will flip an internal bit so that
    /// everything is considered unreachable at the start of the next
    /// collection.
    pub fn toggle_reachability(&mut self) {
        self.reachable_state ^= REACHABLE;
    }

    /// Manually mark every pointer as unreachable. Prefer
    /// `toggle_reachability` when you have the guarantee that all the
    /// pointers in the map are currently considered reachable.
    pub fn inefficient_mark_all_unreachable(&mut self) {
        self.map.each_mut(|_, descr| {
                // invert to mark as unreachable
                descr.set_reachable(self.reachable_state ^ REACHABLE);
                true
            });
    }

    /// Deregister the allocation starting at `ptr`.
    pub fn remove(&mut self, ptr: uint) {
        self.map.remove(&ptr);
    }

    /// Iterate over `(low, &'a ~PtrDescr)`.
    pub fn iter<'a>(&'a self) -> TrieMapIterator<'a, ~PtrDescr> {
        self.map.iter()
    }

    /// The number of pointers registered.
    #[allow(dead_code)]
    pub fn len(&self) -> uint { self.map.len() }
}
