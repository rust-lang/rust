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

pub struct PtrMap {
    // a map from the start of each allocation to a descriptor
    // containing information about it.
    priv map: TrieMap<PtrDescr>,
    // The state of `reachable` that represents whether an allocations
    // is reachable, i.e. descr.reachable_flag == this.reachable_state
    // implies the pointer is reachable.
    priv reachable_state: bool,
}

/// This representation could be optimised.
pub struct PtrDescr {
    // the top edge of the allocation.
    high: uint,
    // the finaliser to run
    finaliser: Option<fn(*mut ())>,
    // whether this allocation is reachable (see
    // PtrMap.reachable_state)
    reachable_flag: bool,
    // whether this allocation should be scanned (i.e. whether it
    // contains rooted references to GC pointers)
    scan: bool
}

impl PtrDescr {
    fn is_used(&self) -> bool {
        self.high != 0
    }
}

impl PtrMap {
    /// Create a new PtrMap.
    pub fn new() -> PtrMap {
        PtrMap {
            map: TrieMap::new(),
            reachable_state: true
        }
    }

    /// Register an allocation starting at `ptr` running for `length`
    /// bytes. `scan` indicates if the allocation should be scanned,
    /// and `finaliser` is the "destructor" to run on the region.
    pub fn insert_alloc(&mut self, ptr: uint, length: uint, scan: bool,
                        finaliser: Option<fn(*mut ())>) {
        let descr = PtrDescr {
            high: ptr + length,
            reachable_flag: self.reachable_state,
            scan: scan,
            finaliser: finaliser
        };
        self.map.insert(ptr, descr);
    }

    /// Attempt to reuse the allocation starting at `ptr`. Returns
    /// `true` if it was successfully registered, otherwise `false`
    /// (attempting to reuse an live allocation, or an allocation that
    /// wasn't found).
    pub fn reuse_alloc(&mut self,
                       ptr: uint, length: uint, scan: bool,
                       finaliser: Option<fn(*mut ())>) -> bool {
        match self.map.find_mut(&ptr) {
            Some(descr) => {
                if descr.is_used() {
                    warn!("attempting to reuse a used allocation")
                    false // don't overwrite
                } else {
                    descr.high = ptr + length;
                    descr.finaliser = finaliser;
                    descr.scan = scan;
                    descr.reachable_flag = self.reachable_state;
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
    pub fn mark_reachable_scan_info(&mut self, ptr: uint) -> Option<(uint, bool)> {
        match self.map.find_mut(&ptr) {
            Some(descr) => {
                if descr.is_used() && descr.reachable_flag != self.reachable_state {
                    descr.reachable_flag = self.reachable_state;
                    Some((descr.high, descr.scan))
                } else {
                    None
                }
            }
            None => None
        }
    }

    /// Find the unreachable pointers in the map, returing `[(low,
    /// size, finaliser)]`.
    pub fn find_unreachable(&mut self) -> ~[(uint, uint, Option<fn(*mut ())>)] {
        self.map.iter()
            .filter_map(|(low, descr)| {
                if descr.is_used() && descr.reachable_flag != self.reachable_state {
                    Some((low, descr.high - low, descr.finaliser))
                } else {
                    None
                }
            }).collect()
    }

    /// Mark an allocation as unused.
    pub fn mark_unused(&mut self, ptr: uint) {
        match self.map.find_mut(&ptr) {
            Some(descr) => descr.high = 0,
            None => {}
        }
    }

    /// After a collection this will flip an internal bit so that
    /// everything is considered unreachable at the start of the next
    /// collection.
    pub fn toggle_reachability(&mut self) {
        self.reachable_state = !self.reachable_state;
    }

    /// Deregister the allocation starting at `ptr`.
    pub fn remove(&mut self, ptr: uint) {
        self.map.remove(&ptr);
    }

    /// Iterate over `(low, &'a PtrDescr)`.
    pub fn iter<'a>(&'a self) -> TrieMapIterator<'a, PtrDescr> {
        self.map.iter()
    }

    /// The number of pointers registered.
    pub fn len(&self) -> uint { self.map.len() }
}
