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
}

/// This representation could be optimised.
pub struct PtrDescr {
    // the top edge of the allocation.
    high: uint,
    // the finaliser to run
    finaliser: Option<fn(*mut ())>,
    // whether this allocation is reachable
    reachable: bool,
    // whether this allocation should be scanned (i.e. whether it
    // contains rooted references to GC pointers)
    scan: bool
}

impl PtrMap {
    /// Create a new PtrMap.
    pub fn new() -> PtrMap {
        PtrMap {
            map: TrieMap::new()
        }
    }

    /// Register an allocation starting at `ptr` running for `length` bytes
    pub fn insert_alloc(&mut self, ptr: uint, length: uint, scan: bool) {
        let descr = PtrDescr { high: ptr + length, reachable: false, scan: scan, finaliser: None };
        self.map.insert(ptr, descr);
    }

    /// Mark every registered allocation as unreachable.
    pub fn mark_all_unreachable(&mut self) {
        for (_, d) in self.map.mut_iter() {
            d.reachable = false;
        }
    }

    /// Look up the allocation starting at `ptr` and, if it is
    /// currently marked as unreachable, mark it as reachable and
    /// retrieve the high end & whether it requires scanning;
    /// otherwise, return None.
    pub fn mark_reachable_scan_info(&mut self, ptr: uint) -> Option<(uint, bool)> {
        match self.map.find_mut(&ptr) {
            Some(descr) => {
                if !descr.reachable {
                    descr.reachable = true;
                    Some((descr.high, descr.scan))
                } else {
                    None
                }
            }
            None => None
        }
    }

    /// Find the unreachable pointers in the map, returing `[(low,
    /// finaliser)]`.
    pub fn find_unreachable(&mut self) -> ~[(uint, Option<fn(*mut ())>)] {
        self.map.iter()
            .filter_map(|(low, descr)|
                        if !descr.reachable {Some((low, descr.finaliser))} else {None})
            .collect()
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
