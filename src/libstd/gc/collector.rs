// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use gc::collector::ptr_map::PtrMap;
use iter::Iterator;
use libc;
use local_data;
use num::BitCount;
use option::{Some, None, Option};
use ops::Drop;
use ptr::RawPtr;
use vec::{OwnedVector, ImmutableVector};
use uint;

use gc::GcTracer;

mod ptr_map;

static DEFAULT_ALLOCS_PER_COLLECTION_MASK: uint = (1 << 10) - 1;

static ALLOC_CACHE_MIN_LOG: uint = 3;
static ALLOC_CACHE_MAX_LOG: uint = 20;

pub type TracingFunc = fn(*(), uint, &mut GcTracer);


/// A thread local garbage collector, precise on the head,
/// conservative on the stack, neither generational nor incremental.
///
/// # Design
///
/// Currently stores two sets of known pointers:
///
/// - managed pointers (i.e. allocations entirely under the control of
///   this GC), and
/// - "roots", which are any other pointers/datastructures/memory
///   regions that have registered themselves as possibly containing
///   GC'd pointers (the registration includes a tracing function
///   pointer with which to find these GC'd pointers)
///
/// A conservative stack-scan is performed where any bitpatterns that
/// look like pointers from either of the two sets above are taken to
/// be actual references and a tracing is initiated from there.
///
/// Any managed pointers that were not visited in this search are
/// considered dead and deallocated.
///
/// Allocations and deallocations are performed directly with malloc
/// and free, with caching of small allocations.
pub struct GarbageCollector {
    /// Non-garbage-collectable roots
    priv roots: PtrMap,
    /// Garbage-collectable pointers.
    priv gc_ptrs: PtrMap,
    /// number of GC-able allocations performed.
    priv gc_allocs: uint,
    /// cached allocations, of sizes 8, 16, 32, 64, ... 1 << 20 (1 MB)
    /// (inclusive, with 8 at index 0). Anything smaller gets rounded
    /// to 8, anything larger is uncached.
    priv alloc_cache: [~[uint], .. ALLOC_CACHE_MAX_LOG - ALLOC_CACHE_MIN_LOG + 1],
    /// the number of allocations to do before collection (in mask
    /// form, i.e. we are detecting `gc_allocs % (1 << n) == 0` for
    /// some n).
    priv gc_allocs_per_collection_mask: uint
}

fn compute_log_rounded_up_size(size: uint) -> uint {
    if size <= (1 << ALLOC_CACHE_MIN_LOG) {
        // round up to the minimum
        ALLOC_CACHE_MIN_LOG
    } else {
        // for powers of two 1 << n, this gives n + 1, otherwise,
        // for a number like `0b101` it gives 3, which is exactly
        // what we want.
        let raw = uint::bits - size.leading_zeros();
        // power of two
        if size & (size - 1) == 0 {
            raw - 1
        } else {
            raw
        }
    }
}

impl GarbageCollector {
    pub fn new() -> GarbageCollector {
        GarbageCollector {
            roots: PtrMap::new(),
            gc_ptrs: PtrMap::new(),
            // :( ... at least the compiler will tell us when we have
            // the wrong number.
            alloc_cache: [~[], ~[], ~[], ~[], ~[], ~[],
                          ~[], ~[], ~[], ~[], ~[], ~[],
                          ~[], ~[], ~[], ~[], ~[], ~[]],
            gc_allocs: 0,
            gc_allocs_per_collection_mask: DEFAULT_ALLOCS_PER_COLLECTION_MASK
        }
    }

    /// Run a garbage collection if we're due for one.
    pub unsafe fn occasional_collection(&mut self, stack_top: uint) {
        if self.gc_allocs & self.gc_allocs_per_collection_mask == 0 {
            self.collect(stack_top)
        }
    }

    /// Allocate `size` bytes of memory such that they are scanned for
    /// other GC'd pointers (for use by types like `Gc<Gc<int>>`).
    ///
    /// `finaliser` is passed the start of the allocation at some
    /// unspecified pointer after the allocation has become
    /// unreachable.
    pub unsafe fn alloc(&mut self, size: uint,
                        tracer: Option<TracingFunc>,
                        finaliser: Option<fn(*mut ())>) -> *mut u8 {
        self.gc_allocs += 1;
        let log_next_power_of_two = compute_log_rounded_up_size(size);

        // it's always larger than ALLOC_CACHE_MIN_LOG
        let alloc_size = if log_next_power_of_two <= ALLOC_CACHE_MAX_LOG {
            match self.alloc_cache[log_next_power_of_two - ALLOC_CACHE_MIN_LOG].pop_opt() {
                Some(ptr) => {
                    // attempt to reuse the metadata we have for that
                    // allocation already.
                    let success = self.gc_ptrs.reuse_alloc(ptr, size, tracer, finaliser);
                    if success {
                        debug!("using cache for allocation of size {}", size);
                        return ptr as *mut u8;
                    }
                }
                None => {}
            }
            // otherwise, just allocate as per usual.
            1 << log_next_power_of_two
        } else {
            // huge allocations allocate exactly what they want.
            size
        };

        let ptr = libc::malloc(alloc_size as libc::size_t);
        if ptr.is_null() {
            fail!("GC failed to allocate.")
        }

        self.gc_ptrs.insert_alloc(ptr as uint, size, tracer, finaliser);

        ptr as *mut u8
    }

    /// Register the block of memory [`start`, `end`) for tracing when
    /// a word matching `start` pointer is seen during a conservative
    /// scan. On such a scan, `tracer` is called, passing in the
    /// pointer and `metadata` (which can be arbitrary).
    pub unsafe fn nongc_register(&mut self, start: *(), metadata: uint, tracer: TracingFunc) {
        self.roots.insert_alloc(start as uint, metadata, Some(tracer), None)
    }

    /// Update the metadata word associated with `start`.
    pub unsafe fn nongc_update_metadata(&mut self, start: *(), metadata: uint) -> bool {
        self.roots.update_metadata(start as uint, metadata)
    }

    /// Stop considering the root starting at `start` for tracing.
    pub unsafe fn nongc_unregister(&mut self, start: *()) {
        self.roots.remove(start as uint);
    }

    /// Check if this is the first time that the non-GC'd pointer
    /// `start` has been traced in this iteration.
    pub fn nongc_first_trace(&mut self, start: *()) -> bool {
        debug!("nongc_first_trace: checking {}", start);
        self.roots.mark_reachable_scan_info(start as uint).is_some()
    }

    /// Check if this is the first time that the GC'd pointer `start`
    /// has been traced in this iteration.
    pub fn gc_first_trace(&mut self, start: *()) -> bool {
        debug!("gc_first_trace: checking {}", start);
        self.gc_ptrs.mark_reachable_scan_info(start as uint).is_some()
    }

    /// Run a conservative scan of the words from `start` to `end`.
    pub unsafe fn conservative_scan(&mut self, mut start: *uint, end: *uint) {
        while start < end {
            let ptr = *start;
            let trace_info = match self.gc_ptrs.mark_reachable_scan_info(ptr) {
                i @ Some(_) => i,
                None => self.roots.mark_reachable_scan_info(ptr)
            };
            match trace_info {
                Some((metadata, Some(tracer))) => {
                    tracer(ptr as *(), metadata, &mut GcTracer { gc: self })
                }
                // don't need no tracing (either not a pointer we
                // recognise, or one without a registered tracer)
                _ => {}
            }

            start = start.offset(1);
        }
    }

    /// Collect garbage. An upper bound on the position of any GC'd
    /// pointers on the stack should be passed as `stack_top`.
    pub unsafe fn collect(&mut self, stack_top: uint) {
        debug!("collecting");
        clear_registers(0, 0, 0, 0, 0, 0);

        let stack: uint = 1;
        let stack_end = &stack as *uint;

        // Step 1. mark any reachable pointers

        // every pointer is considered reachable on this exact line
        // (new allocations are reachable by default)
        self.gc_ptrs.toggle_reachability();
        self.roots.inefficient_mark_all_unreachable();
        // and now everything is considered unreachable.

        self.conservative_scan(stack_end, stack_top as *uint);

        // conservatively search task-local storage; this could
        // possibly use the tydesc to be precise.
        local_data::each_unborrowed_value(|ptr, tydesc| {
                let end = (ptr as *u8).offset((*tydesc).size as int);
                self.conservative_scan(ptr as *uint, end as *uint)
            });

        // Step 2. sweep all the unreachable ones for deallocation.
        let mut bytes_collected = 0u;
        let mut large_allocs = ~[];
        self.gc_ptrs.each_unreachable(|ptr, descr| {
                debug!("unreachable: 0x{:x}", ptr);
                match descr.finaliser {
                    Some(f) => f(ptr as *mut ()),
                    None => {}
                }

                // GC'd pointers use the metadata to store the size
                let log_rounded = compute_log_rounded_up_size(descr.metadata);
                // a "small" allocation so we cache it.
                if log_rounded <= ALLOC_CACHE_MAX_LOG {
                    // the each_unreachable driver marks this as
                    // unused internally.
                    self.alloc_cache[log_rounded - ALLOC_CACHE_MIN_LOG].push(ptr);

                    let actual_size = 1 << log_rounded;
                    bytes_collected += actual_size;
                } else {
                    large_allocs.push(ptr);

                    bytes_collected += descr.metadata;
                }

                true
            });
        // have to do these removals outside that loop
        for &ptr in large_allocs.iter() {
            // a big one, so whatever, the OS can have its memory
            // back.
            self.gc_ptrs.remove(ptr);
            libc::free(ptr as *libc::c_void);
        }
    }
}

impl Drop for GarbageCollector {
    fn drop(&mut self) {
        // free all the pointers we're controlling.
        for (ptr, descr) in self.gc_ptrs.iter() {
            // unused pointers have their finalisers cleared.
            match descr.finaliser {
                Some(f) => f(ptr as *mut ()),
                None => {}
            }
            unsafe {libc::free(ptr as *libc::c_void);}
        }
    }
}

// cargo culted from Boehm.
#[inline(never)]
fn clear_registers(_: uint, _: uint, _: uint,
                   _: uint, _: uint, _: uint) {}
