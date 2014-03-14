// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];

use ptr;
use raw;

static RC_IMMORTAL : uint = 0x77777777;

/*
 * Box annihilation
 *
 * This runs at task death to free all boxes.
 */

unsafe fn each_live_alloc(read_next_before: bool,
                          f: |alloc: *mut raw::Box<()>| -> bool)
                          -> bool {
    //! Walks the internal list of allocations

    use rt::local_heap;

    let mut alloc = local_heap::live_allocs();
    while alloc != ptr::mut_null() {
        let next_before = (*alloc).next;

        if !f(alloc) {
            return false;
        }

        if read_next_before {
            alloc = next_before;
        } else {
            alloc = (*alloc).next;
        }
    }
    return true;
}

#[cfg(unix)]
fn debug_mem() -> bool {
    // FIXME: Need to port the environment struct to newsched
    false
}

#[cfg(windows)]
fn debug_mem() -> bool {
    false
}

/// Destroys all managed memory (i.e. @ boxes) held by the current task.
pub unsafe fn annihilate() {
    use rt::local_heap::local_free;

    let mut n_total_boxes = 0u;

    // Pass 1: Make all boxes immortal.
    //
    // In this pass, nothing gets freed, so it does not matter whether
    // we read the next field before or after the callback.
    each_live_alloc(true, |alloc| {
        n_total_boxes += 1;
        (*alloc).ref_count = RC_IMMORTAL;
        true
    });

    // Pass 2: Drop all boxes.
    //
    // In this pass, unique-managed boxes may get freed, but not
    // managed boxes, so we must read the `next` field *after* the
    // callback, as the original value may have been freed.
    each_live_alloc(false, |alloc| {
        let drop_glue = (*alloc).drop_glue;
        let data = &mut (*alloc).data as *mut ();
        drop_glue(data as *mut u8);
        true
    });

    // Pass 3: Free all boxes.
    //
    // In this pass, managed boxes may get freed (but not
    // unique-managed boxes, though I think that none of those are
    // left), so we must read the `next` field before, since it will
    // not be valid after.
    each_live_alloc(true, |alloc| {
        local_free(alloc as *u8);
        true
    });

    if debug_mem() {
        // We do logging here w/o allocation.
        debug!("total boxes annihilated: {}", n_total_boxes);
    }
}
