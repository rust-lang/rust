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
#[allow(non_uppercase_statics)];

/*! Precise garbage collector

The precise GC exposes two functions, gc and
cleanup_stack_for_failure. The gc function is the entry point to the
garbage collector itself. The cleanup_stack_for_failure is the entry
point for GC-based cleanup.

Precise GC depends on changes to LLVM's GC which add support for
automatic rooting and addrspace-based metadata marking. Rather than
explicitly rooting pointers with LLVM's gcroot intrinsic, the GC
merely creates allocas for pointers, and allows an LLVM pass to
automatically infer roots based on the allocas present in a function
(and live at a given location). The compiler communicates the type of
the pointer to LLVM by setting the addrspace of the pointer type. The
compiler then emits a map from addrspace to tydesc, which LLVM then
uses to match pointers with their tydesc. The GC reads the metadata
table produced by LLVM, and uses it to determine which glue functions
to call to free objects on their respective heaps.

GC-based cleanup is a replacement for landing pads which relies on the
GC infrastructure to find pointers on the stack to cleanup. Whereas
the normal GC needs to walk task-local heap allocations, the cleanup
code needs to walk exchange heap allocations and stack-allocations
with destructors.

*/

use cast;
use container::{Set, MutableSet};
use io;
use libc::{uintptr_t};
use option::{None, Option, Some};
use ptr;
use hashmap::HashSet;
use stackwalk::walk_stack;
use sys;
use unstable::intrinsics::{TyDesc};

pub use stackwalk::Word;

// Mirrors rust_stack.h stk_seg
pub struct StackSegment {
    prev: *StackSegment,
    next: *StackSegment,
    end: uintptr_t,
    // And other fields which we don't care about...
}

pub mod rustrt {
    use stackwalk::Word;
    use super::StackSegment;

    #[link_name = "rustrt"]
    extern {
        #[rust_stack]
        pub unsafe fn rust_gc_metadata() -> *Word;

        pub unsafe fn rust_get_stack_segment() -> *StackSegment;
        pub unsafe fn rust_get_c_stack() -> *StackSegment;
    }
}

unsafe fn bump<T, U>(ptr: *T, count: uint) -> *U {
    return ptr::offset(ptr, count as int) as *U;
}

unsafe fn align_to_pointer<T>(ptr: *T) -> *T {
    let align = sys::min_align_of::<*T>();
    let ptr = ptr as uint;
    let ptr = (ptr + (align - 1)) & -align;
    return ptr as *T;
}

unsafe fn get_safe_point_count() -> uint {
    let module_meta = rustrt::rust_gc_metadata();
    return *module_meta;
}

struct SafePoint {
    sp_meta: *Word,
    fn_meta: *Word,
}

// Returns the safe point metadata for the given program counter, if
// any.
unsafe fn is_safe_point(pc: *Word) -> Option<SafePoint> {
    let module_meta = rustrt::rust_gc_metadata();
    let num_safe_points = *module_meta;
    let safe_points: *Word = bump(module_meta, 1);

    if ptr::is_null(pc) {
        return None;
    }

    // FIXME (#2997): Use binary rather than linear search.
    let mut spi = 0;
    while spi < num_safe_points {
        let sp: **Word = bump(safe_points, spi*3);
        let sp_loc = *sp;
        if sp_loc == pc {
            return Some(SafePoint {
                sp_meta: *bump(sp, 1),
                fn_meta: *bump(sp, 2),
            });
        }
        spi += 1;
    }
    return None;
}

type Visitor<'self> = &'self fn(root: **Word, tydesc: *TyDesc);

// Walks the list of roots for the given safe point, and calls visitor
// on each root.
unsafe fn _walk_safe_point(fp: *Word, sp: SafePoint, visitor: Visitor) {
    let fp_bytes = fp as *u8;
    let sp_meta = sp.sp_meta as *u32;

    let num_stack_roots = *sp_meta as uint;
    let num_reg_roots = *ptr::offset(sp_meta, 1) as uint;

    let stack_roots: *u32 = bump(sp_meta, 2);
    let reg_roots: *u8 = bump(stack_roots, num_stack_roots);
    let addrspaces: *Word = align_to_pointer(bump(reg_roots, num_reg_roots));
    let tydescs: ***TyDesc = bump(addrspaces, num_stack_roots);

    // Stack roots
    let mut sri = 0;
    while sri < num_stack_roots {
        if *ptr::offset(addrspaces, sri as int) >= 1 {
            let root =
                ptr::offset(fp_bytes, *ptr::offset(stack_roots, sri as int) as int)
                as **Word;
            let tydescpp = ptr::offset(tydescs, sri as int);
            let tydesc = if ptr::is_not_null(tydescpp) &&
                ptr::is_not_null(*tydescpp) {
                **tydescpp
            } else {
                ptr::null()
            };
            visitor(root, tydesc);
        }
        sri += 1;
    }

    // Register roots
    let mut rri = 0;
    while rri < num_reg_roots {
        if *ptr::offset(addrspaces, (num_stack_roots + rri) as int) == 1 {
            // FIXME(#2997): Need to find callee saved registers on the stack.
        }
        rri += 1;
    }
}

unsafe fn walk_safe_point(fp: *Word, sp: SafePoint, visitor: Visitor) {
    _walk_safe_point(fp, sp, visitor)
}

// Is fp contained in segment?
unsafe fn is_frame_in_segment(fp: *Word, segment: *StackSegment) -> bool {
    let begin = segment as Word;
    let end = (*segment).end as Word;
    let frame = fp as Word;

    return begin <= frame && frame <= end;
}

struct Segment { segment: *StackSegment, boundary: bool }

// Find and return the segment containing the given frame pointer. At
// stack segment boundaries, returns true for boundary, so that the
// caller can do any special handling to identify where the correct
// return address is in the stack frame.
unsafe fn find_segment_for_frame(fp: *Word, segment: *StackSegment)
    -> Segment {
    // Check if frame is in either current frame or previous frame.
    let in_segment = is_frame_in_segment(fp, segment);
    let in_prev_segment = ptr::is_not_null((*segment).prev) &&
        is_frame_in_segment(fp, (*segment).prev);

    // If frame is not in either segment, walk down segment list until
    // we find the segment containing this frame.
    if !in_segment && !in_prev_segment {
        let mut segment = segment;
        while ptr::is_not_null((*segment).next) &&
            is_frame_in_segment(fp, (*segment).next) {
            segment = (*segment).next;
        }
        return Segment {segment: segment, boundary: false};
    }

    // If frame is in previous frame, then we're at a boundary.
    if !in_segment && in_prev_segment {
        return Segment {segment: (*segment).prev, boundary: true};
    }

    // Otherwise, we're somewhere on the inside of the frame.
    return Segment {segment: segment, boundary: false};
}

type Memory = uint;

static task_local_heap: Memory = 1;
static exchange_heap:   Memory = 2;
static stack:           Memory = 4;

static need_cleanup:    Memory = exchange_heap | stack;

// Walks stack, searching for roots of the requested type, and passes
// each root to the visitor.
unsafe fn _walk_gc_roots(mem: Memory, sentinel: **Word, visitor: Visitor) {
    let mut segment = rustrt::rust_get_stack_segment();
    let mut last_ret: *Word = ptr::null();
    // To avoid collecting memory used by the GC itself, skip stack
    // frames until past the root GC stack frame. The root GC stack
    // frame is marked by a sentinel, which is a box pointer stored on
    // the stack.
    let mut reached_sentinel = ptr::is_null(sentinel);
    do walk_stack |frame| {
        let pc = last_ret;
        let Segment {segment: next_segment, boundary: boundary} =
            find_segment_for_frame(frame.fp, segment);
        segment = next_segment;
        // Each stack segment is bounded by a morestack frame. The
        // morestack frame includes two return addresses, one for
        // morestack itself, at the normal offset from the frame
        // pointer, and then a second return address for the
        // function prologue (which called morestack after
        // determining that it had hit the end of the stack).
        // Since morestack itself takes two parameters, the offset
        // for this second return address is 3 greater than the
        // return address for morestack.
        let ret_offset = if boundary { 4 } else { 1 };
        last_ret = *ptr::offset(frame.fp, ret_offset as int) as *Word;

        if !ptr::is_null(pc) {

            let mut delay_reached_sentinel = reached_sentinel;
            let sp = is_safe_point(pc);
            match sp {
                Some(sp_info) => {
                    do walk_safe_point(frame.fp, sp_info) |root, tydesc| {
                        // Skip roots until we see the sentinel.
                        if !reached_sentinel && root == sentinel {
                            delay_reached_sentinel = true;
                        }

                        // Skip null pointers, which can occur when a
                        // unique pointer has already been freed.
                        if reached_sentinel && !ptr::is_null(*root) {
                            if ptr::is_null(tydesc) {
                                // Root is a generic box.
                                let refcount = **root;
                                if mem | task_local_heap != 0 && refcount != -1 {
                                    visitor(root, tydesc);
                                } else if mem | exchange_heap != 0 && refcount == -1 {
                                    visitor(root, tydesc);
                                }
                            } else {
                                // Root is a non-immediate.
                                if mem | stack != 0 {
                                    visitor(root, tydesc);
                                }
                            }
                        }
                    }
                }
                None => ()
            }
            reached_sentinel = delay_reached_sentinel;
        }
    }
}

unsafe fn walk_gc_roots(mem: Memory, sentinel: **Word, visitor: Visitor) {
    _walk_gc_roots(mem, sentinel, visitor)
}
pub fn gc() {
    unsafe {
        // Abort when GC is disabled.
        if get_safe_point_count() == 0 {
            return;
        }

        do walk_gc_roots(task_local_heap, ptr::null()) |_root, _tydesc| {
            // FIXME(#2997): Walk roots and mark them.
            io::stdout().write([46]); // .
        }
    }
}

#[cfg(gc)]
fn expect_sentinel() -> bool { true }

#[cfg(nogc)]
fn expect_sentinel() -> bool { false }

// Entry point for GC-based cleanup. Walks stack looking for exchange
// heap and stack allocations requiring drop, and runs all
// destructors.
//
// This should only be called from fail!, as it will drop the roots
// which are *live* on the stack, rather than dropping those that are
// dead.
pub fn cleanup_stack_for_failure() {
    unsafe {
        // Abort when GC is disabled.
        if get_safe_point_count() == 0 {
            return;
        }

        // Leave a sentinel on the stack to mark the current frame. The
        // stack walker will ignore any frames above the sentinel, thus
        // avoiding collecting any memory being used by the stack walker
        // itself.
        //
        // However, when core itself is not compiled with GC, then none of
        // the functions in core will have GC metadata, which means we
        // won't be able to find the sentinel root on the stack. In this
        // case, we can safely skip the sentinel since we won't find our
        // own stack roots on the stack anyway.
        let sentinel_box = ~0;
        let sentinel: **Word = if expect_sentinel() {
            cast::transmute(&sentinel_box)
        } else {
            ptr::null()
        };

        let mut roots = HashSet::new();
        do walk_gc_roots(need_cleanup, sentinel) |root, tydesc| {
            // Track roots to avoid double frees.
            if !roots.contains(&*root) {
                roots.insert(*root);

                if ptr::is_null(tydesc) {
                    // FIXME #4420: Destroy this box
                    // FIXME #4330: Destroy this box
                } else {
                    ((*tydesc).drop_glue)(*root as *i8);
                }
            }
        }
    }
}
