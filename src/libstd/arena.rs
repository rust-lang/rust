// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Dynamic arenas.

// Arenas are used to quickly allocate objects that share a
// lifetime. The arena uses ~[u8] vectors as a backing store to
// allocate objects from. For each allocated object, the arena stores
// a pointer to the type descriptor followed by the
// object. (Potentially with alignment padding after each of them.)
// When the arena is destroyed, it iterates through all of its chunks,
// and uses the tydesc information to trace through the objects,
// calling the destructors on them.
// One subtle point that needs to be addressed is how to handle
// failures while running the user provided initializer function. It
// is important to not run the destructor on uninitalized objects, but
// how to detect them is somewhat subtle. Since alloc() can be invoked
// recursively, it is not sufficient to simply exclude the most recent
// object. To solve this without requiring extra space, we use the low
// order bit of the tydesc pointer to encode whether the object it
// describes has been fully initialized.

// As an optimization, objects with destructors are stored in
// different chunks than objects without destructors. This reduces
// overhead when initializing plain-old-data and means we don't need
// to waste time running the destructors of POD.

use list;
use list::{List, Cons, Nil};

use core::at_vec;
use core::cast::transmute;
use core::cast;
use core::libc::size_t;
use core::ptr;
use core::sys::TypeDesc;
use core::sys;
use core::uint;
use core::vec;

pub mod rusti {
    #[abi = "rust-intrinsic"]
    pub extern "rust-intrinsic" {
        fn move_val_init<T>(dst: &mut T, +src: T);
        fn needs_drop<T>() -> bool;
    }
}

pub mod rustrt {
    use core::libc::size_t;
    use core::sys::TypeDesc;

    pub extern {
        #[rust_stack]
        unsafe fn rust_call_tydesc_glue(root: *u8,
                                        tydesc: *TypeDesc,
                                        field: size_t);
    }
}

// This probably belongs somewhere else. Needs to be kept in sync with
// changes to glue...
static tydesc_drop_glue_index: size_t = 3 as size_t;

// The way arena uses arrays is really deeply awful. The arrays are
// allocated, and have capacities reserved, but the fill for the array
// will always stay at 0.
struct Chunk {
    data: @[u8],
    mut fill: uint,
    is_pod: bool,
}

pub struct Arena {
    // The head is seperated out from the list as a unbenchmarked
    // microoptimization, to avoid needing to case on the list to
    // access the head.
    priv mut head: Chunk,
    priv mut pod_head: Chunk,
    priv mut chunks: @List<Chunk>,
}

#[unsafe_destructor]
impl Drop for Arena {
    fn finalize(&self) {
        unsafe {
            destroy_chunk(&self.head);
            for list::each(self.chunks) |chunk| {
                if !chunk.is_pod { destroy_chunk(chunk); }
            }
        }
    }
}

fn chunk(size: uint, is_pod: bool) -> Chunk {
    let mut v: @[u8] = @[];
    unsafe { at_vec::raw::reserve(&mut v, size); }
    Chunk {
        data: unsafe { cast::transmute(v) },
        fill: 0u,
        is_pod: is_pod,
    }
}

pub fn arena_with_size(initial_size: uint) -> Arena {
    Arena {
        head: chunk(initial_size, false),
        pod_head: chunk(initial_size, true),
        chunks: @Nil,
    }
}

pub fn Arena() -> Arena {
    arena_with_size(32u)
}

#[inline(always)]
fn round_up_to(base: uint, align: uint) -> uint {
    (base + (align - 1)) & !(align - 1)
}

// Walk down a chunk, running the destructors for any objects stored
// in it.
unsafe fn destroy_chunk(chunk: &Chunk) {
    let mut idx = 0;
    let buf = vec::raw::to_ptr(chunk.data);
    let fill = chunk.fill;

    while idx < fill {
        let tydesc_data: *uint = transmute(ptr::offset(buf, idx));
        let (tydesc, is_done) = un_bitpack_tydesc_ptr(*tydesc_data);
        let size = (*tydesc).size, align = (*tydesc).align;

        let after_tydesc = idx + sys::size_of::<*TypeDesc>();

        let start = round_up_to(after_tydesc, align);

        //debug!("freeing object: idx = %u, size = %u, align = %u, done = %b",
        //       start, size, align, is_done);
        if is_done {
            rustrt::rust_call_tydesc_glue(
                ptr::offset(buf, start), tydesc, tydesc_drop_glue_index);
        }

        // Find where the next tydesc lives
        idx = round_up_to(start + size, sys::pref_align_of::<*TypeDesc>());
    }
}

// We encode whether the object a tydesc describes has been
// initialized in the arena in the low bit of the tydesc pointer. This
// is necessary in order to properly do cleanup if a failure occurs
// during an initializer.
#[inline(always)]
unsafe fn bitpack_tydesc_ptr(p: *TypeDesc, is_done: bool) -> uint {
    let p_bits: uint = transmute(p);
    p_bits | (is_done as uint)
}
#[inline(always)]
unsafe fn un_bitpack_tydesc_ptr(p: uint) -> (*TypeDesc, bool) {
    (transmute(p & !1), p & 1 == 1)
}

pub impl Arena {
    // Functions for the POD part of the arena
    priv fn alloc_pod_grow(&self, n_bytes: uint, align: uint) -> *u8 {
        // Allocate a new chunk.
        let chunk_size = at_vec::capacity(self.pod_head.data);
        let new_min_chunk_size = uint::max(n_bytes, chunk_size);
        self.chunks = @Cons(copy self.pod_head, self.chunks);
        self.pod_head =
            chunk(uint::next_power_of_two(new_min_chunk_size + 1u), true);

        return self.alloc_pod_inner(n_bytes, align);
    }

    #[inline(always)]
    priv fn alloc_pod_inner(&self, n_bytes: uint, align: uint) -> *u8 {
        let head = &mut self.pod_head;

        let start = round_up_to(head.fill, align);
        let end = start + n_bytes;
        if end > at_vec::capacity(head.data) {
            return self.alloc_pod_grow(n_bytes, align);
        }
        head.fill = end;

        //debug!("idx = %u, size = %u, align = %u, fill = %u",
        //       start, n_bytes, align, head.fill);

        unsafe {
            ptr::offset(vec::raw::to_ptr(head.data), start)
        }
    }

    #[inline(always)]
    #[cfg(stage0)]
    priv fn alloc_pod<T>(&self, op: &fn() -> T) -> &'self T {
        unsafe {
            let tydesc = sys::get_type_desc::<T>();
            let ptr = self.alloc_pod_inner((*tydesc).size, (*tydesc).align);
            let ptr: *mut T = transmute(ptr);
            rusti::move_val_init(&mut (*ptr), op());
            return transmute(ptr);
        }
    }

    #[inline(always)]
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    priv fn alloc_pod<'a, T>(&'a self, op: &fn() -> T) -> &'a T {
        unsafe {
            let tydesc = sys::get_type_desc::<T>();
            let ptr = self.alloc_pod_inner((*tydesc).size, (*tydesc).align);
            let ptr: *mut T = transmute(ptr);
            rusti::move_val_init(&mut (*ptr), op());
            return transmute(ptr);
        }
    }

    // Functions for the non-POD part of the arena
    priv fn alloc_nonpod_grow(&self, n_bytes: uint, align: uint) -> (*u8, *u8) {
        // Allocate a new chunk.
        let chunk_size = at_vec::capacity(self.head.data);
        let new_min_chunk_size = uint::max(n_bytes, chunk_size);
        self.chunks = @Cons(copy self.head, self.chunks);
        self.head =
            chunk(uint::next_power_of_two(new_min_chunk_size + 1u), false);

        return self.alloc_nonpod_inner(n_bytes, align);
    }

    #[inline(always)]
    priv fn alloc_nonpod_inner(&self, n_bytes: uint, align: uint) -> (*u8, *u8) {
        let head = &mut self.head;

        let tydesc_start = head.fill;
        let after_tydesc = head.fill + sys::size_of::<*TypeDesc>();
        let start = round_up_to(after_tydesc, align);
        let end = start + n_bytes;
        if end > at_vec::capacity(head.data) {
            return self.alloc_nonpod_grow(n_bytes, align);
        }
        head.fill = round_up_to(end, sys::pref_align_of::<*TypeDesc>());

        //debug!("idx = %u, size = %u, align = %u, fill = %u",
        //       start, n_bytes, align, head.fill);

        unsafe {
            let buf = vec::raw::to_ptr(head.data);
            return (ptr::offset(buf, tydesc_start), ptr::offset(buf, start));
        }
    }

    #[inline(always)]
    #[cfg(stage0)]
    priv fn alloc_nonpod<T>(&self, op: &fn() -> T) -> &'self T {
        unsafe {
            let tydesc = sys::get_type_desc::<T>();
            let (ty_ptr, ptr) =
                self.alloc_nonpod_inner((*tydesc).size, (*tydesc).align);
            let ty_ptr: *mut uint = transmute(ty_ptr);
            let ptr: *mut T = transmute(ptr);
            // Write in our tydesc along with a bit indicating that it
            // has *not* been initialized yet.
            *ty_ptr = transmute(tydesc);
            // Actually initialize it
            rusti::move_val_init(&mut(*ptr), op());
            // Now that we are done, update the tydesc to indicate that
            // the object is there.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, true);

            return transmute(ptr);
        }
    }

    #[inline(always)]
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    priv fn alloc_nonpod<'a, T>(&'a self, op: &fn() -> T) -> &'a T {
        unsafe {
            let tydesc = sys::get_type_desc::<T>();
            let (ty_ptr, ptr) =
                self.alloc_nonpod_inner((*tydesc).size, (*tydesc).align);
            let ty_ptr: *mut uint = transmute(ty_ptr);
            let ptr: *mut T = transmute(ptr);
            // Write in our tydesc along with a bit indicating that it
            // has *not* been initialized yet.
            *ty_ptr = transmute(tydesc);
            // Actually initialize it
            rusti::move_val_init(&mut(*ptr), op());
            // Now that we are done, update the tydesc to indicate that
            // the object is there.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, true);

            return transmute(ptr);
        }
    }

    // The external interface
    #[inline(always)]
    #[cfg(stage0)]
    fn alloc<T>(&self, op: &fn() -> T) -> &'self T {
        unsafe {
            if !rusti::needs_drop::<T>() {
                self.alloc_pod(op)
            } else {
                self.alloc_nonpod(op)
            }
        }
    }

    // The external interface
    #[inline(always)]
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn alloc<'a, T>(&'a self, op: &fn() -> T) -> &'a T {
        unsafe {
            if !rusti::needs_drop::<T>() {
                self.alloc_pod(op)
            } else {
                self.alloc_nonpod(op)
            }
        }
    }
}

#[test]
fn test_arena_destructors() {
    let arena = Arena();
    for uint::range(0, 10) |i| {
        // Arena allocate something with drop glue to make sure it
        // doesn't leak.
        do arena.alloc { @i };
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        do arena.alloc { [0u8, 1u8, 2u8] };
    }
}

#[test] #[should_fail] #[ignore(cfg(windows))]
fn test_arena_destructors_fail() {
    let arena = Arena();
    // Put some stuff in the arena.
    for uint::range(0, 10) |i| {
        // Arena allocate something with drop glue to make sure it
        // doesn't leak.
        do arena.alloc { @i };
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        do arena.alloc { [0u8, 1u8, 2u8] };
    }
    // Now, fail while allocating
    do arena.alloc::<@int> {
        // First, recursively allocate something else; that needs to
        // get freed too.
        do arena.alloc { @20 };
        // Now fail.
        fail!();
    };
}
