// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
//! The arena, a fast but limited type of allocator.
//!
//! Arenas are a type of allocator that destroy the objects within, all at
//! once, once the arena itself is destroyed. They do not support deallocation
//! of individual objects while the arena itself is still alive. The benefit
//! of an arena is very fast allocation; just a pointer bump.

#[crate_id = "arena#0.10-pre"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[license = "MIT/ASL2"];
#[allow(missing_doc)];
#[feature(managed_boxes)];

extern mod collections;

#[cfg(test)] extern mod extra;

use collections::list::{List, Cons, Nil};
use collections::list;

use std::cast::{transmute, transmute_mut, transmute_mut_region};
use std::cast;
use std::cell::{Cell, RefCell};
use std::num;
use std::ptr;
use std::kinds::marker;
use std::mem;
use std::rc::Rc;
use std::rt::global_heap;
use std::unstable::intrinsics::{TyDesc, get_tydesc};
use std::unstable::intrinsics;
use std::util;
use std::vec;

// The way arena uses arrays is really deeply awful. The arrays are
// allocated, and have capacities reserved, but the fill for the array
// will always stay at 0.
#[deriving(Clone)]
struct Chunk {
    data: Rc<RefCell<~[u8]>>,
    fill: Cell<uint>,
    is_pod: Cell<bool>,
}
impl Chunk {
    fn capacity(&self) -> uint {
        self.data.borrow().borrow().get().capacity()
    }

    unsafe fn as_ptr(&self) -> *u8 {
        self.data.borrow().borrow().get().as_ptr()
    }
}

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
// is important to not run the destructor on uninitialized objects, but
// how to detect them is somewhat subtle. Since alloc() can be invoked
// recursively, it is not sufficient to simply exclude the most recent
// object. To solve this without requiring extra space, we use the low
// order bit of the tydesc pointer to encode whether the object it
// describes has been fully initialized.

// As an optimization, objects with destructors are stored in
// different chunks than objects without destructors. This reduces
// overhead when initializing plain-old-data and means we don't need
// to waste time running the destructors of POD.
pub struct Arena {
    // The head is separated out from the list as a unbenchmarked
    // microoptimization, to avoid needing to case on the list to
    // access the head.
    priv head: Chunk,
    priv pod_head: Chunk,
    priv chunks: RefCell<@List<Chunk>>,
    priv no_freeze: marker::NoFreeze,
}

impl Arena {
    pub fn new() -> Arena {
        Arena::new_with_size(32u)
    }

    pub fn new_with_size(initial_size: uint) -> Arena {
        Arena {
            head: chunk(initial_size, false),
            pod_head: chunk(initial_size, true),
            chunks: RefCell::new(@Nil),
            no_freeze: marker::NoFreeze,
        }
    }
}

fn chunk(size: uint, is_pod: bool) -> Chunk {
    Chunk {
        data: Rc::new(RefCell::new(vec::with_capacity(size))),
        fill: Cell::new(0u),
        is_pod: Cell::new(is_pod),
    }
}

#[unsafe_destructor]
impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            destroy_chunk(&self.head);

            list::each(self.chunks.get(), |chunk| {
                if !chunk.is_pod.get() {
                    destroy_chunk(chunk);
                }
                true
            });
        }
    }
}

#[inline]
fn round_up(base: uint, align: uint) -> uint {
    (base.checked_add(&(align - 1))).unwrap() & !(&(align - 1))
}

// Walk down a chunk, running the destructors for any objects stored
// in it.
unsafe fn destroy_chunk(chunk: &Chunk) {
    let mut idx = 0;
    let buf = chunk.as_ptr();
    let fill = chunk.fill.get();

    while idx < fill {
        let tydesc_data: *uint = transmute(ptr::offset(buf, idx as int));
        let (tydesc, is_done) = un_bitpack_tydesc_ptr(*tydesc_data);
        let (size, align) = ((*tydesc).size, (*tydesc).align);

        let after_tydesc = idx + mem::size_of::<*TyDesc>();

        let start = round_up(after_tydesc, align);

        //debug!("freeing object: idx = {}, size = {}, align = {}, done = {}",
        //       start, size, align, is_done);
        if is_done {
            ((*tydesc).drop_glue)(ptr::offset(buf, start as int) as *i8);
        }

        // Find where the next tydesc lives
        idx = round_up(start + size, mem::pref_align_of::<*TyDesc>());
    }
}

// We encode whether the object a tydesc describes has been
// initialized in the arena in the low bit of the tydesc pointer. This
// is necessary in order to properly do cleanup if a failure occurs
// during an initializer.
#[inline]
unsafe fn bitpack_tydesc_ptr(p: *TyDesc, is_done: bool) -> uint {
    let p_bits: uint = transmute(p);
    p_bits | (is_done as uint)
}
#[inline]
unsafe fn un_bitpack_tydesc_ptr(p: uint) -> (*TyDesc, bool) {
    (transmute(p & !1), p & 1 == 1)
}

impl Arena {
    fn chunk_size(&self) -> uint {
        self.pod_head.capacity()
    }
    // Functions for the POD part of the arena
    fn alloc_pod_grow(&mut self, n_bytes: uint, align: uint) -> *u8 {
        // Allocate a new chunk.
        let new_min_chunk_size = num::max(n_bytes, self.chunk_size());
        self.chunks.set(@Cons(self.pod_head.clone(), self.chunks.get()));
        self.pod_head =
            chunk(num::next_power_of_two(new_min_chunk_size + 1u), true);

        return self.alloc_pod_inner(n_bytes, align);
    }

    #[inline]
    fn alloc_pod_inner(&mut self, n_bytes: uint, align: uint) -> *u8 {
        unsafe {
            let this = transmute_mut_region(self);
            let start = round_up(this.pod_head.fill.get(), align);
            let end = start + n_bytes;
            if end > self.chunk_size() {
                return this.alloc_pod_grow(n_bytes, align);
            }
            this.pod_head.fill.set(end);

            //debug!("idx = {}, size = {}, align = {}, fill = {}",
            //       start, n_bytes, align, head.fill.get());

            this.pod_head.as_ptr().offset(start as int)
        }
    }

    #[inline]
    fn alloc_pod<'a, T>(&'a mut self, op: || -> T) -> &'a T {
        unsafe {
            let tydesc = get_tydesc::<T>();
            let ptr = self.alloc_pod_inner((*tydesc).size, (*tydesc).align);
            let ptr: *mut T = transmute(ptr);
            intrinsics::move_val_init(&mut (*ptr), op());
            return transmute(ptr);
        }
    }

    // Functions for the non-POD part of the arena
    fn alloc_nonpod_grow(&mut self, n_bytes: uint, align: uint)
                         -> (*u8, *u8) {
        // Allocate a new chunk.
        let new_min_chunk_size = num::max(n_bytes, self.chunk_size());
        self.chunks.set(@Cons(self.head.clone(), self.chunks.get()));
        self.head =
            chunk(num::next_power_of_two(new_min_chunk_size + 1u), false);

        return self.alloc_nonpod_inner(n_bytes, align);
    }

    #[inline]
    fn alloc_nonpod_inner(&mut self, n_bytes: uint, align: uint)
                          -> (*u8, *u8) {
        unsafe {
            let start;
            let end;
            let tydesc_start;
            let after_tydesc;

            {
                let head = transmute_mut_region(&mut self.head);

                tydesc_start = head.fill.get();
                after_tydesc = head.fill.get() + mem::size_of::<*TyDesc>();
                start = round_up(after_tydesc, align);
                end = start + n_bytes;
            }

            if end > self.head.capacity() {
                return self.alloc_nonpod_grow(n_bytes, align);
            }

            let head = transmute_mut_region(&mut self.head);
            head.fill.set(round_up(end, mem::pref_align_of::<*TyDesc>()));

            //debug!("idx = {}, size = {}, align = {}, fill = {}",
            //       start, n_bytes, align, head.fill);

            let buf = self.head.as_ptr();
            return (ptr::offset(buf, tydesc_start as int), ptr::offset(buf, start as int));
        }
    }

    #[inline]
    fn alloc_nonpod<'a, T>(&'a mut self, op: || -> T) -> &'a T {
        unsafe {
            let tydesc = get_tydesc::<T>();
            let (ty_ptr, ptr) =
                self.alloc_nonpod_inner((*tydesc).size, (*tydesc).align);
            let ty_ptr: *mut uint = transmute(ty_ptr);
            let ptr: *mut T = transmute(ptr);
            // Write in our tydesc along with a bit indicating that it
            // has *not* been initialized yet.
            *ty_ptr = transmute(tydesc);
            // Actually initialize it
            intrinsics::move_val_init(&mut(*ptr), op());
            // Now that we are done, update the tydesc to indicate that
            // the object is there.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, true);

            return transmute(ptr);
        }
    }

    // The external interface
    #[inline]
    pub fn alloc<'a, T>(&'a self, op: || -> T) -> &'a T {
        unsafe {
            // FIXME: Borrow check
            let this = transmute_mut(self);
            if intrinsics::needs_drop::<T>() {
                this.alloc_nonpod(op)
            } else {
                this.alloc_pod(op)
            }
        }
    }
}

#[test]
fn test_arena_destructors() {
    let arena = Arena::new();
    for i in range(0u, 10) {
        // Arena allocate something with drop glue to make sure it
        // doesn't leak.
        arena.alloc(|| @i);
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        arena.alloc(|| [0u8, 1u8, 2u8]);
    }
}

#[test]
#[should_fail]
fn test_arena_destructors_fail() {
    let arena = Arena::new();
    // Put some stuff in the arena.
    for i in range(0u, 10) {
        // Arena allocate something with drop glue to make sure it
        // doesn't leak.
        arena.alloc(|| { @i });
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        arena.alloc(|| { [0u8, 1u8, 2u8] });
    }
    // Now, fail while allocating
    arena.alloc::<@int>(|| {
        // Now fail.
        fail!();
    });
}

/// An arena that can hold objects of only one type.
///
/// Safety note: Modifying objects in the arena that have already had their
/// `drop` destructors run can cause leaks, because the destructor will not
/// run again for these objects.
pub struct TypedArena<T> {
    /// A pointer to the next object to be allocated.
    priv ptr: *T,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    priv end: *T,

    /// The type descriptor of the objects in the arena. This should not be
    /// necessary, but is until generic destructors are supported.
    priv tydesc: *TyDesc,

    /// A pointer to the first arena segment.
    priv first: Option<~TypedArenaChunk>,
}

struct TypedArenaChunk {
    /// Pointer to the next arena segment.
    next: Option<~TypedArenaChunk>,

    /// The number of elements that this chunk can hold.
    capacity: uint,

    // Objects follow here, suitably aligned.
}

impl TypedArenaChunk {
    #[inline]
    fn new<T>(next: Option<~TypedArenaChunk>, capacity: uint)
           -> ~TypedArenaChunk {
        let mut size = mem::size_of::<TypedArenaChunk>();
        size = round_up(size, mem::min_align_of::<T>());
        let elem_size = mem::size_of::<T>();
        let elems_size = elem_size.checked_mul(&capacity).unwrap();
        size = size.checked_add(&elems_size).unwrap();

        let mut chunk = unsafe {
            let chunk = global_heap::exchange_malloc(size);
            let mut chunk: ~TypedArenaChunk = cast::transmute(chunk);
            intrinsics::move_val_init(&mut chunk.next, next);
            chunk
        };

        chunk.capacity = capacity;
        chunk
    }

    /// Destroys this arena chunk. If the type descriptor is supplied, the
    /// drop glue is called; otherwise, drop glue is not called.
    #[inline]
    unsafe fn destroy(&mut self, len: uint, opt_tydesc: Option<*TyDesc>) {
        // Destroy all the allocated objects.
        match opt_tydesc {
            None => {}
            Some(tydesc) => {
                let mut start = self.start(tydesc);
                for _ in range(0, len) {
                    ((*tydesc).drop_glue)(start as *i8);
                    start = start.offset((*tydesc).size as int)
                }
            }
        }

        // Destroy the next chunk.
        let next_opt = util::replace(&mut self.next, None);
        match next_opt {
            None => {}
            Some(mut next) => {
                // We assume that the next chunk is completely filled.
                next.destroy(next.capacity, opt_tydesc)
            }
        }
    }

    // Returns a pointer to the first allocated object.
    #[inline]
    fn start(&self, tydesc: *TyDesc) -> *u8 {
        let this: *TypedArenaChunk = self;
        unsafe {
            cast::transmute(round_up(this.offset(1) as uint, (*tydesc).align))
        }
    }

    // Returns a pointer to the end of the allocated space.
    #[inline]
    fn end(&self, tydesc: *TyDesc) -> *u8 {
        unsafe {
            let size = (*tydesc).size.checked_mul(&self.capacity).unwrap();
            self.start(tydesc).offset(size as int)
        }
    }
}

impl<T> TypedArena<T> {
    /// Creates a new arena with preallocated space for 8 objects.
    #[inline]
    pub fn new() -> TypedArena<T> {
        TypedArena::with_capacity(8)
    }

    /// Creates a new arena with preallocated space for the given number of
    /// objects.
    #[inline]
    pub fn with_capacity(capacity: uint) -> TypedArena<T> {
        let chunk = TypedArenaChunk::new::<T>(None, capacity);
        let tydesc = unsafe {
            intrinsics::get_tydesc::<T>()
        };
        TypedArena {
            ptr: chunk.start(tydesc) as *T,
            end: chunk.end(tydesc) as *T,
            tydesc: tydesc,
            first: Some(chunk),
        }
    }

    /// Allocates an object into this arena.
    #[inline]
    pub fn alloc<'a>(&'a self, object: T) -> &'a T {
        unsafe {
            let this = cast::transmute_mut(self);
            if this.ptr == this.end {
                this.grow()
            }

            let ptr: &'a mut T = cast::transmute(this.ptr);
            intrinsics::move_val_init(ptr, object);
            this.ptr = this.ptr.offset(1);
            let ptr: &'a T = ptr;
            ptr
        }
    }

    /// Grows the arena.
    #[inline(never)]
    fn grow(&mut self) {
        let chunk = self.first.take_unwrap();
        let new_capacity = chunk.capacity.checked_mul(&2).unwrap();
        let chunk = TypedArenaChunk::new::<T>(Some(chunk), new_capacity);
        self.ptr = chunk.start(self.tydesc) as *T;
        self.end = chunk.end(self.tydesc) as *T;
        self.first = Some(chunk)
    }
}

#[unsafe_destructor]
impl<T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        // Determine how much was filled.
        let start = self.first.get_ref().start(self.tydesc) as uint;
        let end = self.ptr as uint;
        let diff = (end - start) / mem::size_of::<T>();

        // Pass that to the `destroy` method.
        unsafe {
            let opt_tydesc = if intrinsics::needs_drop::<T>() {
                Some(self.tydesc)
            } else {
                None
            };
            self.first.get_mut_ref().destroy(diff, opt_tydesc)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Arena, TypedArena};
    use extra::test::BenchHarness;

    struct Point {
        x: int,
        y: int,
        z: int,
    }

    #[test]
    pub fn test_pod() {
        let arena = TypedArena::new();
        for _ in range(0, 100000) {
            arena.alloc(Point {
                x: 1,
                y: 2,
                z: 3,
            });
        }
    }

    #[bench]
    pub fn bench_pod(bh: &mut BenchHarness) {
        let arena = TypedArena::new();
        bh.iter(|| {
            arena.alloc(Point {
                x: 1,
                y: 2,
                z: 3,
            });
        })
    }

    #[bench]
    pub fn bench_pod_nonarena(bh: &mut BenchHarness) {
        bh.iter(|| {
            let _ = ~Point {
                x: 1,
                y: 2,
                z: 3,
            };
        })
    }

    #[bench]
    pub fn bench_pod_old_arena(bh: &mut BenchHarness) {
        let arena = Arena::new();
        bh.iter(|| {
            arena.alloc(|| {
                Point {
                    x: 1,
                    y: 2,
                    z: 3,
                }
            });
        })
    }

    struct Nonpod {
        string: ~str,
        array: ~[int],
    }

    #[test]
    pub fn test_nonpod() {
        let arena = TypedArena::new();
        for _ in range(0, 100000) {
            arena.alloc(Nonpod {
                string: ~"hello world",
                array: ~[ 1, 2, 3, 4, 5 ],
            });
        }
    }

    #[bench]
    pub fn bench_nonpod(bh: &mut BenchHarness) {
        let arena = TypedArena::new();
        bh.iter(|| {
            arena.alloc(Nonpod {
                string: ~"hello world",
                array: ~[ 1, 2, 3, 4, 5 ],
            });
        })
    }

    #[bench]
    pub fn bench_nonpod_nonarena(bh: &mut BenchHarness) {
        bh.iter(|| {
            let _ = ~Nonpod {
                string: ~"hello world",
                array: ~[ 1, 2, 3, 4, 5 ],
            };
        })
    }

    #[bench]
    pub fn bench_nonpod_old_arena(bh: &mut BenchHarness) {
        let arena = Arena::new();
        bh.iter(|| {
            let _ = arena.alloc(|| Nonpod {
                string: ~"hello world",
                array: ~[ 1, 2, 3, 4, 5 ],
            });
        })
    }
}
