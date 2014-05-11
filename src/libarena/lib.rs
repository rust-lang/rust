// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

#![crate_id = "arena#0.11-pre"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]
#![allow(missing_doc)]

extern crate collections;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::intrinsics::{TyDesc, get_tydesc};
use std::intrinsics;
use std::mem;
use std::mem::min_align_of;
use std::num;
use std::ptr::read;
use std::rc::Rc;
use std::rt::heap::exchange_malloc;

// The way arena uses arrays is really deeply awful. The arrays are
// allocated, and have capacities reserved, but the fill for the array
// will always stay at 0.
#[deriving(Clone, Eq)]
struct Chunk {
    data: Rc<RefCell<Vec<u8> >>,
    fill: Cell<uint>,
    is_copy: Cell<bool>,
}
impl Chunk {
    fn capacity(&self) -> uint {
        self.data.borrow().capacity()
    }

    unsafe fn as_ptr(&self) -> *u8 {
        self.data.borrow().as_ptr()
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
    head: Chunk,
    copy_head: Chunk,
    chunks: RefCell<Vec<Chunk>>,
}

impl Arena {
    pub fn new() -> Arena {
        Arena::new_with_size(32u)
    }

    pub fn new_with_size(initial_size: uint) -> Arena {
        Arena {
            head: chunk(initial_size, false),
            copy_head: chunk(initial_size, true),
            chunks: RefCell::new(Vec::new()),
        }
    }
}

fn chunk(size: uint, is_copy: bool) -> Chunk {
    Chunk {
        data: Rc::new(RefCell::new(Vec::with_capacity(size))),
        fill: Cell::new(0u),
        is_copy: Cell::new(is_copy),
    }
}

#[unsafe_destructor]
impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            destroy_chunk(&self.head);
            for chunk in self.chunks.borrow().iter() {
                if !chunk.is_copy.get() {
                    destroy_chunk(chunk);
                }
            }
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
        let tydesc_data: *uint = mem::transmute(buf.offset(idx as int));
        let (tydesc, is_done) = un_bitpack_tydesc_ptr(*tydesc_data);
        let (size, align) = ((*tydesc).size, (*tydesc).align);

        let after_tydesc = idx + mem::size_of::<*TyDesc>();

        let start = round_up(after_tydesc, align);

        //debug!("freeing object: idx = {}, size = {}, align = {}, done = {}",
        //       start, size, align, is_done);
        if is_done {
            ((*tydesc).drop_glue)(buf.offset(start as int) as *i8);
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
fn bitpack_tydesc_ptr(p: *TyDesc, is_done: bool) -> uint {
    p as uint | (is_done as uint)
}
#[inline]
fn un_bitpack_tydesc_ptr(p: uint) -> (*TyDesc, bool) {
    ((p & !1) as *TyDesc, p & 1 == 1)
}

impl Arena {
    fn chunk_size(&self) -> uint {
        self.copy_head.capacity()
    }
    // Functions for the POD part of the arena
    fn alloc_copy_grow(&mut self, n_bytes: uint, align: uint) -> *u8 {
        // Allocate a new chunk.
        let new_min_chunk_size = cmp::max(n_bytes, self.chunk_size());
        self.chunks.borrow_mut().push(self.copy_head.clone());
        self.copy_head =
            chunk(num::next_power_of_two(new_min_chunk_size + 1u), true);

        return self.alloc_copy_inner(n_bytes, align);
    }

    #[inline]
    fn alloc_copy_inner(&mut self, n_bytes: uint, align: uint) -> *u8 {
        unsafe {
            let start = round_up(self.copy_head.fill.get(), align);
            let end = start + n_bytes;
            if end > self.chunk_size() {
                return self.alloc_copy_grow(n_bytes, align);
            }
            self.copy_head.fill.set(end);

            //debug!("idx = {}, size = {}, align = {}, fill = {}",
            //       start, n_bytes, align, head.fill.get());

            self.copy_head.as_ptr().offset(start as int)
        }
    }

    #[inline]
    fn alloc_copy<'a, T>(&'a mut self, op: || -> T) -> &'a T {
        unsafe {
            let ptr = self.alloc_copy_inner(mem::size_of::<T>(), min_align_of::<T>());
            let ptr = ptr as *mut T;
            mem::move_val_init(&mut (*ptr), op());
            return &*ptr;
        }
    }

    // Functions for the non-POD part of the arena
    fn alloc_noncopy_grow(&mut self, n_bytes: uint, align: uint)
                         -> (*u8, *u8) {
        // Allocate a new chunk.
        let new_min_chunk_size = cmp::max(n_bytes, self.chunk_size());
        self.chunks.borrow_mut().push(self.head.clone());
        self.head =
            chunk(num::next_power_of_two(new_min_chunk_size + 1u), false);

        return self.alloc_noncopy_inner(n_bytes, align);
    }

    #[inline]
    fn alloc_noncopy_inner(&mut self, n_bytes: uint, align: uint)
                          -> (*u8, *u8) {
        unsafe {
            let tydesc_start = self.head.fill.get();
            let after_tydesc = self.head.fill.get() + mem::size_of::<*TyDesc>();
            let start = round_up(after_tydesc, align);
            let end = start + n_bytes;

            if end > self.head.capacity() {
                return self.alloc_noncopy_grow(n_bytes, align);
            }

            self.head.fill.set(round_up(end, mem::pref_align_of::<*TyDesc>()));

            //debug!("idx = {}, size = {}, align = {}, fill = {}",
            //       start, n_bytes, align, head.fill);

            let buf = self.head.as_ptr();
            return (buf.offset(tydesc_start as int), buf.offset(start as int));
        }
    }

    #[inline]
    fn alloc_noncopy<'a, T>(&'a mut self, op: || -> T) -> &'a T {
        unsafe {
            let tydesc = get_tydesc::<T>();
            let (ty_ptr, ptr) =
                self.alloc_noncopy_inner(mem::size_of::<T>(), min_align_of::<T>());
            let ty_ptr = ty_ptr as *mut uint;
            let ptr = ptr as *mut T;
            // Write in our tydesc along with a bit indicating that it
            // has *not* been initialized yet.
            *ty_ptr = mem::transmute(tydesc);
            // Actually initialize it
            mem::move_val_init(&mut(*ptr), op());
            // Now that we are done, update the tydesc to indicate that
            // the object is there.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, true);

            return &*ptr;
        }
    }

    // The external interface
    #[inline]
    pub fn alloc<'a, T>(&'a self, op: || -> T) -> &'a T {
        unsafe {
            // FIXME #13933: Remove/justify all `&T` to `&mut T` transmutes
            let this: &mut Arena = mem::transmute::<&_, &mut _>(self);
            if intrinsics::needs_drop::<T>() {
                this.alloc_noncopy(op)
            } else {
                this.alloc_copy(op)
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
        arena.alloc(|| Rc::new(i));
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
        arena.alloc(|| { Rc::new(i) });
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        arena.alloc(|| { [0u8, 1u8, 2u8] });
    }
    // Now, fail while allocating
    arena.alloc::<Rc<int>>(|| {
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
    ptr: *T,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: *T,

    /// A pointer to the first arena segment.
    first: Option<Box<TypedArenaChunk<T>>>,
}

struct TypedArenaChunk<T> {
    /// Pointer to the next arena segment.
    next: Option<Box<TypedArenaChunk<T>>>,

    /// The number of elements that this chunk can hold.
    capacity: uint,

    // Objects follow here, suitably aligned.
}

impl<T> TypedArenaChunk<T> {
    #[cfg(stage0)]
    #[inline]
    fn new(next: Option<Box<TypedArenaChunk<T>>>, capacity: uint)
           -> Box<TypedArenaChunk<T>> {
        let mut size = mem::size_of::<TypedArenaChunk<T>>();
        size = round_up(size, min_align_of::<T>());
        let elem_size = mem::size_of::<T>();
        let elems_size = elem_size.checked_mul(&capacity).unwrap();
        size = size.checked_add(&elems_size).unwrap();

        let mut chunk = unsafe {
            let chunk = exchange_malloc(size);
            let mut chunk: Box<TypedArenaChunk<T>> = mem::transmute(chunk);
            mem::move_val_init(&mut chunk.next, next);
            chunk
        };

        chunk.capacity = capacity;
        chunk
    }

    #[inline]
    #[cfg(not(stage0))]
    fn new(next: Option<Box<TypedArenaChunk<T>>>, capacity: uint)
           -> Box<TypedArenaChunk<T>> {
        let mut size = mem::size_of::<TypedArenaChunk<T>>();
        size = round_up(size, mem::min_align_of::<T>());
        let elem_size = mem::size_of::<T>();
        let elems_size = elem_size.checked_mul(&capacity).unwrap();
        size = size.checked_add(&elems_size).unwrap();

        let mut chunk = unsafe {
            let chunk = exchange_malloc(size, min_align_of::<TypedArenaChunk<T>>());
            let mut chunk: Box<TypedArenaChunk<T>> = mem::transmute(chunk);
            mem::move_val_init(&mut chunk.next, next);
            chunk
        };

        chunk.capacity = capacity;
        chunk
    }

    /// Destroys this arena chunk. If the type descriptor is supplied, the
    /// drop glue is called; otherwise, drop glue is not called.
    #[inline]
    unsafe fn destroy(&mut self, len: uint) {
        // Destroy all the allocated objects.
        if intrinsics::needs_drop::<T>() {
            let mut start = self.start();
            for _ in range(0, len) {
                read(start as *T); // run the destructor on the pointer
                start = start.offset(mem::size_of::<T>() as int)
            }
        }

        // Destroy the next chunk.
        let next_opt = mem::replace(&mut self.next, None);
        match next_opt {
            None => {}
            Some(mut next) => {
                // We assume that the next chunk is completely filled.
                next.destroy(next.capacity)
            }
        }
    }

    // Returns a pointer to the first allocated object.
    #[inline]
    fn start(&self) -> *u8 {
        let this: *TypedArenaChunk<T> = self;
        unsafe {
            mem::transmute(round_up(this.offset(1) as uint, min_align_of::<T>()))
        }
    }

    // Returns a pointer to the end of the allocated space.
    #[inline]
    fn end(&self) -> *u8 {
        unsafe {
            let size = mem::size_of::<T>().checked_mul(&self.capacity).unwrap();
            self.start().offset(size as int)
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
        let chunk = TypedArenaChunk::<T>::new(None, capacity);
        TypedArena {
            ptr: chunk.start() as *T,
            end: chunk.end() as *T,
            first: Some(chunk),
        }
    }

    /// Allocates an object into this arena.
    #[inline]
    pub fn alloc<'a>(&'a self, object: T) -> &'a T {
        unsafe {
            // FIXME #13933: Remove/justify all `&T` to `&mut T` transmutes
            let this: &mut TypedArena<T> = mem::transmute::<&_, &mut _>(self);
            if this.ptr == this.end {
                this.grow()
            }

            let ptr: &'a mut T = mem::transmute(this.ptr);
            mem::move_val_init(ptr, object);
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
        let chunk = TypedArenaChunk::<T>::new(Some(chunk), new_capacity);
        self.ptr = chunk.start() as *T;
        self.end = chunk.end() as *T;
        self.first = Some(chunk)
    }
}

#[unsafe_destructor]
impl<T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        // Determine how much was filled.
        let start = self.first.get_ref().start() as uint;
        let end = self.ptr as uint;
        let diff = (end - start) / mem::size_of::<T>();

        // Pass that to the `destroy` method.
        unsafe {
            self.first.get_mut_ref().destroy(diff)
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use super::{Arena, TypedArena};

    struct Point {
        x: int,
        y: int,
        z: int,
    }

    #[test]
    pub fn test_copy() {
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
    pub fn bench_copy(b: &mut Bencher) {
        let arena = TypedArena::new();
        b.iter(|| {
            arena.alloc(Point {
                x: 1,
                y: 2,
                z: 3,
            })
        })
    }

    #[bench]
    pub fn bench_copy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            box Point {
                x: 1,
                y: 2,
                z: 3,
            }
        })
    }

    #[bench]
    pub fn bench_copy_old_arena(b: &mut Bencher) {
        let arena = Arena::new();
        b.iter(|| {
            arena.alloc(|| {
                Point {
                    x: 1,
                    y: 2,
                    z: 3,
                }
            })
        })
    }

    struct Noncopy {
        string: ~str,
        array: Vec<int> ,
    }

    #[test]
    pub fn test_noncopy() {
        let arena = TypedArena::new();
        for _ in range(0, 100000) {
            arena.alloc(Noncopy {
                string: "hello world".to_owned(),
                array: vec!( 1, 2, 3, 4, 5 ),
            });
        }
    }

    #[bench]
    pub fn bench_noncopy(b: &mut Bencher) {
        let arena = TypedArena::new();
        b.iter(|| {
            arena.alloc(Noncopy {
                string: "hello world".to_owned(),
                array: vec!( 1, 2, 3, 4, 5 ),
            })
        })
    }

    #[bench]
    pub fn bench_noncopy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            box Noncopy {
                string: "hello world".to_owned(),
                array: vec!( 1, 2, 3, 4, 5 ),
            }
        })
    }

    #[bench]
    pub fn bench_noncopy_old_arena(b: &mut Bencher) {
        let arena = Arena::new();
        b.iter(|| {
            arena.alloc(|| Noncopy {
                string: "hello world".to_owned(),
                array: vec!( 1, 2, 3, 4, 5 ),
            })
        })
    }
}
