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
//!
//! This crate has two arenas implemented: TypedArena, which is a simpler
//! arena but can only hold objects of a single type, and Arena, which is a
//! more complex, slower Arena which can hold objects of any type.

#![crate_name = "arena"]
#![experimental]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/")]

#![feature(unsafe_destructor)]
#![allow(missing_doc)]

use std::cell::{Cell, RefCell};
use std::cmp;
use std::intrinsics::{TyDesc, get_tydesc};
use std::intrinsics;
use std::mem;
use std::num;
use std::ptr;
use std::rc::Rc;
use std::rt::heap::allocate;

// The way arena uses arrays is really deeply awful. The arrays are
// allocated, and have capacities reserved, but the fill for the array
// will always stay at 0.
#[deriving(Clone, PartialEq)]
struct Chunk {
    data: Rc<RefCell<Vec<u8> >>,
    fill: Cell<uint>,
    is_copy: Cell<bool>,
}
impl Chunk {
    fn capacity(&self) -> uint {
        self.data.borrow().capacity()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.data.borrow().as_ptr()
    }
}

/// A slower reflection-based arena that can allocate objects of any type.
///
/// This arena uses Vec<u8> as a backing store to allocate objects from.  For
/// each allocated object, the arena stores a pointer to the type descriptor
/// followed by the object. (Potentially with alignment padding after each
/// element.) When the arena is destroyed, it iterates through all of its
/// chunks, and uses the tydesc information to trace through the objects,
/// calling the destructors on them.  One subtle point that needs to be
/// addressed is how to handle failures while running the user provided
/// initializer function. It is important to not run the destructor on
/// uninitialized objects, but how to detect them is somewhat subtle. Since
/// alloc() can be invoked recursively, it is not sufficient to simply exclude
/// the most recent object. To solve this without requiring extra space, we
/// use the low order bit of the tydesc pointer to encode whether the object
/// it describes has been fully initialized.
///
/// As an optimization, objects with destructors are stored in
/// different chunks than objects without destructors. This reduces
/// overhead when initializing plain-old-data and means we don't need
/// to waste time running the destructors of POD.
pub struct Arena {
    // The head is separated out from the list as a unbenchmarked
    // microoptimization, to avoid needing to case on the list to access the
    // head.
    head: RefCell<Chunk>,
    copy_head: RefCell<Chunk>,
    chunks: RefCell<Vec<Chunk>>,
}

impl Arena {
    /// Allocate a new Arena with 32 bytes preallocated.
    pub fn new() -> Arena {
        Arena::new_with_size(32u)
    }

    /// Allocate a new Arena with `initial_size` bytes preallocated.
    pub fn new_with_size(initial_size: uint) -> Arena {
        Arena {
            head: RefCell::new(chunk(initial_size, false)),
            copy_head: RefCell::new(chunk(initial_size, true)),
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
            destroy_chunk(&*self.head.borrow());
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
        let tydesc_data: *const uint = mem::transmute(buf.offset(idx as int));
        let (tydesc, is_done) = un_bitpack_tydesc_ptr(*tydesc_data);
        let (size, align) = ((*tydesc).size, (*tydesc).align);

        let after_tydesc = idx + mem::size_of::<*const TyDesc>();

        let start = round_up(after_tydesc, align);

        //debug!("freeing object: idx = {}, size = {}, align = {}, done = {}",
        //       start, size, align, is_done);
        if is_done {
            ((*tydesc).drop_glue)(buf.offset(start as int) as *const i8);
        }

        // Find where the next tydesc lives
        idx = round_up(start + size, mem::align_of::<*const TyDesc>());
    }
}

// We encode whether the object a tydesc describes has been
// initialized in the arena in the low bit of the tydesc pointer. This
// is necessary in order to properly do cleanup if a failure occurs
// during an initializer.
#[inline]
fn bitpack_tydesc_ptr(p: *const TyDesc, is_done: bool) -> uint {
    p as uint | (is_done as uint)
}
#[inline]
fn un_bitpack_tydesc_ptr(p: uint) -> (*const TyDesc, bool) {
    ((p & !1) as *const TyDesc, p & 1 == 1)
}

impl Arena {
    fn chunk_size(&self) -> uint {
        self.copy_head.borrow().capacity()
    }

    // Functions for the POD part of the arena
    fn alloc_copy_grow(&self, n_bytes: uint, align: uint) -> *const u8 {
        // Allocate a new chunk.
        let new_min_chunk_size = cmp::max(n_bytes, self.chunk_size());
        self.chunks.borrow_mut().push(self.copy_head.borrow().clone());

        *self.copy_head.borrow_mut() =
            chunk(num::next_power_of_two(new_min_chunk_size + 1u), true);

        return self.alloc_copy_inner(n_bytes, align);
    }

    #[inline]
    fn alloc_copy_inner(&self, n_bytes: uint, align: uint) -> *const u8 {
        let start = round_up(self.copy_head.borrow().fill.get(), align);

        let end = start + n_bytes;
        if end > self.chunk_size() {
            return self.alloc_copy_grow(n_bytes, align);
        }

        let copy_head = self.copy_head.borrow();
        copy_head.fill.set(end);

        unsafe {
            copy_head.as_ptr().offset(start as int)
        }
    }

    #[inline]
    fn alloc_copy<'a, T>(&'a self, op: || -> T) -> &'a T {
        unsafe {
            let ptr = self.alloc_copy_inner(mem::size_of::<T>(),
                                            mem::min_align_of::<T>());
            let ptr = ptr as *mut T;
            ptr::write(&mut (*ptr), op());
            return &*ptr;
        }
    }

    // Functions for the non-POD part of the arena
    fn alloc_noncopy_grow(&self, n_bytes: uint,
                          align: uint) -> (*const u8, *const u8) {
        // Allocate a new chunk.
        let new_min_chunk_size = cmp::max(n_bytes, self.chunk_size());
        self.chunks.borrow_mut().push(self.head.borrow().clone());

        *self.head.borrow_mut() =
            chunk(num::next_power_of_two(new_min_chunk_size + 1u), false);

        return self.alloc_noncopy_inner(n_bytes, align);
    }

    #[inline]
    fn alloc_noncopy_inner(&self, n_bytes: uint,
                           align: uint) -> (*const u8, *const u8) {
        // Be careful to not maintain any `head` borrows active, because
        // `alloc_noncopy_grow` borrows it mutably.
        let (start, end, tydesc_start, head_capacity) = {
            let head = self.head.borrow();
            let fill = head.fill.get();

            let tydesc_start = fill;
            let after_tydesc = fill + mem::size_of::<*const TyDesc>();
            let start = round_up(after_tydesc, align);
            let end = start + n_bytes;

            (start, end, tydesc_start, head.capacity())
        };

        if end > head_capacity {
            return self.alloc_noncopy_grow(n_bytes, align);
        }

        let head = self.head.borrow();
        head.fill.set(round_up(end, mem::align_of::<*const TyDesc>()));

        unsafe {
            let buf = head.as_ptr();
            return (buf.offset(tydesc_start as int), buf.offset(start as int));
        }
    }

    #[inline]
    fn alloc_noncopy<'a, T>(&'a self, op: || -> T) -> &'a T {
        unsafe {
            let tydesc = get_tydesc::<T>();
            let (ty_ptr, ptr) =
                self.alloc_noncopy_inner(mem::size_of::<T>(),
                                         mem::min_align_of::<T>());
            let ty_ptr = ty_ptr as *mut uint;
            let ptr = ptr as *mut T;
            // Write in our tydesc along with a bit indicating that it
            // has *not* been initialized yet.
            *ty_ptr = mem::transmute(tydesc);
            // Actually initialize it
            ptr::write(&mut(*ptr), op());
            // Now that we are done, update the tydesc to indicate that
            // the object is there.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, true);

            return &*ptr;
        }
    }

    /// Allocate a new item in the arena, using `op` to initialize the value
    /// and returning a reference to it.
    #[inline]
    pub fn alloc<'a, T>(&'a self, op: || -> T) -> &'a T {
        unsafe {
            if intrinsics::needs_drop::<T>() {
                self.alloc_noncopy(op)
            } else {
                self.alloc_copy(op)
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
fn test_arena_alloc_nested() {
    struct Inner { value: uint }
    struct Outer<'a> { inner: &'a Inner }

    let arena = Arena::new();

    let result = arena.alloc(|| Outer {
        inner: arena.alloc(|| Inner { value: 10 })
    });

    assert_eq!(result.inner.value, 10);
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

/// A faster arena that can hold objects of only one type.
///
/// Safety note: Modifying objects in the arena that have already had their
/// `drop` destructors run can cause leaks, because the destructor will not
/// run again for these objects.
pub struct TypedArena<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*const T>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*const T>,

    /// A pointer to the first arena segment.
    first: RefCell<TypedArenaChunkRef<T>>,
}
type TypedArenaChunkRef<T> = Option<Box<TypedArenaChunk<T>>>;

struct TypedArenaChunk<T> {
    /// Pointer to the next arena segment.
    next: TypedArenaChunkRef<T>,

    /// The number of elements that this chunk can hold.
    capacity: uint,

    // Objects follow here, suitably aligned.
}

impl<T> TypedArenaChunk<T> {
    #[inline]
    fn new(next: Option<Box<TypedArenaChunk<T>>>, capacity: uint)
           -> Box<TypedArenaChunk<T>> {
        let mut size = mem::size_of::<TypedArenaChunk<T>>();
        size = round_up(size, mem::min_align_of::<T>());
        let elem_size = mem::size_of::<T>();
        let elems_size = elem_size.checked_mul(&capacity).unwrap();
        size = size.checked_add(&elems_size).unwrap();

        let mut chunk = unsafe {
            let chunk = allocate(size, mem::min_align_of::<TypedArenaChunk<T>>());
            let mut chunk: Box<TypedArenaChunk<T>> = mem::transmute(chunk);
            ptr::write(&mut chunk.next, next);
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
                ptr::read(start as *const T); // run the destructor on the pointer
                start = start.offset(mem::size_of::<T>() as int)
            }
        }

        // Destroy the next chunk.
        let next_opt = mem::replace(&mut self.next, None);
        match next_opt {
            None => {}
            Some(mut next) => {
                // We assume that the next chunk is completely filled.
                let capacity = next.capacity;
                next.destroy(capacity)
            }
        }
    }

    // Returns a pointer to the first allocated object.
    #[inline]
    fn start(&self) -> *const u8 {
        let this: *const TypedArenaChunk<T> = self;
        unsafe {
            mem::transmute(round_up(this.offset(1) as uint,
                                    mem::min_align_of::<T>()))
        }
    }

    // Returns a pointer to the end of the allocated space.
    #[inline]
    fn end(&self) -> *const u8 {
        unsafe {
            let size = mem::size_of::<T>().checked_mul(&self.capacity).unwrap();
            self.start().offset(size as int)
        }
    }
}

impl<T> TypedArena<T> {
    /// Creates a new TypedArena with preallocated space for 8 objects.
    #[inline]
    pub fn new() -> TypedArena<T> {
        TypedArena::with_capacity(8)
    }

    /// Creates a new TypedArena with preallocated space for the given number of
    /// objects.
    #[inline]
    pub fn with_capacity(capacity: uint) -> TypedArena<T> {
        let chunk = TypedArenaChunk::<T>::new(None, capacity);
        TypedArena {
            ptr: Cell::new(chunk.start() as *const T),
            end: Cell::new(chunk.end() as *const T),
            first: RefCell::new(Some(chunk)),
        }
    }

    /// Allocates an object in the TypedArena, returning a reference to it.
    #[inline]
    pub fn alloc<'a>(&'a self, object: T) -> &'a T {
        if self.ptr == self.end {
            self.grow()
        }

        let ptr: &'a T = unsafe {
            let ptr: &'a mut T = mem::transmute(self.ptr);
            ptr::write(ptr, object);
            self.ptr.set(self.ptr.get().offset(1));
            ptr
        };

        ptr
    }

    /// Grows the arena.
    #[inline(never)]
    fn grow(&self) {
        let chunk = self.first.borrow_mut().take_unwrap();
        let new_capacity = chunk.capacity.checked_mul(&2).unwrap();
        let chunk = TypedArenaChunk::<T>::new(Some(chunk), new_capacity);
        self.ptr.set(chunk.start() as *const T);
        self.end.set(chunk.end() as *const T);
        *self.first.borrow_mut() = Some(chunk)
    }
}

#[unsafe_destructor]
impl<T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        // Determine how much was filled.
        let start = self.first.borrow().get_ref().start() as uint;
        let end = self.ptr.get() as uint;
        let diff = (end - start) / mem::size_of::<T>();

        // Pass that to the `destroy` method.
        unsafe {
            self.first.borrow_mut().get_mut_ref().destroy(diff)
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
        for _ in range(0u, 100000) {
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
        string: String,
        array: Vec<int> ,
    }

    #[test]
    pub fn test_noncopy() {
        let arena = TypedArena::new();
        for _ in range(0u, 100000) {
            arena.alloc(Noncopy {
                string: "hello world".to_string(),
                array: vec!( 1, 2, 3, 4, 5 ),
            });
        }
    }

    #[bench]
    pub fn bench_noncopy(b: &mut Bencher) {
        let arena = TypedArena::new();
        b.iter(|| {
            arena.alloc(Noncopy {
                string: "hello world".to_string(),
                array: vec!( 1, 2, 3, 4, 5 ),
            })
        })
    }

    #[bench]
    pub fn bench_noncopy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            box Noncopy {
                string: "hello world".to_string(),
                array: vec!( 1, 2, 3, 4, 5 ),
            }
        })
    }

    #[bench]
    pub fn bench_noncopy_old_arena(b: &mut Bencher) {
        let arena = Arena::new();
        b.iter(|| {
            arena.alloc(|| Noncopy {
                string: "hello world".to_string(),
                array: vec!( 1, 2, 3, 4, 5 ),
            })
        })
    }
}
