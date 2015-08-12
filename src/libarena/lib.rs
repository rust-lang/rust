// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The arena, a fast but limited type of allocator.
//!
//! Arenas are a type of allocator that destroy the objects within, all at
//! once, once the arena itself is destroyed. They do not support deallocation
//! of individual objects while the arena itself is still alive. The benefit
//! of an arena is very fast allocation; just a pointer bump.
//!
//! This crate has two arenas implemented: `TypedArena`, which is a simpler
//! arena but can only hold objects of a single type, and `Arena`, which is a
//! more complex, slower arena which can hold objects of any type.

#![crate_name = "arena"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(no_crate_inject, attr(deny(warnings))))]

#![feature(alloc)]
#![feature(box_syntax)]
#![feature(core_intrinsics)]
#![feature(drop_in_place)]
#![feature(raw)]
#![feature(heap_api)]
#![feature(oom)]
#![feature(raw)]
#![feature(staged_api)]
#![feature(dropck_parametricity)]
#![cfg_attr(test, feature(test))]

extern crate alloc;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::intrinsics;
use std::marker;
use std::mem;
use std::ptr;
use std::raw;
use std::raw::Repr;
use std::rc::Rc;
use std::slice;

use alloc::heap;
use alloc::raw_vec::RawVec;

// The way arena uses arrays is really deeply awful. The arrays are
// allocated, and have capacities reserved, but the fill for the array
// will always stay at 0.
#[derive(Clone, PartialEq)]
struct Chunk {
    data: Rc<RefCell<Vec<u8>>>,
    fill: Cell<usize>,
    is_copy: Cell<bool>,
}

impl Chunk {
    fn capacity(&self) -> usize {
        self.data.borrow().capacity()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.data.borrow().as_ptr()
    }
}

/// A slower reflection-based arena that can allocate objects of any type.
///
/// This arena uses `Vec<u8>` as a backing store to allocate objects from. For
/// each allocated object, the arena stores a pointer to the type descriptor
/// followed by the object (potentially with alignment padding after each
/// element). When the arena is destroyed, it iterates through all of its
/// chunks, and uses the tydesc information to trace through the objects,
/// calling the destructors on them. One subtle point that needs to be
/// addressed is how to handle panics while running the user provided
/// initializer function. It is important to not run the destructor on
/// uninitialized objects, but how to detect them is somewhat subtle. Since
/// `alloc()` can be invoked recursively, it is not sufficient to simply exclude
/// the most recent object. To solve this without requiring extra space, we
/// use the low order bit of the tydesc pointer to encode whether the object
/// it describes has been fully initialized.
///
/// As an optimization, objects with destructors are stored in different chunks
/// than objects without destructors. This reduces overhead when initializing
/// plain-old-data (`Copy` types) and means we don't need to waste time running
/// their destructors.
pub struct Arena<'longer_than_self> {
    // The head is separated out from the list as a unbenchmarked
    // microoptimization, to avoid needing to case on the list to access the
    // head.
    head: RefCell<Chunk>,
    copy_head: RefCell<Chunk>,
    chunks: RefCell<Vec<Chunk>>,
    _marker: marker::PhantomData<*mut &'longer_than_self ()>,
}

impl<'a> Arena<'a> {
    /// Allocates a new Arena with 32 bytes preallocated.
    pub fn new() -> Arena<'a> {
        Arena::new_with_size(32)
    }

    /// Allocates a new Arena with `initial_size` bytes preallocated.
    pub fn new_with_size(initial_size: usize) -> Arena<'a> {
        Arena {
            head: RefCell::new(chunk(initial_size, false)),
            copy_head: RefCell::new(chunk(initial_size, true)),
            chunks: RefCell::new(Vec::new()),
            _marker: marker::PhantomData,
        }
    }
}

fn chunk(size: usize, is_copy: bool) -> Chunk {
    Chunk {
        data: Rc::new(RefCell::new(Vec::with_capacity(size))),
        fill: Cell::new(0),
        is_copy: Cell::new(is_copy),
    }
}

impl<'longer_than_self> Drop for Arena<'longer_than_self> {
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
fn round_up(base: usize, align: usize) -> usize {
    (base.checked_add(align - 1)).unwrap() & !(align - 1)
}

// Walk down a chunk, running the destructors for any objects stored
// in it.
unsafe fn destroy_chunk(chunk: &Chunk) {
    let mut idx = 0;
    let buf = chunk.as_ptr();
    let fill = chunk.fill.get();

    while idx < fill {
        let tydesc_data = buf.offset(idx as isize) as *const usize;
        let (tydesc, is_done) = un_bitpack_tydesc_ptr(*tydesc_data);
        let (size, align) = ((*tydesc).size, (*tydesc).align);

        let after_tydesc = idx + mem::size_of::<*const TyDesc>();

        let start = round_up(after_tydesc, align);

        // debug!("freeing object: idx = {}, size = {}, align = {}, done = {}",
        //        start, size, align, is_done);
        if is_done {
            ((*tydesc).drop_glue)(buf.offset(start as isize) as *const i8);
        }

        // Find where the next tydesc lives
        idx = round_up(start + size, mem::align_of::<*const TyDesc>());
    }
}

// We encode whether the object a tydesc describes has been
// initialized in the arena in the low bit of the tydesc pointer. This
// is necessary in order to properly do cleanup if a panic occurs
// during an initializer.
#[inline]
fn bitpack_tydesc_ptr(p: *const TyDesc, is_done: bool) -> usize {
    p as usize | (is_done as usize)
}
#[inline]
fn un_bitpack_tydesc_ptr(p: usize) -> (*const TyDesc, bool) {
    ((p & !1) as *const TyDesc, p & 1 == 1)
}

// HACK(eddyb) TyDesc replacement using a trait object vtable.
// This could be replaced in the future with a custom DST layout,
// or `&'static (drop_glue, size, align)` created by a `const fn`.
struct TyDesc {
    drop_glue: fn(*const i8),
    size: usize,
    align: usize,
}

trait AllTypes {
    fn dummy(&self) {}
}

impl<T: ?Sized> AllTypes for T {}

unsafe fn get_tydesc<T>() -> *const TyDesc {
    use std::raw::TraitObject;

    let ptr = &*(1 as *const T);

    // Can use any trait that is implemented for all types.
    let obj = mem::transmute::<&AllTypes, TraitObject>(ptr);
    obj.vtable as *const TyDesc
}

impl<'longer_than_self> Arena<'longer_than_self> {
    fn chunk_size(&self) -> usize {
        self.copy_head.borrow().capacity()
    }

    // Functions for the POD part of the arena
    fn alloc_copy_grow(&self, n_bytes: usize, align: usize) -> *const u8 {
        // Allocate a new chunk.
        let new_min_chunk_size = cmp::max(n_bytes, self.chunk_size());
        self.chunks.borrow_mut().push(self.copy_head.borrow().clone());

        *self.copy_head.borrow_mut() = chunk((new_min_chunk_size + 1).next_power_of_two(), true);

        self.alloc_copy_inner(n_bytes, align)
    }

    #[inline]
    fn alloc_copy_inner(&self, n_bytes: usize, align: usize) -> *const u8 {
        let start = round_up(self.copy_head.borrow().fill.get(), align);

        let end = start + n_bytes;
        if end > self.chunk_size() {
            return self.alloc_copy_grow(n_bytes, align);
        }

        let copy_head = self.copy_head.borrow();
        copy_head.fill.set(end);

        unsafe { copy_head.as_ptr().offset(start as isize) }
    }

    #[inline]
    fn alloc_copy<T, F>(&self, op: F) -> &mut T
        where F: FnOnce() -> T
    {
        unsafe {
            let ptr = self.alloc_copy_inner(mem::size_of::<T>(), mem::align_of::<T>());
            let ptr = ptr as *mut T;
            ptr::write(&mut (*ptr), op());
            &mut *ptr
        }
    }

    // Functions for the non-POD part of the arena
    fn alloc_noncopy_grow(&self, n_bytes: usize, align: usize) -> (*const u8, *const u8) {
        // Allocate a new chunk.
        let new_min_chunk_size = cmp::max(n_bytes, self.chunk_size());
        self.chunks.borrow_mut().push(self.head.borrow().clone());

        *self.head.borrow_mut() = chunk((new_min_chunk_size + 1).next_power_of_two(), false);

        self.alloc_noncopy_inner(n_bytes, align)
    }

    #[inline]
    fn alloc_noncopy_inner(&self, n_bytes: usize, align: usize) -> (*const u8, *const u8) {
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
            (buf.offset(tydesc_start as isize),
             buf.offset(start as isize))
        }
    }

    #[inline]
    fn alloc_noncopy<T, F>(&self, op: F) -> &mut T
        where F: FnOnce() -> T
    {
        unsafe {
            let tydesc = get_tydesc::<T>();
            let (ty_ptr, ptr) = self.alloc_noncopy_inner(mem::size_of::<T>(), mem::align_of::<T>());
            let ty_ptr = ty_ptr as *mut usize;
            let ptr = ptr as *mut T;
            // Write in our tydesc along with a bit indicating that it
            // has *not* been initialized yet.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, false);
            // Actually initialize it
            ptr::write(&mut (*ptr), op());
            // Now that we are done, update the tydesc to indicate that
            // the object is there.
            *ty_ptr = bitpack_tydesc_ptr(tydesc, true);

            &mut *ptr
        }
    }

    /// Allocates a new item in the arena, using `op` to initialize the value,
    /// and returns a reference to it.
    #[inline]
    pub fn alloc<T: 'longer_than_self, F>(&self, op: F) -> &mut T
        where F: FnOnce() -> T
    {
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
    for i in 0..10 {
        // Arena allocate something with drop glue to make sure it
        // doesn't leak.
        arena.alloc(|| Rc::new(i));
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        arena.alloc(|| [0u8, 1u8, 2u8]);
    }
}

#[test]
#[should_panic]
fn test_arena_destructors_fail() {
    let arena = Arena::new();
    // Put some stuff in the arena.
    for i in 0..10 {
        // Arena allocate something with drop glue to make sure it
        // doesn't leak.
        arena.alloc(|| Rc::new(i));
        // Allocate something with funny size and alignment, to keep
        // things interesting.
        arena.alloc(|| [0u8, 1, 2]);
    }
    // Now, panic while allocating
    arena.alloc::<Rc<i32>, _>(|| {
        panic!();
    });
}

/// A faster arena that can hold objects of only one type.
pub struct TypedArena<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut T>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut T>,

    /// A vector arena segments.
    chunks: RefCell<Vec<TypedArenaChunk<T>>>,

    /// Marker indicating that dropping the arena causes its owned
    /// instances of `T` to be dropped.
    _own: marker::PhantomData<T>,
}

struct TypedArenaChunk<T> {
    /// Pointer to the next arena segment.
    storage: RawVec<T>,
}

impl<T> TypedArenaChunk<T> {
    #[inline]
    unsafe fn new(capacity: usize) -> TypedArenaChunk<T> {
        TypedArenaChunk { storage: RawVec::with_capacity(capacity) }
    }

    /// Destroys this arena chunk.
    #[inline]
    unsafe fn destroy(&mut self, len: usize) {
        // The branch on needs_drop() is an -O1 performance optimization.
        // Without the branch, dropping TypedArena<u8> takes linear time.
        if intrinsics::needs_drop::<T>() {
            let mut start = self.start();
            // Destroy all allocated objects.
            for _ in 0..len {
                ptr::drop_in_place(start);
                start = start.offset(1);
            }
        }
    }

    // Returns a pointer to the first allocated object.
    #[inline]
    fn start(&self) -> *mut T {
        self.storage.ptr()
    }

    // Returns a pointer to the end of the allocated space.
    #[inline]
    fn end(&self) -> *mut T {
        unsafe {
            if mem::size_of::<T>() == 0 {
                // A pointer as large as possible for zero-sized elements.
                !0 as *mut T
            } else {
                self.start().offset(self.storage.cap() as isize)
            }
        }
    }
}

const PAGE: usize = 4096;

impl<T> TypedArena<T> {
    /// Creates a new `TypedArena` with preallocated space for many objects.
    #[inline]
    pub fn new() -> TypedArena<T> {
        // Reserve at least one page.
        let elem_size = cmp::max(1, mem::size_of::<T>());
        TypedArena::with_capacity(PAGE / elem_size)
    }

    /// Creates a new `TypedArena` with preallocated space for the given number of
    /// objects.
    #[inline]
    pub fn with_capacity(capacity: usize) -> TypedArena<T> {
        unsafe {
            let chunk = TypedArenaChunk::<T>::new(cmp::max(1, capacity));
            TypedArena {
                ptr: Cell::new(chunk.start()),
                end: Cell::new(chunk.end()),
                chunks: RefCell::new(vec![chunk]),
                _own: marker::PhantomData,
            }
        }
    }

    /// Allocates an object in the `TypedArena`, returning a reference to it.
    #[inline]
    pub fn alloc(&self, object: T) -> &mut T {
        if self.ptr == self.end {
            self.grow()
        }

        unsafe {
            if mem::size_of::<T>() == 0 {
                self.ptr.set(intrinsics::arith_offset(self.ptr.get() as *mut u8, 1) as *mut T);
                let ptr = heap::EMPTY as *mut T;
                // Don't drop the object. This `write` is equivalent to `forget`.
                ptr::write(ptr, object);
                &mut *ptr
            } else {
                let ptr = self.ptr.get();
                // Advance the pointer.
                self.ptr.set(self.ptr.get().offset(1));
                // Write into uninitialized memory.
                ptr::write(ptr, object);
                &mut *ptr
            }
        }
    }

    /// Grows the arena.
    #[inline(never)]
    #[cold]
    fn grow(&self) {
        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let prev_capacity = chunks.last().unwrap().storage.cap();
            let new_capacity = prev_capacity.checked_mul(2).unwrap();
            if chunks.last_mut().unwrap().storage.double_in_place() {
                self.end.set(chunks.last().unwrap().end());
            } else {
                let chunk = TypedArenaChunk::<T>::new(new_capacity);
                self.ptr.set(chunk.start());
                self.end.set(chunk.end());
                chunks.push(chunk);
            }
        }
    }
}

impl<T> Drop for TypedArena<T> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        unsafe {
            // Determine how much was filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            let mut last_chunk = chunks_borrow.pop().unwrap();
            let start = last_chunk.start() as usize;
            let end = self.ptr.get() as usize;
            let diff = if mem::size_of::<T>() == 0 {
                // Avoid division by zero.
                end - start
            } else {
                (end - start) / mem::size_of::<T>()
            };

            // Pass that to the `destroy` method.
            last_chunk.destroy(diff);
            // Destroy this chunk.
            let _: RawVec<T> = mem::transmute(last_chunk);

            for chunk in chunks_borrow.iter_mut() {
                let cap = chunk.storage.cap();
                chunk.destroy(cap);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use super::{Arena, TypedArena};

    #[allow(dead_code)]
    struct Point {
        x: i32,
        y: i32,
        z: i32,
    }

    #[test]
    fn test_arena_alloc_nested() {
        struct Inner {
            value: u8,
        }
        struct Outer<'a> {
            inner: &'a Inner,
        }
        enum EI<'e> {
            I(Inner),
            O(Outer<'e>),
        }

        struct Wrap<'a>(TypedArena<EI<'a>>);

        impl<'a> Wrap<'a> {
            fn alloc_inner<F: Fn() -> Inner>(&self, f: F) -> &Inner {
                let r: &EI = self.0.alloc(EI::I(f()));
                if let &EI::I(ref i) = r {
                    i
                } else {
                    panic!("mismatch");
                }
            }
            fn alloc_outer<F: Fn() -> Outer<'a>>(&self, f: F) -> &Outer {
                let r: &EI = self.0.alloc(EI::O(f()));
                if let &EI::O(ref o) = r {
                    o
                } else {
                    panic!("mismatch");
                }
            }
        }

        let arena = Wrap(TypedArena::new());

        let result = arena.alloc_outer(|| {
            Outer { inner: arena.alloc_inner(|| Inner { value: 10 }) }
        });

        assert_eq!(result.inner.value, 10);
    }

    #[test]
    pub fn test_copy() {
        let arena = TypedArena::new();
        for _ in 0..100000 {
            arena.alloc(Point { x: 1, y: 2, z: 3 });
        }
    }

    #[bench]
    pub fn bench_copy(b: &mut Bencher) {
        let arena = TypedArena::new();
        b.iter(|| arena.alloc(Point { x: 1, y: 2, z: 3 }))
    }

    #[bench]
    pub fn bench_copy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = box Point { x: 1, y: 2, z: 3 };
        })
    }

    #[bench]
    pub fn bench_copy_old_arena(b: &mut Bencher) {
        let arena = Arena::new();
        b.iter(|| arena.alloc(|| Point { x: 1, y: 2, z: 3 }))
    }

    #[allow(dead_code)]
    struct Noncopy {
        string: String,
        array: Vec<i32>,
    }

    #[test]
    pub fn test_noncopy() {
        let arena = TypedArena::new();
        for _ in 0..100000 {
            arena.alloc(Noncopy {
                string: "hello world".to_string(),
                array: vec![1, 2, 3, 4, 5],
            });
        }
    }

    #[bench]
    pub fn bench_noncopy(b: &mut Bencher) {
        let arena = TypedArena::new();
        b.iter(|| {
            arena.alloc(Noncopy {
                string: "hello world".to_string(),
                array: vec![1, 2, 3, 4, 5],
            })
        })
    }

    #[bench]
    pub fn bench_noncopy_nonarena(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = box Noncopy {
                string: "hello world".to_string(),
                array: vec![1, 2, 3, 4, 5],
            };
        })
    }

    #[bench]
    pub fn bench_noncopy_old_arena(b: &mut Bencher) {
        let arena = Arena::new();
        b.iter(|| {
            arena.alloc(|| {
                Noncopy {
                    string: "hello world".to_string(),
                    array: vec![1, 2, 3, 4, 5],
                }
            })
        })
    }

    #[test]
    pub fn test_zero_sized() {
        let arena = TypedArena::new();
        for _ in 0..100000 {
            arena.alloc(());
        }
    }
}
