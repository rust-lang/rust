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
#![feature(core_intrinsics)]
#![feature(drop_in_place)]
#![feature(heap_api)]
#![feature(raw)]
#![feature(heap_api)]
#![feature(staged_api)]
#![feature(dropck_parametricity)]
#![cfg_attr(test, feature(test))]

#![allow(deprecated)]

extern crate alloc;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::intrinsics;
use std::marker::{PhantomData, Send};
use std::mem;
use std::ptr;
use std::slice;

use alloc::heap;
use alloc::raw_vec::RawVec;

struct Chunk {
    data: RawVec<u8>,
    /// Index of the first unused byte.
    fill: Cell<usize>,
    /// Indicates whether objects with destructors are stored in this chunk.
    is_copy: Cell<bool>,
}

impl Chunk {
    fn new(size: usize, is_copy: bool) -> Chunk {
        Chunk {
            data: RawVec::with_capacity(size),
            fill: Cell::new(0),
            is_copy: Cell::new(is_copy),
        }
    }

    fn capacity(&self) -> usize {
        self.data.cap()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.data.ptr()
    }

    // Walk down a chunk, running the destructors for any objects stored
    // in it.
    unsafe fn destroy(&self) {
        let mut idx = 0;
        let buf = self.as_ptr();
        let fill = self.fill.get();

        while idx < fill {
            let tydesc_data = buf.offset(idx as isize) as *const usize;
            let (tydesc, is_done) = un_bitpack_tydesc_ptr(*tydesc_data);
            let (size, align) = ((*tydesc).size, (*tydesc).align);

            let after_tydesc = idx + mem::size_of::<*const TyDesc>();

            let start = round_up(after_tydesc, align);

            if is_done {
                ((*tydesc).drop_glue)(buf.offset(start as isize) as *const i8);
            }

            // Find where the next tydesc lives
            idx = round_up(start + size, mem::align_of::<*const TyDesc>());
        }
    }
}

/// A slower reflection-based arena that can allocate objects of any type.
///
/// This arena uses `RawVec<u8>` as a backing store to allocate objects from.
/// For each allocated object, the arena stores a pointer to the type descriptor
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
#[unstable(feature = "rustc_private",
           reason = "Private to rustc", issue = "0")]
#[rustc_deprecated(since = "1.6.0-dev", reason =
"The reflection-based arena is superseded by the any-arena crate")]
pub struct Arena<'longer_than_self> {
    // The heads are separated out from the list as a unbenchmarked
    // microoptimization, to avoid needing to case on the list to access a head.
    head: RefCell<Chunk>,
    copy_head: RefCell<Chunk>,
    chunks: RefCell<Vec<Chunk>>,
    _marker: PhantomData<*mut &'longer_than_self ()>,
}

impl<'a> Arena<'a> {
    /// Allocates a new Arena with 32 bytes preallocated.
    pub fn new() -> Arena<'a> {
        Arena::new_with_size(32)
    }

    /// Allocates a new Arena with `initial_size` bytes preallocated.
    pub fn new_with_size(initial_size: usize) -> Arena<'a> {
        Arena {
            head: RefCell::new(Chunk::new(initial_size, false)),
            copy_head: RefCell::new(Chunk::new(initial_size, true)),
            chunks: RefCell::new(Vec::new()),
            _marker: PhantomData,
        }
    }
}

impl<'longer_than_self> Drop for Arena<'longer_than_self> {
    fn drop(&mut self) {
        unsafe {
            self.head.borrow().destroy();
            for chunk in self.chunks.borrow().iter() {
                if !chunk.is_copy.get() {
                    chunk.destroy();
                }
            }
        }
    }
}

#[inline]
fn round_up(base: usize, align: usize) -> usize {
    (base.checked_add(align - 1)).unwrap() & !(align - 1)
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
// Requirements:
// * rvalue promotion (issue #1056)
// * mem::{size_of, align_of} must be const fns
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

    let ptr = &*(heap::EMPTY as *const T);

    // Can use any trait that is implemented for all types.
    let obj = mem::transmute::<&AllTypes, TraitObject>(ptr);
    obj.vtable as *const TyDesc
}

impl<'longer_than_self> Arena<'longer_than_self> {
    // Grows a given chunk and returns `false`, or replaces it with a bigger
    // chunk and returns `true`.
    // This method is shared by both parts of the arena.
    #[cold]
    fn alloc_grow(&self, head: &mut Chunk, used_cap: usize, n_bytes: usize) -> bool {
        if head.data.reserve_in_place(used_cap, n_bytes) {
            // In-place reallocation succeeded.
            false
        } else {
            // Allocate a new chunk.
            let new_min_chunk_size = cmp::max(n_bytes, head.capacity());
            let new_chunk = Chunk::new((new_min_chunk_size + 1).next_power_of_two(), false);
            let old_chunk = mem::replace(head, new_chunk);
            if old_chunk.fill.get() != 0 {
                self.chunks.borrow_mut().push(old_chunk);
            }
            true
        }
    }

    // Functions for the copyable part of the arena.

    #[inline]
    fn alloc_copy_inner(&self, n_bytes: usize, align: usize) -> *const u8 {
        let mut copy_head = self.copy_head.borrow_mut();
        let fill = copy_head.fill.get();
        let mut start = round_up(fill, align);
        let mut end = start + n_bytes;

        if end > copy_head.capacity() {
            if self.alloc_grow(&mut *copy_head, fill, end - fill) {
                // Continuing with a newly allocated chunk
                start = 0;
                end = n_bytes;
                copy_head.is_copy.set(true);
            }
        }

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

    // Functions for the non-copyable part of the arena.

    #[inline]
    fn alloc_noncopy_inner(&self, n_bytes: usize, align: usize) -> (*const u8, *const u8) {
        let mut head = self.head.borrow_mut();
        let fill = head.fill.get();

        let mut tydesc_start = fill;
        let after_tydesc = fill + mem::size_of::<*const TyDesc>();
        let mut start = round_up(after_tydesc, align);
        let mut end = round_up(start + n_bytes, mem::align_of::<*const TyDesc>());

        if end > head.capacity() {
            if self.alloc_grow(&mut *head, tydesc_start, end - tydesc_start) {
                // Continuing with a newly allocated chunk
                tydesc_start = 0;
                start = round_up(mem::size_of::<*const TyDesc>(), align);
                end = round_up(start + n_bytes, mem::align_of::<*const TyDesc>());
            }
        }

        head.fill.set(end);

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

    /// Allocates a slice of bytes of requested length. The bytes are not guaranteed to be zero
    /// if the arena has previously been cleared.
    ///
    /// # Panics
    ///
    /// Panics if the requested length is too large and causes overflow.
    pub fn alloc_bytes(&self, len: usize) -> &mut [u8] {
        unsafe {
            // Check for overflow.
            self.copy_head.borrow().fill.get().checked_add(len).expect("length overflow");
            let ptr = self.alloc_copy_inner(len, 1);
            intrinsics::assume(!ptr.is_null());
            slice::from_raw_parts_mut(ptr as *mut _, len)
        }
    }

    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        unsafe {
            self.head.borrow().destroy();
            self.head.borrow().fill.set(0);
            self.copy_head.borrow().fill.set(0);
            for chunk in self.chunks.borrow().iter() {
                if !chunk.is_copy.get() {
                    chunk.destroy();
                }
            }
            self.chunks.borrow_mut().clear();
        }
    }
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
    _own: PhantomData<T>,
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
                _own: PhantomData,
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
    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        unsafe {
            // Clear the last chunk, which is partially filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            let last_idx = chunks_borrow.len() - 1;
            self.clear_last_chunk(&mut chunks_borrow[last_idx]);
            // If `T` is ZST, code below has no effect.
            for mut chunk in chunks_borrow.drain(..last_idx) {
                let cap = chunk.storage.cap();
                chunk.destroy(cap);
            }
        }
    }

    // Drops the contents of the last chunk. The last chunk is partially empty, unlike all other
    // chunks.
    fn clear_last_chunk(&self, last_chunk: &mut TypedArenaChunk<T>) {
        // Determine how much was filled.
        let start = last_chunk.start() as usize;
        // We obtain the value of the pointer to the first uninitialized element.
        let end = self.ptr.get() as usize;
        // We then calculate the number of elements to be dropped in the last chunk,
        // which is the filled area's length.
        let diff = if mem::size_of::<T>() == 0 {
            // `T` is ZST. It can't have a drop flag, so the value here doesn't matter. We get
            // the number of zero-sized values in the last and only chunk, just out of caution.
            // Recall that `end` was incremented for each allocated value.
            end - start
        } else {
            (end - start) / mem::size_of::<T>()
        };
        // Pass that to the `destroy` method.
        unsafe {
            last_chunk.destroy(diff);
        }
        // Reset the chunk.
        self.ptr.set(last_chunk.start());
    }
}

impl<T> Drop for TypedArena<T> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        unsafe {
            // Determine how much was filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            let mut last_chunk = chunks_borrow.pop().unwrap();
            // Drop the contents of the last chunk.
            self.clear_last_chunk(&mut last_chunk);
            // The last chunk will be dropped. Destroy all other chunks.
            for chunk in chunks_borrow.iter_mut() {
                let cap = chunk.storage.cap();
                chunk.destroy(cap);
            }
            // RawVec handles deallocation of `last_chunk` and `self.chunks`.
        }
    }
}

unsafe impl<T: Send> Send for TypedArena<T> {}

#[cfg(test)]
mod tests {
    extern crate test;
    use self::test::Bencher;
    use super::{Arena, TypedArena};
    use std::cell::Cell;
    use std::rc::Rc;

    #[allow(dead_code)]
    #[derive(Debug, Eq, PartialEq)]
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
            let _: Box<_> = Box::new(Point { x: 1, y: 2, z: 3 });
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

    #[test]
    pub fn test_typed_arena_zero_sized() {
        let arena = TypedArena::new();
        for _ in 0..100000 {
            arena.alloc(());
        }
    }

    #[test]
    pub fn test_arena_zero_sized() {
        let arena = Arena::new();
        let mut points = vec![];
        for _ in 0..1000 {
            for _ in 0..100 {
                arena.alloc(|| ());
            }
            let point = arena.alloc(|| Point { x: 1, y: 2, z: 3 });
            points.push(point);
        }
        for point in &points {
            assert_eq!(**point, Point { x: 1, y: 2, z: 3 });
        }
    }

    #[test]
    pub fn test_typed_arena_clear() {
        let mut arena = TypedArena::new();
        for _ in 0..10 {
            arena.clear();
            for _ in 0..10000 {
                arena.alloc(Point { x: 1, y: 2, z: 3 });
            }
        }
    }

    #[test]
    pub fn test_arena_clear() {
        let mut arena = Arena::new();
        for _ in 0..10 {
            arena.clear();
            for _ in 0..10000 {
                arena.alloc(|| Point { x: 1, y: 2, z: 3 });
                arena.alloc(|| {
                    Noncopy {
                        string: "hello world".to_string(),
                        array: vec![],
                    }
                });
            }
        }
    }

    #[test]
    pub fn test_arena_alloc_bytes() {
        let arena = Arena::new();
        for i in 0..10000 {
            arena.alloc(|| Point { x: 1, y: 2, z: 3 });
            for byte in arena.alloc_bytes(i % 42).iter_mut() {
                *byte = i as u8;
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

    // Drop tests

    struct DropCounter<'a> {
        count: &'a Cell<u32>,
    }

    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            self.count.set(self.count.get() + 1);
        }
    }

    #[test]
    fn test_arena_drop_count() {
        let counter = Cell::new(0);
        {
            let arena = Arena::new();
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(|| DropCounter { count: &counter });
                // Allocate something with funny size and alignment, to keep
                // things interesting.
                arena.alloc(|| [0u8, 1u8, 2u8]);
            }
            // dropping
        };
        assert_eq!(counter.get(), 100);
    }

    #[test]
    fn test_arena_drop_on_clear() {
        let counter = Cell::new(0);
        for i in 0..10 {
            let mut arena = Arena::new();
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(|| DropCounter { count: &counter });
                // Allocate something with funny size and alignment, to keep
                // things interesting.
                arena.alloc(|| [0u8, 1u8, 2u8]);
            }
            arena.clear();
            assert_eq!(counter.get(), i * 100 + 100);
        }
    }

    #[test]
    fn test_typed_arena_drop_count() {
        let counter = Cell::new(0);
        {
            let arena: TypedArena<DropCounter> = TypedArena::new();
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(DropCounter { count: &counter });
            }
        };
        assert_eq!(counter.get(), 100);
    }

    #[test]
    fn test_typed_arena_drop_on_clear() {
        let counter = Cell::new(0);
        let mut arena: TypedArena<DropCounter> = TypedArena::new();
        for i in 0..10 {
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(DropCounter { count: &counter });
            }
            arena.clear();
            assert_eq!(counter.get(), i * 100 + 100);
        }
    }

    thread_local! {
        static DROP_COUNTER: Cell<u32> = Cell::new(0)
    }

    struct SmallDroppable;

    impl Drop for SmallDroppable {
        fn drop(&mut self) {
            DROP_COUNTER.with(|c| c.set(c.get() + 1));
        }
    }

    #[test]
    fn test_arena_drop_small_count() {
        DROP_COUNTER.with(|c| c.set(0));
        {
            let arena = Arena::new();
            for _ in 0..10 {
                for _ in 0..10 {
                    // Allocate something with drop glue to make sure it doesn't leak.
                    arena.alloc(|| SmallDroppable);
                }
                // Allocate something with funny size and alignment, to keep
                // things interesting.
                arena.alloc(|| [0u8, 1u8, 2u8]);
            }
            // dropping
        };
        assert_eq!(DROP_COUNTER.with(|c| c.get()), 100);
    }

    #[test]
    fn test_typed_arena_drop_small_count() {
        DROP_COUNTER.with(|c| c.set(0));
        {
            let arena: TypedArena<SmallDroppable> = TypedArena::new();
            for _ in 0..100 {
                // Allocate something with drop glue to make sure it doesn't leak.
                arena.alloc(SmallDroppable);
            }
            // dropping
        };
        assert_eq!(DROP_COUNTER.with(|c| c.get()), 100);
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
            let _: Box<_> = Box::new(Noncopy {
                string: "hello world".to_string(),
                array: vec![1, 2, 3, 4, 5],
            });
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
}
