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
//! This crate implements `TypedArena`, a simple arena that can only hold
//! objects of a single type.

#![crate_name = "arena"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(no_crate_inject, attr(deny(warnings))))]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(heap_api)]
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

use alloc::heap;
use alloc::raw_vec::RawVec;

/// An arena that can hold objects of only one type.
pub struct TypedArena<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut T>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut T>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<TypedArenaChunk<T>>>,

    /// Marker indicating that dropping the arena causes its owned
    /// instances of `T` to be dropped.
    _own: PhantomData<T>,
}

struct TypedArenaChunk<T> {
    /// The raw storage for the arena chunk.
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
    /// Creates a new `TypedArena`.
    #[inline]
    pub fn new() -> TypedArena<T> {
        TypedArena {
            // We set both `ptr` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            ptr: Cell::new(0 as *mut T),
            end: Cell::new(0 as *mut T),
            chunks: RefCell::new(vec![]),
            _own: PhantomData,
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
            let (chunk, new_capacity);
            if let Some(last_chunk) = chunks.last_mut() {
                if last_chunk.storage.double_in_place() {
                    self.end.set(last_chunk.end());
                    return;
                } else {
                    let prev_capacity = last_chunk.storage.cap();
                    new_capacity = prev_capacity.checked_mul(2).unwrap();
                }
            } else {
                let elem_size = cmp::max(1, mem::size_of::<T>());
                new_capacity = cmp::max(1, PAGE / elem_size);
            }
            chunk = TypedArenaChunk::<T>::new(new_capacity);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        unsafe {
            // Clear the last chunk, which is partially filled.
            let mut chunks_borrow = self.chunks.borrow_mut();
            if let Some(mut last_chunk) = chunks_borrow.pop() {
                self.clear_last_chunk(&mut last_chunk);
                // If `T` is ZST, code below has no effect.
                for mut chunk in chunks_borrow.drain(..) {
                    let cap = chunk.storage.cap();
                    chunk.destroy(cap);
                }
                chunks_borrow.push(last_chunk);
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
            if let Some(mut last_chunk) = chunks_borrow.pop() {
                // Drop the contents of the last chunk.
                self.clear_last_chunk(&mut last_chunk);
                // The last chunk will be dropped. Destroy all other chunks.
                for chunk in chunks_borrow.iter_mut() {
                    let cap = chunk.storage.cap();
                    chunk.destroy(cap);
                }
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
    use super::TypedArena;
    use std::cell::Cell;

    #[allow(dead_code)]
    #[derive(Debug, Eq, PartialEq)]
    struct Point {
        x: i32,
        y: i32,
        z: i32,
    }

    #[test]
    pub fn test_unused() {
        let arena: TypedArena<Point> = TypedArena::new();
        assert!(arena.chunks.borrow().is_empty());
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
    pub fn test_typed_arena_clear() {
        let mut arena = TypedArena::new();
        for _ in 0..10 {
            arena.clear();
            for _ in 0..10000 {
                arena.alloc(Point { x: 1, y: 2, z: 3 });
            }
        }
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
}
