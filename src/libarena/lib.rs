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

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(no_crate_inject, attr(deny(warnings))))]

#![feature(alloc)]
#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(nll)]
#![feature(raw_vec_internals)]
#![cfg_attr(test, feature(test))]

#![allow(deprecated)]

extern crate alloc;
extern crate rustc_data_structures;

use rustc_data_structures::defer_deallocs::DeferDeallocs;
use rustc_data_structures::sync::{MTLock, WorkerLocal};

use std::cell::{Cell, RefCell};
use std::cmp;
use std::intrinsics;
use std::marker::{PhantomData, Send};
use std::mem;
use std::ptr;
use std::slice;
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
        TypedArenaChunk {
            storage: RawVec::with_capacity(capacity),
        }
    }

    /// Destroys this arena chunk.
    #[inline]
    unsafe fn destroy(&mut self, len: usize) {
        // The branch on needs_drop() is an -O1 performance optimization.
        // Without the branch, dropping TypedArena<u8> takes linear time.
        if mem::needs_drop::<T>() {
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
                self.start().add(self.storage.cap())
            }
        }
    }
}

const PAGE: usize = 4096;

impl<T> Default for TypedArena<T> {
    /// Creates a new `TypedArena`.
    fn default() -> TypedArena<T> {
        TypedArena {
            // We set both `ptr` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            ptr: Cell::new(0 as *mut T),
            end: Cell::new(0 as *mut T),
            chunks: RefCell::new(vec![]),
            _own: PhantomData,
        }
    }
}

impl<T> TypedArena<T> {
    /// Allocates an object in the `TypedArena`, returning a reference to it.
    #[inline]
    pub fn alloc(&self, object: T) -> &mut T {
        // Zero sized path
        if mem::size_of::<T>() == 0 {
            if self.ptr == self.end {
                self.grow(1)
            }

            unsafe {
                self.ptr
                    .set(intrinsics::arith_offset(self.ptr.get() as *mut u8, 1)
                        as *mut T);
                let ptr = mem::align_of::<T>() as *mut T;
                // Don't drop the object. This `write` is equivalent to `forget`.
                ptr::write(ptr, object);
                return &mut *ptr;
            }
        }

        let ptr = self.ptr.get();

        unsafe {
            if std::intrinsics::unlikely(ptr == self.end.get()) {
                self.grow_and_alloc(object)
            } else {
                self.alloc_unchecked(ptr, object)
            }
        }
    }

    #[inline(always)]
    unsafe fn alloc_unchecked(&self, ptr: *mut T, object: T) -> &mut T {
        // Advance the pointer.
        self.ptr.set(ptr.offset(1));
        // Write into uninitialized memory.
        ptr::write(ptr, object);
        &mut *ptr
    }

    #[inline(never)]
    #[cold]
    fn grow_and_alloc(&self, object: T) -> &mut T {
        // We move the object in this function so if it has a destructor
        // the fast path need not have an unwind handler to destroy it
        self.grow(1);
        unsafe {
            self.alloc_unchecked(self.ptr.get(), object)
        }
    }

    /// Allocates a slice of objects that are copied into the `TypedArena`, returning a mutable
    /// reference to it. Will panic if passed a zero-sized types.
    ///
    /// Panics:
    ///
    ///  - Zero-sized types
    ///  - Zero-length slices
    #[inline]
    pub fn alloc_slice(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        assert!(mem::size_of::<T>() != 0);
        assert!(slice.len() != 0);

        let available_capacity_bytes = self.end.get() as usize - self.ptr.get() as usize;
        let at_least_bytes = slice.len() * mem::size_of::<T>();
        if available_capacity_bytes < at_least_bytes {
            self.grow_slice(slice.len());
        }

        unsafe {
            let start_ptr = self.ptr.get();
            let arena_slice = slice::from_raw_parts_mut(start_ptr, slice.len());
            self.ptr.set(start_ptr.add(arena_slice.len()));
            arena_slice.copy_from_slice(slice);
            arena_slice
        }
    }

    #[inline(never)]
    #[cold]
    fn grow_slice(&self, n: usize) {
        self.grow(n)
    }
    
    /// Grows the arena.
    #[inline(always)]
    fn grow(&self, n: usize) {
        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let (chunk, mut new_capacity);
            if let Some(last_chunk) = chunks.last_mut() {
                let used_bytes = self.ptr.get() as usize - last_chunk.start() as usize;
                let currently_used_cap = used_bytes / mem::size_of::<T>();
                if last_chunk.storage.reserve_in_place(currently_used_cap, n) {
                    self.end.set(last_chunk.end());
                    return;
                } else {
                    new_capacity = last_chunk.storage.cap();
                    loop {
                        new_capacity = new_capacity.checked_mul(2).unwrap();
                        if new_capacity >= currently_used_cap + n {
                            break;
                        }
                    }
                }
            } else {
                let elem_size = cmp::max(1, mem::size_of::<T>());
                new_capacity = cmp::max(n, PAGE / elem_size);
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

unsafe impl<#[may_dangle] T> Drop for TypedArena<T> {
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

type BackingType = usize;
const BLOCK_SIZE: usize = std::mem::size_of::<BackingType>();

#[inline(always)]
fn required_backing_types(bytes: usize) -> usize {
    assert!(BLOCK_SIZE.is_power_of_two());
    // FIXME: This addition could overflow
    (bytes + BLOCK_SIZE - 1) / BLOCK_SIZE
}

#[inline(always)]
fn align(val: usize, align: usize) -> usize {
    assert!(align.is_power_of_two());
    (val + align - 1) & !(align - 1)
}

pub struct DroplessArena {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut u8>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut u8>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<TypedArenaChunk<BackingType>>>,
}

#[no_mangle]
pub fn tatest1(a: &TypedArena<usize>) -> &usize {
    a.alloc(64usize)
}

#[no_mangle]
pub fn atest1(a: &DroplessArena) -> &usize {
    a.alloc(64usize)
}

#[no_mangle]
pub fn atest2(a: &SyncDroplessArena, b: Box<usize>) -> &Box<usize> {
    a.promote(b)
}

#[no_mangle]
pub fn atest6(a: &SyncDroplessArena, b: usize) -> &usize {
    a.promote(b)
}

#[no_mangle]
pub fn atest3(a: &DroplessArena) {
    a.align(8);
}

#[no_mangle]
pub fn atest4(a: &DroplessArena) {
    a.align(16);
}

#[no_mangle]
pub fn atest5(a: &DroplessArena) {
    a.align(4);
}

unsafe impl Send for DroplessArena {}

impl Default for DroplessArena {
    #[inline]
    fn default() -> DroplessArena {
        DroplessArena {
            ptr: Cell::new(0 as *mut u8),
            end: Cell::new(0 as *mut u8),
            chunks: Default::default(),
        }
    }
}

impl DroplessArena {
    pub fn in_arena<T: ?Sized>(&self, ptr: *const T) -> bool {
        let ptr = ptr as *const u8 as *mut BackingType;
        for chunk in &*self.chunks.borrow() {
            if chunk.start() <= ptr && ptr < chunk.end() {
                return true;
            }
        }

        false
    }

    #[inline]
    fn align(&self, align: usize) {
        // FIXME: The addition of `align` could overflow, in which case final_address
        // will be 0. Do we have any guarantee that our chunk won't end up as the final
        // bytes in our memory space?
        let final_address = ((self.ptr.get() as usize) + align - 1) & !(align - 1);
        self.ptr.set(final_address as *mut u8);

        // Aligning to the block_size cannot go outside our current chuck, just to its end
        if align > BLOCK_SIZE {
            // For larger alignments we have to check that we didn't go out of bounds
            assert!(self.ptr <= self.end);
        }
    }

    fn grow(&self, needed_bytes: usize) {
        unsafe {
            let needed_vals = required_backing_types(needed_bytes);
            let mut chunks = self.chunks.borrow_mut();
            let (chunk, mut new_capacity);
            if let Some(last_chunk) = chunks.last_mut() {
                let used_bytes = self.ptr.get() as usize - last_chunk.start() as usize;
                let used_vals = required_backing_types(used_bytes);
                if last_chunk
                    .storage
                    .reserve_in_place(used_vals, needed_vals)
                {
                    self.end.set(last_chunk.end() as *mut u8);
                    return;
                } else {
                    new_capacity = last_chunk.storage.cap();
                    loop {
                        new_capacity = new_capacity.checked_mul(2).unwrap();
                        if new_capacity >= used_vals + needed_vals {
                            break;
                        }
                    }
                }
            } else {
                new_capacity = cmp::max(needed_vals, required_backing_types(PAGE));
            }
            chunk = TypedArenaChunk::<BackingType>::new(new_capacity);
            self.ptr.set(chunk.start() as *mut u8);
            self.end.set(chunk.end() as *mut u8);
            chunks.push(chunk);
        }
    }

    #[inline(never)]
    #[cold]
    fn grow_and_alloc_raw(&self, bytes: usize) -> &mut [u8] {
        self.grow(bytes);
        unsafe {
            self.alloc_raw_unchecked(self.ptr.get(), bytes)
        }
    }

    #[inline(always)]
    unsafe fn alloc_raw_unchecked(&self, start: *mut u8, bytes: usize) -> &mut [u8] {
        // Tell LLVM that `start` is aligned to BLOCK_SIZE
        std::intrinsics::assume(start as usize == align(start as usize, BLOCK_SIZE));

        // Set the pointer past ourselves and align it
        let end = start.offset(bytes as isize) as usize;
        let end = align(end, BLOCK_SIZE) as *mut u8;
        self.ptr.set(end);

        // Return the result
        slice::from_raw_parts_mut(start, bytes)
    }

    #[inline]
    pub fn alloc_raw(&self, bytes: usize, align: usize) -> &mut [u8] {
        // FIXME: Always align to 8 bytes here? Or usize alignment
        unsafe {
            assert!(bytes != 0);
            assert!(align <= BLOCK_SIZE);
            assert!(std::mem::align_of::<BackingType>() == std::mem::size_of::<BackingType>());
            // FIXME: Check that `bytes` fit in a isize

            // FIXME: arith_offset could overflow here.
            // Find some way to guarantee this doesn't happen for small fixed size types
            let ptr = self.ptr.get();
            let future_end = intrinsics::arith_offset(ptr, bytes as isize);
            if std::intrinsics::unlikely((future_end as *mut u8) >= self.end.get()) {
                self.grow_and_alloc_raw(bytes)
            } else {
                self.alloc_raw_unchecked(ptr, bytes)
            }
        }
    }

    #[inline]
    pub fn alloc<T>(&self, object: T) -> &mut T {
        assert!(!mem::needs_drop::<T>());

        let mem = self.alloc_raw(
            mem::size_of::<T>(),
            mem::align_of::<T>()) as *mut _ as *mut T;

        unsafe {
            // Write into uninitialized memory.
            ptr::write(mem, object);
            &mut *mem
        }
    }

    /// Allocates a slice of objects that are copied into the `DroplessArena`, returning a mutable
    /// reference to it. Will panic if passed a zero-sized type.
    ///
    /// Panics:
    ///
    ///  - Zero-sized types
    ///  - Zero-length slices
    #[inline]
    pub fn alloc_slice<T>(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        assert!(!mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() != 0);
        assert!(slice.len() != 0);

        let mem = self.alloc_raw(
            slice.len() * mem::size_of::<T>(),
            mem::align_of::<T>()) as *mut _ as *mut T;

        unsafe {
            let arena_slice = slice::from_raw_parts_mut(mem, slice.len());
            arena_slice.copy_from_slice(slice);
            arena_slice
        }
    }
}

#[derive(Default)]
// FIXME(@Zoxc): this type is entirely unused in rustc
pub struct SyncTypedArena<T> {
    lock: MTLock<TypedArena<T>>,
}

impl<T> SyncTypedArena<T> {
    #[inline(always)]
    pub fn alloc(&self, object: T) -> &mut T {
        // Extend the lifetime of the result since it's limited to the lock guard
        unsafe { &mut *(self.lock.lock().alloc(object) as *mut T) }
    }

    #[inline(always)]
    pub fn alloc_slice(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        // Extend the lifetime of the result since it's limited to the lock guard
        unsafe { &mut *(self.lock.lock().alloc_slice(slice) as *mut [T]) }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.lock.get_mut().clear();
    }
}

struct DropType {
    drop_fn: unsafe fn(*mut u8),
    obj: *mut u8,
}

unsafe fn drop_for_type<T>(to_drop: *mut u8) {
    std::ptr::drop_in_place(to_drop as *mut T)
}

impl Drop for DropType {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self.obj)
        }
    }
}

pub struct SyncDroplessArena {
    // Ordered so `deferred` gets dropped before the arena
    // since its destructor can reference memory in the arena
    deferred: WorkerLocal<TypedArena<DropType>>,
    lock: MTLock<DroplessArena>,
}

impl SyncDroplessArena {
    #[inline]
    pub fn new() -> Self {
        SyncDroplessArena {
            lock: Default::default(),
            deferred: WorkerLocal::new(|_| Default::default()),
        }
    }

    #[inline(always)]
    pub fn in_arena<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.lock.lock().in_arena(ptr)
    }

    #[inline(always)]
    pub fn alloc_raw(&self, bytes: usize, align: usize) -> &mut [u8] {
        // Extend the lifetime of the result since it's limited to the lock guard
        unsafe { &mut *(self.lock.lock().alloc_raw(bytes, align) as *mut [u8]) }
    }

    #[inline(always)]
    pub fn alloc<T>(&self, object: T) -> &mut T {
        // Extend the lifetime of the result since it's limited to the lock guard
        unsafe { &mut *(self.lock.lock().alloc(object) as *mut T) }
    }

    #[inline(always)]
    pub fn alloc_slice<T>(&self, slice: &[T]) -> &mut [T]
    where
        T: Copy,
    {
        // Extend the lifetime of the result since it's limited to the lock guard
        unsafe { &mut *(self.lock.lock().alloc_slice(slice) as *mut [T]) }
    }

    #[inline]
    pub fn promote<T: DeferDeallocs>(&self, object: T) -> &T {
        let mem = self.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) as *mut _ as *mut T;
        let result = unsafe {
            // Write into uninitialized memory.
            ptr::write(mem, object);
            &mut *mem
        };
        // Record the destructor after doing the allocation as that may panic
        // and would cause `object` destuctor to run twice if it was recorded before
        self.deferred.alloc(DropType {
            drop_fn: drop_for_type::<T>,
            obj: result as *mut T as *mut u8,
        });
        result
    }

    #[inline(always)]
    pub fn promote_vec<T: DeferDeallocs>(&self, vec: Vec<T>) -> &[T] {
        &self.promote(vec)[..]
    }
}

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
        let arena: TypedArena<Point> = TypedArena::default();
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

        let arena = Wrap(TypedArena::default());

        let result = arena.alloc_outer(|| Outer {
            inner: arena.alloc_inner(|| Inner { value: 10 }),
        });

        assert_eq!(result.inner.value, 10);
    }

    #[test]
    pub fn test_copy() {
        let arena = TypedArena::default();
        for _ in 0..100000 {
            arena.alloc(Point { x: 1, y: 2, z: 3 });
        }
    }

    #[bench]
    pub fn bench_copy(b: &mut Bencher) {
        let arena = TypedArena::default();
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
        let arena = TypedArena::default();
        for _ in 0..100000 {
            arena.alloc(Noncopy {
                string: "hello world".to_string(),
                array: vec![1, 2, 3, 4, 5],
            });
        }
    }

    #[test]
    pub fn test_typed_arena_zero_sized() {
        let arena = TypedArena::default();
        for _ in 0..100000 {
            arena.alloc(());
        }
    }

    #[test]
    pub fn test_typed_arena_clear() {
        let mut arena = TypedArena::default();
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
            let arena: TypedArena<DropCounter> = TypedArena::default();
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
        let mut arena: TypedArena<DropCounter> = TypedArena::default();
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
            let arena: TypedArena<SmallDroppable> = TypedArena::default();
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
        let arena = TypedArena::default();
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
