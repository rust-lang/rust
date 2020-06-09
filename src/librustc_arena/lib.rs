//! The arena, a fast but limited type of allocator.
//!
//! Arenas are a type of allocator that destroy the objects within, all at
//! once, once the arena itself is destroyed. They do not support deallocation
//! of individual objects while the arena itself is still alive. The benefit
//! of an arena is very fast allocation; just a pointer bump.
//!
//! This crate implements several kinds of arena.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/",
    test(no_crate_inject, attr(deny(warnings)))
)]
#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(raw_vec_internals)]
#![cfg_attr(test, feature(test))]
#![allow(deprecated)]

extern crate alloc;

use rustc_data_structures::cold_path;
use smallvec::SmallVec;

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
    /// The number of valid entries in the chunk.
    entries: usize,
}

impl<T> TypedArenaChunk<T> {
    #[inline]
    unsafe fn new(capacity: usize) -> TypedArenaChunk<T> {
        TypedArenaChunk { storage: RawVec::with_capacity(capacity), entries: 0 }
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
                self.start().add(self.storage.capacity())
            }
        }
    }
}

// The arenas start with PAGE-sized chunks, and then each new chunk is twice as
// big as its predecessor, up until we reach HUGE_PAGE-sized chunks, whereupon
// we stop growing. This scales well, from arenas that are barely used up to
// arenas that are used for 100s of MiBs. Note also that the chosen sizes match
// the usual sizes of pages and huge pages on Linux.
const PAGE: usize = 4096;
const HUGE_PAGE: usize = 2 * 1024 * 1024;

impl<T> Default for TypedArena<T> {
    /// Creates a new `TypedArena`.
    fn default() -> TypedArena<T> {
        TypedArena {
            // We set both `ptr` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: RefCell::new(vec![]),
            _own: PhantomData,
        }
    }
}

impl<T> TypedArena<T> {
    /// Allocates an object in the `TypedArena`, returning a reference to it.
    #[inline]
    pub fn alloc(&self, object: T) -> &mut T {
        if self.ptr == self.end {
            self.grow(1)
        }

        unsafe {
            if mem::size_of::<T>() == 0 {
                self.ptr.set(intrinsics::arith_offset(self.ptr.get() as *mut u8, 1) as *mut T);
                let ptr = mem::align_of::<T>() as *mut T;
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

    #[inline]
    fn can_allocate(&self, additional: usize) -> bool {
        let available_bytes = self.end.get() as usize - self.ptr.get() as usize;
        let additional_bytes = additional.checked_mul(mem::size_of::<T>()).unwrap();
        available_bytes >= additional_bytes
    }

    /// Ensures there's enough space in the current chunk to fit `len` objects.
    #[inline]
    fn ensure_capacity(&self, additional: usize) {
        if !self.can_allocate(additional) {
            self.grow(additional);
            debug_assert!(self.can_allocate(additional));
        }
    }

    #[inline]
    unsafe fn alloc_raw_slice(&self, len: usize) -> *mut T {
        assert!(mem::size_of::<T>() != 0);
        assert!(len != 0);

        self.ensure_capacity(len);

        let start_ptr = self.ptr.get();
        self.ptr.set(start_ptr.add(len));
        start_ptr
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
        unsafe {
            let len = slice.len();
            let start_ptr = self.alloc_raw_slice(len);
            slice.as_ptr().copy_to_nonoverlapping(start_ptr, len);
            slice::from_raw_parts_mut(start_ptr, len)
        }
    }

    #[inline]
    pub fn alloc_from_iter<I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        assert!(mem::size_of::<T>() != 0);
        let mut vec: SmallVec<[_; 8]> = iter.into_iter().collect();
        if vec.is_empty() {
            return &mut [];
        }
        // Move the content to the arena by copying it and then forgetting
        // the content of the SmallVec
        unsafe {
            let len = vec.len();
            let start_ptr = self.alloc_raw_slice(len);
            vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
            vec.set_len(0);
            slice::from_raw_parts_mut(start_ptr, len)
        }
    }

    /// Grows the arena.
    #[inline(never)]
    #[cold]
    fn grow(&self, additional: usize) {
        unsafe {
            // We need the element size to convert chunk sizes (ranging from
            // PAGE to HUGE_PAGE bytes) to element counts.
            let elem_size = cmp::max(1, mem::size_of::<T>());
            let mut chunks = self.chunks.borrow_mut();
            let mut new_cap;
            if let Some(last_chunk) = chunks.last_mut() {
                let used_bytes = self.ptr.get() as usize - last_chunk.start() as usize;
                last_chunk.entries = used_bytes / mem::size_of::<T>();

                // If the previous chunk's capacity is less than HUGE_PAGE
                // bytes, then this chunk will be least double the previous
                // chunk's size.
                new_cap = last_chunk.storage.capacity();
                if new_cap < HUGE_PAGE / elem_size {
                    new_cap = new_cap.checked_mul(2).unwrap();
                }
            } else {
                new_cap = PAGE / elem_size;
            }
            // Also ensure that this chunk can fit `additional`.
            new_cap = cmp::max(additional, new_cap);

            let chunk = TypedArenaChunk::<T>::new(new_cap);
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
            if let Some(mut last_chunk) = chunks_borrow.last_mut() {
                self.clear_last_chunk(&mut last_chunk);
                let len = chunks_borrow.len();
                // If `T` is ZST, code below has no effect.
                for mut chunk in chunks_borrow.drain(..len - 1) {
                    chunk.destroy(chunk.entries);
                }
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
                    chunk.destroy(chunk.entries);
                }
            }
            // RawVec handles deallocation of `last_chunk` and `self.chunks`.
        }
    }
}

unsafe impl<T: Send> Send for TypedArena<T> {}

pub struct DroplessArena {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut u8>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut u8>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<TypedArenaChunk<u8>>>,
}

unsafe impl Send for DroplessArena {}

impl Default for DroplessArena {
    #[inline]
    fn default() -> DroplessArena {
        DroplessArena {
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: Default::default(),
        }
    }
}

impl DroplessArena {
    #[inline]
    fn align(&self, align: usize) {
        let final_address = ((self.ptr.get() as usize) + align - 1) & !(align - 1);
        self.ptr.set(final_address as *mut u8);
        assert!(self.ptr <= self.end);
    }

    #[inline(never)]
    #[cold]
    fn grow(&self, additional: usize) {
        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let mut new_cap;
            if let Some(last_chunk) = chunks.last_mut() {
                // There is no need to update `last_chunk.entries` because that
                // field isn't used by `DroplessArena`.

                // If the previous chunk's capacity is less than HUGE_PAGE
                // bytes, then this chunk will be least double the previous
                // chunk's size.
                new_cap = last_chunk.storage.capacity();
                if new_cap < HUGE_PAGE {
                    new_cap = new_cap.checked_mul(2).unwrap();
                }
            } else {
                new_cap = PAGE;
            }
            // Also ensure that this chunk can fit `additional`.
            new_cap = cmp::max(additional, new_cap);

            let chunk = TypedArenaChunk::<u8>::new(new_cap);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    #[inline]
    pub fn alloc_raw(&self, bytes: usize, align: usize) -> &mut [u8] {
        unsafe {
            assert!(bytes != 0);

            self.align(align);

            let future_end = intrinsics::arith_offset(self.ptr.get(), bytes as isize);
            if (future_end as *mut u8) > self.end.get() {
                self.grow(bytes);
            }

            let ptr = self.ptr.get();
            // Set the pointer past ourselves
            self.ptr.set(intrinsics::arith_offset(self.ptr.get(), bytes as isize) as *mut u8);
            slice::from_raw_parts_mut(ptr, bytes)
        }
    }

    #[inline]
    pub fn alloc<T>(&self, object: T) -> &mut T {
        assert!(!mem::needs_drop::<T>());

        let mem = self.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) as *mut _ as *mut T;

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
        assert!(!slice.is_empty());

        let mem = self.alloc_raw(slice.len() * mem::size_of::<T>(), mem::align_of::<T>()) as *mut _
            as *mut T;

        unsafe {
            let arena_slice = slice::from_raw_parts_mut(mem, slice.len());
            arena_slice.copy_from_slice(slice);
            arena_slice
        }
    }

    #[inline]
    unsafe fn write_from_iter<T, I: Iterator<Item = T>>(
        &self,
        mut iter: I,
        len: usize,
        mem: *mut T,
    ) -> &mut [T] {
        let mut i = 0;
        // Use a manual loop since LLVM manages to optimize it better for
        // slice iterators
        loop {
            let value = iter.next();
            if i >= len || value.is_none() {
                // We only return as many items as the iterator gave us, even
                // though it was supposed to give us `len`
                return slice::from_raw_parts_mut(mem, i);
            }
            ptr::write(mem.add(i), value.unwrap());
            i += 1;
        }
    }

    #[inline]
    pub fn alloc_from_iter<T, I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        let iter = iter.into_iter();
        assert!(mem::size_of::<T>() != 0);
        assert!(!mem::needs_drop::<T>());

        let size_hint = iter.size_hint();

        match size_hint {
            (min, Some(max)) if min == max => {
                // We know the exact number of elements the iterator will produce here
                let len = min;

                if len == 0 {
                    return &mut [];
                }
                let size = len.checked_mul(mem::size_of::<T>()).unwrap();
                let mem = self.alloc_raw(size, mem::align_of::<T>()) as *mut _ as *mut T;
                unsafe { self.write_from_iter(iter, len, mem) }
            }
            (_, _) => {
                cold_path(move || -> &mut [T] {
                    let mut vec: SmallVec<[_; 8]> = iter.collect();
                    if vec.is_empty() {
                        return &mut [];
                    }
                    // Move the content to the arena by copying it and then forgetting
                    // the content of the SmallVec
                    unsafe {
                        let len = vec.len();
                        let start_ptr = self
                            .alloc_raw(len * mem::size_of::<T>(), mem::align_of::<T>())
                            as *mut _ as *mut T;
                        vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
                        vec.set_len(0);
                        slice::from_raw_parts_mut(start_ptr, len)
                    }
                })
            }
        }
    }
}

/// Calls the destructor for an object when dropped.
struct DropType {
    drop_fn: unsafe fn(*mut u8),
    obj: *mut u8,
}

unsafe fn drop_for_type<T>(to_drop: *mut u8) {
    std::ptr::drop_in_place(to_drop as *mut T)
}

impl Drop for DropType {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.obj) }
    }
}

/// An arena which can be used to allocate any type.
/// Allocating in this arena is unsafe since the type system
/// doesn't know which types it contains. In order to
/// allocate safely, you must store a PhantomData<T>
/// alongside this arena for each type T you allocate.
#[derive(Default)]
pub struct DropArena {
    /// A list of destructors to run when the arena drops.
    /// Ordered so `destructors` gets dropped before the arena
    /// since its destructor can reference memory in the arena.
    destructors: RefCell<Vec<DropType>>,
    arena: DroplessArena,
}

impl DropArena {
    #[inline]
    pub unsafe fn alloc<T>(&self, object: T) -> &mut T {
        let mem =
            self.arena.alloc_raw(mem::size_of::<T>(), mem::align_of::<T>()) as *mut _ as *mut T;
        // Write into uninitialized memory.
        ptr::write(mem, object);
        let result = &mut *mem;
        // Record the destructor after doing the allocation as that may panic
        // and would cause `object`'s destuctor to run twice if it was recorded before
        self.destructors
            .borrow_mut()
            .push(DropType { drop_fn: drop_for_type::<T>, obj: result as *mut T as *mut u8 });
        result
    }

    #[inline]
    pub unsafe fn alloc_from_iter<T, I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        let mut vec: SmallVec<[_; 8]> = iter.into_iter().collect();
        if vec.is_empty() {
            return &mut [];
        }
        let len = vec.len();

        let start_ptr = self
            .arena
            .alloc_raw(len.checked_mul(mem::size_of::<T>()).unwrap(), mem::align_of::<T>())
            as *mut _ as *mut T;

        let mut destructors = self.destructors.borrow_mut();
        // Reserve space for the destructors so we can't panic while adding them
        destructors.reserve(len);

        // Move the content to the arena by copying it and then forgetting
        // the content of the SmallVec
        vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
        mem::forget(vec.drain(..));

        // Record the destructors after doing the allocation as that may panic
        // and would cause `object`'s destuctor to run twice if it was recorded before
        for i in 0..len {
            destructors.push(DropType {
                drop_fn: drop_for_type::<T>,
                obj: start_ptr.offset(i as isize) as *mut u8,
            });
        }

        slice::from_raw_parts_mut(start_ptr, len)
    }
}

#[macro_export]
macro_rules! arena_for_type {
    ([][$ty:ty]) => {
        $crate::TypedArena<$ty>
    };
    ([few $(, $attrs:ident)*][$ty:ty]) => {
        ::std::marker::PhantomData<$ty>
    };
    ([$ignore:ident $(, $attrs:ident)*]$args:tt) => {
        $crate::arena_for_type!([$($attrs),*]$args)
    };
}

#[macro_export]
macro_rules! which_arena_for_type {
    ([][$arena:expr]) => {
        ::std::option::Option::Some($arena)
    };
    ([few$(, $attrs:ident)*][$arena:expr]) => {
        ::std::option::Option::None
    };
    ([$ignore:ident$(, $attrs:ident)*]$args:tt) => {
        $crate::which_arena_for_type!([$($attrs),*]$args)
    };
}

#[macro_export]
macro_rules! declare_arena {
    ([], [$($a:tt $name:ident: $ty:ty,)*], $tcx:lifetime) => {
        #[derive(Default)]
        pub struct Arena<$tcx> {
            pub dropless: $crate::DroplessArena,
            drop: $crate::DropArena,
            $($name: $crate::arena_for_type!($a[$ty]),)*
        }

        #[marker]
        pub trait ArenaAllocatable {}

        impl<T: Copy> ArenaAllocatable for T {}

        unsafe trait ArenaField<'tcx>: Sized {
            /// Returns a specific arena to allocate from.
            /// If `None` is returned, the `DropArena` will be used.
            fn arena<'a>(arena: &'a Arena<'tcx>) -> Option<&'a $crate::TypedArena<Self>>;
        }

        unsafe impl<'tcx, T> ArenaField<'tcx> for T {
            #[inline]
            default fn arena<'a>(_: &'a Arena<'tcx>) -> Option<&'a $crate::TypedArena<Self>> {
                panic!()
            }
        }

        $(
            #[allow(unused_lifetimes)]
            impl<$tcx> ArenaAllocatable for $ty {}
            unsafe impl<$tcx> ArenaField<$tcx> for $ty {
                #[inline]
                fn arena<'a>(_arena: &'a Arena<$tcx>) -> Option<&'a $crate::TypedArena<Self>> {
                    $crate::which_arena_for_type!($a[&_arena.$name])
                }
            }
        )*

        impl<'tcx> Arena<'tcx> {
            #[inline]
            pub fn alloc<T: ArenaAllocatable>(&self, value: T) -> &mut T {
                if !::std::mem::needs_drop::<T>() {
                    return self.dropless.alloc(value);
                }
                match <T as ArenaField<'tcx>>::arena(self) {
                    ::std::option::Option::Some(arena) => arena.alloc(value),
                    ::std::option::Option::None => unsafe { self.drop.alloc(value) },
                }
            }

            #[inline]
            pub fn alloc_slice<T: ::std::marker::Copy>(&self, value: &[T]) -> &mut [T] {
                if value.is_empty() {
                    return &mut [];
                }
                self.dropless.alloc_slice(value)
            }

            pub fn alloc_from_iter<'a, T: ArenaAllocatable>(
                &'a self,
                iter: impl ::std::iter::IntoIterator<Item = T>,
            ) -> &'a mut [T] {
                if !::std::mem::needs_drop::<T>() {
                    return self.dropless.alloc_from_iter(iter);
                }
                match <T as ArenaField<'tcx>>::arena(self) {
                    ::std::option::Option::Some(arena) => arena.alloc_from_iter(iter),
                    ::std::option::Option::None => unsafe { self.drop.alloc_from_iter(iter) },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
