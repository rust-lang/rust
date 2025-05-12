//! The arena, a fast but limited type of allocator.
//!
//! Arenas are a type of allocator that destroy the objects within, all at
//! once, once the arena itself is destroyed. They do not support deallocation
//! of individual objects while the arena itself is still alive. The benefit
//! of an arena is very fast allocation; just a pointer bump.
//!
//! This crate implements several kinds of arena.

// tidy-alphabetical-start
#![allow(clippy::mut_from_ref)] // Arena allocators are one place where this pattern is fine.
#![allow(internal_features)]
#![cfg_attr(test, feature(test))]
#![deny(unsafe_op_in_unsafe_fn)]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    test(no_crate_inject, attr(deny(warnings)))
)]
#![doc(rust_logo)]
#![feature(core_intrinsics)]
#![feature(decl_macro)]
#![feature(dropck_eyepatch)]
#![feature(maybe_uninit_slice)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(unwrap_infallible)]
// tidy-alphabetical-end

use std::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ptr::{self, NonNull};
use std::{cmp, intrinsics, slice};

use smallvec::SmallVec;

/// This calls the passed function while ensuring it won't be inlined into the caller.
#[inline(never)]
#[cold]
fn outline<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

struct ArenaChunk<T = u8> {
    /// The raw storage for the arena chunk.
    storage: NonNull<[MaybeUninit<T>]>,
    /// The number of valid entries in the chunk.
    entries: usize,
}

unsafe impl<#[may_dangle] T> Drop for ArenaChunk<T> {
    fn drop(&mut self) {
        unsafe { drop(Box::from_raw(self.storage.as_mut())) }
    }
}

impl<T> ArenaChunk<T> {
    #[inline]
    unsafe fn new(capacity: usize) -> ArenaChunk<T> {
        ArenaChunk {
            storage: NonNull::from(Box::leak(Box::new_uninit_slice(capacity))),
            entries: 0,
        }
    }

    /// Destroys this arena chunk.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len` elements of this chunk have been initialized.
    #[inline]
    unsafe fn destroy(&mut self, len: usize) {
        // The branch on needs_drop() is an -O1 performance optimization.
        // Without the branch, dropping TypedArena<T> takes linear time.
        if mem::needs_drop::<T>() {
            // SAFETY: The caller must ensure that `len` elements of this chunk have
            // been initialized.
            unsafe {
                let slice = self.storage.as_mut();
                slice[..len].assume_init_drop();
            }
        }
    }

    // Returns a pointer to the first allocated object.
    #[inline]
    fn start(&mut self) -> *mut T {
        self.storage.as_ptr() as *mut T
    }

    // Returns a pointer to the end of the allocated space.
    #[inline]
    fn end(&mut self) -> *mut T {
        unsafe {
            if size_of::<T>() == 0 {
                // A pointer as large as possible for zero-sized elements.
                ptr::without_provenance_mut(!0)
            } else {
                self.start().add(self.storage.len())
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

/// An arena that can hold objects of only one type.
pub struct TypedArena<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut T>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut T>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<ArenaChunk<T>>>,

    /// Marker indicating that dropping the arena causes its owned
    /// instances of `T` to be dropped.
    _own: PhantomData<T>,
}

impl<T> Default for TypedArena<T> {
    /// Creates a new `TypedArena`.
    fn default() -> TypedArena<T> {
        TypedArena {
            // We set both `ptr` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: Default::default(),
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
            if size_of::<T>() == 0 {
                self.ptr.set(self.ptr.get().wrapping_byte_add(1));
                let ptr = ptr::NonNull::<T>::dangling().as_ptr();
                // Don't drop the object. This `write` is equivalent to `forget`.
                ptr::write(ptr, object);
                &mut *ptr
            } else {
                let ptr = self.ptr.get();
                // Advance the pointer.
                self.ptr.set(self.ptr.get().add(1));
                // Write into uninitialized memory.
                ptr::write(ptr, object);
                &mut *ptr
            }
        }
    }

    #[inline]
    fn can_allocate(&self, additional: usize) -> bool {
        // FIXME: this should *likely* use `offset_from`, but more
        // investigation is needed (including running tests in miri).
        let available_bytes = self.end.get().addr() - self.ptr.get().addr();
        let additional_bytes = additional.checked_mul(size_of::<T>()).unwrap();
        available_bytes >= additional_bytes
    }

    #[inline]
    fn alloc_raw_slice(&self, len: usize) -> *mut T {
        assert!(size_of::<T>() != 0);
        assert!(len != 0);

        // Ensure the current chunk can fit `len` objects.
        if !self.can_allocate(len) {
            self.grow(len);
            debug_assert!(self.can_allocate(len));
        }

        let start_ptr = self.ptr.get();
        // SAFETY: `can_allocate`/`grow` ensures that there is enough space for
        // `len` elements.
        unsafe { self.ptr.set(start_ptr.add(len)) };
        start_ptr
    }

    /// Allocates the elements of this iterator into a contiguous slice in the `TypedArena`.
    ///
    /// Note: for reasons of reentrancy and panic safety we collect into a `SmallVec<[_; 8]>` before
    /// storing the elements in the arena.
    #[inline]
    pub fn alloc_from_iter<I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        self.try_alloc_from_iter(iter.into_iter().map(Ok::<T, !>)).into_ok()
    }

    /// Allocates the elements of this iterator into a contiguous slice in the `TypedArena`.
    ///
    /// Note: for reasons of reentrancy and panic safety we collect into a `SmallVec<[_; 8]>` before
    /// storing the elements in the arena.
    #[inline]
    pub fn try_alloc_from_iter<E>(
        &self,
        iter: impl IntoIterator<Item = Result<T, E>>,
    ) -> Result<&mut [T], E> {
        // Despite the similarlty with `DroplessArena`, we cannot reuse their fast case. The reason
        // is subtle: these arenas are reentrant. In other words, `iter` may very well be holding a
        // reference to `self` and adding elements to the arena during iteration.
        //
        // For this reason, if we pre-allocated any space for the elements of this iterator, we'd
        // have to track that some uninitialized elements are followed by some initialized elements,
        // else we might accidentally drop uninitialized memory if something panics or if the
        // iterator doesn't fill all the length we expected.
        //
        // So we collect all the elements beforehand, which takes care of reentrancy and panic
        // safety. This function is much less hot than `DroplessArena::alloc_from_iter`, so it
        // doesn't need to be hyper-optimized.
        assert!(size_of::<T>() != 0);

        let vec: Result<SmallVec<[T; 8]>, E> = iter.into_iter().collect();
        let mut vec = vec?;
        if vec.is_empty() {
            return Ok(&mut []);
        }
        // Move the content to the arena by copying and then forgetting it.
        let len = vec.len();
        let start_ptr = self.alloc_raw_slice(len);
        Ok(unsafe {
            vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
            vec.set_len(0);
            slice::from_raw_parts_mut(start_ptr, len)
        })
    }

    /// Grows the arena.
    #[inline(never)]
    #[cold]
    fn grow(&self, additional: usize) {
        unsafe {
            // We need the element size to convert chunk sizes (ranging from
            // PAGE to HUGE_PAGE bytes) to element counts.
            let elem_size = cmp::max(1, size_of::<T>());
            let mut chunks = self.chunks.borrow_mut();
            let mut new_cap;
            if let Some(last_chunk) = chunks.last_mut() {
                // If a type is `!needs_drop`, we don't need to keep track of how many elements
                // the chunk stores - the field will be ignored anyway.
                if mem::needs_drop::<T>() {
                    // FIXME: this should *likely* use `offset_from`, but more
                    // investigation is needed (including running tests in miri).
                    let used_bytes = self.ptr.get().addr() - last_chunk.start().addr();
                    last_chunk.entries = used_bytes / size_of::<T>();
                }

                // If the previous chunk's len is less than HUGE_PAGE
                // bytes, then this chunk will be least double the previous
                // chunk's size.
                new_cap = last_chunk.storage.len().min(HUGE_PAGE / elem_size / 2);
                new_cap *= 2;
            } else {
                new_cap = PAGE / elem_size;
            }
            // Also ensure that this chunk can fit `additional`.
            new_cap = cmp::max(additional, new_cap);

            let mut chunk = ArenaChunk::<T>::new(new_cap);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    // Drops the contents of the last chunk. The last chunk is partially empty, unlike all other
    // chunks.
    fn clear_last_chunk(&self, last_chunk: &mut ArenaChunk<T>) {
        // Determine how much was filled.
        let start = last_chunk.start().addr();
        // We obtain the value of the pointer to the first uninitialized element.
        let end = self.ptr.get().addr();
        // We then calculate the number of elements to be dropped in the last chunk,
        // which is the filled area's length.
        let diff = if size_of::<T>() == 0 {
            // `T` is ZST. It can't have a drop flag, so the value here doesn't matter. We get
            // the number of zero-sized values in the last and only chunk, just out of caution.
            // Recall that `end` was incremented for each allocated value.
            end - start
        } else {
            // FIXME: this should *likely* use `offset_from`, but more
            // investigation is needed (including running tests in miri).
            (end - start) / size_of::<T>()
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
            // Box handles deallocation of `last_chunk` and `self.chunks`.
        }
    }
}

unsafe impl<T: Send> Send for TypedArena<T> {}

#[inline(always)]
fn align_down(val: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    val & !(align - 1)
}

#[inline(always)]
fn align_up(val: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (val + align - 1) & !(align - 1)
}

// Pointer alignment is common in compiler types, so keep `DroplessArena` aligned to them
// to optimize away alignment code.
const DROPLESS_ALIGNMENT: usize = align_of::<usize>();

/// An arena that can hold objects of multiple different types that impl `Copy`
/// and/or satisfy `!mem::needs_drop`.
pub struct DroplessArena {
    /// A pointer to the start of the free space.
    start: Cell<*mut u8>,

    /// A pointer to the end of free space.
    ///
    /// The allocation proceeds downwards from the end of the chunk towards the
    /// start. (This is slightly simpler and faster than allocating upwards,
    /// see <https://fitzgeraldnick.com/2019/11/01/always-bump-downwards.html>.)
    /// When this pointer crosses the start pointer, a new chunk is allocated.
    ///
    /// This is kept aligned to DROPLESS_ALIGNMENT.
    end: Cell<*mut u8>,

    /// A vector of arena chunks.
    chunks: RefCell<Vec<ArenaChunk>>,
}

unsafe impl Send for DroplessArena {}

impl Default for DroplessArena {
    #[inline]
    fn default() -> DroplessArena {
        DroplessArena {
            // We set both `start` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            start: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: Default::default(),
        }
    }
}

impl DroplessArena {
    #[inline(never)]
    #[cold]
    fn grow(&self, layout: Layout) {
        // Add some padding so we can align `self.end` while
        // still fitting in a `layout` allocation.
        let additional = layout.size() + cmp::max(DROPLESS_ALIGNMENT, layout.align()) - 1;

        unsafe {
            let mut chunks = self.chunks.borrow_mut();
            let mut new_cap;
            if let Some(last_chunk) = chunks.last_mut() {
                // There is no need to update `last_chunk.entries` because that
                // field isn't used by `DroplessArena`.

                // If the previous chunk's len is less than HUGE_PAGE
                // bytes, then this chunk will be least double the previous
                // chunk's size.
                new_cap = last_chunk.storage.len().min(HUGE_PAGE / 2);
                new_cap *= 2;
            } else {
                new_cap = PAGE;
            }
            // Also ensure that this chunk can fit `additional`.
            new_cap = cmp::max(additional, new_cap);

            let mut chunk = ArenaChunk::new(align_up(new_cap, PAGE));
            self.start.set(chunk.start());

            // Align the end to DROPLESS_ALIGNMENT.
            let end = align_down(chunk.end().addr(), DROPLESS_ALIGNMENT);

            // Make sure we don't go past `start`. This should not happen since the allocation
            // should be at least DROPLESS_ALIGNMENT - 1 bytes.
            debug_assert!(chunk.start().addr() <= end);

            self.end.set(chunk.end().with_addr(end));

            chunks.push(chunk);
        }
    }

    #[inline]
    pub fn alloc_raw(&self, layout: Layout) -> *mut u8 {
        assert!(layout.size() != 0);

        // This loop executes once or twice: if allocation fails the first
        // time, the `grow` ensures it will succeed the second time.
        loop {
            let start = self.start.get().addr();
            let old_end = self.end.get();
            let end = old_end.addr();

            // Align allocated bytes so that `self.end` stays aligned to
            // DROPLESS_ALIGNMENT.
            let bytes = align_up(layout.size(), DROPLESS_ALIGNMENT);

            // Tell LLVM that `end` is aligned to DROPLESS_ALIGNMENT.
            unsafe { intrinsics::assume(end == align_down(end, DROPLESS_ALIGNMENT)) };

            if let Some(sub) = end.checked_sub(bytes) {
                let new_end = align_down(sub, layout.align());
                if start <= new_end {
                    let new_end = old_end.with_addr(new_end);
                    // `new_end` is aligned to DROPLESS_ALIGNMENT as `align_down`
                    // preserves alignment as both `end` and `bytes` are already
                    // aligned to DROPLESS_ALIGNMENT.
                    self.end.set(new_end);
                    return new_end;
                }
            }

            // No free space left. Allocate a new chunk to satisfy the request.
            // On failure the grow will panic or abort.
            self.grow(layout);
        }
    }

    #[inline]
    pub fn alloc<T>(&self, object: T) -> &mut T {
        assert!(!mem::needs_drop::<T>());
        assert!(size_of::<T>() != 0);

        let mem = self.alloc_raw(Layout::new::<T>()) as *mut T;

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
        assert!(size_of::<T>() != 0);
        assert!(!slice.is_empty());

        let mem = self.alloc_raw(Layout::for_value::<[T]>(slice)) as *mut T;

        unsafe {
            mem.copy_from_nonoverlapping(slice.as_ptr(), slice.len());
            slice::from_raw_parts_mut(mem, slice.len())
        }
    }

    /// Used by `Lift` to check whether this slice is allocated
    /// in this arena.
    #[inline]
    pub fn contains_slice<T>(&self, slice: &[T]) -> bool {
        for chunk in self.chunks.borrow_mut().iter_mut() {
            let ptr = slice.as_ptr().cast::<u8>().cast_mut();
            if chunk.start() <= ptr && chunk.end() >= ptr {
                return true;
            }
        }
        false
    }

    /// Allocates a string slice that is copied into the `DroplessArena`, returning a
    /// reference to it. Will panic if passed an empty string.
    ///
    /// Panics:
    ///
    ///  - Zero-length string
    #[inline]
    pub fn alloc_str(&self, string: &str) -> &str {
        let slice = self.alloc_slice(string.as_bytes());

        // SAFETY: the result has a copy of the same valid UTF-8 bytes.
        unsafe { std::str::from_utf8_unchecked(slice) }
    }

    /// # Safety
    ///
    /// The caller must ensure that `mem` is valid for writes up to `size_of::<T>() * len`, and that
    /// that memory stays allocated and not shared for the lifetime of `self`. This must hold even
    /// if `iter.next()` allocates onto `self`.
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
            // SAFETY: The caller must ensure that `mem` is valid for writes up to
            // `size_of::<T>() * len`.
            unsafe {
                match iter.next() {
                    Some(value) if i < len => mem.add(i).write(value),
                    Some(_) | None => {
                        // We only return as many items as the iterator gave us, even
                        // though it was supposed to give us `len`
                        return slice::from_raw_parts_mut(mem, i);
                    }
                }
            }
            i += 1;
        }
    }

    #[inline]
    pub fn alloc_from_iter<T, I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        // Warning: this function is reentrant: `iter` could hold a reference to `&self` and
        // allocate additional elements while we're iterating.
        let iter = iter.into_iter();
        assert!(size_of::<T>() != 0);
        assert!(!mem::needs_drop::<T>());

        let size_hint = iter.size_hint();

        match size_hint {
            (min, Some(max)) if min == max => {
                // We know the exact number of elements the iterator expects to produce here.
                let len = min;

                if len == 0 {
                    return &mut [];
                }

                let mem = self.alloc_raw(Layout::array::<T>(len).unwrap()) as *mut T;
                // SAFETY: `write_from_iter` doesn't touch `self`. It only touches the slice we just
                // reserved. If the iterator panics or doesn't output `len` elements, this will
                // leave some unallocated slots in the arena, which is fine because we do not call
                // `drop`.
                unsafe { self.write_from_iter(iter, len, mem) }
            }
            (_, _) => outline(move || self.try_alloc_from_iter(iter.map(Ok::<T, !>)).into_ok()),
        }
    }

    #[inline]
    pub fn try_alloc_from_iter<T, E>(
        &self,
        iter: impl IntoIterator<Item = Result<T, E>>,
    ) -> Result<&mut [T], E> {
        // Despite the similarlty with `alloc_from_iter`, we cannot reuse their fast case, as we
        // cannot know the minimum length of the iterator in this case.
        assert!(size_of::<T>() != 0);

        // Takes care of reentrancy.
        let vec: Result<SmallVec<[T; 8]>, E> = iter.into_iter().collect();
        let mut vec = vec?;
        if vec.is_empty() {
            return Ok(&mut []);
        }
        // Move the content to the arena by copying and then forgetting it.
        let len = vec.len();
        Ok(unsafe {
            let start_ptr = self.alloc_raw(Layout::for_value::<[T]>(vec.as_slice())) as *mut T;
            vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
            vec.set_len(0);
            slice::from_raw_parts_mut(start_ptr, len)
        })
    }
}

/// Declare an `Arena` containing one dropless arena and many typed arenas (the
/// types of the typed arenas are specified by the arguments).
///
/// There are three cases of interest.
/// - Types that are `Copy`: these need not be specified in the arguments. They
///   will use the `DroplessArena`.
/// - Types that are `!Copy` and `!Drop`: these must be specified in the
///   arguments. An empty `TypedArena` will be created for each one, but the
///   `DroplessArena` will always be used and the `TypedArena` will stay empty.
///   This is odd but harmless, because an empty arena allocates no memory.
/// - Types that are `!Copy` and `Drop`: these must be specified in the
///   arguments. The `TypedArena` will be used for them.
///
#[rustc_macro_transparency = "semitransparent"]
pub macro declare_arena([$($a:tt $name:ident: $ty:ty,)*]) {
    #[derive(Default)]
    pub struct Arena<'tcx> {
        pub dropless: $crate::DroplessArena,
        $($name: $crate::TypedArena<$ty>,)*
    }

    pub trait ArenaAllocatable<'tcx, C = rustc_arena::IsNotCopy>: Sized {
        #[allow(clippy::mut_from_ref)]
        fn allocate_on(self, arena: &'tcx Arena<'tcx>) -> &'tcx mut Self;
        #[allow(clippy::mut_from_ref)]
        fn allocate_from_iter(
            arena: &'tcx Arena<'tcx>,
            iter: impl ::std::iter::IntoIterator<Item = Self>,
        ) -> &'tcx mut [Self];
    }

    // Any type that impls `Copy` can be arena-allocated in the `DroplessArena`.
    impl<'tcx, T: Copy> ArenaAllocatable<'tcx, rustc_arena::IsCopy> for T {
        #[inline]
        #[allow(clippy::mut_from_ref)]
        fn allocate_on(self, arena: &'tcx Arena<'tcx>) -> &'tcx mut Self {
            arena.dropless.alloc(self)
        }
        #[inline]
        #[allow(clippy::mut_from_ref)]
        fn allocate_from_iter(
            arena: &'tcx Arena<'tcx>,
            iter: impl ::std::iter::IntoIterator<Item = Self>,
        ) -> &'tcx mut [Self] {
            arena.dropless.alloc_from_iter(iter)
        }
    }
    $(
        impl<'tcx> ArenaAllocatable<'tcx, rustc_arena::IsNotCopy> for $ty {
            #[inline]
            fn allocate_on(self, arena: &'tcx Arena<'tcx>) -> &'tcx mut Self {
                if !::std::mem::needs_drop::<Self>() {
                    arena.dropless.alloc(self)
                } else {
                    arena.$name.alloc(self)
                }
            }

            #[inline]
            #[allow(clippy::mut_from_ref)]
            fn allocate_from_iter(
                arena: &'tcx Arena<'tcx>,
                iter: impl ::std::iter::IntoIterator<Item = Self>,
            ) -> &'tcx mut [Self] {
                if !::std::mem::needs_drop::<Self>() {
                    arena.dropless.alloc_from_iter(iter)
                } else {
                    arena.$name.alloc_from_iter(iter)
                }
            }
        }
    )*

    impl<'tcx> Arena<'tcx> {
        #[inline]
        #[allow(clippy::mut_from_ref)]
        pub fn alloc<T: ArenaAllocatable<'tcx, C>, C>(&'tcx self, value: T) -> &mut T {
            value.allocate_on(self)
        }

        // Any type that impls `Copy` can have slices be arena-allocated in the `DroplessArena`.
        #[inline]
        #[allow(clippy::mut_from_ref)]
        pub fn alloc_slice<T: ::std::marker::Copy>(&self, value: &[T]) -> &mut [T] {
            if value.is_empty() {
                return &mut [];
            }
            self.dropless.alloc_slice(value)
        }

        #[inline]
        pub fn alloc_str(&self, string: &str) -> &str {
            if string.is_empty() {
                return "";
            }
            self.dropless.alloc_str(string)
        }

        #[allow(clippy::mut_from_ref)]
        pub fn alloc_from_iter<T: ArenaAllocatable<'tcx, C>, C>(
            &'tcx self,
            iter: impl ::std::iter::IntoIterator<Item = T>,
        ) -> &mut [T] {
            T::allocate_from_iter(self, iter)
        }
    }
}

// Marker types that let us give different behaviour for arenas allocating
// `Copy` types vs `!Copy` types.
pub struct IsCopy;
pub struct IsNotCopy;

#[cfg(test)]
mod tests;
