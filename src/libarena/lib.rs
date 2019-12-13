//! The arena, a fast but limited type of allocator.
//!
//! Arenas are a type of allocator that destroy the objects within, all at
//! once, once the arena itself is destroyed. They do not support deallocation
//! of individual objects while the arena itself is still alive. The benefit
//! of an arena is very fast allocation; just a pointer bump.
//!
//! This crate implements `TypedArena`, a simple arena that can only hold
//! objects of a single type.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       test(no_crate_inject, attr(deny(warnings))))]

#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(ptr_offset_from)]
#![feature(raw_vec_internals)]
#![feature(untagged_unions)]
#![cfg_attr(test, feature(test))]

#![allow(deprecated)]

extern crate alloc;

use rustc_data_structures::cold_path;
use rustc_data_structures::sync::{SharedWorkerLocal, WorkerLocal, Lock};
use smallvec::SmallVec;

use std::cell::{Cell, RefCell};
use std::cmp;
use std::marker::{PhantomData, Send};
use std::mem;
use std::ptr;
use std::slice;

use alloc::raw_vec::RawVec;

trait ChunkBackend<T>: Sized {
    type ChunkVecType: Sized;

    /// Create new vec.
    fn new() -> Self;

    /// Create new vec.
    fn new_vec() -> Self::ChunkVecType;

    /// Check current chunk has enough space for next allocation.
    fn can_allocate(&self, len: usize, align: usize) -> bool;

    /// Allocate a new chunk and point to it.
    fn grow(&self, len: usize, align: usize, chunks: &mut Self::ChunkVecType);

    /// Allocate a slice from this chunk. Panic if space lacks.
    unsafe fn alloc_raw_slice(&self, len: usize, align: usize) -> *mut T;

    /// Clear the arena.
    fn clear(&self, chunks: &mut Self::ChunkVecType);
}

struct NOPCurrentChunk<T> {
    phantom: PhantomData<*mut T>,
}

impl<T> ChunkBackend<T> for NOPCurrentChunk<T> {
    type ChunkVecType = ();

    #[inline]
    fn new() -> Self {
        let phantom = PhantomData;
        NOPCurrentChunk { phantom }
    }

    #[inline]
    fn new_vec() {}

    #[inline]
    fn can_allocate(&self, _len: usize, _align: usize) -> bool
    { true }

    #[inline]
    fn grow(&self, _len: usize, _align: usize, _chunks: &mut Self::ChunkVecType)
    {}

    #[inline]
    unsafe fn alloc_raw_slice(&self, _len: usize, align: usize) -> *mut T {
        assert!(align >= mem::align_of::<T>());
        align as *mut T
    }

    #[inline]
    fn clear(&self, _chunk: &mut Self::ChunkVecType)
    {}
}

struct ZSTCurrentChunk<T> {
    counter: Cell<usize>,
    phantom: PhantomData<*mut T>,
}

impl<T> ChunkBackend<T> for ZSTCurrentChunk<T> {
    type ChunkVecType = ();

    #[inline]
    fn new() -> Self {
        ZSTCurrentChunk {
            counter: Cell::new(0),
            phantom: PhantomData,
        }
    }

    #[inline]
    fn new_vec() {}

    #[inline]
    fn can_allocate(&self, _len: usize, _align: usize) -> bool
    { true }

    #[inline]
    fn grow(&self, _len: usize, _align: usize, _chunks: &mut Self::ChunkVecType)
    {}

    #[inline]
    unsafe fn alloc_raw_slice(&self, len: usize, align: usize) -> *mut T {
        assert!(align >= mem::align_of::<T>());
        let count = self.counter.get();
        self.counter.set(count+len);
        align as *mut T
    }

    #[inline]
    fn clear(&self, _chunks: &mut Self::ChunkVecType) {
        assert!(mem::needs_drop::<T>());

        let count = self.counter.get();
        for _ in 0..count {
            let ptr = mem::align_of::<T>() as *mut T;
            unsafe { ptr::drop_in_place(ptr) }
        }

        self.counter.set(0)
    }
}

struct TypedCurrentChunk<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut T>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut T>,
}

impl<T> ChunkBackend<T> for TypedCurrentChunk<T> {
    type ChunkVecType = Vec<TypedArenaChunk<T>>;

    #[inline]
    fn new() -> Self {
        TypedCurrentChunk {
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
        }
    }

    #[inline]
    fn new_vec() -> Self::ChunkVecType {
        vec![]
    }

    #[inline]
    fn can_allocate(&self, len: usize, align: usize) -> bool {
        assert!(mem::size_of::<T>() > 0);
        assert!(mem::align_of::<T>() == align);
        let available_capacity = unsafe { self.end.get().offset_from(self.ptr.get()) };
        assert!(available_capacity >= 0);
        let available_capacity = available_capacity as usize;
        available_capacity >= len
    }

    /// Grows the arena.
    #[inline(never)]
    #[cold]
    fn grow(&self, len: usize, align: usize, chunks: &mut Self::ChunkVecType) {
        assert!(mem::size_of::<T>() > 0);
        assert!(mem::align_of::<T>() == align);
        unsafe {
            let mut new_capacity;
            if let Some(last_chunk) = chunks.last_mut() {
                let used_bytes = self.ptr.get() as usize - last_chunk.start() as usize;
                let currently_used_cap = used_bytes / mem::size_of::<T>();
                last_chunk.entries = currently_used_cap;
                if last_chunk.storage.reserve_in_place(currently_used_cap, len) {
                    self.end.set(last_chunk.end());
                    return;
                } else {
                    new_capacity = last_chunk.storage.capacity();
                    loop {
                        new_capacity = new_capacity.checked_mul(2).unwrap();
                        if new_capacity >= currently_used_cap + len {
                            break;
                        }
                    }
                }
            } else {
                let elem_size = cmp::max(1, mem::size_of::<T>());
                new_capacity = cmp::max(len, PAGE / elem_size);
            }

            let chunk = TypedArenaChunk::new(new_capacity);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    #[inline]
    unsafe fn alloc_raw_slice(&self, len: usize, align: usize) -> *mut T {
        assert!(mem::size_of::<T>() > 0);
        assert!(align == mem::align_of::<T>());

        let ptr = self.ptr.get();
        let end = ptr.add(len);
        assert!(end <= self.end.get());

        self.ptr.set(end);
        ptr
    }

    fn clear(&self, chunks: &mut Self::ChunkVecType) {
        assert!(mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() > 0);

        if let Some(last_chunk) = chunks.last_mut() {
            // Clear the last chunk, which is partially filled.
            unsafe {
                let start = last_chunk.start();
                let len = self.ptr.get().offset_from(start);
                assert!(len >= 0);
                let slice = slice::from_raw_parts_mut(start, len as usize);
                ptr::drop_in_place(slice);
                self.ptr.set(start);
            }

            let len = chunks.len();
            // If `T` is ZST, code below has no effect.
            for mut chunk in chunks.drain(..len-1) {
                unsafe { chunk.destroy(chunk.entries) }
            }
        }
    }
}

struct DroplessCurrentChunk<T> {
    /// A pointer to the next object to be allocated.
    ptr: Cell<*mut u8>,

    /// A pointer to the end of the allocated area. When this pointer is
    /// reached, a new chunk is allocated.
    end: Cell<*mut u8>,

    /// Ensure correct semantics.
    _own: PhantomData<*mut T>,
}

impl<T> ChunkBackend<T> for DroplessCurrentChunk<T> {
    type ChunkVecType = Vec<TypedArenaChunk<u8>>;

    #[inline]
    fn new() -> Self {
        DroplessCurrentChunk {
            // We set both `ptr` and `end` to 0 so that the first call to
            // alloc() will trigger a grow().
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            _own: PhantomData,
        }
    }

    #[inline]
    fn new_vec() -> Self::ChunkVecType {
        vec![]
    }

    #[inline]
    fn can_allocate(&self, len: usize, align: usize) -> bool {
        let len = len * mem::size_of::<T>();
        let ptr = self.ptr.get();
        let ptr = unsafe { ptr.add(ptr.align_offset(align)) };
        let available_capacity = unsafe { self.end.get().offset_from(ptr) };
        assert!(available_capacity >= 0);
        let available_capacity = available_capacity as usize;
        available_capacity >= len
    }

    /// Grows the arena.
    #[inline(never)]
    #[cold]
    fn grow(&self, len: usize, _align: usize, chunks: &mut Self::ChunkVecType) {
        let len = len * mem::size_of::<T>();
        unsafe {
            let mut new_capacity;
            if let Some(last_chunk) = chunks.last_mut() {
                let currently_used_cap = self.ptr.get() as usize - last_chunk.start() as usize;
                last_chunk.entries = currently_used_cap;
                if last_chunk.storage.reserve_in_place(currently_used_cap, len) {
                    self.end.set(last_chunk.end());
                    return;
                } else {
                    new_capacity = last_chunk.storage.capacity();
                    loop {
                        new_capacity = new_capacity.checked_mul(2).unwrap();
                        if new_capacity >= currently_used_cap + len {
                            break;
                        }
                    }
                }
            } else {
                new_capacity = cmp::max(len, PAGE);
            }

            let chunk = TypedArenaChunk::new(new_capacity);
            self.ptr.set(chunk.start());
            self.end.set(chunk.end());
            chunks.push(chunk);
        }
    }

    #[inline]
    unsafe fn alloc_raw_slice(&self, len: usize, align: usize) -> *mut T {
        let len = len * mem::size_of::<T>();
        let ptr = self.ptr.get();
        let ptr = ptr.add(ptr.align_offset(align));
        let end = ptr.add(len);
        assert!(end <= self.end.get());

        self.ptr.set(end);
        ptr as *mut T
    }

    #[inline]
    fn clear(&self, chunks: &mut Self::ChunkVecType) {
        if let Some(last_chunk) = chunks.last_mut() {
            // Clear the last chunk, which is partially filled.
            self.ptr.set(last_chunk.start())
        }
    }
}

struct GenericArena<T, Chunk: ChunkBackend<T>> {
    /// Current chunk for next allocation.
    current: Chunk,

    /// A vector of arena chunks.
    chunks: RefCell<Chunk::ChunkVecType>,

    /// Marker indicating that dropping the arena causes its owned
    /// instances of `T` to be dropped.
    _own: PhantomData<T>,
}

impl<T, Chunk: ChunkBackend<T>> Default for GenericArena<T, Chunk> {
    fn default() -> Self {
        let current = Chunk::new();
        let chunks = RefCell::new(Chunk::new_vec());
        GenericArena {
            current, chunks,
            _own: PhantomData,
        }
    }
}

impl<T, Chunk: ChunkBackend<T>> GenericArena<T, Chunk> {
    #[inline]
    unsafe fn alloc_raw_slice(&self, len: usize) -> *mut T {
        let align = mem::align_of::<T>();
        if !self.current.can_allocate(len, align) {
            self.current.grow(len, align, &mut *self.chunks.borrow_mut());
            debug_assert!(self.current.can_allocate(len, align));
        }

        self.current.alloc_raw_slice(len, align)
    }

    /// Allocates an object in the `GenericArena`, returning a reference to it.
    #[inline]
    pub fn alloc(&self, object: T) -> &mut T {
        unsafe {
            let ptr = self.alloc_raw_slice(1);
            ptr::write(ptr, object);
            &mut *ptr
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

    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        self.current.clear(&mut *self.chunks.borrow_mut())
    }
}

unsafe impl<#[may_dangle] T, Chunk: ChunkBackend<T>> Drop for GenericArena<T, Chunk> {
    fn drop(&mut self) {
        self.clear()
        // RawVec handles deallocation of `last_chunk` and `self.chunks`.
    }
}

pub union TypedArena<T> {
    nop: mem::ManuallyDrop<GenericArena<T, NOPCurrentChunk<T>>>,
    zst: mem::ManuallyDrop<GenericArena<T, ZSTCurrentChunk<T>>>,
    dropless: mem::ManuallyDrop<GenericArena<T, DroplessCurrentChunk<T>>>,
    typed: mem::ManuallyDrop<GenericArena<T, TypedCurrentChunk<T>>>,
}

impl<T> Default for TypedArena<T> {
    fn default() -> Self {
        match (mem::needs_drop::<T>(), mem::size_of::<T>() > 0) {
            (true, true) => Self { typed: mem::ManuallyDrop::new(GenericArena::default()) },
            (true, false) => Self { zst: mem::ManuallyDrop::new(GenericArena::default()) },
            (false, true) => Self { dropless: mem::ManuallyDrop::new(GenericArena::default()) },
            (false, false) => Self { nop: mem::ManuallyDrop::new(GenericArena::default()) },
        }
    }
}

macro_rules! forward_impl {
    ($t:ty, &$self:ident; $call:ident($($e:expr),*)) => {
        match (mem::needs_drop::<$t>(), mem::size_of::<$t>() > 0) {
            (true, true)   => unsafe { &*$self.typed }.$call($($e),*),
            (true, false)  => unsafe { &*$self.zst }.$call($($e),*),
            (false, true)  => unsafe { &*$self.dropless }.$call($($e),*),
            (false, false) => unsafe { &*$self.nop }.$call($($e),*),
        }
    };
    ($t:ty, &mut $self:ident; $call:ident($($e:expr),*)) => {
        match (mem::needs_drop::<$t>(), mem::size_of::<$t>() > 0) {
            (true, true)   => unsafe { &mut*$self.typed }.$call($($e),*),
            (true, false)  => unsafe { &mut*$self.zst }.$call($($e),*),
            (false, true)  => unsafe { &mut*$self.dropless }.$call($($e),*),
            (false, false) => unsafe { &mut*$self.nop }.$call($($e),*),
        }
    };
}

impl<T> TypedArena<T> {
    /// Allocates an object in the `GenericArena`, returning a reference to it.
    #[inline]
    pub fn alloc(&self, object: T) -> &mut T {
        forward_impl!(T, &self; alloc(object))
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
        forward_impl!(T, &self; alloc_slice(slice))
    }

    #[inline]
    pub fn alloc_from_iter<I: IntoIterator<Item = T>>(&self, iter: I) -> &mut [T] {
        forward_impl!(T, &self; alloc_from_iter(iter))
    }

    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        forward_impl!(T, &mut self; clear())
    }
}

unsafe impl<#[may_dangle] T> Drop for TypedArena<T> {
    fn drop(&mut self) {
        match (mem::needs_drop::<T>(), mem::size_of::<T>() > 0) {
            (true, true) =>   unsafe { let _ = mem::ManuallyDrop::drop(&mut self.typed); },
            (true, false) =>  unsafe { let _ = mem::ManuallyDrop::drop(&mut self.zst); },
            (false, true) =>  unsafe { let _ = mem::ManuallyDrop::drop(&mut self.dropless); },
            (false, false) => unsafe { let _ = mem::ManuallyDrop::drop(&mut self.nop); },
        }
    }
}

unsafe impl<T: Send> Send for TypedArena<T> {}

struct TypedArenaChunk<T> {
    /// The raw storage for the arena chunk.
    storage: RawVec<T>,
    /// The number of valid entries in the chunk.
    entries: usize,
}

impl<T> TypedArenaChunk<T> {
    #[inline]
    unsafe fn new(capacity: usize) -> TypedArenaChunk<T> {
        TypedArenaChunk {
            storage: RawVec::with_capacity(capacity),
            entries: 0,
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
                self.start().add(self.storage.capacity())
            }
        }
    }
}

const PAGE: usize = 4096;

#[derive(Default)]
pub struct DroplessArena {
    backend: GenericArena<u8, DroplessCurrentChunk<u8>>,
}

unsafe impl Send for DroplessArena {}

impl DroplessArena {
    #[inline]
    pub fn alloc_raw(&self, bytes: usize, align: usize) -> &mut [u8] {
        if !self.backend.current.can_allocate(bytes, align) {
            self.backend.current.grow(bytes, align, &mut *self.backend.chunks.borrow_mut());
            debug_assert!(self.backend.current.can_allocate(bytes, align));
        }

        unsafe {
            let ptr = self.backend.current.alloc_raw_slice(bytes, align);
            slice::from_raw_parts_mut(ptr, bytes)
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
        assert!(!slice.is_empty());

        let mem = self.alloc_raw(
            slice.len() * mem::size_of::<T>(),
            mem::align_of::<T>()) as *mut _ as *mut T;

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
                    return &mut []
                }
                let size = len.checked_mul(mem::size_of::<T>()).unwrap();
                let mem = self.alloc_raw(size, mem::align_of::<T>()) as *mut _ as *mut T;
                unsafe {
                    self.write_from_iter(iter, len, mem)
                }
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
                        let start_ptr = self.alloc_raw(
                            len * mem::size_of::<T>(),
                            mem::align_of::<T>()
                        ) as *mut _ as *mut T;
                        vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
                        vec.set_len(0);
                        slice::from_raw_parts_mut(start_ptr, len)
                    }
                })
            }
        }
    }
}

pub struct SyncDroplessArena {
    /// Current chunk for next allocation.
    current: WorkerLocal<DroplessCurrentChunk<u8>>,

    /// A vector of arena chunks.
    chunks: Lock<SharedWorkerLocal<Vec<TypedArenaChunk<u8>>>>,
}

impl Default for SyncDroplessArena {
    #[inline]
    fn default() -> SyncDroplessArena {
        SyncDroplessArena {
            current: WorkerLocal::new(|_| DroplessCurrentChunk::new()),
            chunks: Default::default(),
        }
    }
}

impl SyncDroplessArena {
    pub fn in_arena<T: ?Sized>(&self, ptr: *const T) -> bool {
        let ptr = ptr as *const u8 as *mut u8;

        self.chunks.lock().iter().any(|chunks| chunks.iter().any(|chunk| {
            chunk.start() <= ptr && ptr < chunk.end()
        }))
    }

    #[inline]
    pub fn alloc_raw(&self, bytes: usize, align: usize) -> &mut [u8] {
        if !self.current.can_allocate(bytes, align) {
            self.current.grow(bytes, align, &mut *self.chunks.borrow_mut());
            debug_assert!(self.current.can_allocate(bytes, align));
        }

        unsafe {
            let ptr = self.current.alloc_raw_slice(bytes, align);
            slice::from_raw_parts_mut(ptr, bytes)
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
        assert!(!slice.is_empty());

        let mem = self.alloc_raw(
            slice.len() * mem::size_of::<T>(),
            mem::align_of::<T>()) as *mut _ as *mut T;

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
                    return &mut []
                }
                let size = len.checked_mul(mem::size_of::<T>()).unwrap();
                let mem = self.alloc_raw(size, mem::align_of::<T>()) as *mut _ as *mut T;
                unsafe {
                    self.write_from_iter(iter, len, mem)
                }
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
                        let start_ptr = self.alloc_raw(
                            len * mem::size_of::<T>(),
                            mem::align_of::<T>()
                        ) as *mut _ as *mut T;
                        vec.as_ptr().copy_to_nonoverlapping(start_ptr, len);
                        vec.set_len(0);
                        slice::from_raw_parts_mut(start_ptr, len)
                    }
                })
            }
        }
    }

    /// Clears the arena. Deallocates all but the longest chunk which may be reused.
    pub fn clear(&mut self) {
        self.current.clear(&mut *self.chunks.borrow_mut())
    }
}

impl Drop for SyncDroplessArena {
    fn drop(&mut self) {
        self.clear()
        // RawVec handles deallocation of `last_chunk` and `self.chunks`.
    }
}

#[cfg(test)]
mod tests;
