//! A contiguous push-only array type with lock-free reads.

use crate::{
    collect::{self, Pin},
    scopeguard::guard,
};
use core::ptr::NonNull;
use parking_lot::{Mutex, MutexGuard};
use std::{
    alloc::{Allocator, Global, Layout, LayoutError, handle_alloc_error},
    cell::UnsafeCell,
    intrinsics::unlikely,
    iter::FromIterator,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicPtr, Ordering},
};
use std::{
    cmp,
    ptr::slice_from_raw_parts,
    sync::{Arc, atomic::AtomicUsize},
};

mod code;
mod tests;

/// A handle to a [SyncPushVec] with read access.
///
/// It is acquired either by a pin, or by exclusive access to the vector.
pub struct Read<'a, T> {
    table: &'a SyncPushVec<T>,
}

impl<T> Copy for Read<'_, T> {}
impl<T> Clone for Read<'_, T> {
    fn clone(&self) -> Self {
        Self { table: self.table }
    }
}

/// A handle to a [SyncPushVec] with write access.
pub struct Write<'a, T> {
    table: &'a SyncPushVec<T>,
}

/// A handle to a [SyncPushVec] with write access protected by a lock.
pub struct LockedWrite<'a, T> {
    table: Write<'a, T>,
    _guard: MutexGuard<'a, ()>,
}

impl<'a, T> Deref for LockedWrite<'a, T> {
    type Target = Write<'a, T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.table
    }
}

impl<'a, T> DerefMut for LockedWrite<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.table
    }
}

/// A contiguous push-only array type with lock-free reads.
pub struct SyncPushVec<T> {
    current: AtomicPtr<TableInfo>,

    lock: Mutex<()>,

    old: UnsafeCell<Vec<Arc<DestroyTable<T>>>>,

    // Tell dropck that we own instances of T.
    marker: PhantomData<T>,
}

struct TableInfo {
    items: AtomicUsize,
    capacity: usize,
}

#[repr(transparent)]
struct TableRef<T> {
    data: NonNull<TableInfo>,

    marker: PhantomData<*mut T>,
}

impl<T> Copy for TableRef<T> {}
impl<T> Clone for TableRef<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            marker: self.marker,
        }
    }
}

impl<T> TableRef<T> {
    #[inline]
    fn empty() -> Self {
        if cfg!(debug_assertions) {
            let real = Self::layout(0).unwrap().0;
            let dummy = Layout::new::<TableInfo>().align_to(real.align()).unwrap();
            debug_assert_eq!(real, dummy);
        }

        #[repr(C, align(64))]
        struct EmptyTable {
            info: TableInfo,
        }

        static EMPTY: EmptyTable = EmptyTable {
            info: TableInfo {
                capacity: 0,
                items: AtomicUsize::new(0),
            },
        };

        Self {
            data: unsafe {
                NonNull::new_unchecked(&EMPTY.info as *const TableInfo as *mut TableInfo)
            },
            marker: PhantomData,
        }
    }

    #[inline]
    fn layout(capacity: usize) -> Result<(Layout, usize), LayoutError> {
        let data = Layout::new::<T>().repeat(capacity)?.0;
        let info = Layout::new::<TableInfo>();
        data.extend(info)
    }

    #[inline]
    fn allocate(capacity: usize) -> Self {
        let (layout, info_offset) = Self::layout(capacity).expect("capacity overflow");

        let ptr: NonNull<u8> = Global
            .allocate(layout)
            .map(|ptr| ptr.cast())
            .unwrap_or_else(|_| handle_alloc_error(layout));

        let info =
            unsafe { NonNull::new_unchecked(ptr.as_ptr().add(info_offset) as *mut TableInfo) };

        let mut result = Self {
            data: info,
            marker: PhantomData,
        };

        unsafe {
            *result.info_mut() = TableInfo {
                capacity,
                items: AtomicUsize::new(0),
            };
        }

        result
    }

    #[inline]
    unsafe fn free(self) {
        unsafe {
            let items = self.info().items.load(Ordering::Relaxed);
            if items > 0 {
                if mem::needs_drop::<T>() {
                    for i in 0..items {
                        self.data(i).drop_in_place();
                    }
                }

                let (layout, info_offset) = Self::layout(self.info().capacity).unwrap_unchecked();

                Global.deallocate(
                    NonNull::new_unchecked((self.data.as_ptr() as *mut u8).sub(info_offset)),
                    layout,
                )
            }
        }
    }

    fn from_maybe_empty_iter<I: Iterator<Item = T>, const CHECK_LEN: bool>(
        iter: I,
        iter_size: usize,
        capacity: usize,
    ) -> TableRef<T> {
        if iter_size == 0 {
            TableRef::empty()
        } else {
            let capacity = cmp::max(iter_size, capacity);
            unsafe { TableRef::from_iter::<_, CHECK_LEN>(iter, capacity) }
        }
    }

    /// Allocates a new table and fills it with the content of an iterator
    unsafe fn from_iter<I: Iterator<Item = T>, const CHECK_LEN: bool>(
        iter: I,
        new_capacity: usize,
    ) -> TableRef<T> {
        unsafe {
            debug_assert!(new_capacity > 0);

            let mut new_table = TableRef::<T>::allocate(new_capacity);

            let mut guard = guard(Some(new_table), |new_table| {
                new_table.map(|new_table| new_table.free());
            });

            // Copy all elements to the new table.
            for (index, item) in iter.enumerate() {
                debug_assert!(index < new_capacity);
                if CHECK_LEN && index >= new_capacity {
                    break;
                }

                new_table.first().add(index).write(item);

                // Write items per iteration in case `next` on the iterator panics.

                *new_table.info_mut().items.get_mut() = index + 1;
            }

            *guard = None;

            new_table
        }
    }

    unsafe fn info(&self) -> &TableInfo {
        unsafe { self.data.as_ref() }
    }

    unsafe fn info_mut(&mut self) -> &mut TableInfo {
        unsafe { self.data.as_mut() }
    }

    #[inline]
    unsafe fn first(&self) -> *mut T {
        unsafe { (self.data.as_ptr() as *mut T).sub(self.info().capacity) }
    }

    /// Returns a pointer to an element in the table.
    #[inline]
    unsafe fn slice(&self) -> *const [T] {
        unsafe {
            let items = self.info().items.load(Ordering::Acquire);
            let base = if items == 0 && mem::align_of::<T>() > 64 {
                // Need a special case here since our empty allocation isn't aligned to T.
                // It only has an alignment of 64.
                mem::align_of::<T>() as *const T
            } else {
                self.first() as *const T
            };
            slice_from_raw_parts(base, items)
        }
    }

    /// Returns a pointer to an element in the table.
    #[inline]
    unsafe fn data(&self, index: usize) -> *mut T {
        unsafe {
            debug_assert!(index < self.info().items.load(Ordering::Acquire));

            self.first().add(index)
        }
    }
}

impl<T: Clone> TableRef<T> {
    /// Allocates a new table of a different size and moves the contents of the
    /// current table into it.
    unsafe fn clone(&self, new_capacity: usize) -> TableRef<T> {
        unsafe {
            debug_assert!(new_capacity >= self.info().capacity);

            TableRef::from_iter::<_, false>((*self.slice()).iter().cloned(), new_capacity)
        }
    }
}

struct DestroyTable<T> {
    table: TableRef<T>,
    lock: Mutex<bool>,
}

unsafe impl<T> Sync for DestroyTable<T> {}
unsafe impl<T: Send> Send for DestroyTable<T> {}

impl<T> DestroyTable<T> {
    unsafe fn run(&self) {
        unsafe {
            let mut status = self.lock.lock();
            if !*status {
                *status = true;
                self.table.free();
            }
        }
    }
}

unsafe impl<#[may_dangle] T> Drop for SyncPushVec<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.current().free();
            for table in self.old.get_mut() {
                table.run();
            }
        }
    }
}

unsafe impl<T: Send> Send for SyncPushVec<T> {}
unsafe impl<T: Sync> Sync for SyncPushVec<T> {}

impl<T> Default for SyncPushVec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SyncPushVec<T> {
    /// Constructs a new, empty vector with zero capacity.
    ///
    /// The vector will not allocate until elements are pushed onto it.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Constructs a new, empty vector with the specified capacity.
    ///
    /// The vector will be able to hold exactly `capacity` elements without reallocating. If `capacity` is 0, the vector will not allocate.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            current: AtomicPtr::new(
                if capacity > 0 {
                    TableRef::<T>::allocate(capacity)
                } else {
                    TableRef::empty()
                }
                .data
                .as_ptr(),
            ),
            old: UnsafeCell::new(Vec::new()),
            marker: PhantomData,
            lock: Mutex::new(()),
        }
    }

    /// Gets a reference to the underlying mutex that protects writes.
    #[inline]
    pub fn mutex(&self) -> &Mutex<()> {
        &self.lock
    }

    /// Creates a [Read] handle from a pinned region.
    ///
    /// Use [crate::collect::pin] to get a `Pin` instance.
    #[inline]
    pub fn read<'a>(&'a self, pin: Pin<'a>) -> Read<'a, T> {
        let _pin = pin;
        Read { table: self }
    }

    /// Creates a [Write] handle without checking for exclusive access.
    ///
    /// # Safety
    /// It's up to the caller to ensure only one thread writes to the vector at a time.
    #[inline]
    pub unsafe fn unsafe_write(&self) -> Write<'_, T> {
        Write { table: self }
    }

    /// Creates a [Write] handle from a mutable reference.
    #[inline]
    pub fn write(&mut self) -> Write<'_, T> {
        Write { table: self }
    }

    /// Creates a [LockedWrite] handle by taking the underlying mutex that protects writes.
    #[inline]
    pub fn lock(&self) -> LockedWrite<'_, T> {
        LockedWrite {
            table: Write { table: self },
            _guard: self.lock.lock(),
        }
    }

    /// Creates a [LockedWrite] handle from a guard protecting the underlying mutex that protects writes.
    #[inline]
    pub fn lock_from_guard<'a>(&'a self, guard: MutexGuard<'a, ()>) -> LockedWrite<'a, T> {
        // Verify that we are target of the guard
        assert_eq!(
            &self.lock as *const _,
            MutexGuard::mutex(&guard) as *const _
        );

        LockedWrite {
            table: Write { table: self },
            _guard: guard,
        }
    }

    /// Extracts a mutable slice of the entire vector.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { &mut *(self.current().slice() as *mut [T]) }
    }

    #[inline]
    fn current(&self) -> TableRef<T> {
        TableRef {
            data: unsafe { NonNull::new_unchecked(self.current.load(Ordering::Acquire)) },
            marker: PhantomData,
        }
    }
}

impl<'a, T> Read<'a, T> {
    /// Returns the number of elements the map can hold without reallocating.
    #[inline]
    pub fn capacity(self) -> usize {
        unsafe { self.table.current().info().capacity }
    }

    /// Returns the number of elements in the table.
    #[inline]
    pub fn len(self) -> usize {
        unsafe { self.table.current().info().items.load(Ordering::Acquire) }
    }

    /// Extracts a slice containing the entire vector.
    #[inline]
    pub fn as_slice(self) -> &'a [T] {
        let table = self.table.current();
        unsafe { &*table.slice() }
    }
}

impl<T> Write<'_, T> {
    /// Creates a [Read] handle which gives access to read operations.
    #[inline]
    pub fn read(&self) -> Read<'_, T> {
        Read { table: self.table }
    }
}

impl<'a, T: Send + Clone> Write<'a, T> {
    /// Inserts a new element into the end of the table, and returns a refernce to it along
    /// with its index.
    #[inline]
    pub fn push(&mut self, value: T) -> (&'a T, usize) {
        let mut table = self.table.current();
        unsafe {
            let items = table.info().items.load(Ordering::Relaxed);

            if unlikely(items == table.info().capacity) {
                table = self.expand_by_one();
            }

            let result = table.first().add(items);

            result.write(value);

            table.info().items.store(items + 1, Ordering::Release);

            (&*result, items)
        }
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given vector. The collection may reserve more space to avoid
    /// frequent reallocations. Does nothing if the capacity is already sufficient.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let table = self.table.current();
        unsafe {
            let required = table
                .info()
                .items
                .load(Ordering::Relaxed)
                .checked_add(additional)
                .expect("capacity overflow");

            if table.info().capacity < required {
                self.expand_by(additional);
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn expand_by_one(&mut self) -> TableRef<T> {
        self.expand_by(1)
    }

    // Tiny Vecs are dumb. Skip to:
    // - 8 if the element size is 1, because any heap allocators is likely
    //   to round up a request of less than 8 bytes to at least 8 bytes.
    // - 4 if elements are moderate-sized (<= 1 KiB).
    // - 1 otherwise, to avoid wasting too much space for very short Vecs.
    const MIN_NON_ZERO_CAP: usize = if mem::size_of::<T>() == 1 {
        8
    } else if mem::size_of::<T>() <= 1024 {
        4
    } else {
        1
    };

    fn expand_by(&mut self, additional: usize) -> TableRef<T> {
        let table = self.table.current();

        let items = unsafe { table.info().items.load(Ordering::Relaxed) };
        let capacity = unsafe { table.info().capacity };

        // Avoid `Option::ok_or_else` because it bloats LLVM IR.
        let required_cap = match items.checked_add(additional) {
            Some(required_cap) => required_cap,
            None => panic!("capacity overflow"),
        };

        // This guarantees exponential growth. The doubling cannot overflow
        // because `cap <= isize::MAX` and the type of `cap` is `usize`.
        let cap = cmp::max(capacity * 2, required_cap);
        let cap = cmp::max(Self::MIN_NON_ZERO_CAP, cap);

        let new_table = unsafe { table.clone(cap) };

        self.replace_table(new_table);

        new_table
    }
}

impl<T: Send> Write<'_, T> {
    fn replace_table(&mut self, new_table: TableRef<T>) {
        let table = self.table.current();

        self.table
            .current
            .store(new_table.data.as_ptr(), Ordering::Release);

        let destroy = Arc::new(DestroyTable {
            table,
            lock: Mutex::new(false),
        });

        unsafe {
            (*self.table.old.get()).push(destroy.clone());

            collect::defer_unchecked(move || destroy.run());
        }
    }

    /// Replaces the content of the vector with the content of the iterator.
    /// `capacity` specifies the new capacity if it's greater than the length of the iterator.
    #[inline]
    pub fn replace<I: IntoIterator<Item = T>>(&mut self, iter: I, capacity: usize) {
        let iter = iter.into_iter();

        let table = if let Some(max) = iter.size_hint().1 {
            TableRef::from_maybe_empty_iter::<_, true>(iter, max, capacity)
        } else {
            let elements: Vec<_> = iter.collect();
            let len = elements.len();
            TableRef::from_maybe_empty_iter::<_, false>(elements.into_iter(), len, capacity)
        };

        self.replace_table(table);
    }
}

impl<T: Clone + Send> Extend<T> for Write<'_, T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        iter.for_each(|v| {
            self.push(v);
        });
    }

    #[inline]
    fn extend_one(&mut self, item: T) {
        self.push(item);
    }

    #[inline]
    fn extend_reserve(&mut self, additional: usize) {
        self.reserve(additional);
    }
}

impl<T: Clone + Send> FromIterator<T> for SyncPushVec<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut map = Self::with_capacity(iter.size_hint().0);
        let mut write = map.write();
        iter.for_each(|v| {
            write.push(v);
        });
        map
    }
}
