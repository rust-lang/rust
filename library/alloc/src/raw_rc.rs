//! Base implementation for `rc::{Rc, Weak}` and `sync::{Arc, Weak}`.
//!
//! The memory layout of an reference counted allocation is designed so that the reference counts
//! has fixed offsets to the value. In this way, many operations on reference counted pointers can
//! share the same code in order to reduce binary size.
//!
//! This is done by using a layout like the following structure:
//!
//! ```ignore (illustrative)
//! #[repr(C)]
//! struct RcAllocation<T> {
//!     padding: MaybeUninit<[u8; const { align_of::<T>().saturating_sub(size_of::<RefCounts>()) }]>,
//!     ref_counts: RefCounts,
//!     value: T,
//! }
//! ```
//!
//! In this way, for all `T`:
//! `mem::offset_of!(RcAllocation<T>, value) - mem::offset_of!(RcAllocation<T>, ref_counts)`
//! always equal to `size_of::<RefCounts>()`.

use core::alloc::{AllocError, Allocator, Layout, LayoutError};
use core::any::Any;
use core::cell::UnsafeCell;
#[cfg(not(no_global_oom_handling))]
use core::clone::CloneToUninit;
use core::fmt::{self, Debug, Display, Formatter, Pointer};
use core::hash::{Hash, Hasher};
use core::hint;
#[cfg(not(no_global_oom_handling))]
use core::iter::TrustedLen;
use core::marker::{PhantomData, Unsize};
#[cfg(not(no_global_oom_handling))]
use core::mem::ManuallyDrop;
use core::mem::{self, MaybeUninit, SizedTypeProperties};
use core::num::NonZeroUsize;
use core::ops::{CoerceUnsized, DispatchFromDyn};
use core::pin::PinCoerceUnsized;
use core::ptr::{self, NonNull};

#[cfg(not(no_global_oom_handling))]
use crate::alloc;
use crate::alloc::Global;
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

/// A trait for `rc` and `sync` module to inject their concrete implementations of reference
/// count operations.
pub unsafe trait RcOps: Sized {
    /// Increment strong or weak reference pointers. Used by `{RawRc,RawWeak}::clone`.
    ///
    /// # Safety
    ///
    /// - `count` should only be handled by the same `RcOps` implementation.
    /// - The value of `count` should be non-zero.
    unsafe fn increment_ref_count(count: &UnsafeCell<usize>);

    /// Decrement strong or weak reference pointers, returns whether the reference count becomes
    /// zero after decrementing. Used by `{RawRc,RawWeak}::drop`.
    ///
    /// # Safety
    ///
    /// - `count` should only be handled by the same `RcOps` implementation.
    /// - The value of `count` should be non-zero.
    unsafe fn decrement_ref_count(count: &UnsafeCell<usize>) -> bool;

    /// Increment `strong_count` if and only if `strong_count` is non-zero, returns whether
    /// incrementing is performed. Used by `RawWeak::upgrade`.
    ///
    /// # Safety
    ///
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    unsafe fn upgrade(strong_count: &UnsafeCell<usize>) -> bool;

    /// Increment `weak_count`. This is required instead of `increment_ref_count` because `Arc`
    /// requires additional synchronization with `is_unique`.
    ///
    /// # Safety
    ///
    /// - `weak_count` should only be handled by the same `RcOps` implementation.
    /// - Caller should provide a `weak_count` value from a `RawRc` object.
    unsafe fn downgrade(weak_count: &UnsafeCell<usize>);

    /// Decrement `strong_count` if and only if `strong_count` is 1, returns true if decrementing
    /// is performed. Used by `RawRc::try_unwrap`.
    ///
    /// # Safety
    ///
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    unsafe fn lock_strong_count(strong_count: &UnsafeCell<usize>) -> bool;

    /// Set `strong_count` to 1.
    ///
    /// # Safety
    ///
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    /// - `strong_count` is 0.
    unsafe fn unlock_strong_count(strong_count: &UnsafeCell<usize>);

    /// Returns whether both `strong_count` are 1 and `weak_count` is 1. Used by `RawRc::get_mut`.
    ///
    /// # Safety
    ///
    /// - Both `strong_count` and `weak_count` should only be handled by the same `RcOps`
    ///   implementation.
    unsafe fn is_unique(strong_count: &UnsafeCell<usize>, weak_count: &UnsafeCell<usize>) -> bool;

    /// Makes `rc` the sole owner of a value by:
    ///
    /// - If both strong count and weak count are 1, nothing will be done because caller is
    ///   already the sole owner of the value.
    /// - If strong count is 1 and weak count is greater than 1, implementor will first
    ///   decrement both strong count and weak count, then `MakeMut::by_move` be called
    ///   to notify the caller moving is needed in order to make caller the sole owner.
    /// - If strong count is greater than 1, `Make::by_clone` will be called to notify the caller
    ///   cloning is needed in order to make caller the sole owner.
    ///
    /// # Safety
    ///
    /// - The reference counts in `MakeMut` should only be handled by the same `RcOps`
    ///   implementation.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn make_mut<T, A>(make_mut: MakeMut<'_, T, A, Self>)
    where
        T: CloneToUninit + ?Sized,
        A: Allocator;
}

// `Arc` requires atomic reference counting, which requires `usize` reference counters having
// the same alignment as `core::sync::atomic::AtomicUsize`.
macro_rules! define_ref_counts {
    ($($target_pointer_width:literal => $align:literal,)*) => {
        $(
            /// Stores the strong and weak reference counts to a shared value.
            #[cfg(target_pointer_width = $target_pointer_width)]
            #[repr(C, align($align))]
            pub struct RefCounts {
                /// Weak reference count (plus one if there are non-zero strong reference count).
                pub weak: UnsafeCell<usize>,
                /// Strong reference count.
                pub strong: UnsafeCell<usize>,
            }
        )*
    };
}

define_ref_counts! {
    "16" => 2,
    "32" => 4,
    "64" => 8,
}

impl RefCounts {
    /// Creates a `RefCounts` with weak count of `1` and strong count of `strong_count`.
    pub const fn new(strong_cont: usize) -> Self {
        Self { weak: UnsafeCell::new(1), strong: UnsafeCell::new(strong_cont) }
    }
}

/// Describes the allocation of a reference counted value.
struct RcLayout {
    /// The layout of the allocation.
    allocation_layout: Layout,
    /// The offset of the value from beginning of the allocation.
    value_offset_bytes: usize,
}

impl RcLayout {
    const fn from_value_layout(value_layout: Layout) -> Result<Self, LayoutError> {
        match RefCounts::LAYOUT.extend(value_layout) {
            Ok((unaligned_allocation_layout, value_offset_bytes)) => {
                Ok(Self { allocation_layout: unaligned_allocation_layout, value_offset_bytes })
            }
            Err(error) => Err(error),
        }
    }

    /// # Safety
    ///
    /// - `RcLayout::from(value_layout)` must return `Ok(...)`.
    const unsafe fn from_value_layout_unchecked(value_layout: Layout) -> Self {
        match Self::from_value_layout(value_layout) {
            Ok(rc_layout) => rc_layout,
            Err(_) => unsafe { hint::unreachable_unchecked() },
        }
    }

    #[cfg(not(no_global_oom_handling))]
    const fn from_value_ref<T>(value_ref: &T) -> Result<Self, LayoutError>
    where
        T: ?Sized,
    {
        Self::from_value_layout(Layout::for_value(value_ref))
    }

    /// # Safety
    ///
    /// - `value_ptr` points to a value that is contained in a reference counted allocation.
    /// - `value_ptr` contains correct metadata for the memory layout of `T`.
    const unsafe fn from_value_ptr_unchecked<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        unsafe { Self::from_value_layout_unchecked(Layout::for_value_raw(value_ptr.as_ptr())) }
    }

    const fn of<T>() -> Result<Self, LayoutError> {
        Self::from_value_layout(T::LAYOUT)
    }

    #[cfg(not(no_global_oom_handling))]
    const fn of_slice<T>(length: usize) -> Result<Self, LayoutError> {
        match Layout::array::<T>(length) {
            Ok(layout) => Self::from_value_layout(layout),
            Err(error) => Err(error),
        }
    }
}

trait RcLayoutExt {
    const RC_LAYOUT: RcLayout;
}

impl<T> RcLayoutExt for T {
    const RC_LAYOUT: RcLayout = match RcLayout::of::<Self>() {
        Ok(rc_layout) => rc_layout,
        Err(_) => panic!("layout size is too large"),
    };
}

unsafe fn ref_counts_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<RefCounts> {
    const REF_COUNTS_OFFSET_BYTES: usize = RefCounts::LAYOUT.size();

    unsafe { value_ptr.byte_sub(REF_COUNTS_OFFSET_BYTES).cast() }
}

unsafe fn strong_count_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<UnsafeCell<usize>> {
    const STRONG_COUNT_OFFSET_BYTES: usize =
        RefCounts::LAYOUT.size() - mem::offset_of!(RefCounts, strong);

    unsafe { value_ptr.byte_sub(STRONG_COUNT_OFFSET_BYTES).cast() }
}

unsafe fn weak_count_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<UnsafeCell<usize>> {
    const WEAK_COUNT_OFFSET_BYTES: usize =
        RefCounts::LAYOUT.size() - mem::offset_of!(RefCounts, weak);

    unsafe { value_ptr.byte_sub(WEAK_COUNT_OFFSET_BYTES).cast() }
}

unsafe fn init_rc_allocation<const STRONG_COUNT: usize>(
    allocation_ptr: NonNull<[u8]>,
    rc_layout: &RcLayout,
) -> NonNull<()> {
    let allocation_ptr = allocation_ptr.cast::<()>();
    let value_ptr = unsafe { allocation_ptr.byte_add(rc_layout.value_offset_bytes) };
    let ref_counts = const { RefCounts::new(STRONG_COUNT) };

    unsafe { ref_counts_ptr_from_value_ptr(value_ptr).write(ref_counts) };

    value_ptr
}

unsafe fn try_handle_rc_allocation_result<const STRONG_COUNT: usize>(
    allocation_result: Result<NonNull<[u8]>, AllocError>,
    rc_layout: &RcLayout,
) -> Result<NonNull<()>, AllocError> {
    allocation_result.map(|allocation_ptr| unsafe {
        init_rc_allocation::<STRONG_COUNT>(allocation_ptr, rc_layout)
    })
}

fn try_allocate_uninit_for_rc<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: &RcLayout,
) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    unsafe {
        try_handle_rc_allocation_result::<STRONG_COUNT>(
            alloc.allocate(rc_layout.allocation_layout),
            rc_layout,
        )
    }
}

fn try_allocate_zeroed_for_rc<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: &RcLayout,
) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    unsafe {
        try_handle_rc_allocation_result::<STRONG_COUNT>(
            alloc.allocate_zeroed(rc_layout.allocation_layout),
            rc_layout,
        )
    }
}

#[cfg(not(no_global_oom_handling))]
unsafe fn handle_rc_allocation_result<const STRONG_COUNT: usize>(
    allocation_result: Result<NonNull<[u8]>, AllocError>,
    rc_layout: &RcLayout,
) -> NonNull<()> {
    match allocation_result {
        Ok(allocation_ptr) => unsafe {
            init_rc_allocation::<STRONG_COUNT>(allocation_ptr, rc_layout)
        },
        Err(AllocError) => alloc::handle_alloc_error(rc_layout.allocation_layout),
    }
}

#[cfg(not(no_global_oom_handling))]
fn allocate_uninit_for_rc<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: &RcLayout,
) -> NonNull<()>
where
    A: Allocator,
{
    unsafe {
        handle_rc_allocation_result::<STRONG_COUNT>(
            alloc.allocate(rc_layout.allocation_layout),
            rc_layout,
        )
    }
}

#[cfg(not(no_global_oom_handling))]
fn allocate_zeroed_for_rc<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: &RcLayout,
) -> NonNull<()>
where
    A: Allocator,
{
    unsafe {
        handle_rc_allocation_result::<STRONG_COUNT>(
            alloc.allocate_zeroed(rc_layout.allocation_layout),
            rc_layout,
        )
    }
}

/// Allocate a memory block for storing a reference counted value according to `rc_layout` and
/// initialize the value with `f`. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
fn allocate_for_rc_with<A, F, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: &RcLayout,
    f: F,
) -> NonNull<()>
where
    A: Allocator,
    F: FnOnce(NonNull<()>),
{
    struct Guard<'a, A>
    where
        A: Allocator,
    {
        alloc: &'a A,
        ptr: NonNull<()>,
        rc_layout: &'a RcLayout,
    }

    impl<'a, A> Drop for Guard<'a, A>
    where
        A: Allocator,
    {
        fn drop(&mut self) {
            unsafe { deallocate_rc_ptr::<A>(self.alloc, self.ptr, self.rc_layout) };
        }
    }

    let ptr = allocate_uninit_for_rc::<A, STRONG_COUNT>(alloc, &rc_layout);
    let guard = Guard { alloc, ptr, rc_layout };

    f(ptr);

    mem::forget(guard);

    ptr
}

/// # Safety
///
/// - Memory starts with `ptr` is valid to read `size` bytes.
/// - `rc_layout` has enough space for storing a value of `size` bytes.
#[cfg(not(no_global_oom_handling))]
unsafe fn allocate_for_rc_with_bytes<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: &RcLayout,
    ptr: NonNull<()>,
    size: usize,
) -> NonNull<()>
where
    A: Allocator,
{
    allocate_for_rc_with::<A, _, STRONG_COUNT>(alloc, rc_layout, |dst_ptr| unsafe {
        ptr::copy_nonoverlapping::<u8>(ptr.as_ptr().cast(), dst_ptr.as_ptr().cast(), size);
    })
}

#[cfg(not(no_global_oom_handling))]
fn allocate_for_rc_with_value<A, T, const STRONG_COUNT: usize>(alloc: &A, value: &T) -> NonNull<T>
where
    A: Allocator,
    T: ?Sized,
{
    let rc_layout = RcLayout::from_value_ref(value).unwrap();

    unsafe {
        let ptr = allocate_for_rc_with_bytes::<A, STRONG_COUNT>(
            alloc,
            &rc_layout,
            NonNull::from(value).cast(),
            size_of_val(value),
        );

        NonNull::new_unchecked(ptr.as_ptr().with_metadata_of(value))
    }
}

#[cfg(not(no_global_oom_handling))]
unsafe fn allocate_for_rc_with_value_unchecked<A, T, const STRONG_COUNT: usize>(
    alloc: &A,
    value: &T,
) -> NonNull<T>
where
    A: Allocator,
    T: ?Sized,
{
    unsafe {
        let rc_layout = RcLayout::from_value_ptr_unchecked(NonNull::from(value));

        let ptr = allocate_for_rc_with_bytes::<A, STRONG_COUNT>(
            alloc,
            &rc_layout,
            NonNull::from(value).cast(),
            size_of_val(value),
        );

        NonNull::new_unchecked(ptr.as_ptr().with_metadata_of(value))
    }
}

unsafe fn deallocate_rc_ptr<A>(alloc: &A, ptr: NonNull<()>, rc_layout: &RcLayout)
where
    A: Allocator,
{
    unsafe {
        alloc.deallocate(
            ptr.cast().byte_sub(rc_layout.value_offset_bytes),
            rc_layout.allocation_layout,
        );
    }
}

fn is_dangling(ptr: NonNull<()>) -> bool {
    ptr.addr() == NonZeroUsize::MAX
}

struct GuardedWeak<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    weak: &'a mut RawWeak<T, A>,
    _phantom_data: PhantomData<R>,
}

impl<'a, T, A, R> GuardedWeak<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    unsafe fn new(weak: &'a mut RawWeak<T, A>) -> Self {
        Self { weak, _phantom_data: PhantomData }
    }
}

impl<T, A, R> Drop for GuardedWeak<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    fn drop(&mut self) {
        unsafe { self.weak.drop_unchecked::<R>() };
    }
}

struct GuardedRc<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    rc: &'a mut RawRc<T, A>,
    _phantom_data: PhantomData<R>,
}

impl<'a, T, A, R> GuardedRc<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    unsafe fn new(rc: &'a mut RawRc<T, A>) -> Self {
        Self { rc, _phantom_data: PhantomData }
    }
}

impl<T, A, R> Drop for GuardedRc<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    fn drop(&mut self) {
        unsafe { self.rc.drop::<R>() };
    }
}

#[cfg(not(no_global_oom_handling))]
pub struct MakeMut<'a, T, A, R>
where
    T: ?Sized,
{
    rc: &'a mut RawRc<T, A>,
    _phantom_data: PhantomData<R>,
}

#[cfg(not(no_global_oom_handling))]
impl<'a, T, A, R> MakeMut<'a, T, A, R>
where
    T: ?Sized,
{
    pub fn ref_counts(&self) -> &RefCounts {
        self.rc.ref_counts()
    }

    /// # Safety
    ///
    /// - strong count in ref counts has been set to 0.
    pub unsafe fn by_move(self)
    where
        A: Allocator,
        R: RcOps,
    {
        let (ptr_ref, alloc) = self.rc.borrow_raw_parts();
        let old_ptr = *ptr_ref;

        unsafe {
            let mut weak = RawWeak::from_raw_parts(old_ptr, &*alloc);
            let guard = GuardedWeak::<T, &A, R>::new(&mut weak);
            let new_ptr = allocate_for_rc_with_value_unchecked::<A, T, 1>(alloc, old_ptr.as_ref());

            *ptr_ref = new_ptr;

            drop(guard);
        }
    }

    pub fn by_clone(self)
    where
        T: CloneToUninit,
        A: Allocator,
        R: RcOps,
    {
        let (ptr_ref, alloc) = self.rc.borrow_raw_parts();
        let old_ptr = *ptr_ref;

        unsafe {
            let rc_layout = RcLayout::from_value_ptr_unchecked(old_ptr);

            let new_ptr = allocate_for_rc_with::<A, _, 1>(alloc, &rc_layout, |new_ptr| {
                T::clone_to_uninit(old_ptr.as_ref(), new_ptr.as_ptr().cast());
            });

            *ptr_ref = NonNull::new_unchecked(new_ptr.as_ptr().with_metadata_of(old_ptr.as_ptr()));

            RawRc::from_raw_parts(old_ptr, &*alloc).drop::<R>();
        }
    }
}

pub struct RawWeak<T, A>
where
    T: ?Sized,
{
    ptr: NonNull<T>,
    alloc: A,
}

impl<T, A> RawWeak<T, A>
where
    T: ?Sized,
{
    pub const unsafe fn from_raw_parts(ptr: NonNull<T>, alloc: A) -> Self {
        Self { ptr, alloc }
    }

    pub unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        A: Default,
    {
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    pub fn allocator(&self) -> &A {
        &self.alloc
    }

    pub fn as_ptr(&self) -> NonNull<T> {
        self.ptr
    }

    unsafe fn as_ref_unchecked(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }

    unsafe fn assume_init_drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            let guard = GuardedWeak::<T, A, R>::new(self);

            guard.weak.as_ptr().drop_in_place();
        };
    }

    #[cfg(not(no_global_oom_handling))]
    fn borrow_raw_parts(&mut self) -> (&mut NonNull<T>, &mut A) {
        (&mut self.ptr, &mut self.alloc)
    }

    pub unsafe fn cast<U>(self) -> RawWeak<U, A> {
        unsafe { self.cast_with(NonNull::cast) }
    }

    pub unsafe fn cast_with<U, F>(self, f: F) -> RawWeak<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        unsafe { RawWeak::from_raw_parts(f(self.ptr), self.alloc) }
    }

    pub unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<A, R>(ptr: NonNull<()>, alloc: &A) -> A
        where
            A: Clone,
            R: RcOps,
        {
            if !is_dangling(ptr) {
                unsafe { R::increment_ref_count(weak_count_ptr_from_value_ptr(ptr).as_ref()) };
            }

            alloc.clone()
        }

        unsafe {
            let alloc = inner::<A, R>(self.ptr.cast(), &self.alloc);

            Self::from_raw_parts(self.ptr, alloc)
        }
    }

    pub unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        if !is_dangling(self.ptr.cast()) {
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    unsafe fn drop_unchecked<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            if R::decrement_ref_count(self.weak_count_unchecked()) {
                let rc_layout = RcLayout::from_value_ptr_unchecked(self.ptr);

                deallocate_rc_ptr::<A>(&self.alloc, self.ptr.cast(), &rc_layout);
            }
        };
    }

    unsafe fn get_mut_unchecked(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }

    pub fn into_raw(self) -> NonNull<T> {
        self.ptr
    }

    pub fn into_raw_parts(self) -> (NonNull<T>, A) {
        (self.ptr, self.alloc)
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    pub fn ptr_ne(&self, other: &Self) -> bool {
        !ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    #[cfg(not(no_sync))]
    pub fn ref_counts(&self) -> Option<&RefCounts> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.ref_counts_unchecked() })
    }

    unsafe fn ref_counts_unchecked(&self) -> &RefCounts {
        unsafe { ref_counts_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    pub fn strong_count(&self) -> Option<&UnsafeCell<usize>> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.strong_count_unchecked() })
    }

    unsafe fn strong_count_unchecked(&self) -> &UnsafeCell<usize> {
        unsafe { strong_count_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    pub fn weak_count(&self) -> Option<&UnsafeCell<usize>> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.weak_count_unchecked() })
    }

    unsafe fn weak_count_unchecked(&self) -> &UnsafeCell<usize> {
        unsafe { weak_count_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    pub fn upgrade<R>(&self) -> Option<RawRc<T, A>>
    where
        A: Clone,
        R: RcOps,
    {
        if is_dangling(self.ptr.cast()) { None } else { unsafe { self.upgrade_unchecked::<R>() } }
    }

    unsafe fn upgrade_unchecked<R>(&self) -> Option<RawRc<T, A>>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe {
            R::upgrade(self.strong_count_unchecked())
                .then(|| RawRc::from_raw_parts(self.ptr, self.alloc.clone()))
        }
    }
}

impl<T, A> RawWeak<T, A> {
    pub fn new_dangling() -> Self
    where
        A: Default,
    {
        Self::new_dangling_in(A::default())
    }

    pub const fn new_dangling_in(alloc: A) -> Self {
        unsafe { Self::from_raw_parts(NonNull::without_provenance(NonZeroUsize::MAX), alloc) }
    }

    pub fn try_new_uninit<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        Self::try_new_uninit_in::<STRONG_COUNT>(A::default())
    }

    pub fn try_new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        try_allocate_uninit_for_rc::<A, STRONG_COUNT>(&alloc, &T::RC_LAYOUT)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub fn try_new_zeroed<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        Self::try_new_zeroed_in::<STRONG_COUNT>(A::default())
    }

    pub fn try_new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        try_allocate_zeroed_for_rc::<A, STRONG_COUNT>(&alloc, &T::RC_LAYOUT)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        Self::new_uninit_in::<STRONG_COUNT>(A::default())
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe {
            Self::from_raw_parts(
                allocate_uninit_for_rc::<A, STRONG_COUNT>(&alloc, &T::RC_LAYOUT).cast(),
                alloc,
            )
        }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        Self::new_zeroed_in::<STRONG_COUNT>(A::default())
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe {
            Self::from_raw_parts(
                allocate_zeroed_for_rc::<A, STRONG_COUNT>(&alloc, &T::RC_LAYOUT).cast(),
                alloc,
            )
        }
    }

    unsafe fn assume_init_into_inner<R>(mut self) -> T
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            let result = self.ptr.read();

            self.drop_unchecked::<R>();

            result
        }
    }
}

impl<T, A> RawWeak<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    fn allocate_in<F>(length: usize, alloc: A, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(&A, &RcLayout) -> NonNull<()>,
    {
        let rc_layout = RcLayout::of_slice::<T>(length).unwrap();
        let ptr = allocate_fn(&alloc, &rc_layout);

        unsafe { Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast(), length), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::new_uninit_slice_in::<STRONG_COUNT>(length, A::default())
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, allocate_uninit_for_rc::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::new_zeroed_slice_in::<STRONG_COUNT>(length, A::default())
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, allocate_zeroed_for_rc::<A, STRONG_COUNT>)
    }
}

impl<T, U, A> CoerceUnsized<RawWeak<U, A>> for RawWeak<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

impl<T, A> Debug for RawWeak<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("(Weak)")
    }
}

impl<T, A> Default for RawWeak<T, A>
where
    A: Default,
{
    fn default() -> Self {
        Self::new_dangling()
    }
}

impl<T, U> DispatchFromDyn<RawWeak<U, Global>> for RawWeak<T, Global>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

#[repr(transparent)]
pub struct RawRc<T, A>
where
    T: ?Sized,
{
    weak: RawWeak<T, A>,
    _phantom_data: PhantomData<T>,
}

impl<T, A> RawRc<T, A>
where
    T: ?Sized,
{
    pub unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        A: Default,
    {
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    pub unsafe fn from_raw_parts(ptr: NonNull<T>, alloc: A) -> Self {
        unsafe { Self::from_weak(RawWeak::from_raw_parts(ptr, alloc)) }
    }

    unsafe fn from_weak(weak: RawWeak<T, A>) -> Self {
        Self { weak, _phantom_data: PhantomData }
    }

    pub fn allocator(&self) -> &A {
        self.weak.allocator()
    }

    pub fn as_ptr(&self) -> NonNull<T> {
        self.weak.as_ptr()
    }

    #[cfg(not(no_global_oom_handling))]
    fn borrow_raw_parts(&mut self) -> (&mut NonNull<T>, &mut A) {
        self.weak.borrow_raw_parts()
    }

    pub unsafe fn cast<U>(self) -> RawRc<U, A> {
        unsafe { RawRc::from_weak(self.weak.cast()) }
    }

    pub unsafe fn cast_with<U, F>(self, f: F) -> RawRc<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        unsafe { RawRc::from_weak(self.weak.cast_with(f)) }
    }

    pub unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<A, R>(ptr: NonNull<()>, alloc: &A) -> A
        where
            A: Clone,
            R: RcOps,
        {
            unsafe { R::increment_ref_count(strong_count_ptr_from_value_ptr(ptr).as_ref()) };

            alloc.clone()
        }

        unsafe {
            let ptr = self.as_ptr();
            let alloc = inner::<A, R>(ptr.cast(), self.allocator());

            Self::from_raw_parts(ptr, alloc)
        }
    }

    pub unsafe fn decrement_strong_count<R: RcOps>(ptr: NonNull<T>)
    where
        A: Allocator + Default,
    {
        unsafe { Self::decrement_strong_count_in::<R>(ptr, A::default()) };
    }

    pub unsafe fn decrement_strong_count_in<R: RcOps>(ptr: NonNull<T>, alloc: A)
    where
        A: Allocator,
    {
        unsafe { RawRc::from_raw_parts(ptr, alloc).drop::<R>() };
    }

    pub unsafe fn increment_strong_count<R: RcOps>(ptr: NonNull<T>) {
        unsafe { R::increment_ref_count(strong_count_ptr_from_value_ptr(ptr.cast()).as_ref()) };
    }

    pub unsafe fn downgrade<R>(&self) -> RawWeak<T, A>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<A, R>(ptr: NonNull<()>, alloc: &A) -> A
        where
            A: Clone,
            R: RcOps,
        {
            unsafe { R::downgrade(weak_count_ptr_from_value_ptr(ptr).as_ref()) };

            alloc.clone()
        }

        unsafe {
            let ptr = self.as_ptr();
            let alloc = inner::<A, R>(ptr.cast(), self.allocator());

            RawWeak::from_raw_parts(ptr, alloc)
        }
    }

    pub unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            if R::decrement_ref_count(self.strong_count()) {
                self.drop_slow::<R>();
            }
        };
    }

    #[inline(never)]
    unsafe fn drop_slow<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe { self.weak.assume_init_drop::<R>() }
    }

    pub unsafe fn get_mut<R>(&mut self) -> Option<&mut T>
    where
        R: RcOps,
    {
        unsafe {
            R::is_unique(self.strong_count(), self.weak_count()).then(|| self.get_mut_unchecked())
        }
    }

    pub unsafe fn get_mut_unchecked(&mut self) -> &mut T {
        unsafe { self.weak.get_mut_unchecked() }
    }

    pub fn into_raw(self) -> NonNull<T> {
        self.weak.into_raw()
    }

    pub fn into_raw_parts(self) -> (NonNull<T>, A) {
        self.weak.into_raw_parts()
    }

    #[cfg(not(no_global_oom_handling))]
    pub unsafe fn make_mut<R>(&mut self) -> &mut T
    where
        T: CloneToUninit,
        A: Allocator + Clone,
        R: RcOps,
    {
        unsafe {
            R::make_mut(MakeMut { rc: self, _phantom_data: PhantomData });

            self.get_mut_unchecked()
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        RawWeak::ptr_eq(&self.weak, &other.weak)
    }

    pub fn ptr_ne(&self, other: &Self) -> bool {
        RawWeak::ptr_ne(&self.weak, &other.weak)
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn ref_counts(&self) -> &RefCounts {
        unsafe { self.weak.ref_counts_unchecked() }
    }

    pub fn strong_count(&self) -> &UnsafeCell<usize> {
        unsafe { self.weak.strong_count_unchecked() }
    }

    pub fn weak_count(&self) -> &UnsafeCell<usize> {
        unsafe { self.weak.weak_count_unchecked() }
    }
}

impl<T, A> RawRc<T, A> {
    unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        unsafe {
            weak.as_ptr().write(value);

            Self::from_weak(weak)
        }
    }

    pub fn try_new(value: T) -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>()
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    pub fn try_new_in(value: T, alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc)
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<1>(), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit_in::<1>(alloc), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    fn new_with<F>(f: F) -> Self
    where
        A: Allocator + Default,
        F: FnOnce() -> T,
    {
        let alloc = A::default();

        unsafe {
            let ptr = allocate_for_rc_with::<A, _, 1>(&alloc, &T::RC_LAYOUT, |ptr| {
                ptr.cast().write(f());
            });

            Self::from_raw_parts(ptr.cast(), alloc)
        }
    }

    #[cfg(not(no_global_oom_handling))]
    pub unsafe fn new_cyclic<F, R>(data_fn: F) -> Self
    where
        A: Allocator + Default,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RcOps,
    {
        unsafe { Self::new_cyclic_in::<F, R>(data_fn, A::default()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub unsafe fn new_cyclic_in<F, R>(data_fn: F, alloc: A) -> Self
    where
        A: Allocator,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RcOps,
    {
        let mut weak = RawWeak::new_uninit_in::<0>(alloc);
        let guard = unsafe { GuardedWeak::<T, A, R>::new(&mut weak) };
        let data = data_fn(&guard.weak);

        mem::forget(guard);

        unsafe { RawUniqueRc::from_weak_with_value(weak, data).into_rc::<R>() }
    }

    pub unsafe fn into_inner<R>(self) -> Option<T>
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            R::decrement_ref_count(self.strong_count())
                .then(|| self.weak.assume_init_into_inner::<R>())
        }
    }

    pub unsafe fn try_unwrap<R>(self) -> Result<T, RawRc<T, A>>
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            if R::lock_strong_count(self.strong_count()) {
                Ok(self.weak.assume_init_into_inner::<R>())
            } else {
                Err(self)
            }
        }
    }

    pub unsafe fn unwrap_or_clone<R>(self) -> T
    where
        T: Clone,
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            self.try_unwrap::<R>().unwrap_or_else(|mut rc| {
                let guard = GuardedRc::<T, A, R>::new(&mut rc);

                T::clone(guard.rc.as_ref())
            })
        }
    }
}

impl<T, A> RawRc<MaybeUninit<T>, A> {
    pub fn try_new_uninit() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub fn try_new_uninit_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub fn try_new_zeroed() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_zeroed::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub fn try_new_zeroed_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_zeroed_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_in::<1>(alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_in::<1>(alloc)) }
    }

    pub unsafe fn assume_init(self) -> RawRc<T, A> {
        unsafe { self.cast() }
    }
}

impl<T, A> RawRc<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_iter_exact<I>(iter: I, length: usize) -> Self
    where
        A: Allocator + Default,
        I: Iterator<Item = T>,
    {
        struct Guard<T> {
            head: NonNull<T>,
            tail: NonNull<T>,
        }

        impl<T> Drop for Guard<T> {
            fn drop(&mut self) {
                unsafe {
                    let length = self.tail.sub_ptr(self.head);

                    NonNull::<[T]>::slice_from_raw_parts(self.head, length).drop_in_place();
                }
            }
        }

        let rc_layout = RcLayout::of_slice::<T>(length).unwrap();
        let alloc = A::default();

        unsafe {
            let ptr = allocate_for_rc_with::<A, _, 1>(&alloc, &rc_layout, |ptr| {
                let ptr = ptr.cast::<T>();
                let mut guard = Guard::<T> { head: ptr, tail: ptr };

                iter.for_each(|value| {
                    guard.tail.write(value);
                    guard.tail = guard.tail.add(1);
                });

                mem::forget(guard);
            });

            Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast::<T>(), length), alloc)
        }
    }
}

impl<T, A> RawRc<[MaybeUninit<T>], A> {
    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_uninit_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice_in::<1>(length, alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_zeroed_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice_in::<1>(length, alloc)) }
    }

    pub unsafe fn assume_init(self) -> RawRc<[T], A> {
        unsafe { self.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

impl<A> RawRc<dyn Any, A> {
    pub fn downcast<T>(self) -> Result<RawRc<T, A>, Self>
    where
        T: Any,
    {
        if self.as_ref().is::<T>() { Ok(unsafe { self.downcast_unchecked() }) } else { Err(self) }
    }

    pub unsafe fn downcast_unchecked<T>(self) -> RawRc<T, A>
    where
        T: Any,
    {
        unsafe { self.cast() }
    }
}

#[cfg(not(no_sync))]
impl<A> RawRc<dyn Any + Send + Sync, A> {
    pub fn downcast<T>(self) -> Result<RawRc<T, A>, Self>
    where
        T: Any,
    {
        if self.as_ref().is::<T>() { Ok(unsafe { self.downcast_unchecked() }) } else { Err(self) }
    }

    pub unsafe fn downcast_unchecked<T>(self) -> RawRc<T, A>
    where
        T: Any,
    {
        unsafe { self.cast() }
    }
}

impl<T, A> AsRef<T> for RawRc<T, A>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { self.weak.as_ref_unchecked() }
    }
}

impl<T, U, A> CoerceUnsized<RawRc<U, A>> for RawRc<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

impl<T, A> Debug for RawRc<T, A>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_ref(), f)
    }
}

impl<T, A> Display for RawRc<T, A>
where
    T: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.as_ref(), f)
    }
}

impl<T, U> DispatchFromDyn<RawRc<U, Global>> for RawRc<T, Global>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}

impl<T, A> Pointer for RawRc<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Pointer::fmt(&self.as_ptr(), f)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> Default for RawRc<T, A>
where
    T: Default,
    A: Allocator + Default,
{
    fn default() -> Self {
        Self::new_with(T::default)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> Default for RawRc<[T], A>
where
    A: Allocator + Default,
{
    fn default() -> Self {
        RawRc::new([])
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> Default for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn default() -> Self {
        RawRc::from("")
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<T> for RawRc<T, A>
where
    A: Allocator + Default,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<Box<T, A>> for RawRc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
    fn from(value: Box<T, A>) -> Self {
        let value_ref = &*value;
        let alloc_ref = Box::allocator(&value);

        unsafe {
            let rc_ptr = allocate_for_rc_with_value::<A, T, 1>(alloc_ref, value_ref);
            let (box_ptr, alloc) = Box::into_raw_with_allocator(value);

            drop(Box::from_raw_in(box_ptr as *mut ManuallyDrop<T>, &alloc));

            Self::from_raw_parts(rc_ptr, alloc)
        }
    }
}

#[cfg(not(no_global_oom_handling))]
trait SpecRawRcFromSlice<T> {
    fn spec_from_slice(slice: &[T]) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> SpecRawRcFromSlice<T> for RawRc<[T], A>
where
    A: Allocator + Default,
    T: Clone,
{
    default fn spec_from_slice(slice: &[T]) -> Self {
        unsafe { Self::from_iter_exact(slice.iter().cloned(), slice.len()) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> SpecRawRcFromSlice<T> for RawRc<[T], A>
where
    A: Allocator + Default,
    T: Copy,
{
    fn spec_from_slice(slice: &[T]) -> Self {
        let alloc = A::default();
        let ptr = allocate_for_rc_with_value::<A, [T], 1>(&alloc, slice);

        unsafe { Self::from_raw_parts(ptr, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<&[T]> for RawRc<[T], A>
where
    A: Allocator + Default,
    T: Clone,
{
    fn from(value: &[T]) -> Self {
        Self::spec_from_slice(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<&mut [T]> for RawRc<[T], A>
where
    A: Allocator + Default,
    T: Clone,
{
    fn from(value: &mut [T]) -> Self {
        Self::from(&*value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> From<&str> for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn from(value: &str) -> Self {
        let rc_of_bytes = RawRc::<[u8], A>::from(value.as_bytes());

        unsafe { rc_of_bytes.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> From<&mut str> for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn from(value: &mut str) -> Self {
        Self::from(&*value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl From<String> for RawRc<str, Global> {
    fn from(value: String) -> Self {
        let rc_of_bytes = RawRc::<[u8], Global>::from(value.into_bytes());

        unsafe { rc_of_bytes.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

impl<A> From<RawRc<str, A>> for RawRc<[u8], A> {
    fn from(value: RawRc<str, A>) -> Self {
        unsafe { value.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, const N: usize, A> From<[T; N]> for RawRc<[T], A>
where
    A: Allocator + Default,
{
    fn from(value: [T; N]) -> Self {
        RawRc::new(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<Vec<T, A>> for RawRc<[T], A>
where
    A: Allocator,
{
    fn from(value: Vec<T, A>) -> Self {
        let value_ref = &*value;
        let alloc_ref = value.allocator();
        let rc_ptr = allocate_for_rc_with_value::<A, [T], 1>(alloc_ref, value_ref);
        let (vec_ptr, _length, capacity, alloc) = value.into_raw_parts_with_alloc();

        unsafe {
            drop(Vec::from_raw_parts_in(vec_ptr, 0, capacity, &alloc));

            Self::from_raw_parts(rc_ptr, alloc)
        }
    }
}

impl<T, const N: usize, A> TryFrom<RawRc<[T], A>> for RawRc<[T; N], A> {
    type Error = RawRc<[T], A>;

    fn try_from(value: RawRc<[T], A>) -> Result<Self, Self::Error> {
        if value.as_ref().len() == N { Ok(unsafe { value.cast() }) } else { Err(value) }
    }
}

#[cfg(not(no_global_oom_handling))]
trait SpecRawRcFromIter<I> {
    fn spec_from_iter(iter: I) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<I> SpecRawRcFromIter<I> for RawRc<[I::Item], Global>
where
    I: Iterator,
{
    default fn spec_from_iter(iter: I) -> Self {
        Self::from(iter.collect::<Vec<_>>())
    }
}

#[cfg(not(no_global_oom_handling))]
impl<I> SpecRawRcFromIter<I> for RawRc<[I::Item], Global>
where
    I: TrustedLen,
{
    fn spec_from_iter(iter: I) -> Self {
        // This is the case for a `TrustedLen` iterator.

        if let (low, Some(high)) = iter.size_hint() {
            debug_assert_eq!(
                low,
                high,
                "TrustedLen iterator's size hint is not exact: {:?}",
                (low, high)
            );

            // SAFETY: We need to ensure that the iterator has an exact length and we have.
            unsafe { Self::from_iter_exact(iter, low) }
        } else {
            // TrustedLen contract guarantees that `upper_bound == None` implies an iterator
            // length exceeding `usize::MAX`.
            // The default implementation would collect into a vec which would panic.
            // Thus we panic here immediately without invoking `Vec` code.
            panic!("capacity overflow");
        }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T> FromIterator<T> for RawRc<[T], Global> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::spec_from_iter(iter.into_iter())
    }
}

impl<T, A> Hash for RawRc<T, A>
where
    T: Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        T::hash(self.as_ref(), state);
    }
}

// Hack to allow specializing on `Eq` even though `Eq` has a method.
#[rustc_unsafe_specialization_marker]
trait MarkerEq: PartialEq<Self> {}

impl<T> MarkerEq for T where T: Eq {}

trait SpecPartialEq {
    fn spec_eq(&self, other: &Self) -> bool;
    fn spec_ne(&self, other: &Self) -> bool;
}

impl<T, A> SpecPartialEq for RawRc<T, A>
where
    T: PartialEq + ?Sized,
{
    default fn spec_eq(&self, other: &Self) -> bool {
        T::eq(self.as_ref(), other.as_ref())
    }

    default fn spec_ne(&self, other: &Self) -> bool {
        T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> SpecPartialEq for RawRc<T, A>
where
    T: MarkerEq + ?Sized,
{
    fn spec_eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other) || T::eq(self.as_ref(), other.as_ref())
    }

    fn spec_ne(&self, other: &Self) -> bool {
        Self::ptr_ne(self, other) && T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialEq for RawRc<T, A>
where
    T: PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        Self::spec_eq(self, other)
    }

    fn ne(&self, other: &Self) -> bool {
        Self::spec_ne(self, other)
    }
}

impl<T, A> Eq for RawRc<T, A> where T: Eq + ?Sized {}

impl<T, A> PartialOrd for RawRc<T, A>
where
    T: PartialOrd + ?Sized,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        T::partial_cmp(self.as_ref(), other.as_ref())
    }

    fn lt(&self, other: &Self) -> bool {
        T::lt(self.as_ref(), other.as_ref())
    }

    fn le(&self, other: &Self) -> bool {
        T::le(self.as_ref(), other.as_ref())
    }

    fn gt(&self, other: &Self) -> bool {
        T::gt(self.as_ref(), other.as_ref())
    }

    fn ge(&self, other: &Self) -> bool {
        T::ge(self.as_ref(), other.as_ref())
    }
}

impl<T, A> Ord for RawRc<T, A>
where
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        T::cmp(self.as_ref(), other.as_ref())
    }
}

unsafe impl<T, A> PinCoerceUnsized for RawRc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
}

#[repr(transparent)]
pub struct RawUniqueRc<T, A>
where
    T: ?Sized,
{
    weak: RawWeak<T, A>,
    // Define the ownership of `T` for drop-check
    _marker: PhantomData<T>,
    // Invariance is necessary for soundness: once other `RawWeak`
    // references exist, we already have a form of shared mutability!
    _marker2: PhantomData<*mut T>,
}

impl<T, A> RawUniqueRc<T, A>
where
    T: ?Sized,
{
    pub unsafe fn downgrade<R>(&self) -> RawWeak<T, A>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe { self.weak.clone::<R>() }
    }

    pub unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe { self.weak.assume_init_drop::<R>() };
    }

    pub unsafe fn into_rc<R>(self) -> RawRc<T, A>
    where
        R: RcOps,
    {
        unsafe {
            R::unlock_strong_count(self.weak.strong_count_unchecked());

            RawRc::from_weak(self.weak)
        }
    }
}

impl<T, A> RawUniqueRc<T, A> {
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        unsafe { weak.as_ptr().write(value) };

        Self { weak, _marker: PhantomData, _marker2: PhantomData }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<0>(), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit_in::<0>(alloc), value) }
    }
}

impl<T, A> AsRef<T> for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { self.weak.as_ref_unchecked() }
    }
}

impl<T, A> AsMut<T> for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.weak.get_mut_unchecked() }
    }
}

impl<T, U, A> CoerceUnsized<RawUniqueRc<U, A>> for RawUniqueRc<T, A>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
    A: Allocator,
{
}

impl<T, A> Debug for RawUniqueRc<T, A>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_ref(), f)
    }
}

impl<T, U> DispatchFromDyn<RawUniqueRc<U, Global>> for RawUniqueRc<T, Global>
where
    T: ?Sized + Unsize<U>,
    U: ?Sized,
{
}
