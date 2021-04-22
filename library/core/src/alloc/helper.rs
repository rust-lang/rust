#![unstable(feature = "allocator_api_internals", issue = "none")]
#![doc(hidden)]

use crate::{
    alloc::{AllocError, Allocator, Layout},
    fmt,
    marker::PhantomData,
    mem,
    ptr::NonNull,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocInit {
    /// The contents of the new memory are uninitialized.
    Uninitialized,
    /// The new memory is guaranteed to be zeroed.
    Zeroed,
}

/// An allocator that requests some extra memory from the parent allocator for storing a prefix and/or a suffix.
///
/// The alignment of the memory block is the maximum of the alignment of `Prefix` and the requested
/// alignment. This may introduce an unused padding between `Prefix` and the returned memory.
///
/// To get a pointer to the prefix, [`prefix()`] may be called.
///
/// [`prefix()`]: Self::prefix
///
/// Consider
///
/// ```rust,ignore (not real code)
/// #[repr(C)]
/// struct Struct {
///     t: T,
///     data: Data,
/// }
/// ```
///
/// where `Data` is a type with layout `layout`.
///
/// When this allocator creates an allocation for layout `layout`, the pointer can be
/// offset by `-offsetof(Struct, data)` and the resulting pointer points is an allocation
/// of `A` for `Layout::new::<Struct>()`.
pub struct PrefixAllocator<Alloc, Prefix = ()> {
    /// The parent allocator to be used as backend
    pub parent: Alloc,
    _prefix: PhantomData<*const Prefix>,
}

impl<Alloc: fmt::Debug, Prefix> fmt::Debug for PrefixAllocator<Alloc, Prefix> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefixAllocator").field("parent", &self.parent).finish()
    }
}

impl<Alloc: Default, Prefix> Default for PrefixAllocator<Alloc, Prefix> {
    fn default() -> Self {
        Self::new(Alloc::default())
    }
}

impl<Alloc: Clone, Prefix> Clone for PrefixAllocator<Alloc, Prefix> {
    fn clone(&self) -> Self {
        Self::new(self.parent.clone())
    }
}

impl<Alloc: Copy, Prefix> Copy for PrefixAllocator<Alloc, Prefix> {}

impl<Alloc: PartialEq, Prefix> PartialEq for PrefixAllocator<Alloc, Prefix> {
    fn eq(&self, other: &Self) -> bool {
        self.parent.eq(&other.parent)
    }
}

impl<Alloc: Eq, Prefix> Eq for PrefixAllocator<Alloc, Prefix> {}

unsafe impl<Alloc: Send, Prefix> Send for PrefixAllocator<Alloc, Prefix> {}
unsafe impl<Alloc: Sync, Prefix> Sync for PrefixAllocator<Alloc, Prefix> {}
impl<Alloc: Unpin, Prefix> Unpin for PrefixAllocator<Alloc, Prefix> {}

impl<Alloc, Prefix> PrefixAllocator<Alloc, Prefix> {
    pub const fn new(parent: Alloc) -> Self {
        Self { parent, _prefix: PhantomData }
    }

    /// Returns the offset between the `Prefix` and the stored data.
    #[inline]
    pub fn prefix_offset(layout: Layout) -> usize {
        let prefix_layout = Layout::new::<Prefix>();
        prefix_layout.size() + prefix_layout.padding_needed_for(layout.align())
    }

    /// Returns a pointer to the prefix.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory *[currently allocated]* via this allocator, and
    /// * `ptr` must point to (and have valid metadata for) a previously valid instance of `T`,
    ///   but the `T` is allowed to be dropped.
    ///
    /// [currently allocated]: https://doc.rust-lang.org/nightly/core/alloc/trait.AllocRef.html#currently-allocated-memory
    #[inline]
    pub unsafe fn prefix<T: ?Sized>(ptr: NonNull<T>) -> NonNull<Prefix> {
        let prefix_layout = Layout::new::<Prefix>();

        // SAFETY: since the only unsized types possible are slices, trait objects,
        //   and extern types, the input safety requirement is currently enough to
        //   satisfy the requirements of for_value_raw; this is an implementation
        //   detail of the language that may not be relied upon outside of std.
        let align = unsafe { mem::align_of_val_raw(ptr.as_ptr()) };

        let offset = prefix_layout.size() + prefix_layout.padding_needed_for(align);
        let ptr = ptr.as_ptr() as *mut u8;

        // SAFETY: `ptr` was allocated with this allocator thus, `ptr - offset` points to the
        //   prefix and is non-null.
        unsafe { NonNull::new_unchecked(ptr.sub(offset)).cast() }
    }

    fn create_ptr(ptr: NonNull<[u8]>, offset_prefix: usize) -> NonNull<[u8]> {
        let len = ptr.len() - offset_prefix;

        // SAFETY: `prefix_offset` is smaller (and not equal to) `ptr` as the same function for calculating `prefix_offset` is used when allocating.
        let ptr = unsafe { NonNull::new_unchecked(ptr.as_mut_ptr().add(offset_prefix)) };

        NonNull::slice_from_raw_parts(ptr, len)
    }

    #[inline]
    fn alloc_impl(
        layout: Layout,
        alloc: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (layout, offset_prefix) =
            Layout::new::<Prefix>().extend(layout).map_err(|_| AllocError)?;

        Ok(Self::create_ptr(alloc(layout)?, offset_prefix))
    }
}

unsafe impl<Alloc, Prefix> Allocator for PrefixAllocator<Alloc, Prefix>
where
    Alloc: Allocator,
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::alloc_impl(layout, |l| self.parent.allocate(l))
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::alloc_impl(layout, |l| self.parent.allocate_zeroed(l))
    }

    unsafe fn grow(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // For (A)Rc it's not needed. When implementing please take care, if the alignment changes.
        unimplemented!("PrefixAllocator currently does not implement growing.");
    }

    unsafe fn grow_zeroed(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // For (A)Rc it's not needed. When implementing please take care, if the alignment changes.
        unimplemented!("PrefixAllocator currently does not implement growing.");
    }

    unsafe fn shrink(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // For (A)Rc it's not needed. When implementing please take care, if the alignment changes.
        unimplemented!("PrefixAllocator currently does not implement shrinking.");
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let (layout, prefix_offset) = Layout::new::<Prefix>().extend(layout).unwrap();
        // SAFETY: `prefix_offset` is smaller (and not equal to) `ptr` as the same function for calculating `prefix_offset` is used when allocating.
        unsafe {
            let base_ptr = NonNull::new_unchecked(ptr.as_ptr().sub(prefix_offset));
            self.parent.deallocate(base_ptr, layout)
        };
    }
}
