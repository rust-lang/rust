use crate::{
    alloc::{AllocError, Allocator, Layout},
    fmt,
    marker::PhantomData,
    ptr::NonNull,
};

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
#[unstable(feature = "allocator_api_internals", issue = "none")]
pub struct PrefixAllocator<Alloc, Prefix = ()> {
    /// The parent allocator to be used as backend
    pub parent: Alloc,
    _prefix: PhantomData<*const Prefix>,
}

impl<Alloc: fmt::Debug, Prefix> fmt::Debug for PrefixAllocator<Alloc, Prefix> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affix").field("parent", &self.parent).finish()
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
    /// * `layout` must *[fit]* that block of memory.
    ///
    /// [currently allocated]: https://doc.rust-lang.org/nightly/core/alloc/trait.AllocRef.html#currently-allocated-memory
    /// [fit]: https://doc.rust-lang.org/nightly/core/alloc/trait.AllocRef.html#memory-fitting
    #[inline]
    pub unsafe fn prefix(ptr: NonNull<u8>, layout: Layout) -> NonNull<Prefix> {
        let prefix_offset = Self::prefix_offset(layout);
        // SAFETY: `prefix_offset` is smaller (and not equal to) `ptr` as the same function for calculating `prefix_offset` is used when allocating.
        unsafe { NonNull::new_unchecked(ptr.as_ptr().sub(prefix_offset)).cast() }
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
