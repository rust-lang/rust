#![unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]

use crate::fmt;
use crate::hash::{Hash, Hasher};

/// FIXME docs
#[lang = "pointee_trait"]
pub trait Pointee {
    /// The type for metadata in pointers and references to `Self`.
    #[lang = "metadata_type"]
    // NOTE: Keep trait bounds in `static_assert_expected_bounds_for_metadata`
    // in `library/core/src/ptr/metadata.rs`
    // in sync with those here:
    type Metadata: Copy + Send + Sync + Ord + Hash + Unpin;
}

/// Pointers to types implementing this trait alias are “thin”
///
/// ```rust
/// #![feature(ptr_metadata)]
///
/// fn this_never_panics<T: std::ptr::Thin>() {
///     assert_eq!(std::mem::size_of::<&T>(), std::mem::size_of::<usize>())
/// }
/// ```
#[unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]
// NOTE: don’t stabilize this before trait aliases are stable in the language?
pub trait Thin = Pointee<Metadata = ()>;

/// Extract the metadata component of a pointer.
#[rustc_const_unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]
#[inline]
pub const fn metadata<T: ?Sized>(ptr: *const T) -> <T as Pointee>::Metadata {
    // SAFETY: Accessing the value from the `PtrRepr` union is safe since *const T
    // and PtrComponents<T> have the same memory layouts. Only std can make this
    // guarantee.
    unsafe { PtrRepr { const_ptr: ptr }.components.metadata }
}

/// Forms a raw pointer from a data address and metadata.
#[unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]
#[rustc_const_unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]
#[inline]
pub const fn from_raw_parts<T: ?Sized>(
    data_address: *const (),
    metadata: <T as Pointee>::Metadata,
) -> *const T {
    // SAFETY: Accessing the value from the `PtrRepr` union is safe since *const T
    // and PtrComponents<T> have the same memory layouts. Only std can make this
    // guarantee.
    unsafe { PtrRepr { components: PtrComponents { data_address, metadata } }.const_ptr }
}

/// Performs the same functionality as [`from_raw_parts`], except that a
/// raw `*mut` pointer is returned, as opposed to a raw `*const` pointer.
///
/// See the documentation of [`from_raw_parts`] for more details.
#[unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]
#[rustc_const_unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]
#[inline]
pub const fn from_raw_parts_mut<T: ?Sized>(
    data_address: *mut (),
    metadata: <T as Pointee>::Metadata,
) -> *mut T {
    // SAFETY: Accessing the value from the `PtrRepr` union is safe since *const T
    // and PtrComponents<T> have the same memory layouts. Only std can make this
    // guarantee.
    unsafe { PtrRepr { components: PtrComponents { data_address, metadata } }.mut_ptr }
}

#[repr(C)]
pub(crate) union PtrRepr<T: ?Sized> {
    pub(crate) const_ptr: *const T,
    pub(crate) mut_ptr: *mut T,
    pub(crate) components: PtrComponents<T>,
}

#[repr(C)]
pub(crate) struct PtrComponents<T: ?Sized> {
    pub(crate) data_address: *const (),
    pub(crate) metadata: <T as Pointee>::Metadata,
}

// Manual impl needed to avoid `T: Copy` bound.
impl<T: ?Sized> Copy for PtrComponents<T> {}

// Manual impl needed to avoid `T: Clone` bound.
impl<T: ?Sized> Clone for PtrComponents<T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// The metadata for a `dyn SomeTrait` trait object type.
#[lang = "dyn_metadata"]
pub struct DynMetadata<Dyn: ?Sized> {
    vtable_ptr: &'static VTable,
    phantom: crate::marker::PhantomData<Dyn>,
}

/// The common prefix of all vtables. It is followed by function pointers for trait methods.
///
/// Private implementation detail of `DynMetadata::size_of` etc.
#[repr(C)]
struct VTable {
    drop_in_place: fn(*mut ()),
    size_of: usize,
    align_of: usize,
}

impl<Dyn: ?Sized> DynMetadata<Dyn> {
    /// Returns the size of the type associated with this vtable.
    #[inline]
    pub fn size_of(self) -> usize {
        self.vtable_ptr.size_of
    }

    /// Returns the alignment of the type associated with this vtable.
    #[inline]
    pub fn align_of(self) -> usize {
        self.vtable_ptr.align_of
    }

    /// Returns the size and alignment together as a `Layout`
    #[inline]
    pub fn layout(self) -> crate::alloc::Layout {
        // SAFETY: the compiler emitted this vtable for a concrete Rust type which
        // is known to have a valid layout. Same rationale as in `Layout::for_value`.
        unsafe { crate::alloc::Layout::from_size_align_unchecked(self.size_of(), self.align_of()) }
    }
}

unsafe impl<Dyn: ?Sized> Send for DynMetadata<Dyn> {}
unsafe impl<Dyn: ?Sized> Sync for DynMetadata<Dyn> {}

impl<Dyn: ?Sized> fmt::Debug for DynMetadata<Dyn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DynMetadata").field(&(self.vtable_ptr as *const VTable)).finish()
    }
}

// Manual impls needed to avoid `Dyn: $Trait` bounds.

impl<Dyn: ?Sized> Unpin for DynMetadata<Dyn> {}

impl<Dyn: ?Sized> Copy for DynMetadata<Dyn> {}

impl<Dyn: ?Sized> Clone for DynMetadata<Dyn> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<Dyn: ?Sized> Eq for DynMetadata<Dyn> {}

impl<Dyn: ?Sized> PartialEq for DynMetadata<Dyn> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        crate::ptr::eq::<VTable>(self.vtable_ptr, other.vtable_ptr)
    }
}

impl<Dyn: ?Sized> Ord for DynMetadata<Dyn> {
    #[inline]
    fn cmp(&self, other: &Self) -> crate::cmp::Ordering {
        (self.vtable_ptr as *const VTable).cmp(&(other.vtable_ptr as *const VTable))
    }
}

impl<Dyn: ?Sized> PartialOrd for DynMetadata<Dyn> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<crate::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Dyn: ?Sized> Hash for DynMetadata<Dyn> {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        crate::ptr::hash::<VTable, _>(self.vtable_ptr, hasher)
    }
}
