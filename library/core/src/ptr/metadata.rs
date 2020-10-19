#![unstable(feature = "ptr_metadata", issue = /* FIXME */ "none")]

use crate::fmt;
use crate::hash::Hash;
use crate::ptr::NonNull;

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
#[inline]
pub fn metadata<T: ?Sized>(ptr: *const T) -> <T as Pointee>::Metadata {
    // SAFETY: Accessing the value from the `PtrRepr` union is safe since *const T
    // and PtrComponents<T> have the same memory layouts. Only std can make this
    // guarantee.
    unsafe { PtrRepr { const_ptr: ptr }.components.metadata }
}

#[repr(C)]
union PtrRepr<T: ?Sized> {
    const_ptr: *const T,
    components: PtrComponents<T>,
}

#[repr(C)]
struct PtrComponents<T: ?Sized> {
    data_address: usize,
    metadata: <T as Pointee>::Metadata,
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
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct DynMetadata {
    #[allow(unused)]
    vtable_ptr: NonNull<()>,
}

unsafe impl Send for DynMetadata {}
unsafe impl Sync for DynMetadata {}

impl fmt::Debug for DynMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DynMetadata { … }")
    }
}
