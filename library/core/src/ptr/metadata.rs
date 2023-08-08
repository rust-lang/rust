#![unstable(feature = "ptr_metadata", issue = "81513")]

use crate::fmt;
use crate::hash::{Hash, Hasher};

/// Provides the pointer metadata type of any pointed-to type.
///
/// # Pointer metadata
///
/// Raw pointer types and reference types in Rust can be thought of as made of two parts:
/// a data pointer that contains the memory address of the value, and some metadata.
///
/// For statically-sized types (that implement the `Sized` traits)
/// as well as for `extern` types,
/// pointers are said to be “thin”: metadata is zero-sized and its type is `()`.
///
/// Pointers to [dynamically-sized types][dst] are said to be “wide” or “fat”,
/// they have non-zero-sized metadata:
///
/// * For structs whose last field is a DST, metadata is the metadata for the last field
/// * For the `str` type, metadata is the length in bytes as `usize`
/// * For slice types like `[T]`, metadata is the length in items as `usize`
/// * For trait objects like `dyn SomeTrait`, metadata is [`DynMetadata<Self>`][DynMetadata]
///   (e.g. `DynMetadata<dyn SomeTrait>`)
///
/// In the future, the Rust language may gain new kinds of types
/// that have different pointer metadata.
///
/// [dst]: https://doc.rust-lang.org/nomicon/exotic-sizes.html#dynamically-sized-types-dsts
///
///
/// # The `Pointee` trait
///
/// The point of this trait is its `Metadata` associated type,
/// which is `()` or `usize` or `DynMetadata<_>` as described above.
/// It is automatically implemented for every type.
/// It can be assumed to be implemented in a generic context, even without a corresponding bound.
///
///
/// # Usage
///
/// Raw pointers can be decomposed into the data address and metadata components
/// with their [`to_raw_parts`] method.
///
/// Alternatively, metadata alone can be extracted with the [`metadata`] function.
/// A reference can be passed to [`metadata`] and implicitly coerced.
///
/// A (possibly-wide) pointer can be put back together from its address and metadata
/// with [`from_raw_parts`] or [`from_raw_parts_mut`].
///
/// [`to_raw_parts`]: *const::to_raw_parts
#[lang = "pointee_trait"]
#[rustc_deny_explicit_impl(implement_via_object = false)]
pub trait Pointee {
    /// The type for metadata in pointers and references to `Self`.
    #[lang = "metadata_type"]
    // NOTE: Keep trait bounds in `static_assert_expected_bounds_for_metadata`
    // in `library/core/src/ptr/metadata.rs`
    // in sync with those here:
    type Metadata: Copy + Send + Sync + Ord + Hash + Unpin;
}

/// Pointers to types implementing this trait alias are “thin”.
///
/// This includes statically-`Sized` types and `extern` types.
///
/// # Example
///
/// ```rust
/// #![feature(ptr_metadata)]
///
/// fn this_never_panics<T: std::ptr::Thin>() {
///     assert_eq!(std::mem::size_of::<&T>(), std::mem::size_of::<usize>())
/// }
/// ```
#[unstable(feature = "ptr_metadata", issue = "81513")]
// NOTE: don’t stabilize this before trait aliases are stable in the language?
pub trait Thin = Pointee<Metadata = ()>;

/// Extract the metadata component of a pointer.
///
/// Values of type `*mut T`, `&T`, or `&mut T` can be passed directly to this function
/// as they implicitly coerce to `*const T`.
///
/// # Example
///
/// ```
/// #![feature(ptr_metadata)]
///
/// assert_eq!(std::ptr::metadata("foo"), 3_usize);
/// ```
#[rustc_const_unstable(feature = "ptr_metadata", issue = "81513")]
#[inline]
pub const fn metadata<T: ?Sized>(ptr: *const T) -> <T as Pointee>::Metadata {
    // SAFETY: Accessing the value from the `PtrRepr` union is safe since *const T
    // and PtrComponents<T> have the same memory layouts. Only std can make this
    // guarantee.
    unsafe { PtrRepr { const_ptr: ptr }.components.metadata }
}

/// Forms a (possibly-wide) raw pointer from a data address and metadata.
///
/// This function is safe but the returned pointer is not necessarily safe to dereference.
/// For slices, see the documentation of [`slice::from_raw_parts`] for safety requirements.
/// For trait objects, the metadata must come from a pointer to the same underlying erased type.
///
/// [`slice::from_raw_parts`]: crate::slice::from_raw_parts
#[unstable(feature = "ptr_metadata", issue = "81513")]
#[rustc_const_unstable(feature = "ptr_metadata", issue = "81513")]
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
#[unstable(feature = "ptr_metadata", issue = "81513")]
#[rustc_const_unstable(feature = "ptr_metadata", issue = "81513")]
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
union PtrRepr<T: ?Sized> {
    const_ptr: *const T,
    mut_ptr: *mut T,
    components: PtrComponents<T>,
}

#[repr(C)]
struct PtrComponents<T: ?Sized> {
    data_address: *const (),
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

/// The metadata for a `Dyn = dyn SomeTrait` trait object type.
///
/// It is a pointer to a vtable (virtual call table)
/// that represents all the necessary information
/// to manipulate the concrete type stored inside a trait object.
/// The vtable notably it contains:
///
/// * type size
/// * type alignment
/// * a pointer to the type’s `drop_in_place` impl (may be a no-op for plain-old-data)
/// * pointers to all the methods for the type’s implementation of the trait
///
/// Note that the first three are special because they’re necessary to allocate, drop,
/// and deallocate any trait object.
///
/// It is possible to name this struct with a type parameter that is not a `dyn` trait object
/// (for example `DynMetadata<u64>`) but not to obtain a meaningful value of that struct.
#[lang = "dyn_metadata"]
pub struct DynMetadata<Dyn: ?Sized> {
    vtable_ptr: &'static VTable,
    phantom: crate::marker::PhantomData<Dyn>,
}

extern "C" {
    /// Opaque type for accessing vtables.
    ///
    /// Private implementation detail of `DynMetadata::size_of` etc.
    /// There is conceptually not actually any Abstract Machine memory behind this pointer.
    type VTable;
}

impl<Dyn: ?Sized> DynMetadata<Dyn> {
    /// Returns the size of the type associated with this vtable.
    #[inline]
    pub fn size_of(self) -> usize {
        // Note that "size stored in vtable" is *not* the same as "result of size_of_val_raw".
        // Consider a reference like `&(i32, dyn Send)`: the vtable will only store the size of the
        // `Send` part!
        // SAFETY: DynMetadata always contains a valid vtable pointer
        return unsafe {
            crate::intrinsics::vtable_size(self.vtable_ptr as *const VTable as *const ())
        };
    }

    /// Returns the alignment of the type associated with this vtable.
    #[inline]
    pub fn align_of(self) -> usize {
        // SAFETY: DynMetadata always contains a valid vtable pointer
        return unsafe {
            crate::intrinsics::vtable_align(self.vtable_ptr as *const VTable as *const ())
        };
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
