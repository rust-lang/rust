#![unstable(feature = "ptr_metadata", issue = "81513")]

use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::intrinsics::{aggregate_raw_ptr, ptr_metadata};
use crate::marker::Freeze;
use crate::ptr::NonNull;

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
/// Raw pointers can be decomposed into the data pointer and metadata components
/// with their [`to_raw_parts`] method.
///
/// Alternatively, metadata alone can be extracted with the [`metadata`] function.
/// A reference can be passed to [`metadata`] and implicitly coerced.
///
/// A (possibly-wide) pointer can be put back together from its data pointer and metadata
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
    type Metadata: fmt::Debug + Copy + Send + Sync + Ord + Hash + Unpin + Freeze;
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

/// Extracts the metadata component of a pointer.
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
#[inline]
pub const fn metadata<T: ?Sized>(ptr: *const T) -> <T as Pointee>::Metadata {
    ptr_metadata(ptr)
}

/// Forms a (possibly-wide) raw pointer from a data pointer and metadata.
///
/// This function is safe but the returned pointer is not necessarily safe to dereference.
/// For slices, see the documentation of [`slice::from_raw_parts`] for safety requirements.
/// For trait objects, the metadata must come from a pointer to the same underlying erased type.
///
/// [`slice::from_raw_parts`]: crate::slice::from_raw_parts
#[unstable(feature = "ptr_metadata", issue = "81513")]
#[inline]
pub const fn from_raw_parts<T: ?Sized>(
    data_pointer: *const impl Thin,
    metadata: <T as Pointee>::Metadata,
) -> *const T {
    aggregate_raw_ptr(data_pointer, metadata)
}

/// Performs the same functionality as [`from_raw_parts`], except that a
/// raw `*mut` pointer is returned, as opposed to a raw `*const` pointer.
///
/// See the documentation of [`from_raw_parts`] for more details.
#[unstable(feature = "ptr_metadata", issue = "81513")]
#[inline]
pub const fn from_raw_parts_mut<T: ?Sized>(
    data_pointer: *mut impl Thin,
    metadata: <T as Pointee>::Metadata,
) -> *mut T {
    aggregate_raw_ptr(data_pointer, metadata)
}

/// The metadata for a `Dyn = dyn SomeTrait` trait object type.
///
/// It is a pointer to a vtable (virtual call table)
/// that represents all the necessary information
/// to manipulate the concrete type stored inside a trait object.
/// The vtable notably contains:
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
///
/// Note that while this type implements `PartialEq`, comparing vtable pointers is unreliable:
/// pointers to vtables of the same type for the same trait can compare inequal (because vtables are
/// duplicated in multiple codegen units), and pointers to vtables of *different* types/traits can
/// compare equal (since identical vtables can be deduplicated within a codegen unit).
#[lang = "dyn_metadata"]
pub struct DynMetadata<Dyn: ?Sized> {
    _vtable_ptr: NonNull<VTable>,
    _phantom: crate::marker::PhantomData<Dyn>,
}

extern "C" {
    /// Opaque type for accessing vtables.
    ///
    /// Private implementation detail of `DynMetadata::size_of` etc.
    /// There is conceptually not actually any Abstract Machine memory behind this pointer.
    type VTable;
}

impl<Dyn: ?Sized> DynMetadata<Dyn> {
    /// When `DynMetadata` appears as the metadata field of a wide pointer, the rustc_middle layout
    /// computation does magic and the resulting layout is *not* a `FieldsShape::Aggregate`, instead
    /// it is a `FieldsShape::Primitive`. This means that the same type can have different layout
    /// depending on whether it appears as the metadata field of a wide pointer or as a stand-alone
    /// type, which understandably confuses codegen and leads to ICEs when trying to project to a
    /// field of `DynMetadata`. To work around that issue, we use `transmute` instead of using a
    /// field projection.
    #[inline]
    fn vtable_ptr(self) -> *const VTable {
        // SAFETY: this layout assumption is hard-coded into the compiler.
        // If it's somehow not a size match, the transmute will error.
        unsafe { crate::mem::transmute::<Self, *const VTable>(self) }
    }

    /// Returns the size of the type associated with this vtable.
    #[inline]
    pub fn size_of(self) -> usize {
        // Note that "size stored in vtable" is *not* the same as "result of size_of_val_raw".
        // Consider a reference like `&(i32, dyn Send)`: the vtable will only store the size of the
        // `Send` part!
        // SAFETY: DynMetadata always contains a valid vtable pointer
        unsafe { crate::intrinsics::vtable_size(self.vtable_ptr() as *const ()) }
    }

    /// Returns the alignment of the type associated with this vtable.
    #[inline]
    pub fn align_of(self) -> usize {
        // SAFETY: DynMetadata always contains a valid vtable pointer
        unsafe { crate::intrinsics::vtable_align(self.vtable_ptr() as *const ()) }
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
        f.debug_tuple("DynMetadata").field(&self.vtable_ptr()).finish()
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
        crate::ptr::eq::<VTable>(self.vtable_ptr(), other.vtable_ptr())
    }
}

impl<Dyn: ?Sized> Ord for DynMetadata<Dyn> {
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn cmp(&self, other: &Self) -> crate::cmp::Ordering {
        <*const VTable>::cmp(&self.vtable_ptr(), &other.vtable_ptr())
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
        crate::ptr::hash::<VTable, _>(self.vtable_ptr(), hasher)
    }
}
