use std::alloc::Layout;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::{fmt, iter, mem, ptr, slice};

use rustc_data_structures::aligned::{Aligned, align_of};
use rustc_data_structures::sync::DynSync;
use rustc_serialize::{Encodable, Encoder};
use rustc_type_ir::FlagComputation;

use super::{DebruijnIndex, TyCtxt, TypeFlags};
use crate::arena::Arena;

/// `List<T>` is a bit like `&[T]`, but with some critical differences.
/// - IMPORTANT: Every `List<T>` is *required* to have unique contents. The
///   type's correctness relies on this, *but it does not enforce it*.
///   Therefore, any code that creates a `List<T>` must ensure uniqueness
///   itself. In practice this is achieved by interning.
/// - The length is stored within the `List<T>`, so `&List<Ty>` is a thin
///   pointer.
/// - Because of this, you cannot get a `List<T>` that is a sub-list of another
///   `List<T>`. You can get a sub-slice `&[T]`, however.
/// - `List<T>` can be used with `TaggedRef`, which is useful within
///   structs whose size must be minimized.
/// - Because of the uniqueness assumption, we can use the address of a
///   `List<T>` for faster equality comparisons and hashing.
/// - `T` must be `Copy`. This lets `List<T>` be stored in a dropless arena and
///   iterators return a `T` rather than a `&T`.
/// - `T` must not be zero-sized.
pub type List<T> = RawList<(), T>;

/// A generic type that can be used to prepend a [`List`] with some header.
///
/// The header will be ignored for value-based operations like [`PartialEq`],
/// [`Hash`] and [`Encodable`].
#[repr(C)]
pub struct RawList<H, T> {
    skel: ListSkeleton<H, T>,
    opaque: OpaqueListContents,
}

/// A [`RawList`] without the unsized tail. This type is used for layout computation
/// and constructing empty lists.
#[repr(C)]
struct ListSkeleton<H, T> {
    header: H,
    len: usize,
    /// Although this claims to be a zero-length array, in practice `len`
    /// elements are actually present.
    data: [T; 0],
}

impl<T> Default for &List<T> {
    fn default() -> Self {
        List::empty()
    }
}

unsafe extern "C" {
    /// A dummy type used to force `List` to be unsized while not requiring
    /// references to it be wide pointers.
    type OpaqueListContents;
}

impl<H, T> RawList<H, T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.skel.len
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Allocates a list from `arena` and copies the contents of `slice` into it.
    ///
    /// WARNING: the contents *must be unique*, such that no list with these
    /// contents has been previously created. If not, operations such as `eq`
    /// and `hash` might give incorrect results.
    ///
    /// Panics if `T` is `Drop`, or `T` is zero-sized, or the slice is empty
    /// (because the empty list exists statically, and is available via
    /// `empty()`).
    #[inline]
    pub(super) fn from_arena<'tcx>(
        arena: &'tcx Arena<'tcx>,
        header: H,
        slice: &[T],
    ) -> &'tcx RawList<H, T>
    where
        T: Copy,
    {
        assert!(!mem::needs_drop::<T>());
        assert!(size_of::<T>() != 0);
        assert!(!slice.is_empty());

        let (layout, _offset) =
            Layout::new::<ListSkeleton<H, T>>().extend(Layout::for_value::<[T]>(slice)).unwrap();

        let mem = arena.dropless.alloc_raw(layout) as *mut RawList<H, T>;
        unsafe {
            // Write the header
            (&raw mut (*mem).skel.header).write(header);

            // Write the length
            (&raw mut (*mem).skel.len).write(slice.len());

            // Write the elements
            (&raw mut (*mem).skel.data)
                .cast::<T>()
                .copy_from_nonoverlapping(slice.as_ptr(), slice.len());

            &*mem
        }
    }

    // If this method didn't exist, we would use `slice.iter` due to
    // deref coercion.
    //
    // This would be weird, as `self.into_iter` iterates over `T` directly.
    #[inline(always)]
    pub fn iter(&self) -> <&'_ RawList<H, T> as IntoIterator>::IntoIter
    where
        T: Copy,
    {
        self.into_iter()
    }
}

impl<'a, H, T: Copy> rustc_type_ir::inherent::SliceLike for &'a RawList<H, T> {
    type Item = T;

    type IntoIter = iter::Copied<<&'a [T] as IntoIterator>::IntoIter>;

    fn iter(self) -> Self::IntoIter {
        (*self).iter()
    }

    fn as_slice(&self) -> &[Self::Item] {
        (*self).as_slice()
    }
}

macro_rules! impl_list_empty {
    ($header_ty:ty, $header_init:expr) => {
        impl<T> RawList<$header_ty, T> {
            /// Returns a reference to the (per header unique, static) empty list.
            #[inline(always)]
            pub fn empty<'a>() -> &'a RawList<$header_ty, T> {
                #[repr(align(64))]
                struct MaxAlign;

                static EMPTY: ListSkeleton<$header_ty, MaxAlign> =
                    ListSkeleton { header: $header_init, len: 0, data: [] };

                assert!(align_of::<T>() <= align_of::<MaxAlign>());

                // SAFETY: `EMPTY` is sufficiently aligned to be an empty list for all
                // types with `align_of(T) <= align_of(MaxAlign)`, which we checked above.
                unsafe { &*((&raw const EMPTY) as *const RawList<$header_ty, T>) }
            }
        }
    };
}

impl_list_empty!((), ());

impl<H, T: fmt::Debug> fmt::Debug for RawList<H, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<H, S: Encoder, T: Encodable<S>> Encodable<S> for RawList<H, T> {
    #[inline]
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}

impl<H, T: PartialEq> PartialEq for RawList<H, T> {
    #[inline]
    fn eq(&self, other: &RawList<H, T>) -> bool {
        // Pointer equality implies list equality (due to the unique contents
        // assumption).
        ptr::eq(self, other)
    }
}

impl<H, T: Eq> Eq for RawList<H, T> {}

impl<H, T> Ord for RawList<H, T>
where
    T: Ord,
{
    fn cmp(&self, other: &RawList<H, T>) -> Ordering {
        // Pointer equality implies list equality (due to the unique contents
        // assumption), but the contents must be compared otherwise.
        if self == other { Ordering::Equal } else { <[T] as Ord>::cmp(&**self, &**other) }
    }
}

impl<H, T> PartialOrd for RawList<H, T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &RawList<H, T>) -> Option<Ordering> {
        // Pointer equality implies list equality (due to the unique contents
        // assumption), but the contents must be compared otherwise.
        if self == other {
            Some(Ordering::Equal)
        } else {
            <[T] as PartialOrd>::partial_cmp(&**self, &**other)
        }
    }
}

impl<Hdr, T> Hash for RawList<Hdr, T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // Pointer hashing is sufficient (due to the unique contents
        // assumption).
        ptr::from_ref(self).hash(s)
    }
}

impl<H, T> Deref for RawList<H, T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_ref()
    }
}

impl<H, T> AsRef<[T]> for RawList<H, T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        let data_ptr = (&raw const self.skel.data).cast::<T>();
        // SAFETY: `data_ptr` has the same provenance as `self` and can therefore
        // access the `self.skel.len` elements stored at `self.skel.data`.
        // Note that we specifically don't reborrow `&self.skel.data`, because that
        // would give us a pointer with provenance over 0 bytes.
        unsafe { slice::from_raw_parts(data_ptr, self.skel.len) }
    }
}

impl<'a, H, T: Copy> IntoIterator for &'a RawList<H, T> {
    type Item = T;
    type IntoIter = iter::Copied<<&'a [T] as IntoIterator>::IntoIter>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self[..].iter().copied()
    }
}

unsafe impl<H: Sync, T: Sync> Sync for RawList<H, T> {}

// We need this since `List` uses extern type `OpaqueListContents`.
unsafe impl<H: DynSync, T: DynSync> DynSync for RawList<H, T> {}

// Safety:
// Layouts of `ListSkeleton<H, T>` and `RawList<H, T>` are the same, modulo opaque tail,
// thus aligns of `ListSkeleton<H, T>` and `RawList<H, T>` must be the same.
unsafe impl<H, T> Aligned for RawList<H, T> {
    const ALIGN: ptr::Alignment = align_of::<ListSkeleton<H, T>>();
}

/// A [`List`] that additionally stores type information inline to speed up
/// [`TypeVisitableExt`](super::TypeVisitableExt) operations.
pub type ListWithCachedTypeInfo<T> = RawList<TypeInfo, T>;

impl<T> ListWithCachedTypeInfo<T> {
    #[inline(always)]
    pub fn flags(&self) -> TypeFlags {
        self.skel.header.flags
    }

    #[inline(always)]
    pub fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.skel.header.outer_exclusive_binder
    }
}

impl_list_empty!(TypeInfo, TypeInfo::empty());

/// The additional info that is stored in [`ListWithCachedTypeInfo`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeInfo {
    flags: TypeFlags,
    outer_exclusive_binder: DebruijnIndex,
}

impl TypeInfo {
    const fn empty() -> Self {
        Self { flags: TypeFlags::empty(), outer_exclusive_binder: super::INNERMOST }
    }
}

impl<'tcx> From<FlagComputation<TyCtxt<'tcx>>> for TypeInfo {
    fn from(computation: FlagComputation<TyCtxt<'tcx>>) -> TypeInfo {
        TypeInfo {
            flags: computation.flags,
            outer_exclusive_binder: computation.outer_exclusive_binder,
        }
    }
}
