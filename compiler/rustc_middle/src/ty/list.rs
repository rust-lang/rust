use super::flags::FlagComputation;
use super::{DebruijnIndex, DebugWithInfcx, InferCtxtLike, TyCtxt, TypeFlags, WithInfcx};
use crate::arena::Arena;
use rustc_data_structures::aligned::{align_of, Aligned};
use rustc_serialize::{Encodable, Encoder};
use std::alloc::Layout;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::slice;

#[cfg(parallel_compiler)]
use rustc_data_structures::sync::DynSync;

/// `List<T>` is a bit like `&[T]`, but with some critical differences.
/// - IMPORTANT: Every `List<T>` is *required* to have unique contents. The
///   type's correctness relies on this, *but it does not enforce it*.
///   Therefore, any code that creates a `List<T>` must ensure uniqueness
///   itself. In practice this is achieved by interning.
/// - The length is stored within the `List<T>`, so `&List<Ty>` is a thin
///   pointer.
/// - Because of this, you cannot get a `List<T>` that is a sub-list of another
///   `List<T>`. You can get a sub-slice `&[T]`, however.
/// - `List<T>` can be used with `CopyTaggedPtr`, which is useful within
///   structs whose size must be minimized.
/// - Because of the uniqueness assumption, we can use the address of a
///   `List<T>` for faster equality comparisons and hashing.
/// - `T` must be `Copy`. This lets `List<T>` be stored in a dropless arena and
///   iterators return a `T` rather than a `&T`.
/// - `T` must not be zero-sized.
#[repr(C)]
pub struct List<T> {
    skel: ListSkeleton<T>,
    opaque: OpaqueListContents,
}

/// This type defines the statically known field offsets and alignment of a [`List`].
///
/// The alignment of a `List` cannot be determined, because it has an extern type tail.
/// We use `ListSkeleton` instead of `List` whenever computing the alignment is required,
/// for example:
/// - Implementing the [`Aligned`] trait, which is required for [`CopyTaggedPtr`].
/// - Projecting from [`ListWithCachedTypeInfo`] to `List`, which requires computing the padding
///   between the cached type info and the list, which requires computing the list's alignment.
///
/// Note that `ListSkeleton` is `Sized`, but **it's size is not correct**, as it is missing the
/// dynamically sized list tail. Do not create a `ListSkeleton` on the stack.
///
/// FIXME: This can be removed once we properly support `!Sized + Aligned + Thin` types.
///
/// [`CopyTaggedPtr`]: rustc_data_structures::tagged_ptr::CopyTaggedPtr
#[repr(C)]
struct ListSkeleton<T> {
    len: usize,

    /// Although this claims to be a zero-length array, in practice `len`
    /// elements are actually present.
    data: [T; 0],
}

extern "C" {
    /// A dummy type used to force `List` to be unsized while not requiring
    /// references to it be wide pointers.
    type OpaqueListContents;
}

impl<T> List<T> {
    /// Returns a reference to the (unique, static) empty list.
    #[inline(always)]
    pub fn empty<'a>() -> &'a List<T> {
        ListWithCachedTypeInfo::empty()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.skel.len
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T: Copy> List<T> {
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
    pub(super) fn from_arena<'tcx>(arena: &'tcx Arena<'tcx>, slice: &[T]) -> &'tcx List<T> {
        assert!(!mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() != 0);
        assert!(!slice.is_empty());

        let (layout, _offset) =
            Layout::new::<usize>().extend(Layout::for_value::<[T]>(slice)).unwrap();
        let mem = arena.dropless.alloc_raw(layout) as *mut List<T>;
        unsafe {
            // Write the length
            ptr::addr_of_mut!((*mem).skel.len).write(slice.len());

            // Write the elements
            ptr::addr_of_mut!((*mem).skel.data)
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
    pub fn iter(&self) -> <&'_ List<T> as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<T: fmt::Debug> fmt::Debug for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}
impl<'tcx, T: DebugWithInfcx<TyCtxt<'tcx>>> DebugWithInfcx<TyCtxt<'tcx>> for List<T> {
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        fmt::Debug::fmt(&this.map(|this| this.as_slice()), f)
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for List<T> {
    #[inline]
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}

impl<T: PartialEq> PartialEq for List<T> {
    #[inline]
    fn eq(&self, other: &List<T>) -> bool {
        // Pointer equality implies list equality (due to the unique contents
        // assumption).
        ptr::eq(self, other)
    }
}

impl<T: Eq> Eq for List<T> {}

impl<T> Ord for List<T>
where
    T: Ord,
{
    fn cmp(&self, other: &List<T>) -> Ordering {
        // Pointer equality implies list equality (due to the unique contents
        // assumption), but the contents must be compared otherwise.
        if self == other { Ordering::Equal } else { <[T] as Ord>::cmp(&**self, &**other) }
    }
}

impl<T> PartialOrd for List<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &List<T>) -> Option<Ordering> {
        // Pointer equality implies list equality (due to the unique contents
        // assumption), but the contents must be compared otherwise.
        if self == other {
            Some(Ordering::Equal)
        } else {
            <[T] as PartialOrd>::partial_cmp(&**self, &**other)
        }
    }
}

impl<T> Hash for List<T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // Pointer hashing is sufficient (due to the unique contents
        // assumption).
        ptr::from_ref(self).hash(s)
    }
}

impl<T> Deref for List<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T> AsRef<[T]> for List<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        let data_ptr = ptr::addr_of!(self.skel.data).cast::<T>();
        // SAFETY: `data_ptr` has the same provenance as `self` and can therefore
        // access the `self.skel.len` elements stored at `self.skel.data`.
        // Note that we specifically don't reborrow `&self.skel.data`, because that
        // would give us a pointer with provenance over 0 bytes.
        unsafe { slice::from_raw_parts(data_ptr, self.skel.len) }
    }
}

impl<'a, T: Copy> IntoIterator for &'a List<T> {
    type Item = T;
    type IntoIter = iter::Copied<<&'a [T] as IntoIterator>::IntoIter>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self[..].iter().copied()
    }
}

unsafe impl<T: Sync> Sync for List<T> {}

// We need this since `List` uses extern type `OpaqueListContents`.
#[cfg(parallel_compiler)]
unsafe impl<T: DynSync> DynSync for List<T> {}

// Safety:
// Layouts of `ListSkeleton<T>` and `List<T>` are the same, modulo opaque tail,
// thus aligns of `ListSkeleton<T>` and `List<T>` must be the same.
unsafe impl<T> Aligned for List<T> {
    const ALIGN: ptr::Alignment = align_of::<ListSkeleton<T>>();
}

/// A [`List`] that additionally stores type information inline to speed up
/// [`TypeVisitableExt`](super::TypeVisitableExt) operations.
#[repr(C)]
pub struct ListWithCachedTypeInfo<T> {
    skel: ListWithCachedTypeInfoSkeleton<T>,
    opaque: OpaqueListContents,
}

/// The additional info that is stored in [`ListWithCachedTypeInfo`].
#[repr(C)]
pub struct TypeInfo {
    flags: TypeFlags,
    outer_exclusive_binder: DebruijnIndex,
}

impl From<FlagComputation> for TypeInfo {
    fn from(computation: FlagComputation) -> TypeInfo {
        TypeInfo {
            flags: computation.flags,
            outer_exclusive_binder: computation.outer_exclusive_binder,
        }
    }
}

/// This type is similar to [`ListSkeleton`], but for [`ListWithCachedTypeInfo`].
/// It is used for computing the alignment of a [`ListWithCachedTypeInfo`].
#[repr(C)]
struct ListWithCachedTypeInfoSkeleton<T> {
    info: TypeInfo,
    // N.B.: There may be padding between these two fields. We cannot use `List` directly
    // here, because it has an unknown alignment which makes computing the amount of padding
    // and therefore projecting from `&ListWithCachedTypeInfo` to `&List` impossible.
    list: ListSkeleton<T>,
}

impl<T> ListWithCachedTypeInfo<T> {
    #[inline(always)]
    pub fn empty<'a>() -> &'a ListWithCachedTypeInfo<T> {
        #[repr(align(64))]
        struct MaxAlign;

        #[repr(C)]
        struct Empty {
            info: TypeInfo,
            zero: [u8; 2 * mem::align_of::<MaxAlign>() - mem::size_of::<TypeInfo>()],
            align: MaxAlign,
        }

        static EMPTY: Empty = Empty {
            info: TypeInfo { flags: TypeFlags::empty(), outer_exclusive_binder: super::INNERMOST },
            zero: [0; 2 * mem::align_of::<MaxAlign>() - mem::size_of::<TypeInfo>()],
            align: MaxAlign,
        };

        assert!(mem::align_of::<T>() <= mem::align_of::<MaxAlign>());

        // The layout of the empty `ListWithCachedTypeInfo<T>` must be one of the following,
        // depending on the alignment of `T`:
        //
        // On 64-bit platforms:
        // F = flags (32 bit), B = outer_exclusive_binder (32 bit), LL = len (64 bit)
        // align(T) <= 8: FBLL
        // align(T) = 16: FB..LL..
        // align(T) = 32: FB......LL......
        // align(T) = 64: FB..............LL..............
        //
        // On 32-bit platforms:
        // F = flags (32 bit), B = outer_exclusive_binder (32 bit), L = len (32 bit)
        // align(T) <= 4: FBL
        // align(T) =  8: FBL.
        // align(T) = 16: FB..L...
        // align(T) = 32: FB......L.......
        // align(T) = 64: FB..............L...............
        //
        // We zero out every possible location of `len` so that `EMPTY` is a valid
        // `ListWithCachedTypeInfo<T>` for all `T` with alignment up to 64 bytes.
        unsafe { &*(std::ptr::addr_of!(EMPTY) as *const ListWithCachedTypeInfo<T>) }
    }

    #[inline]
    pub(super) fn from_arena<'tcx>(
        arena: &'tcx Arena<'tcx>,
        info: TypeInfo,
        slice: &[T],
    ) -> &'tcx ListWithCachedTypeInfo<T>
    where
        T: Copy,
    {
        assert!(!mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() != 0);
        assert!(!slice.is_empty());

        let (list_layout, _offset) =
            Layout::new::<usize>().extend(Layout::for_value::<[T]>(slice)).unwrap();

        let (layout, _offset) = Layout::new::<TypeInfo>().extend(list_layout).unwrap();

        let mem = arena.dropless.alloc_raw(layout) as *mut ListWithCachedTypeInfo<T>;
        unsafe {
            // Write the cached type info
            ptr::addr_of_mut!((*mem).skel.info).write(info);

            // Write the length
            ptr::addr_of_mut!((*mem).skel.list.len).write(slice.len());

            // Write the elements
            ptr::addr_of_mut!((*mem).skel.list.data)
                .cast::<T>()
                .copy_from_nonoverlapping(slice.as_ptr(), slice.len());

            &*mem
        }
    }

    #[inline(always)]
    pub fn as_list(&self) -> &List<T> {
        self
    }

    #[inline(always)]
    pub fn flags(&self) -> TypeFlags {
        self.skel.info.flags
    }

    #[inline(always)]
    pub fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.skel.info.outer_exclusive_binder
    }
}

impl<T> Deref for ListWithCachedTypeInfo<T> {
    type Target = List<T>;
    #[inline(always)]
    fn deref(&self) -> &List<T> {
        let list_ptr = ptr::addr_of!(self.skel.list) as *const List<T>;
        unsafe { &*list_ptr }
    }
}

impl<T> AsRef<[T]> for ListWithCachedTypeInfo<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        (&**self).as_ref()
    }
}

impl<T: fmt::Debug> fmt::Debug for ListWithCachedTypeInfo<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}
impl<'tcx, T: DebugWithInfcx<TyCtxt<'tcx>>> DebugWithInfcx<TyCtxt<'tcx>>
    for ListWithCachedTypeInfo<T>
{
    fn fmt<Infcx: InferCtxtLike<Interner = TyCtxt<'tcx>>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        DebugWithInfcx::fmt(this.map(|this| &**this), f)
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for ListWithCachedTypeInfo<T> {
    #[inline]
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}

impl<T: PartialEq> PartialEq for ListWithCachedTypeInfo<T> {
    #[inline]
    fn eq(&self, other: &ListWithCachedTypeInfo<T>) -> bool {
        // Pointer equality implies list equality (due to the unique contents
        // assumption).
        ptr::eq(self, other)
    }
}

impl<T: Eq> Eq for ListWithCachedTypeInfo<T> {}

impl<T> Hash for ListWithCachedTypeInfo<T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // Pointer hashing is sufficient (due to the unique contents
        // assumption).
        ptr::from_ref(self).hash(s)
    }
}

impl<'a, T: Copy> IntoIterator for &'a ListWithCachedTypeInfo<T> {
    type Item = T;
    type IntoIter = iter::Copied<<&'a [T] as IntoIterator>::IntoIter>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        (**self).into_iter()
    }
}

unsafe impl<T: Sync> Sync for ListWithCachedTypeInfo<T> {}

#[cfg(parallel_compiler)]
unsafe impl<T: DynSync> DynSync for ListWithCachedTypeInfo<T> {}

// Safety:
// Layouts of `ListWithCachedTypeInfoSkeleton<T>` and `ListWithCachedTypeInfo<T>`
// are the same, modulo opaque tail, thus their aligns must be the same.
unsafe impl<T> Aligned for ListWithCachedTypeInfo<T> {
    const ALIGN: ptr::Alignment = align_of::<ListWithCachedTypeInfoSkeleton<T>>();
}
