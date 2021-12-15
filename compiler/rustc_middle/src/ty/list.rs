use crate::arena::Arena;
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
    len: usize,

    /// Although this claims to be a zero-length array, in practice `len`
    /// elements are actually present.
    data: [T; 0],

    opaque: OpaqueListContents,
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
        #[repr(align(64))]
        struct MaxAlign;

        assert!(mem::align_of::<T>() <= mem::align_of::<MaxAlign>());

        #[repr(C)]
        struct InOrder<T, U>(T, U);

        // The empty slice is static and contains a single `0` usize (for the
        // length) that is 64-byte aligned, thus featuring the necessary
        // trailing padding for elements with up to 64-byte alignment.
        static EMPTY_SLICE: InOrder<usize, MaxAlign> = InOrder(0, MaxAlign);
        unsafe { &*(&EMPTY_SLICE as *const _ as *const List<T>) }
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
            ptr::addr_of_mut!((*mem).len).write(slice.len());

            // Write the elements
            ptr::addr_of_mut!((*mem).data)
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

impl<S: Encoder, T: Encodable<S>> Encodable<S> for List<T> {
    #[inline]
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for &List<T> {
    #[inline]
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
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
        (self as *const List<T>).hash(s)
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
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
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

unsafe impl<'a, T: 'a> rustc_data_structures::tagged_ptr::Pointer for &'a List<T> {
    const BITS: usize = std::mem::align_of::<usize>().trailing_zeros() as usize;

    #[inline]
    fn into_usize(self) -> usize {
        self as *const List<T> as usize
    }

    #[inline]
    unsafe fn from_usize(ptr: usize) -> &'a List<T> {
        &*(ptr as *const List<T>)
    }

    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        // `Self` is `&'a List<T>` which impls `Copy`, so this is fine.
        let ptr = Self::from_usize(ptr);
        f(&ptr)
    }
}
