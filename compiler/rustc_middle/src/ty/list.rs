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

extern "C" {
    /// A dummy type used to force `List` to be unsized while not requiring references to it be wide
    /// pointers.
    type OpaqueListContents;
}

/// A wrapper for slices with the additional invariant
/// that the slice is interned and no other slice with
/// the same contents can exist in the same context.
/// This means we can use pointer for both
/// equality comparisons and hashing.
///
/// Unlike slices, the types contained in `List` are expected to be `Copy`
/// and iterating over a `List` returns `T` instead of a reference.
///
/// Note: `Slice` was already taken by the `Ty`.
#[repr(C)]
pub struct List<T> {
    len: usize,
    data: [T; 0],
    opaque: OpaqueListContents,
}

unsafe impl<'a, T: 'a> rustc_data_structures::tagged_ptr::Pointer for &'a List<T> {
    const BITS: usize = std::mem::align_of::<usize>().trailing_zeros() as usize;
    fn into_usize(self) -> usize {
        self as *const List<T> as usize
    }
    unsafe fn from_usize(ptr: usize) -> Self {
        &*(ptr as *const List<T>)
    }
    unsafe fn with_ref<R, F: FnOnce(&Self) -> R>(ptr: usize, f: F) -> R {
        // Self: Copy so this is fine
        let ptr = Self::from_usize(ptr);
        f(&ptr)
    }
}

unsafe impl<T: Sync> Sync for List<T> {}

impl<T: Copy> List<T> {
    #[inline]
    pub(super) fn from_arena<'tcx>(arena: &'tcx Arena<'tcx>, slice: &[T]) -> &'tcx List<T> {
        assert!(!mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() != 0);
        assert!(!slice.is_empty());

        let (layout, _offset) =
            Layout::new::<usize>().extend(Layout::for_value::<[T]>(slice)).unwrap();
        let mem = arena.dropless.alloc_raw(layout);
        unsafe {
            let result = &mut *(mem as *mut List<T>);
            // Write the length
            result.len = slice.len();

            // Write the elements
            let arena_slice = slice::from_raw_parts_mut(result.data.as_mut_ptr(), result.len);
            arena_slice.copy_from_slice(slice);

            result
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

impl<T> Ord for List<T>
where
    T: Ord,
{
    fn cmp(&self, other: &List<T>) -> Ordering {
        if self == other { Ordering::Equal } else { <[T] as Ord>::cmp(&**self, &**other) }
    }
}

impl<T> PartialOrd for List<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &List<T>) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else {
            <[T] as PartialOrd>::partial_cmp(&**self, &**other)
        }
    }
}

impl<T: PartialEq> PartialEq for List<T> {
    #[inline]
    fn eq(&self, other: &List<T>) -> bool {
        ptr::eq(self, other)
    }
}
impl<T: Eq> Eq for List<T> {}

impl<T> Hash for List<T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
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

impl<T> List<T> {
    #[inline(always)]
    pub fn empty<'a>() -> &'a List<T> {
        #[repr(align(64), C)]
        struct EmptySlice([u8; 64]);
        static EMPTY_SLICE: EmptySlice = EmptySlice([0; 64]);
        assert!(mem::align_of::<T>() <= 64);
        unsafe { &*(&EMPTY_SLICE as *const _ as *const List<T>) }
    }
}
