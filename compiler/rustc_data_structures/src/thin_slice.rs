use rustc_serialize::{Encodable, Encoder};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::slice;

/// A thin length-prefixed slice. Pointers to this slice are just
/// one ptr-size wide.
#[repr(C)]
pub struct ThinSlice<T> {
    len: usize,

    /// Although this claims to be a zero-length array, in practice `len`
    /// elements are actually present.
    data: [T; 0],

    opaque: OpaqueSliceContents,
}

extern "C" {
    /// A dummy type used to force `List` to be unsized while not requiring
    /// references to it be wide pointers.
    type OpaqueSliceContents;
}

impl<T> ThinSlice<T> {
    /// Returns a reference to the (unique, static) empty list.
    #[inline(always)]
    pub fn empty<'a>() -> &'a Self {
        #[repr(align(64))]
        struct MaxAlign;

        assert!(mem::align_of::<T>() <= mem::align_of::<MaxAlign>());

        #[repr(C)]
        struct InOrder<T, U>(T, U);

        // The empty slice is static and contains a single `0` usize (for the
        // length) that is 64-byte aligned, thus featuring the necessary
        // trailing padding for elements with up to 64-byte alignment.
        static EMPTY_SLICE: InOrder<usize, MaxAlign> = InOrder(0, MaxAlign);
        unsafe { &*(&EMPTY_SLICE as *const _ as *const Self) }
    }

        /// Returns a reference an empty list. Is not guaranteed to be the same for all empty slices.
        #[inline(always)]
        pub const fn const_empty<'a>() -> &'a Self {
            #[repr(align(64))]
            struct MaxAlign;
    
            assert!(mem::align_of::<T>() <= mem::align_of::<MaxAlign>());
    
            #[repr(C)]
            struct InOrder<T, U>(T, U);
    
            // The empty slice is static and contains a single `0` usize (for the
            // length) that is 64-byte aligned, thus featuring the necessary
            // trailing padding for elements with up to 64-byte alignment.
            const EMPTY_SLICE: InOrder<usize, MaxAlign> = InOrder(0, MaxAlign);
            unsafe { &*(&EMPTY_SLICE as *const _ as *const Self) }
        }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Initializes the thin slice from the slice into the memory location specified in `mem`.
    ///
    /// # Safety
    /// `mem` must be valid for writes of the length of a usize + padding + the slice.
    /// The caller must ensure that the memory remains valid of the duration of the lifetime.
    pub unsafe fn initialize<'a>(mem: *mut Self, slice: &[T]) -> &'a Self {
        // Write the length
        ptr::addr_of_mut!((*mem).len).write(slice.len());

        // Write the elements
        ptr::addr_of_mut!((*mem).data)
            .cast::<T>()
            .copy_from_nonoverlapping(slice.as_ptr(), slice.len());

        &*mem
    }

    // If this method didn't exist, we would use `slice.iter` due to
    // deref coercion.
    //
    // This would be weird, as `self.into_iter` iterates over `T` directly.
    #[inline(always)]
    pub fn iter(&self) -> <&'_ Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<T> Default for &ThinSlice<T> {
    fn default() -> Self {
        ThinSlice::empty()
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for ThinSlice<T> {
    #[inline]
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}

impl<T: PartialEq> PartialEq for ThinSlice<T> {
    #[inline]
    fn eq(&self, other: &ThinSlice<T>) -> bool {
        &*self == &*other
    }
}

impl<T: Eq> Eq for ThinSlice<T> {}

impl<T> Ord for ThinSlice<T>
where
    T: Ord,
{
    fn cmp(&self, other: &ThinSlice<T>) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T> PartialOrd for ThinSlice<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &ThinSlice<T>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Hash> Hash for ThinSlice<T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        self.as_slice().hash(s)
    }
}

impl<T> Deref for ThinSlice<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T> AsRef<[T]> for ThinSlice<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl<'a, T> IntoIterator for &'a ThinSlice<T> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self[..].iter()
    }
}

unsafe impl<T: Sync> Sync for ThinSlice<T> {}

impl<T: fmt::Debug> fmt::Debug for ThinSlice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}
