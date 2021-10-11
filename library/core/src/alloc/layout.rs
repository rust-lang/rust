use crate::cmp;
use crate::fmt;
use crate::mem;
use crate::num::NonZeroUsize;
use crate::ptr::NonNull;

// While this function is used in one place and its implementation
// could be inlined, the previous attempts to do so made rustc
// slower:
//
// * https://github.com/rust-lang/rust/pull/72189
// * https://github.com/rust-lang/rust/pull/79827
const fn size_align<T>() -> (usize, usize) {
    (mem::size_of::<T>(), mem::align_of::<T>())
}

/// Layout of a block of memory.
///
/// An instance of `Layout` describes a particular layout of memory.
/// You build a `Layout` up as an input to give to an allocator.
///
/// All layouts have an associated size and a power-of-two alignment.
///
/// (Note that layouts are *not* required to have non-zero size,
/// even though `GlobalAlloc` requires that all memory requests
/// be non-zero in size. A caller must either ensure that conditions
/// like this are met, use specific allocators with looser
/// requirements, or use the more lenient `Allocator` interface.)
#[stable(feature = "alloc_layout", since = "1.28.0")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[lang = "alloc_layout"]
pub struct Layout {
    // size of the requested block of memory, measured in bytes.
    size_: usize,

    // alignment of the requested block of memory, measured in bytes.
    // we ensure that this is always a power-of-two, because API's
    // like `posix_memalign` require it and it is a reasonable
    // constraint to impose on Layout constructors.
    //
    // (However, we do not analogously require `align >= sizeof(void*)`,
    //  even though that is *also* a requirement of `posix_memalign`.)
    align_: NonZeroUsize,
}

impl Layout {
    /// Constructs a `Layout` from a given `size` and `align`,
    /// or returns `LayoutError` if any of the following conditions
    /// are not met:
    ///
    /// * `align` must not be zero,
    ///
    /// * `align` must be a power of two,
    ///
    /// * `size`, when rounded up to the nearest multiple of `align`,
    ///    must not overflow (i.e., the rounded value must be less than
    ///    or equal to `usize::MAX`).
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout", since = "1.50.0")]
    #[inline]
    pub const fn from_size_align(size: usize, align: usize) -> Result<Self, LayoutError> {
        if !align.is_power_of_two() {
            return Err(LayoutError);
        }

        // (power-of-two implies align != 0.)

        // Rounded up size is:
        //   size_rounded_up = (size + align - 1) & !(align - 1);
        //
        // We know from above that align != 0. If adding (align - 1)
        // does not overflow, then rounding up will be fine.
        //
        // Conversely, &-masking with !(align - 1) will subtract off
        // only low-order-bits. Thus if overflow occurs with the sum,
        // the &-mask cannot subtract enough to undo that overflow.
        //
        // Above implies that checking for summation overflow is both
        // necessary and sufficient.
        if size > usize::MAX - (align - 1) {
            return Err(LayoutError);
        }

        // SAFETY: the conditions for `from_size_align_unchecked` have been
        // checked above.
        unsafe { Ok(Layout::from_size_align_unchecked(size, align)) }
    }

    /// Creates a layout, bypassing all checks.
    ///
    /// # Safety
    ///
    /// This function is unsafe as it does not verify the preconditions from
    /// [`Layout::from_size_align`].
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "alloc_layout", since = "1.36.0")]
    #[must_use]
    #[inline]
    pub const unsafe fn from_size_align_unchecked(size: usize, align: usize) -> Self {
        // SAFETY: the caller must ensure that `align` is greater than zero.
        Layout { size_: size, align_: unsafe { NonZeroUsize::new_unchecked(align) } }
    }

    /// The minimum size in bytes for a memory block of this layout.
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout", since = "1.50.0")]
    #[inline]
    pub const fn size(&self) -> usize {
        self.size_
    }

    /// The minimum byte alignment for a memory block of this layout.
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout", since = "1.50.0")]
    #[inline]
    pub const fn align(&self) -> usize {
        self.align_.get()
    }

    /// Constructs a `Layout` suitable for holding a value of type `T`.
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "alloc_layout_const_new", since = "1.42.0")]
    #[must_use]
    #[inline]
    pub const fn new<T>() -> Self {
        let (size, align) = size_align::<T>();
        // SAFETY: the align is guaranteed by Rust to be a power of two and
        // the size+align combo is guaranteed to fit in our address space. As a
        // result use the unchecked constructor here to avoid inserting code
        // that panics if it isn't optimized well enough.
        unsafe { Layout::from_size_align_unchecked(size, align) }
    }

    /// Produces layout describing a record that could be used to
    /// allocate backing structure for `T` (which could be a trait
    /// or other unsized type like a slice).
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[inline]
    pub fn for_value<T: ?Sized>(t: &T) -> Self {
        let (size, align) = (mem::size_of_val(t), mem::align_of_val(t));
        debug_assert!(Layout::from_size_align(size, align).is_ok());
        // SAFETY: see rationale in `new` for why this is using the unsafe variant
        unsafe { Layout::from_size_align_unchecked(size, align) }
    }

    /// Produces layout describing a record that could be used to
    /// allocate backing structure for `T` (which could be a trait
    /// or other unsized type like a slice).
    ///
    /// # Safety
    ///
    /// This function is only safe to call if the following conditions hold:
    ///
    /// - If `T` is `Sized`, this function is always safe to call.
    /// - If the unsized tail of `T` is:
    ///     - a [slice], then the length of the slice tail must be an intialized
    ///       integer, and the size of the *entire value*
    ///       (dynamic tail length + statically sized prefix) must fit in `isize`.
    ///     - a [trait object], then the vtable part of the pointer must point
    ///       to a valid vtable for the type `T` acquired by an unsizing coersion,
    ///       and the size of the *entire value*
    ///       (dynamic tail length + statically sized prefix) must fit in `isize`.
    ///     - an (unstable) [extern type], then this function is always safe to
    ///       call, but may panic or otherwise return the wrong value, as the
    ///       extern type's layout is not known. This is the same behavior as
    ///       [`Layout::for_value`] on a reference to an extern type tail.
    ///     - otherwise, it is conservatively not allowed to call this function.
    ///
    /// [trait object]: ../../book/ch17-02-trait-objects.html
    /// [extern type]: ../../unstable-book/language-features/extern-types.html
    #[unstable(feature = "layout_for_ptr", issue = "69835")]
    pub unsafe fn for_value_raw<T: ?Sized>(t: *const T) -> Self {
        // SAFETY: we pass along the prerequisites of these functions to the caller
        let (size, align) = unsafe { (mem::size_of_val_raw(t), mem::align_of_val_raw(t)) };
        debug_assert!(Layout::from_size_align(size, align).is_ok());
        // SAFETY: see rationale in `new` for why this is using the unsafe variant
        unsafe { Layout::from_size_align_unchecked(size, align) }
    }

    /// Creates a `NonNull` that is dangling, but well-aligned for this Layout.
    ///
    /// Note that the pointer value may potentially represent a valid pointer,
    /// which means this must not be used as a "not yet initialized"
    /// sentinel value. Types that lazily allocate must track initialization by
    /// some other means.
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[rustc_const_unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    pub const fn dangling(&self) -> NonNull<u8> {
        // SAFETY: align is guaranteed to be non-zero
        unsafe { NonNull::new_unchecked(self.align() as *mut u8) }
    }

    /// Creates a layout describing the record that can hold a value
    /// of the same layout as `self`, but that also is aligned to
    /// alignment `align` (measured in bytes).
    ///
    /// If `self` already meets the prescribed alignment, then returns
    /// `self`.
    ///
    /// Note that this method does not add any padding to the overall
    /// size, regardless of whether the returned layout has a different
    /// alignment. In other words, if `K` has size 16, `K.align_to(32)`
    /// will *still* have size 16.
    ///
    /// Returns an error if the combination of `self.size()` and the given
    /// `align` violates the conditions listed in [`Layout::from_size_align`].
    #[stable(feature = "alloc_layout_manipulation", since = "1.44.0")]
    #[inline]
    pub fn align_to(&self, align: usize) -> Result<Self, LayoutError> {
        Layout::from_size_align(self.size(), cmp::max(self.align(), align))
    }

    /// Returns the amount of padding we must insert after `self`
    /// to ensure that the following address will satisfy `align`
    /// (measured in bytes).
    ///
    /// e.g., if `self.size()` is 9, then `self.padding_needed_for(4)`
    /// returns 3, because that is the minimum number of bytes of
    /// padding required to get a 4-aligned address (assuming that the
    /// corresponding memory block starts at a 4-aligned address).
    ///
    /// The return value of this function has no meaning if `align` is
    /// not a power-of-two.
    ///
    /// Note that the utility of the returned value requires `align`
    /// to be less than or equal to the alignment of the starting
    /// address for the whole allocated block of memory. One way to
    /// satisfy this constraint is to ensure `align <= self.align()`.
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[rustc_const_unstable(feature = "const_alloc_layout", issue = "67521")]
    #[inline]
    pub const fn padding_needed_for(&self, align: usize) -> usize {
        let len = self.size();

        // Rounded up value is:
        //   len_rounded_up = (len + align - 1) & !(align - 1);
        // and then we return the padding difference: `len_rounded_up - len`.
        //
        // We use modular arithmetic throughout:
        //
        // 1. align is guaranteed to be > 0, so align - 1 is always
        //    valid.
        //
        // 2. `len + align - 1` can overflow by at most `align - 1`,
        //    so the &-mask with `!(align - 1)` will ensure that in the
        //    case of overflow, `len_rounded_up` will itself be 0.
        //    Thus the returned padding, when added to `len`, yields 0,
        //    which trivially satisfies the alignment `align`.
        //
        // (Of course, attempts to allocate blocks of memory whose
        // size and padding overflow in the above manner should cause
        // the allocator to yield an error anyway.)

        let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
        len_rounded_up.wrapping_sub(len)
    }

    /// Creates a layout by rounding the size of this layout up to a multiple
    /// of the layout's alignment.
    ///
    /// This is equivalent to adding the result of `padding_needed_for`
    /// to the layout's current size.
    #[stable(feature = "alloc_layout_manipulation", since = "1.44.0")]
    #[inline]
    pub fn pad_to_align(&self) -> Layout {
        let pad = self.padding_needed_for(self.align());
        // This cannot overflow. Quoting from the invariant of Layout:
        // > `size`, when rounded up to the nearest multiple of `align`,
        // > must not overflow (i.e., the rounded value must be less than
        // > `usize::MAX`)
        let new_size = self.size() + pad;

        Layout::from_size_align(new_size, self.align()).unwrap()
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with a suitable amount of padding between each to
    /// ensure that each instance is given its requested size and
    /// alignment. On success, returns `(k, offs)` where `k` is the
    /// layout of the array and `offs` is the distance between the start
    /// of each element in the array.
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    pub fn repeat(&self, n: usize) -> Result<(Self, usize), LayoutError> {
        // This cannot overflow. Quoting from the invariant of Layout:
        // > `size`, when rounded up to the nearest multiple of `align`,
        // > must not overflow (i.e., the rounded value must be less than
        // > `usize::MAX`)
        let padded_size = self.size() + self.padding_needed_for(self.align());
        let alloc_size = padded_size.checked_mul(n).ok_or(LayoutError)?;

        // SAFETY: self.align is already known to be valid and alloc_size has been
        // padded already.
        unsafe { Ok((Layout::from_size_align_unchecked(alloc_size, self.align()), padded_size)) }
    }

    /// Creates a layout describing the record for `self` followed by
    /// `next`, including any necessary padding to ensure that `next`
    /// will be properly aligned, but *no trailing padding*.
    ///
    /// In order to match C representation layout `repr(C)`, you should
    /// call `pad_to_align` after extending the layout with all fields.
    /// (There is no way to match the default Rust representation
    /// layout `repr(Rust)`, as it is unspecified.)
    ///
    /// Note that the alignment of the resulting layout will be the maximum of
    /// those of `self` and `next`, in order to ensure alignment of both parts.
    ///
    /// Returns `Ok((k, offset))`, where `k` is layout of the concatenated
    /// record and `offset` is the relative location, in bytes, of the
    /// start of the `next` embedded within the concatenated record
    /// (assuming that the record itself starts at offset 0).
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    ///
    /// # Examples
    ///
    /// To calculate the layout of a `#[repr(C)]` structure and the offsets of
    /// the fields from its fields' layouts:
    ///
    /// ```rust
    /// # use std::alloc::{Layout, LayoutError};
    /// pub fn repr_c(fields: &[Layout]) -> Result<(Layout, Vec<usize>), LayoutError> {
    ///     let mut offsets = Vec::new();
    ///     let mut layout = Layout::from_size_align(0, 1)?;
    ///     for &field in fields {
    ///         let (new_layout, offset) = layout.extend(field)?;
    ///         layout = new_layout;
    ///         offsets.push(offset);
    ///     }
    ///     // Remember to finalize with `pad_to_align`!
    ///     Ok((layout.pad_to_align(), offsets))
    /// }
    /// # // test that it works
    /// # #[repr(C)] struct S { a: u64, b: u32, c: u16, d: u32 }
    /// # let s = Layout::new::<S>();
    /// # let u16 = Layout::new::<u16>();
    /// # let u32 = Layout::new::<u32>();
    /// # let u64 = Layout::new::<u64>();
    /// # assert_eq!(repr_c(&[u64, u32, u16, u32]), Ok((s, vec![0, 8, 12, 16])));
    /// ```
    #[stable(feature = "alloc_layout_manipulation", since = "1.44.0")]
    #[inline]
    pub fn extend(&self, next: Self) -> Result<(Self, usize), LayoutError> {
        let new_align = cmp::max(self.align(), next.align());
        let pad = self.padding_needed_for(next.align());

        let offset = self.size().checked_add(pad).ok_or(LayoutError)?;
        let new_size = offset.checked_add(next.size()).ok_or(LayoutError)?;

        let layout = Layout::from_size_align(new_size, new_align)?;
        Ok((layout, offset))
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with no padding between each instance.
    ///
    /// Note that, unlike `repeat`, `repeat_packed` does not guarantee
    /// that the repeated instances of `self` will be properly
    /// aligned, even if a given instance of `self` is properly
    /// aligned. In other words, if the layout returned by
    /// `repeat_packed` is used to allocate an array, it is not
    /// guaranteed that all elements in the array will be properly
    /// aligned.
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    pub fn repeat_packed(&self, n: usize) -> Result<Self, LayoutError> {
        let size = self.size().checked_mul(n).ok_or(LayoutError)?;
        Layout::from_size_align(size, self.align())
    }

    /// Creates a layout describing the record for `self` followed by
    /// `next` with no additional padding between the two. Since no
    /// padding is inserted, the alignment of `next` is irrelevant,
    /// and is not incorporated *at all* into the resulting layout.
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    pub fn extend_packed(&self, next: Self) -> Result<Self, LayoutError> {
        let new_size = self.size().checked_add(next.size()).ok_or(LayoutError)?;
        Layout::from_size_align(new_size, self.align())
    }

    /// Creates a layout describing the record for a `[T; n]`.
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    #[stable(feature = "alloc_layout_manipulation", since = "1.44.0")]
    #[inline]
    pub fn array<T>(n: usize) -> Result<Self, LayoutError> {
        let (layout, offset) = Layout::new::<T>().repeat(n)?;
        debug_assert_eq!(offset, mem::size_of::<T>());
        Ok(layout.pad_to_align())
    }
}

#[stable(feature = "alloc_layout", since = "1.28.0")]
#[rustc_deprecated(
    since = "1.52.0",
    reason = "Name does not follow std convention, use LayoutError",
    suggestion = "LayoutError"
)]
pub type LayoutErr = LayoutError;

/// The parameters given to `Layout::from_size_align`
/// or some other `Layout` constructor
/// do not satisfy its documented constraints.
#[stable(feature = "alloc_layout_error", since = "1.50.0")]
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayoutError;

// (we need this for downstream impl of trait Error)
#[stable(feature = "alloc_layout", since = "1.28.0")]
impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid parameters to Layout::from_size_align")
    }
}
