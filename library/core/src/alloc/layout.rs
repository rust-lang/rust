// Seemingly inconsequential code changes to this file can lead to measurable
// performance impact on compilation times, due at least in part to the fact
// that the layout code gets called from many instantiations of the various
// collections, resulting in having to optimize down excess IR multiple times.
// Your performance intuition is useless. Run perf.

use crate::error::Error;
use crate::intrinsics::{unchecked_add, unchecked_mul, unchecked_sub};
use crate::mem::SizedTypeProperties;
use crate::ptr::{Alignment, NonNull};
use crate::{assert_unsafe_precondition, fmt, mem};

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
/// All layouts have an associated size and a power-of-two alignment. The size, when rounded up to
/// the nearest multiple of `align`, does not overflow `isize` (i.e., the rounded value will always be
/// less than or equal to `isize::MAX`).
///
/// (Note that layouts are *not* required to have non-zero size,
/// even though `GlobalAlloc` requires that all memory requests
/// be non-zero in size. A caller must either ensure that conditions
/// like this are met, use specific allocators with looser
/// requirements, or use the more lenient `Allocator` interface.)
#[stable(feature = "alloc_layout", since = "1.28.0")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[lang = "alloc_layout"]
pub struct Layout {
    // size of the requested block of memory, measured in bytes.
    size: usize,

    // alignment of the requested block of memory, measured in bytes.
    // we ensure that this is always a power-of-two, because API's
    // like `posix_memalign` require it and it is a reasonable
    // constraint to impose on Layout constructors.
    //
    // (However, we do not analogously require `align >= sizeof(void*)`,
    //  even though that is *also* a requirement of `posix_memalign`.)
    align: Alignment,
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
    ///    must not overflow `isize` (i.e., the rounded value must be
    ///    less than or equal to `isize::MAX`).
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout_size_align", since = "1.50.0")]
    #[inline]
    pub const fn from_size_align(size: usize, align: usize) -> Result<Self, LayoutError> {
        if Layout::is_size_align_valid(size, align) {
            // SAFETY: Layout::is_size_align_valid checks the preconditions for this call.
            unsafe { Ok(Layout { size, align: mem::transmute(align) }) }
        } else {
            Err(LayoutError)
        }
    }

    const fn is_size_align_valid(size: usize, align: usize) -> bool {
        let Some(align) = Alignment::new(align) else { return false };
        if size > Self::max_size_for_align(align) {
            return false;
        }
        true
    }

    #[inline(always)]
    const fn max_size_for_align(align: Alignment) -> usize {
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

        // SAFETY: the maximum possible alignment is `isize::MAX + 1`,
        // so the subtraction cannot overflow.
        unsafe { unchecked_sub(isize::MAX as usize + 1, align.as_usize()) }
    }

    /// Internal helper constructor to skip revalidating alignment validity.
    #[inline]
    const fn from_size_alignment(size: usize, align: Alignment) -> Result<Self, LayoutError> {
        if size > Self::max_size_for_align(align) {
            return Err(LayoutError);
        }

        // SAFETY: Layout::size invariants checked above.
        Ok(Layout { size, align })
    }

    /// Creates a layout, bypassing all checks.
    ///
    /// # Safety
    ///
    /// This function is unsafe as it does not verify the preconditions from
    /// [`Layout::from_size_align`].
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout_unchecked", since = "1.36.0")]
    #[must_use]
    #[inline]
    pub const unsafe fn from_size_align_unchecked(size: usize, align: usize) -> Self {
        assert_unsafe_precondition!(
            check_library_ub,
            "Layout::from_size_align_unchecked requires that align is a power of 2 \
            and the rounded-up allocation size does not exceed isize::MAX",
            (
                size: usize = size,
                align: usize = align,
            ) => Layout::is_size_align_valid(size, align)
        );
        // SAFETY: the caller is required to uphold the preconditions.
        unsafe { Layout { size, align: mem::transmute(align) } }
    }

    /// The minimum size in bytes for a memory block of this layout.
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout_size_align", since = "1.50.0")]
    #[must_use]
    #[inline]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// The minimum byte alignment for a memory block of this layout.
    ///
    /// The returned alignment is guaranteed to be a power of two.
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout_size_align", since = "1.50.0")]
    #[must_use = "this returns the minimum alignment, \
                  without modifying the layout"]
    #[inline]
    pub const fn align(&self) -> usize {
        self.align.as_usize()
    }

    /// Constructs a `Layout` suitable for holding a value of type `T`.
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "alloc_layout_const_new", since = "1.42.0")]
    #[must_use]
    #[inline]
    pub const fn new<T>() -> Self {
        let (size, align) = size_align::<T>();
        // SAFETY: if the type is instantiated, rustc already ensures that its
        // layout is valid. Use the unchecked constructor to avoid inserting a
        // panicking codepath that needs to be optimized out.
        unsafe { Layout::from_size_align_unchecked(size, align) }
    }

    /// Produces layout describing a record that could be used to
    /// allocate backing structure for `T` (which could be a trait
    /// or other unsized type like a slice).
    #[stable(feature = "alloc_layout", since = "1.28.0")]
    #[rustc_const_stable(feature = "const_alloc_layout", since = "CURRENT_RUSTC_VERSION")]
    #[must_use]
    #[inline]
    pub const fn for_value<T: ?Sized>(t: &T) -> Self {
        let (size, align) = (mem::size_of_val(t), mem::align_of_val(t));
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
    ///     - a [slice], then the length of the slice tail must be an initialized
    ///       integer, and the size of the *entire value*
    ///       (dynamic tail length + statically sized prefix) must fit in `isize`.
    ///       For the special case where the dynamic tail length is 0, this function
    ///       is safe to call.
    ///     - a [trait object], then the vtable part of the pointer must point
    ///       to a valid vtable for the type `T` acquired by an unsizing coercion,
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
    #[must_use]
    pub const unsafe fn for_value_raw<T: ?Sized>(t: *const T) -> Self {
        // SAFETY: we pass along the prerequisites of these functions to the caller
        let (size, align) = unsafe { (mem::size_of_val_raw(t), mem::align_of_val_raw(t)) };
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
    #[must_use]
    #[inline]
    pub const fn dangling(&self) -> NonNull<u8> {
        // SAFETY: align is guaranteed to be non-zero
        unsafe { NonNull::new_unchecked(crate::ptr::without_provenance_mut::<u8>(self.align())) }
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
    #[rustc_const_stable(feature = "const_alloc_layout", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn align_to(&self, align: usize) -> Result<Self, LayoutError> {
        if let Some(align) = Alignment::new(align) {
            Layout::from_size_alignment(self.size, Alignment::max(self.align, align))
        } else {
            Err(LayoutError)
        }
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
    #[must_use = "this returns the padding needed, \
                  without modifying the `Layout`"]
    #[inline]
    pub const fn padding_needed_for(&self, align: usize) -> usize {
        // FIXME: Can we just change the type on this to `Alignment`?
        let Some(align) = Alignment::new(align) else { return usize::MAX };
        let len_rounded_up = self.size_rounded_up_to_custom_align(align);
        // SAFETY: Cannot overflow because the rounded-up value is never less
        unsafe { unchecked_sub(len_rounded_up, self.size) }
    }

    /// Returns the smallest multiple of `align` greater than or equal to `self.size()`.
    ///
    /// This can return at most `Alignment::MAX` (aka `isize::MAX + 1`)
    /// because the original size is at most `isize::MAX`.
    #[inline]
    const fn size_rounded_up_to_custom_align(&self, align: Alignment) -> usize {
        // SAFETY:
        // Rounded up value is:
        //   size_rounded_up = (size + align - 1) & !(align - 1);
        //
        // The arithmetic we do here can never overflow:
        //
        // 1. align is guaranteed to be > 0, so align - 1 is always
        //    valid.
        //
        // 2. size is at most `isize::MAX`, so adding `align - 1` (which is at
        //    most `isize::MAX`) can never overflow a `usize`.
        //
        // 3. masking by the alignment can remove at most `align - 1`,
        //    which is what we just added, thus the value we return is never
        //    less than the original `size`.
        //
        // (Size 0 Align MAX is already aligned, so stays the same, but things like
        // Size 1 Align MAX or Size isize::MAX Align 2 round up to `isize::MAX + 1`.)
        unsafe {
            let align_m1 = unchecked_sub(align.as_usize(), 1);
            let size_rounded_up = unchecked_add(self.size, align_m1) & !align_m1;
            size_rounded_up
        }
    }

    /// Creates a layout by rounding the size of this layout up to a multiple
    /// of the layout's alignment.
    ///
    /// This is equivalent to adding the result of `padding_needed_for`
    /// to the layout's current size.
    #[stable(feature = "alloc_layout_manipulation", since = "1.44.0")]
    #[rustc_const_stable(feature = "const_alloc_layout", since = "CURRENT_RUSTC_VERSION")]
    #[must_use = "this returns a new `Layout`, \
                  without modifying the original"]
    #[inline]
    pub const fn pad_to_align(&self) -> Layout {
        // This cannot overflow. Quoting from the invariant of Layout:
        // > `size`, when rounded up to the nearest multiple of `align`,
        // > must not overflow isize (i.e., the rounded value must be
        // > less than or equal to `isize::MAX`)
        let new_size = self.size_rounded_up_to_custom_align(self.align);

        // SAFETY: padded size is guaranteed to not exceed `isize::MAX`.
        unsafe { Layout::from_size_align_unchecked(new_size, self.align()) }
    }

    /// Creates a layout describing the record for `n` instances of
    /// `self`, with a suitable amount of padding between each to
    /// ensure that each instance is given its requested size and
    /// alignment. On success, returns `(k, offs)` where `k` is the
    /// layout of the array and `offs` is the distance between the start
    /// of each element in the array.
    ///
    /// (That distance between elements is sometimes known as "stride".)
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(alloc_layout_extra)]
    /// use std::alloc::Layout;
    ///
    /// // All rust types have a size that's a multiple of their alignment.
    /// let normal = Layout::from_size_align(12, 4).unwrap();
    /// let repeated = normal.repeat(3).unwrap();
    /// assert_eq!(repeated, (Layout::from_size_align(36, 4).unwrap(), 12));
    ///
    /// // But you can manually make layouts which don't meet that rule.
    /// let padding_needed = Layout::from_size_align(6, 4).unwrap();
    /// let repeated = padding_needed.repeat(3).unwrap();
    /// assert_eq!(repeated, (Layout::from_size_align(24, 4).unwrap(), 8));
    /// ```
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    pub const fn repeat(&self, n: usize) -> Result<(Self, usize), LayoutError> {
        let padded = self.pad_to_align();
        if let Ok(repeated) = padded.repeat_packed(n) {
            Ok((repeated, padded.size()))
        } else {
            Err(LayoutError)
        }
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
    #[rustc_const_stable(feature = "const_alloc_layout", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn extend(&self, next: Self) -> Result<(Self, usize), LayoutError> {
        let new_align = Alignment::max(self.align, next.align);
        let offset = self.size_rounded_up_to_custom_align(next.align);

        // SAFETY: `offset` is at most `isize::MAX + 1` (such as from aligning
        // to `Alignment::MAX`) and `next.size` is at most `isize::MAX` (from the
        // `Layout` type invariant).  Thus the largest possible `new_size` is
        // `isize::MAX + 1 + isize::MAX`, which is `usize::MAX`, and cannot overflow.
        let new_size = unsafe { unchecked_add(offset, next.size) };

        if let Ok(layout) = Layout::from_size_alignment(new_size, new_align) {
            Ok((layout, offset))
        } else {
            Err(LayoutError)
        }
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
    pub const fn repeat_packed(&self, n: usize) -> Result<Self, LayoutError> {
        if let Some(size) = self.size.checked_mul(n) {
            // The safe constructor is called here to enforce the isize size limit.
            Layout::from_size_alignment(size, self.align)
        } else {
            Err(LayoutError)
        }
    }

    /// Creates a layout describing the record for `self` followed by
    /// `next` with no additional padding between the two. Since no
    /// padding is inserted, the alignment of `next` is irrelevant,
    /// and is not incorporated *at all* into the resulting layout.
    ///
    /// On arithmetic overflow, returns `LayoutError`.
    #[unstable(feature = "alloc_layout_extra", issue = "55724")]
    #[inline]
    pub const fn extend_packed(&self, next: Self) -> Result<Self, LayoutError> {
        // SAFETY: each `size` is at most `isize::MAX == usize::MAX/2`, so the
        // sum is at most `usize::MAX/2*2 == usize::MAX - 1`, and cannot overflow.
        let new_size = unsafe { unchecked_add(self.size, next.size) };
        // The safe constructor enforces that the new size isn't too big for the alignment
        Layout::from_size_alignment(new_size, self.align)
    }

    /// Creates a layout describing the record for a `[T; n]`.
    ///
    /// On arithmetic overflow or when the total size would exceed
    /// `isize::MAX`, returns `LayoutError`.
    #[stable(feature = "alloc_layout_manipulation", since = "1.44.0")]
    #[rustc_const_stable(feature = "const_alloc_layout", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn array<T>(n: usize) -> Result<Self, LayoutError> {
        // Reduce the amount of code we need to monomorphize per `T`.
        return inner(T::LAYOUT, n);

        #[inline]
        const fn inner(element_layout: Layout, n: usize) -> Result<Layout, LayoutError> {
            let Layout { size: element_size, align } = element_layout;

            // We need to check two things about the size:
            //  - That the total size won't overflow a `usize`, and
            //  - That the total size still fits in an `isize`.
            // By using division we can check them both with a single threshold.
            // That'd usually be a bad idea, but thankfully here the element size
            // and alignment are constants, so the compiler will fold all of it.
            if element_size != 0 && n > Layout::max_size_for_align(align) / element_size {
                return Err(LayoutError);
            }

            // SAFETY: We just checked that we won't overflow `usize` when we multiply.
            // This is a useless hint inside this function, but after inlining this helps
            // deduplicate checks for whether the overall capacity is zero (e.g., in RawVec's
            // allocation path) before/after this multiplication.
            let array_size = unsafe { unchecked_mul(element_size, n) };

            // SAFETY: We just checked above that the `array_size` will not
            // exceed `isize::MAX` even when rounded up to the alignment.
            // And `Alignment` guarantees it's a power of two.
            unsafe { Ok(Layout::from_size_align_unchecked(array_size, align.as_usize())) }
        }
    }
}

#[stable(feature = "alloc_layout", since = "1.28.0")]
#[deprecated(
    since = "1.52.0",
    note = "Name does not follow std convention, use LayoutError",
    suggestion = "LayoutError"
)]
pub type LayoutErr = LayoutError;

/// The `LayoutError` is returned when the parameters given
/// to `Layout::from_size_align`
/// or some other `Layout` constructor
/// do not satisfy its documented constraints.
#[stable(feature = "alloc_layout_error", since = "1.50.0")]
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayoutError;

#[stable(feature = "alloc_layout", since = "1.28.0")]
impl Error for LayoutError {}

// (we need this for downstream impl of trait Error)
#[stable(feature = "alloc_layout", since = "1.28.0")]
impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid parameters to Layout::from_size_align")
    }
}
