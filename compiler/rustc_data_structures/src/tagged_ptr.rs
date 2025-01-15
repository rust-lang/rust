//! This module implements tagged pointers. In order to utilize the pointer
//! packing, you must have a tag type implementing the [`Tag`] trait.
//!
//! We assert that the tag and the reference type is compatible at compile
//! time.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::Deref;
use std::ptr::NonNull;

use crate::aligned::Aligned;
use crate::stable_hasher::{HashStable, StableHasher};

/// This describes tags that the [`TaggedRef`] struct can hold.
///
/// # Safety
///
/// - The [`BITS`] constant must be correct.
/// - No more than [`BITS`] least-significant bits may be set in the returned usize.
/// - [`Eq`] and [`Hash`] must be implementable with the returned `usize` from `into_usize`.
///
/// [`BITS`]: Tag::BITS
pub unsafe trait Tag: Copy {
    /// Number of least-significant bits in the return value of [`into_usize`]
    /// which may be non-zero. In other words this is the bit width of the
    /// value.
    ///
    /// [`into_usize`]: Tag::into_usize
    const BITS: u32;

    /// Turns this tag into an integer.
    ///
    /// The inverse of this function is [`from_usize`].
    ///
    /// This function guarantees that only the least-significant [`Self::BITS`]
    /// bits can be non-zero.
    ///
    /// [`from_usize`]: Tag::from_usize
    /// [`Self::BITS`]: Tag::BITS
    fn into_usize(self) -> usize;

    /// Re-creates the tag from the integer returned by [`into_usize`].
    ///
    /// # Safety
    ///
    /// The passed `tag` must be returned from [`into_usize`].
    ///
    /// [`into_usize`]: Tag::into_usize
    unsafe fn from_usize(tag: usize) -> Self;
}

/// Returns the number of bits available for use for tags in a pointer to `T`
/// (this is based on `T`'s alignment).
pub const fn bits_for<T: ?Sized + Aligned>() -> u32 {
    crate::aligned::align_of::<T>().as_nonzero().trailing_zeros()
}

/// Returns the correct [`Tag::BITS`] constant for a set of tag values.
pub const fn bits_for_tags(mut tags: &[usize]) -> u32 {
    let mut bits = 0;

    while let &[tag, ref rest @ ..] = tags {
        tags = rest;

        // bits required to represent `tag`,
        // position of the most significant 1
        let b = usize::BITS - tag.leading_zeros();
        if b > bits {
            bits = b;
        }
    }

    bits
}

/// A covariant [`Copy`] tagged borrow. This is essentially `{ pointer: &'a P, tag: T }` packed
/// in a single reference.
pub struct TaggedRef<'a, Pointee: Aligned + ?Sized, T: Tag> {
    /// This is semantically a pair of `pointer: &'a P` and `tag: T` fields,
    /// however we pack them in a single pointer, to save space.
    ///
    /// We pack the tag into the **most**-significant bits of the pointer to
    /// ease retrieval of the value. A left shift is a multiplication and
    /// those are embeddable in instruction encoding, for example:
    ///
    /// ```asm
    /// // (<https://godbolt.org/z/jqcYPWEr3>)
    /// example::shift_read3:
    ///     mov     eax, dword ptr [8*rdi]
    ///     ret
    ///
    /// example::mask_read3:
    ///     and     rdi, -8
    ///     mov     eax, dword ptr [rdi]
    ///     ret
    /// ```
    ///
    /// This is ASM outputted by rustc for reads of values behind tagged
    /// pointers for different approaches of tagging:
    /// - `shift_read3` uses `<< 3` (the tag is in the most-significant bits)
    /// - `mask_read3` uses `& !0b111` (the tag is in the least-significant bits)
    ///
    /// The shift approach thus produces less instructions and is likely faster
    /// (see <https://godbolt.org/z/Y913sMdWb>).
    ///
    /// Encoding diagram:
    /// ```text
    /// [ packed.addr                     ]
    /// [ tag ] [ pointer.addr >> T::BITS ] <-- usize::BITS - T::BITS bits
    ///    ^
    ///    |
    /// T::BITS bits
    /// ```
    ///
    /// The tag can be retrieved by `packed.addr() >> T::BITS` and the pointer
    /// can be retrieved by `packed.map_addr(|addr| addr << T::BITS)`.
    packed: NonNull<Pointee>,
    tag_pointer_ghost: PhantomData<(&'a Pointee, T)>,
}

impl<'a, P, T> TaggedRef<'a, P, T>
where
    P: Aligned + ?Sized,
    T: Tag,
{
    /// Tags `pointer` with `tag`.
    ///
    /// [`TaggedRef`]: crate::tagged_ptr::TaggedRef
    #[inline]
    pub fn new(pointer: &'a P, tag: T) -> Self {
        Self { packed: Self::pack(NonNull::from(pointer), tag), tag_pointer_ghost: PhantomData }
    }

    /// Retrieves the pointer.
    #[inline]
    pub fn pointer(self) -> &'a P {
        // SAFETY: pointer_raw returns the original pointer
        unsafe { self.pointer_raw().as_ref() }
    }

    /// Retrieves the tag.
    #[inline]
    pub fn tag(&self) -> T {
        // Unpack the tag, according to the `self.packed` encoding scheme
        let tag = self.packed.addr().get() >> Self::TAG_BIT_SHIFT;

        // Safety:
        // The shift retrieves the original value from `T::into_usize`,
        // satisfying `T::from_usize`'s preconditions.
        unsafe { T::from_usize(tag) }
    }

    /// Sets the tag to a new value.
    #[inline]
    pub fn set_tag(&mut self, tag: T) {
        self.packed = Self::pack(self.pointer_raw(), tag);
    }

    const TAG_BIT_SHIFT: u32 = usize::BITS - T::BITS;
    const ASSERTION: () = { assert!(T::BITS <= bits_for::<P>()) };

    /// Pack pointer `ptr` with a `tag`, according to `self.packed` encoding scheme.
    #[inline]
    fn pack(ptr: NonNull<P>, tag: T) -> NonNull<P> {
        // Trigger assert!
        let () = Self::ASSERTION;

        let packed_tag = tag.into_usize() << Self::TAG_BIT_SHIFT;

        ptr.map_addr(|addr| {
            // Safety:
            // - The pointer is `NonNull` => it's address is `NonZero<usize>`
            // - `P::BITS` least significant bits are always zero (`Pointer` contract)
            // - `T::BITS <= P::BITS` (from `Self::ASSERTION`)
            //
            // Thus `addr >> T::BITS` is guaranteed to be non-zero.
            //
            // `{non_zero} | packed_tag` can't make the value zero.

            let packed = (addr.get() >> T::BITS) | packed_tag;
            unsafe { NonZero::new_unchecked(packed) }
        })
    }

    /// Retrieves the original raw pointer from `self.packed`.
    #[inline]
    pub(super) fn pointer_raw(&self) -> NonNull<P> {
        self.packed.map_addr(|addr| unsafe { NonZero::new_unchecked(addr.get() << T::BITS) })
    }
}

impl<P, T> Copy for TaggedRef<'_, P, T>
where
    P: Aligned + ?Sized,
    T: Tag,
{
}

impl<P, T> Clone for TaggedRef<'_, P, T>
where
    P: Aligned + ?Sized,
    T: Tag,
{
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, T> Deref for TaggedRef<'_, P, T>
where
    P: Aligned + ?Sized,
    T: Tag,
{
    type Target = P;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.pointer()
    }
}

impl<P, T> fmt::Debug for TaggedRef<'_, P, T>
where
    P: Aligned + fmt::Debug + ?Sized,
    T: Tag + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaggedRef")
            .field("pointer", &self.pointer())
            .field("tag", &self.tag())
            .finish()
    }
}

impl<P, T> PartialEq for TaggedRef<'_, P, T>
where
    P: Aligned + ?Sized,
    T: Tag,
{
    #[inline]
    #[allow(ambiguous_wide_pointer_comparisons)]
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed
    }
}

impl<P, T: Tag> Eq for TaggedRef<'_, P, T> {}

impl<P, T: Tag> Hash for TaggedRef<'_, P, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.packed.hash(state);
    }
}

impl<'a, P, T, HCX> HashStable<HCX> for TaggedRef<'a, P, T>
where
    P: HashStable<HCX> + Aligned + ?Sized,
    T: Tag + HashStable<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.pointer().hash_stable(hcx, hasher);
        self.tag().hash_stable(hcx, hasher);
    }
}

// Safety:
// `TaggedRef<P, T, ..>` is semantically just `{ ptr: P, tag: T }`, as such
// it's ok to implement `Sync` as long as `P: Sync, T: Sync`
unsafe impl<P, T> Sync for TaggedRef<'_, P, T>
where
    P: Sync + Aligned + ?Sized,
    T: Sync + Tag,
{
}

// Safety:
// `TaggedRef<P, T, ..>` is semantically just `{ ptr: P, tag: T }`, as such
// it's ok to implement `Send` as long as `P: Send, T: Send`
unsafe impl<P, T> Send for TaggedRef<'_, P, T>
where
    P: Sync + Aligned + ?Sized,
    T: Send + Tag,
{
}

#[cfg(test)]
mod tests;
