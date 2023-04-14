use super::{Pointer, Tag};
use crate::stable_hasher::{HashStable, StableHasher};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A [`Copy`] tagged pointer.
///
/// This is essentially `{ pointer: P, tag: T }` packed in a single pointer.
///
/// You should use this instead of the [`TaggedPtr`] type in all cases where
/// `P` implements [`Copy`].
///
/// If `COMPARE_PACKED` is true, then the pointers will be compared and hashed without
/// unpacking. Otherwise we don't implement [`PartialEq`], [`Eq`] and [`Hash`];
/// if you want that, wrap the [`CopyTaggedPtr`].
///
/// [`TaggedPtr`]: crate::tagged_ptr::TaggedPtr
pub struct CopyTaggedPtr<P, T, const COMPARE_PACKED: bool>
where
    P: Pointer,
    T: Tag,
{
    /// This is semantically a pair of `pointer: P` and `tag: T` fields,
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
    packed: NonNull<P::Target>,
    tag_ghost: PhantomData<T>,
}

// Note that even though `CopyTaggedPtr` is only really expected to work with
// `P: Copy`, can't add `P: Copy` bound, because `CopyTaggedPtr` is used in the
// `TaggedPtr`'s implementation.
impl<P, T, const CP: bool> CopyTaggedPtr<P, T, CP>
where
    P: Pointer,
    T: Tag,
{
    /// Tags `pointer` with `tag`.
    ///
    /// Note that this leaks `pointer`: it won't be dropped when
    /// `CopyTaggedPtr` is dropped. If you have a pointer with a significant
    /// drop, use [`TaggedPtr`] instead.
    ///
    /// [`TaggedPtr`]: crate::tagged_ptr::TaggedPtr
    pub fn new(pointer: P, tag: T) -> Self {
        Self { packed: Self::pack(P::into_ptr(pointer), tag), tag_ghost: PhantomData }
    }

    /// Retrieves the pointer.
    pub fn pointer(self) -> P
    where
        P: Copy,
    {
        // SAFETY: pointer_raw returns the original pointer
        //
        // Note that this isn't going to double-drop or anything because we have
        // P: Copy
        unsafe { P::from_ptr(self.pointer_raw()) }
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
    const ASSERTION: () = { assert!(T::BITS <= P::BITS) };

    /// Pack pointer `ptr` that comes from [`P::into_ptr`] with a `tag`,
    /// according to `self.packed` encoding scheme.
    ///
    /// [`P::into_ptr`]: Pointer::into_ptr
    fn pack(ptr: NonNull<P::Target>, tag: T) -> NonNull<P::Target> {
        // Trigger assert!
        let () = Self::ASSERTION;

        let packed_tag = tag.into_usize() << Self::TAG_BIT_SHIFT;

        ptr.map_addr(|addr| {
            // Safety:
            // - The pointer is `NonNull` => it's address is `NonZeroUsize`
            // - `P::BITS` least significant bits are always zero (`Pointer` contract)
            // - `T::BITS <= P::BITS` (from `Self::ASSERTION`)
            //
            // Thus `addr >> T::BITS` is guaranteed to be non-zero.
            //
            // `{non_zero} | packed_tag` can't make the value zero.

            let packed = (addr.get() >> T::BITS) | packed_tag;
            unsafe { NonZeroUsize::new_unchecked(packed) }
        })
    }

    /// Retrieves the original raw pointer from `self.packed`.
    pub(super) fn pointer_raw(&self) -> NonNull<P::Target> {
        self.packed.map_addr(|addr| unsafe { NonZeroUsize::new_unchecked(addr.get() << T::BITS) })
    }

    /// This provides a reference to the `P` pointer itself, rather than the
    /// `Deref::Target`. It is used for cases where we want to call methods
    /// that may be implement differently for the Pointer than the Pointee
    /// (e.g., `Rc::clone` vs cloning the inner value).
    pub(super) fn with_pointer_ref<R>(&self, f: impl FnOnce(&P) -> R) -> R {
        // Safety:
        // - `self.raw.pointer_raw()` is originally returned from `P::into_ptr`
        //   and as such is valid for `P::from_ptr`.
        //   - This also allows us to not care whatever `f` panics or not.
        // - Even though we create a copy of the pointer, we store it inside
        //   `ManuallyDrop` and only access it by-ref, so we don't double-drop.
        //
        // Semantically this is just `f(&self.pointer)` (where `self.pointer`
        // is non-packed original pointer).
        //
        // Note that even though `CopyTaggedPtr` is only really expected to
        // work with `P: Copy`, we have to assume `P: ?Copy`, because
        // `CopyTaggedPtr` is used in the `TaggedPtr`'s implementation.
        let ptr = unsafe { ManuallyDrop::new(P::from_ptr(self.pointer_raw())) };
        f(&ptr)
    }
}

impl<P, T, const CP: bool> Copy for CopyTaggedPtr<P, T, CP>
where
    P: Pointer + Copy,
    T: Tag,
{
}

impl<P, T, const CP: bool> Clone for CopyTaggedPtr<P, T, CP>
where
    P: Pointer + Copy,
    T: Tag,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, T, const CP: bool> Deref for CopyTaggedPtr<P, T, CP>
where
    P: Pointer,
    T: Tag,
{
    type Target = P::Target;

    fn deref(&self) -> &Self::Target {
        // Safety:
        // `pointer_raw` returns the original pointer from `P::into_ptr` which,
        // by the `Pointer`'s contract, must be valid.
        unsafe { self.pointer_raw().as_ref() }
    }
}

impl<P, T, const CP: bool> DerefMut for CopyTaggedPtr<P, T, CP>
where
    P: Pointer + DerefMut,
    T: Tag,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety:
        // `pointer_raw` returns the original pointer from `P::into_ptr` which,
        // by the `Pointer`'s contract, must be valid for writes if
        // `P: DerefMut`.
        unsafe { self.pointer_raw().as_mut() }
    }
}

impl<P, T, const CP: bool> fmt::Debug for CopyTaggedPtr<P, T, CP>
where
    P: Pointer + fmt::Debug,
    T: Tag + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with_pointer_ref(|ptr| {
            f.debug_struct("CopyTaggedPtr").field("pointer", ptr).field("tag", &self.tag()).finish()
        })
    }
}

impl<P, T> PartialEq for CopyTaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed
    }
}

impl<P, T> Eq for CopyTaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
}

impl<P, T> Hash for CopyTaggedPtr<P, T, true>
where
    P: Pointer,
    T: Tag,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.packed.hash(state);
    }
}

impl<P, T, HCX, const CP: bool> HashStable<HCX> for CopyTaggedPtr<P, T, CP>
where
    P: Pointer + HashStable<HCX>,
    T: Tag + HashStable<HCX>,
{
    fn hash_stable(&self, hcx: &mut HCX, hasher: &mut StableHasher) {
        self.with_pointer_ref(|ptr| ptr.hash_stable(hcx, hasher));
        self.tag().hash_stable(hcx, hasher);
    }
}

// Safety:
// `CopyTaggedPtr<P, T, ..>` is semantically just `{ ptr: P, tag: T }`, as such
// it's ok to implement `Sync` as long as `P: Sync, T: Sync`
unsafe impl<P, T, const CP: bool> Sync for CopyTaggedPtr<P, T, CP>
where
    P: Sync + Pointer,
    T: Sync + Tag,
{
}

// Safety:
// `CopyTaggedPtr<P, T, ..>` is semantically just `{ ptr: P, tag: T }`, as such
// it's ok to implement `Send` as long as `P: Send, T: Send`
unsafe impl<P, T, const CP: bool> Send for CopyTaggedPtr<P, T, CP>
where
    P: Send + Pointer,
    T: Send + Tag,
{
}

/// Test that `new` does not compile if there is not enough alignment for the
/// tag in the pointer.
///
/// ```compile_fail,E0080
/// use rustc_data_structures::tagged_ptr::{CopyTaggedPtr, Tag};
///
/// #[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// enum Tag2 { B00 = 0b00, B01 = 0b01, B10 = 0b10, B11 = 0b11 };
///
/// unsafe impl Tag for Tag2 {
///     const BITS: u32 = 2;
///
///     fn into_usize(self) -> usize { todo!() }
///     unsafe fn from_usize(tag: usize) -> Self { todo!() }
/// }
///
/// let value = 12u16;
/// let reference = &value;
/// let tag = Tag2::B01;
///
/// let _ptr = CopyTaggedPtr::<_, _, true>::new(reference, tag);
/// ```
// For some reason miri does not get the compile error
// probably it `check`s instead of `build`ing?
#[cfg(not(miri))]
const _: () = ();

#[cfg(test)]
mod tests;
