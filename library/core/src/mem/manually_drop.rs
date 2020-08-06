use crate::ops::{Deref, DerefMut};
use crate::ptr;

/// A wrapper to inhibit compiler from automatically calling `T`’s destructor.
/// This wrapper is 0-cost.
///
/// `ManuallyDrop<T>` is subject to the same layout optimizations as `T`.
/// As a consequence, it has *no effect* on the assumptions that the compiler makes
/// about its contents. For example, initializing a `ManuallyDrop<&mut T>`
/// with [`mem::zeroed`] is undefined behavior.
/// If you need to handle uninitialized data, use [`MaybeUninit<T>`] instead.
///
/// Note that accessing the value inside a `ManuallyDrop<T>` is safe.
/// This means that a `ManuallyDrop<T>` whose content has been dropped must not
/// be exposed through a public safe API.
/// Correspondingly, `ManuallyDrop::drop` is unsafe.
///
/// # Examples
///
/// This wrapper can be used to enforce a particular drop order on fields, regardless
/// of how they are defined in the struct:
///
/// ```rust
/// use std::mem::ManuallyDrop;
/// struct Peach;
/// struct Banana;
/// struct Melon;
/// struct FruitBox {
///     // Immediately clear there’s something non-trivial going on with these fields.
///     peach: ManuallyDrop<Peach>,
///     melon: Melon, // Field that’s independent of the other two.
///     banana: ManuallyDrop<Banana>,
/// }
///
/// impl Drop for FruitBox {
///     fn drop(&mut self) {
///         unsafe {
///             // Explicit ordering in which field destructors are run specified in the intuitive
///             // location – the destructor of the structure containing the fields.
///             // Moreover, one can now reorder fields within the struct however much they want.
///             ManuallyDrop::drop(&mut self.peach);
///             ManuallyDrop::drop(&mut self.banana);
///         }
///         // After destructor for `FruitBox` runs (this function), the destructor for Melon gets
///         // invoked in the usual manner, as it is not wrapped in `ManuallyDrop`.
///     }
/// }
/// ```
///
/// However, care should be taken when using this pattern as it can lead to *leak amplification*.
/// In this example, if the `Drop` implementation for `Peach` were to panic, the `banana` field
/// would also be leaked.
///
/// In contrast, the automatically-generated compiler drop implementation would have ensured
/// that all fields are dropped even in the presence of panics. This is especially important when
/// working with [pinned] data, where reusing the memory without calling the destructor could lead
/// to Undefined Behaviour.
///
/// [`mem::zeroed`]: crate::mem::zeroed
/// [`MaybeUninit<T>`]: crate::mem::MaybeUninit
/// [pinned]: crate::pin
#[stable(feature = "manually_drop", since = "1.20.0")]
#[lang = "manually_drop"]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ManuallyDrop<T: ?Sized> {
    value: T,
}

impl<T> ManuallyDrop<T> {
    /// Wrap a value to be manually dropped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::mem::ManuallyDrop;
    /// ManuallyDrop::new(Box::new(()));
    /// ```
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[rustc_const_stable(feature = "const_manually_drop", since = "1.36.0")]
    #[inline(always)]
    pub const fn new(value: T) -> ManuallyDrop<T> {
        ManuallyDrop { value }
    }

    /// Extracts the value from the `ManuallyDrop` container.
    ///
    /// This allows the value to be dropped again.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::mem::ManuallyDrop;
    /// let x = ManuallyDrop::new(Box::new(()));
    /// let _: Box<()> = ManuallyDrop::into_inner(x); // This drops the `Box`.
    /// ```
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[rustc_const_stable(feature = "const_manually_drop", since = "1.36.0")]
    #[inline(always)]
    pub const fn into_inner(slot: ManuallyDrop<T>) -> T {
        slot.value
    }

    /// Takes the value from the `ManuallyDrop<T>` container out.
    ///
    /// This method is primarily intended for moving out values in drop.
    /// Instead of using [`ManuallyDrop::drop`] to manually drop the value,
    /// you can use this method to take the value and use it however desired.
    ///
    /// Whenever possible, it is preferable to use [`into_inner`][`ManuallyDrop::into_inner`]
    /// instead, which prevents duplicating the content of the `ManuallyDrop<T>`.
    ///
    /// # Safety
    ///
    /// This function semantically moves out the contained value without preventing further usage,
    /// leaving the state of this container unchanged.
    /// It is your responsibility to ensure that this `ManuallyDrop` is not used again.
    ///
    #[must_use = "if you don't need the value, you can use `ManuallyDrop::drop` instead"]
    #[stable(feature = "manually_drop_take", since = "1.42.0")]
    #[inline]
    pub unsafe fn take(slot: &mut ManuallyDrop<T>) -> T {
        // SAFETY: we are reading from a reference, which is guaranteed
        // to be valid for reads.
        unsafe { ptr::read(&slot.value) }
    }
}

impl<T: ?Sized> ManuallyDrop<T> {
    /// Manually drops the contained value. This is exactly equivalent to calling
    /// [`ptr::drop_in_place`] with a pointer to the contained value. As such, unless
    /// the contained value is a packed struct, the destructor will be called in-place
    /// without moving the value, and thus can be used to safely drop [pinned] data.
    ///
    /// If you have ownership of the value, you can use [`ManuallyDrop::into_inner`] instead.
    ///
    /// # Safety
    ///
    /// This function runs the destructor of the contained value. Other than changes made by
    /// the destructor itself, the memory is left unchanged, and so as far as the compiler is
    /// concerned still holds a bit-pattern which is valid for the type `T`.
    ///
    /// However, this "zombie" value should not be exposed to safe code, and this function
    /// should not be called more than once. To use a value after it's been dropped, or drop
    /// a value multiple times, can cause Undefined Behavior (depending on what `drop` does).
    /// This is normally prevented by the type system, but users of `ManuallyDrop` must
    /// uphold those guarantees without assistance from the compiler.
    ///
    /// [pinned]: crate::pin
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline]
    pub unsafe fn drop(slot: &mut ManuallyDrop<T>) {
        // SAFETY: we are dropping the value pointed to by a mutable reference
        // which is guaranteed to be valid for writes.
        // It is up to the caller to make sure that `slot` isn't dropped again.
        unsafe { ptr::drop_in_place(&mut slot.value) }
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T: ?Sized> Deref for ManuallyDrop<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T: ?Sized> DerefMut for ManuallyDrop<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
