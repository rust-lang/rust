use crate::ptr;
use crate::ops::{Deref, DerefMut};

/// A wrapper to inhibit compiler from automatically calling `T`’s destructor.
///
/// This wrapper is 0-cost.
///
/// `ManuallyDrop<T>` is subject to the same layout optimizations as `T`.
/// As a consequence, it has *no effect* on the assumptions that the compiler makes
/// about all values being initialized at their type.  In particular, initializing
/// a `ManuallyDrop<&mut T>` with [`mem::zeroed`] is undefined behavior.
/// If you need to handle uninitialized data, use [`MaybeUninit<T>`] instead.
///
/// # Examples
///
/// This wrapper helps with explicitly documenting the drop order dependencies between fields of
/// the type:
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
/// [`mem::zeroed`]: fn.zeroed.html
/// [`MaybeUninit<T>`]: union.MaybeUninit.html
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
    #[inline(always)]
    pub const fn into_inner(slot: ManuallyDrop<T>) -> T {
        slot.value
    }

    /// Takes the contained value out.
    ///
    /// This method is primarily intended for moving out values in drop.
    /// Instead of using [`ManuallyDrop::drop`] to manually drop the value,
    /// you can use this method to take the value and use it however desired.
    /// `Drop` will be invoked on the returned value following normal end-of-scope rules.
    ///
    /// If you have ownership of the container, you can use [`ManuallyDrop::into_inner`] instead.
    ///
    /// # Safety
    ///
    /// This function semantically moves out the contained value without preventing further usage.
    /// It is up to the user of this method to ensure that this container is not used again.
    ///
    /// [`ManuallyDrop::drop`]: #method.drop
    /// [`ManuallyDrop::into_inner`]: #method.into_inner
    #[must_use = "if you don't need the value, you can use `ManuallyDrop::drop` instead"]
    #[unstable(feature = "manually_drop_take", issue = "55422")]
    #[inline]
    pub unsafe fn take(slot: &mut ManuallyDrop<T>) -> T {
        ManuallyDrop::into_inner(ptr::read(slot))
    }
}

impl<T: ?Sized> ManuallyDrop<T> {
    /// Manually drops the contained value.
    ///
    /// If you have ownership of the value, you can use [`ManuallyDrop::into_inner`] instead.
    ///
    /// # Safety
    ///
    /// This function runs the destructor of the contained value and thus the wrapped value
    /// now represents uninitialized data. It is up to the user of this method to ensure the
    /// uninitialized data is not actually used.
    ///
    /// [`ManuallyDrop::into_inner`]: #method.into_inner
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline]
    pub unsafe fn drop(slot: &mut ManuallyDrop<T>) {
        ptr::drop_in_place(&mut slot.value)
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
