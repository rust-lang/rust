use crate::fmt;
use crate::marker::{PhantomData, PointeeSized, Unsize};
use crate::ops::{CoerceUnsized, DispatchFromDyn};
use crate::pin::PinCoerceUnsized;
use crate::ptr::NonNull;

/// A wrapper around a raw non-null `*mut T` that indicates that the possessor
/// of this wrapper owns the referent. Useful for building abstractions like
/// `Box<T>`, `Vec<T>`, `String`, and `HashMap<K, V>`.
///
/// Unlike `*mut T`, `Unique<T>` behaves "as if" it were an instance of `T`.
/// It implements `Send`/`Sync` if `T` is `Send`/`Sync`. It also implies
/// the kind of strong aliasing guarantees an instance of `T` can expect:
/// the referent of the pointer should not be modified without a unique path to
/// its owning Unique.
///
/// If you're uncertain of whether it's correct to use `Unique` for your purposes,
/// consider using `NonNull`, which has weaker semantics.
///
/// Unlike `*mut T`, the pointer must always be non-null, even if the pointer
/// is never dereferenced. This is so that enums may use this forbidden value
/// as a discriminant -- `Option<Unique<T>>` has the same size as `Unique<T>`.
/// However the pointer may still dangle if it isn't dereferenced.
///
/// Unlike `*mut T`, `Unique<T>` is covariant over `T`. This should always be correct
/// for any type which upholds Unique's aliasing requirements.
#[unstable(
    feature = "ptr_internals",
    issue = "none",
    reason = "use `NonNull` instead and consider `PhantomData<T>` \
              (if you also use `#[may_dangle]`), `Send`, and/or `Sync`"
)]
#[doc(hidden)]
#[repr(transparent)]
pub struct Unique<T: PointeeSized> {
    pointer: NonNull<T>,
    // NOTE: this marker has no consequences for variance, but is necessary
    // for dropck to understand that we logically own a `T`.
    //
    // For details, see:
    // https://github.com/rust-lang/rfcs/blob/master/text/0769-sound-generic-drop.md#phantom-data
    _marker: PhantomData<T>,
}

/// `Unique` pointers are `Send` if `T` is `Send` because the data they
/// reference is unaliased. Note that this aliasing invariant is
/// unenforced by the type system; the abstraction using the
/// `Unique` must enforce it.
#[unstable(feature = "ptr_internals", issue = "none")]
unsafe impl<T: Send + PointeeSized> Send for Unique<T> {}

/// `Unique` pointers are `Sync` if `T` is `Sync` because the data they
/// reference is unaliased. Note that this aliasing invariant is
/// unenforced by the type system; the abstraction using the
/// `Unique` must enforce it.
#[unstable(feature = "ptr_internals", issue = "none")]
unsafe impl<T: Sync + PointeeSized> Sync for Unique<T> {}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: Sized> Unique<T> {
    /// Creates a new `Unique` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    ///
    /// Note that the address of the returned pointer may potentially
    /// be that of a valid pointer, which means this must not be used
    /// as a "not yet initialized" sentinel value.
    /// Types that lazily allocate must track initialization by some other means.
    #[must_use]
    #[inline]
    pub const fn dangling() -> Self {
        // FIXME(const-hack) replace with `From`
        Unique { pointer: NonNull::dangling(), _marker: PhantomData }
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized> Unique<T> {
    /// Creates a new `Unique`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    #[inline]
    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        // SAFETY: the caller must guarantee that `ptr` is non-null.
        unsafe { Unique { pointer: NonNull::new_unchecked(ptr), _marker: PhantomData } }
    }

    /// Creates a new `Unique` if `ptr` is non-null.
    #[inline]
    pub const fn new(ptr: *mut T) -> Option<Self> {
        if let Some(pointer) = NonNull::new(ptr) {
            Some(Unique { pointer, _marker: PhantomData })
        } else {
            None
        }
    }

    /// Create a new `Unique` from a `NonNull` in const context.
    #[inline]
    pub const fn from_non_null(pointer: NonNull<T>) -> Self {
        Unique { pointer, _marker: PhantomData }
    }

    /// Acquires the underlying `*mut` pointer.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub const fn as_ptr(self) -> *mut T {
        self.pointer.as_ptr()
    }

    /// Acquires the underlying `*mut` pointer.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub const fn as_non_null_ptr(self) -> NonNull<T> {
        self.pointer
    }

    /// Dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&*my_ptr.as_ptr()`.
    #[must_use]
    #[inline]
    pub const unsafe fn as_ref(&self) -> &T {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a reference.
        unsafe { self.pointer.as_ref() }
    }

    /// Mutably dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&mut *my_ptr.as_ptr()`.
    #[must_use]
    #[inline]
    pub const unsafe fn as_mut(&mut self) -> &mut T {
        // SAFETY: the caller must guarantee that `self` meets all the
        // requirements for a mutable reference.
        unsafe { self.pointer.as_mut() }
    }

    /// Casts to a pointer of another type.
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    pub const fn cast<U>(self) -> Unique<U> {
        // FIXME(const-hack): replace with `From`
        // SAFETY: is `NonNull`
        Unique { pointer: self.pointer.cast(), _marker: PhantomData }
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized> Clone for Unique<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized> Copy for Unique<T> {}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized, U: PointeeSized> CoerceUnsized<Unique<U>> for Unique<T> where T: Unsize<U> {}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized, U: PointeeSized> DispatchFromDyn<Unique<U>> for Unique<T> where T: Unsize<U> {}

#[unstable(feature = "pin_coerce_unsized_trait", issue = "123430")]
unsafe impl<T: PointeeSized> PinCoerceUnsized for Unique<T> {}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized> fmt::Debug for Unique<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
impl<T: PointeeSized> fmt::Pointer for Unique<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: PointeeSized> const From<&mut T> for Unique<T> {
    /// Converts a `&mut T` to a `Unique<T>`.
    ///
    /// This conversion is infallible since references cannot be null.
    #[inline]
    fn from(reference: &mut T) -> Self {
        Self::from(NonNull::from(reference))
    }
}

#[unstable(feature = "ptr_internals", issue = "none")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: PointeeSized> const From<NonNull<T>> for Unique<T> {
    /// Converts a `NonNull<T>` to a `Unique<T>`.
    ///
    /// This conversion is infallible since `NonNull` cannot be null.
    #[inline]
    fn from(pointer: NonNull<T>) -> Self {
        Unique::from_non_null(pointer)
    }
}
