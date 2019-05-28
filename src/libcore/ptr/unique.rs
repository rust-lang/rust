use crate::convert::From;
use crate::ops::{CoerceUnsized, DispatchFromDyn};
use crate::fmt;
use crate::marker::{PhantomData, Unsize};
use crate::mem;
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
#[unstable(feature = "ptr_internals", issue = "0",
           reason = "use NonNull instead and consider PhantomData<T> \
                     (if you also use #[may_dangle]), Send, and/or Sync")]
#[doc(hidden)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
pub struct Unique<T: ?Sized> {
    pointer: *const T,
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
#[unstable(feature = "ptr_internals", issue = "0")]
unsafe impl<T: Send + ?Sized> Send for Unique<T> { }

/// `Unique` pointers are `Sync` if `T` is `Sync` because the data they
/// reference is unaliased. Note that this aliasing invariant is
/// unenforced by the type system; the abstraction using the
/// `Unique` must enforce it.
#[unstable(feature = "ptr_internals", issue = "0")]
unsafe impl<T: Sync + ?Sized> Sync for Unique<T> { }

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: Sized> Unique<T> {
    /// Creates a new `Unique` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    ///
    /// Note that the pointer value may potentially represent a valid pointer to
    /// a `T`, which means this must not be used as a "not yet initialized"
    /// sentinel value. Types that lazily allocate must track initialization by
    /// some other means.
    // FIXME: rename to dangling() to match NonNull?
    #[inline]
    pub const fn empty() -> Self {
        unsafe {
            Unique::new_unchecked(mem::align_of::<T>() as *mut T)
        }
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> Unique<T> {
    /// Creates a new `Unique`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    #[inline]
    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        Unique { pointer: ptr as _, _marker: PhantomData }
    }

    /// Creates a new `Unique` if `ptr` is non-null.
    #[inline]
    pub fn new(ptr: *mut T) -> Option<Self> {
        if !ptr.is_null() {
            Some(unsafe { Unique { pointer: ptr as _, _marker: PhantomData } })
        } else {
            None
        }
    }

    /// Acquires the underlying `*mut` pointer.
    #[inline]
    pub const fn as_ptr(self) -> *mut T {
        self.pointer as *mut T
    }

    /// Dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&*my_ptr.as_ptr()`.
    #[inline]
    pub unsafe fn as_ref(&self) -> &T {
        &*self.as_ptr()
    }

    /// Mutably dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&mut *my_ptr.as_ptr()`.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.as_ptr()
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> Clone for Unique<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> Copy for Unique<T> { }

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized, U: ?Sized> CoerceUnsized<Unique<U>> for Unique<T> where T: Unsize<U> { }

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<Unique<U>> for Unique<T> where T: Unsize<U> { }

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> fmt::Debug for Unique<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> fmt::Pointer for Unique<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> From<&mut T> for Unique<T> {
    #[inline]
    fn from(reference: &mut T) -> Self {
        unsafe { Unique { pointer: reference as *mut T, _marker: PhantomData } }
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> From<&T> for Unique<T> {
    #[inline]
    fn from(reference: &T) -> Self {
        unsafe { Unique { pointer: reference as *const T, _marker: PhantomData } }
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<'a, T: ?Sized> From<NonNull<T>> for Unique<T> {
    #[inline]
    fn from(p: NonNull<T>) -> Self {
        unsafe { Unique::new_unchecked(p.as_ptr()) }
    }
}
