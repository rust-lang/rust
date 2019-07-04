use crate::convert::From;
use crate::ops::{CoerceUnsized, DispatchFromDyn};
use crate::fmt;
use crate::hash;
use crate::marker::Unsize;
use crate::mem;
use crate::ptr::Unique;
use crate::cmp::Ordering;

/// `*mut T` but non-zero and covariant.
///
/// This is often the correct thing to use when building data structures using
/// raw pointers, but is ultimately more dangerous to use because of its additional
/// properties. If you're not sure if you should use `NonNull<T>`, just use `*mut T`!
///
/// Unlike `*mut T`, the pointer must always be non-null, even if the pointer
/// is never dereferenced. This is so that enums may use this forbidden value
/// as a discriminant -- `Option<NonNull<T>>` has the same size as `*mut T`.
/// However the pointer may still dangle if it isn't dereferenced.
///
/// Unlike `*mut T`, `NonNull<T>` is covariant over `T`. If this is incorrect
/// for your use case, you should include some [`PhantomData`] in your type to
/// provide invariance, such as `PhantomData<Cell<T>>` or `PhantomData<&'a mut T>`.
/// Usually this won't be necessary; covariance is correct for most safe abstractions,
/// such as `Box`, `Rc`, `Arc`, `Vec`, and `LinkedList`. This is the case because they
/// provide a public API that follows the normal shared XOR mutable rules of Rust.
///
/// Notice that `NonNull<T>` has a `From` instance for `&T`. However, this does
/// not change the fact that mutating through a (pointer derived from a) shared
/// reference is undefined behavior unless the mutation happens inside an
/// [`UnsafeCell<T>`]. The same goes for creating a mutable reference from a shared
/// reference. When using this `From` instance without an `UnsafeCell<T>`,
/// it is your responsibility to ensure that `as_mut` is never called, and `as_ptr`
/// is never used for mutation.
///
/// [`PhantomData`]: ../marker/struct.PhantomData.html
/// [`UnsafeCell<T>`]: ../cell/struct.UnsafeCell.html
#[stable(feature = "nonnull", since = "1.25.0")]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(1)]
#[rustc_nonnull_optimization_guaranteed]
pub struct NonNull<T: ?Sized> {
    pointer: *const T,
}

/// `NonNull` pointers are not `Send` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> !Send for NonNull<T> { }

/// `NonNull` pointers are not `Sync` because the data they reference may be aliased.
// N.B., this impl is unnecessary, but should provide better error messages.
#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> !Sync for NonNull<T> { }

impl<T: Sized> NonNull<T> {
    /// Creates a new `NonNull` that is dangling, but well-aligned.
    ///
    /// This is useful for initializing types which lazily allocate, like
    /// `Vec::new` does.
    ///
    /// Note that the pointer value may potentially represent a valid pointer to
    /// a `T`, which means this must not be used as a "not yet initialized"
    /// sentinel value. Types that lazily allocate must track initialization by
    /// some other means.
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[inline]
    pub const fn dangling() -> Self {
        unsafe {
            let ptr = mem::align_of::<T>() as *mut T;
            NonNull::new_unchecked(ptr)
        }
    }
}

impl<T: ?Sized> NonNull<T> {
    /// Creates a new `NonNull`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null.
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[inline]
    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        NonNull { pointer: ptr as _ }
    }

    /// Creates a new `NonNull` if `ptr` is non-null.
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[inline]
    pub fn new(ptr: *mut T) -> Option<Self> {
        if !ptr.is_null() {
            Some(unsafe { Self::new_unchecked(ptr) })
        } else {
            None
        }
    }

    /// Acquires the underlying `*mut` pointer.
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[inline]
    pub const fn as_ptr(self) -> *mut T {
        self.pointer as *mut T
    }

    /// Dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&*my_ptr.as_ptr()`.
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[inline]
    pub unsafe fn as_ref(&self) -> &T {
        &*self.as_ptr()
    }

    /// Mutably dereferences the content.
    ///
    /// The resulting lifetime is bound to self so this behaves "as if"
    /// it were actually an instance of T that is getting borrowed. If a longer
    /// (unbound) lifetime is needed, use `&mut *my_ptr.as_ptr()`.
    #[stable(feature = "nonnull", since = "1.25.0")]
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.as_ptr()
    }

    /// Cast to a pointer of another type
    #[stable(feature = "nonnull_cast", since = "1.27.0")]
    #[inline]
    pub const fn cast<U>(self) -> NonNull<U> {
        unsafe {
            NonNull::new_unchecked(self.as_ptr() as *mut U)
        }
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> Clone for NonNull<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> Copy for NonNull<T> { }

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized, U: ?Sized> CoerceUnsized<NonNull<U>> for NonNull<T> where T: Unsize<U> { }

#[unstable(feature = "dispatch_from_dyn", issue = "0")]
impl<T: ?Sized, U: ?Sized> DispatchFromDyn<NonNull<U>> for NonNull<T> where T: Unsize<U> { }

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> fmt::Debug for NonNull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> fmt::Pointer for NonNull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.as_ptr(), f)
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> Eq for NonNull<T> {}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> PartialEq for NonNull<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> Ord for NonNull<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ptr().cmp(&other.as_ptr())
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> PartialOrd for NonNull<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ptr().partial_cmp(&other.as_ptr())
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> hash::Hash for NonNull<T> {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state)
    }
}

#[unstable(feature = "ptr_internals", issue = "0")]
impl<T: ?Sized> From<Unique<T>> for NonNull<T> {
    #[inline]
    fn from(unique: Unique<T>) -> Self {
        unsafe { NonNull::new_unchecked(unique.as_ptr()) }
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> From<&mut T> for NonNull<T> {
    #[inline]
    fn from(reference: &mut T) -> Self {
        unsafe { NonNull { pointer: reference as *mut T } }
    }
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T: ?Sized> From<&T> for NonNull<T> {
    #[inline]
    fn from(reference: &T) -> Self {
        unsafe { NonNull { pointer: reference as *const T } }
    }
}
