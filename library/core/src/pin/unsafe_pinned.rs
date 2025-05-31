use crate::cell::UnsafeCell;
use crate::marker::{PointerLike, Unpin};
use crate::ops::{CoerceUnsized, DispatchFromDyn};
use crate::pin::Pin;
use crate::{fmt, ptr};

/// This type provides a way to entirely opt-out of typical aliasing rules;
/// specifically, `&mut UnsafePinned<T>` is not guaranteed to be a unique pointer.
/// This also subsumes the effects of `UnsafeCell`, i.e., `&UnsafePinned<T>` may point to data
/// that is being mutated.
///
/// However, even if you define your type like `pub struct Wrapper(UnsafePinned<...>)`, it is still
/// very risky to have an `&mut Wrapper` that aliases anything else. Many functions that work
/// generically on `&mut T` assume that the memory that stores `T` is uniquely owned (such as
/// `mem::swap`). In other words, while having aliasing with `&mut Wrapper` is not immediate
/// Undefined Behavior, it is still unsound to expose such a mutable reference to code you do not
/// control! Techniques such as pinning via [`Pin`] are needed to ensure soundness.
///
/// Similar to [`UnsafeCell`](crate::cell::UnsafeCell), `UnsafePinned` will not usually show up in
/// the public API of a library. It is an internal implementation detail of libraries that need to
/// support aliasing mutable references.
///
/// This type blocks niches the same way `UnsafeCell` does.
#[lang = "unsafe_pinned"]
#[repr(transparent)]
#[unstable(feature = "unsafe_pinned", issue = "125735")]
pub struct UnsafePinned<T: ?Sized> {
    value: UnsafeCell<T>,
}

// Override the manual `!Sync` in `UnsafeCell`.
#[unstable(feature = "unsafe_pinned", issue = "125735")]
unsafe impl<T: ?Sized + Sync> Sync for UnsafePinned<T> {}

/// When this type is used, that almost certainly means safe APIs need to use pinning to avoid the
/// aliases from becoming invalidated. Therefore let's mark this as `!Unpin`. You can always opt
/// back in to `Unpin` with an `impl` block, provided your API is still sound while unpinned.
#[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T: ?Sized> !Unpin for UnsafePinned<T> {}

// `Send` and `Sync` are inherited from `T`. This is similar to `SyncUnsafeCell`, since
// we eventually concluded that `UnsafeCell` implicitly making things `!Sync` is sometimes
// unergonomic. A type that needs to be `!Send`/`!Sync` should really have an explicit
// opt-out itself, e.g. via an `PhantomData<*mut T>` or (one day) via `impl !Send`/`impl !Sync`.

impl<T> UnsafePinned<T> {
    /// Constructs a new instance of `UnsafePinned` which will wrap the specified value.
    ///
    /// All access to the inner value through `&UnsafePinned<T>` or `&mut UnsafePinned<T>` or
    /// `Pin<&mut UnsafePinned<T>>` requires `unsafe` code.
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    pub const fn new(value: T) -> Self {
        UnsafePinned { value: UnsafeCell::new(value) }
    }

    /// Unwraps the value, consuming this `UnsafePinned`.
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    pub const fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

impl<T: ?Sized> UnsafePinned<T> {
    /// Get read-write access to the contents of a pinned `UnsafePinned`.
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    pub const fn get_mut_pinned(self: Pin<&mut Self>) -> *mut T {
        // SAFETY: we're not using `get_unchecked_mut` to unpin anything
        unsafe { self.get_unchecked_mut() }.get_mut_unchecked()
    }

    /// Get read-write access to the contents of an `UnsafePinned`.
    ///
    /// You should usually be using `get_mut_pinned` instead to explicitly track the fact that this
    /// memory is "pinned" due to there being aliases.
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    pub const fn get_mut_unchecked(&mut self) -> *mut T {
        ptr::from_mut(self) as *mut T
    }

    /// Get read-only access to the contents of a shared `UnsafePinned`.
    ///
    /// Note that `&UnsafePinned<T>` is read-only if `&T` is read-only. This means that if there is
    /// mutation of the `T`, future reads from the `*const T` returned here are UB! Use
    /// [`UnsafeCell`] if you also need interior mutability.
    ///
    /// [`UnsafeCell`]: crate::cell::UnsafeCell
    ///
    /// ```rust,no_run
    /// #![feature(unsafe_pinned)]
    /// use std::pin::UnsafePinned;
    ///
    /// unsafe {
    ///     let mut x = UnsafePinned::new(0);
    ///     let ptr = x.get(); // read-only pointer, assumes immutability
    ///     x.get_mut_unchecked().write(1);
    ///     ptr.read(); // UB!
    /// }
    /// ```
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    pub const fn get(&self) -> *const T {
        ptr::from_ref(self) as *const T
    }

    /// Gets an immutable pointer to the wrapped value.
    ///
    /// The difference from [`get`] is that this function accepts a raw pointer, which is useful to
    /// avoid the creation of temporary references.
    ///
    /// [`get`]: UnsafePinned::get
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    pub const fn raw_get(this: *const Self) -> *const T {
        this as *const T
    }

    /// Gets a mutable pointer to the wrapped value.
    ///
    /// The difference from [`get_mut_pinned`] and [`get_mut_unchecked`] is that this function
    /// accepts a raw pointer, which is useful to avoid the creation of temporary references.
    ///
    /// [`get_mut_pinned`]: UnsafePinned::get_mut_pinned
    /// [`get_mut_unchecked`]: UnsafePinned::get_mut_unchecked
    #[inline(always)]
    #[must_use]
    #[unstable(feature = "unsafe_pinned", issue = "125735")]
    pub const fn raw_get_mut(this: *mut Self) -> *mut T {
        this as *mut T
    }
}

#[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T: Default> Default for UnsafePinned<T> {
    /// Creates an `UnsafePinned`, with the `Default` value for T.
    fn default() -> Self {
        UnsafePinned::new(T::default())
    }
}

#[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T> From<T> for UnsafePinned<T> {
    /// Creates a new `UnsafePinned<T>` containing the given value.
    fn from(value: T) -> Self {
        UnsafePinned::new(value)
    }
}

#[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T: ?Sized> fmt::Debug for UnsafePinned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnsafePinned").finish_non_exhaustive()
    }
}

#[unstable(feature = "coerce_unsized", issue = "18598")]
// #[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T: CoerceUnsized<U>, U> CoerceUnsized<UnsafePinned<U>> for UnsafePinned<T> {}

// Allow types that wrap `UnsafePinned` to also implement `DispatchFromDyn`
// and become dyn-compatible method receivers.
// Note that currently `UnsafePinned` itself cannot be a method receiver
// because it does not implement Deref.
// In other words:
// `self: UnsafePinned<&Self>` won't work
// `self: UnsafePinned<Self>` becomes possible
// FIXME(unsafe_pinned) this logic is copied from UnsafeCell, is it still sound?
#[unstable(feature = "dispatch_from_dyn", issue = "none")]
// #[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T: DispatchFromDyn<U>, U> DispatchFromDyn<UnsafePinned<U>> for UnsafePinned<T> {}

#[unstable(feature = "pointer_like_trait", issue = "none")]
// #[unstable(feature = "unsafe_pinned", issue = "125735")]
impl<T: PointerLike> PointerLike for UnsafePinned<T> {}

// FIXME(unsafe_pinned): impl PinCoerceUnsized for UnsafePinned<T>?
