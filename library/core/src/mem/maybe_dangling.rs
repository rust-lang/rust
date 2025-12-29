#![unstable(feature = "maybe_dangling", issue = "118166")]

use crate::{mem, ptr};

/// Allows wrapped [references] and [boxes] to dangle.
///
/// <section class="warning">
/// This type is not properly implemented yet, and the documentation below is thus not accurate.
/// </section>
///
/// That is, if a reference (or a `Box`) is wrapped in `MaybeDangling` (including when in a
/// (nested) field of a compound type wrapped in `MaybeDangling`), it does not have to follow
/// pointer aliasing rules or be dereferenceable.
///
/// This can be useful when the value can become dangling while the function holding it is still
/// executing (particularly in concurrent code). As a somewhat absurd example, consider this code:
///
/// ```rust,no_run
/// #![feature(box_as_ptr)]
/// # use std::alloc::{dealloc, Layout};
/// # use std::mem;
///
/// let mut boxed = Box::new(0_u32);
/// let ptr = Box::as_mut_ptr(&mut boxed);
///
/// // Safety: the pointer comes from a box and thus was allocated before; `box` is not used afterwards
/// unsafe { dealloc(ptr.cast(), Layout::new::<u32>()) };
///
/// mem::forget(boxed); // <-- this is UB!
/// ```
///
/// Even though the `Box`e's destructor is not run (and thus we don't have a double free bug), this
/// code is still UB. This is because when moving `boxed` into `forget`, its validity invariants
/// are asserted, causing UB since the `Box` is dangling. The safety comment is as such wrong, as
/// moving the `boxed` variable as part of the `forget` call *is* a use.
///
/// To fix this we could use `MaybeDangling`:
///
/// ```rust
/// #![feature(maybe_dangling, box_as_ptr)]
/// # use std::alloc::{dealloc, Layout};
/// # use std::mem::{self, MaybeDangling};
///
/// let mut boxed = MaybeDangling::new(Box::new(0_u32));
/// let ptr = Box::as_mut_ptr(boxed.as_mut());
///
/// // Safety: the pointer comes from a box and thus was allocated before; `box` is not used afterwards
/// unsafe { dealloc(ptr.cast(), Layout::new::<u32>()) };
///
/// mem::forget(boxed); // <-- this is OK!
/// ```
///
/// Note that the bit pattern must still be valid for the wrapped type. That is, [references]
/// (and [boxes]) still must be aligned and non-null.
///
/// Additionally note that safe code can still assume that the inner value in a `MaybeDangling` is
/// **not** dangling -- functions like [`as_ref`] and [`into_inner`] are safe. It is not sound to
/// return a dangling reference in a `MaybeDangling` to safe code. However, it *is* sound
/// to hold such values internally inside your code -- and there's no way to do that without
/// this type. Note that other types can use this type and thus get the same effect; in particular,
/// [`ManuallyDrop`] will use `MaybeDangling`.
///
/// Note that `MaybeDangling` doesn't prevent drops from being run, which can lead to UB if the
/// drop observes a dangling value. If you need to prevent drops from being run use [`ManuallyDrop`]
/// instead.
///
/// [references]: prim@reference
/// [boxes]: ../../std/boxed/struct.Box.html
/// [`into_inner`]: MaybeDangling::into_inner
/// [`as_ref`]: MaybeDangling::as_ref
/// [`ManuallyDrop`]: crate::mem::ManuallyDrop
#[repr(transparent)]
#[rustc_pub_transparent]
#[derive(Debug, Copy, Clone, Default)]
pub struct MaybeDangling<P: ?Sized>(P);

impl<P: ?Sized> MaybeDangling<P> {
    /// Wraps a value in a `MaybeDangling`, allowing it to dangle.
    pub const fn new(x: P) -> Self
    where
        P: Sized,
    {
        MaybeDangling(x)
    }

    /// Returns a reference to the inner value.
    ///
    /// Note that this is UB if the inner value is currently dangling.
    pub const fn as_ref(&self) -> &P {
        &self.0
    }

    /// Returns a mutable reference to the inner value.
    ///
    /// Note that this is UB if the inner value is currently dangling.
    pub const fn as_mut(&mut self) -> &mut P {
        &mut self.0
    }

    /// Extracts the value from the `MaybeDangling` container.
    ///
    /// Note that this is UB if the inner value is currently dangling.
    pub const fn into_inner(self) -> P
    where
        P: Sized,
    {
        // FIXME: replace this with `self.0` when const checker can figure out that `self` isn't actually dropped
        // SAFETY: this is equivalent to `self.0`
        let x = unsafe { ptr::read(&self.0) };
        mem::forget(self);
        x
    }
}
