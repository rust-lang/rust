use core::marker::PhantomData;
use core::ptr::NonNull;

/// Models a reborrow of some unique reference, when you know that the reborrow
/// and all its descendants (i.e., all pointers and references derived from it)
/// will not be used any more at some point, after which you want to use the
/// original unique reference again.
///
/// The borrow checker usually handles this stacking of borrows for you, but
/// some control flows that accomplish this stacking are too complicated for
/// the compiler to follow. A `DormantMutRef` allows you to check borrowing
/// yourself, while still expressing its stacked nature, and encapsulating
/// the raw pointer code needed to do this without undefined behavior.
pub struct DormantMutRef<'a, T> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<'a, T> Sync for DormantMutRef<'a, T> where &'a mut T: Sync {}
unsafe impl<'a, T> Send for DormantMutRef<'a, T> where &'a mut T: Send {}

impl<'a, T> DormantMutRef<'a, T> {
    /// Capture a unique borrow, and immediately reborrow it. For the compiler,
    /// the lifetime of the new reference is the same as the lifetime of the
    /// original reference, but you promise to use it for a shorter period.
    pub fn new(t: &'a mut T) -> (&'a mut T, Self) {
        let ptr = NonNull::from(t);
        // SAFETY: we hold the borrow throughout 'a via `_marker`, and we expose
        // only this reference, so it is unique.
        let new_ref = unsafe { &mut *ptr.as_ptr() };
        (new_ref, Self { ptr, _marker: PhantomData })
    }

    /// Revert to the unique borrow initially captured.
    ///
    /// # Safety
    ///
    /// The reborrow must have ended, i.e., the reference returned by `new` and
    /// all pointers and references derived from it, must not be used anymore.
    pub unsafe fn awaken(self) -> &'a mut T {
        // SAFETY: our own safety conditions imply this reference is again unique.
        unsafe { &mut *self.ptr.as_ptr() }
    }

    /// Borrows a new mutable reference from the unique borrow initially captured.
    ///
    /// # Safety
    ///
    /// The reborrow must have ended, i.e., the reference returned by `new` and
    /// all pointers and references derived from it, must not be used anymore.
    pub unsafe fn reborrow(&mut self) -> &'a mut T {
        // SAFETY: our own safety conditions imply this reference is again unique.
        unsafe { &mut *self.ptr.as_ptr() }
    }

    /// Borrows a new shared reference from the unique borrow initially captured.
    ///
    /// # Safety
    ///
    /// The reborrow must have ended, i.e., the reference returned by `new` and
    /// all pointers and references derived from it, must not be used anymore.
    pub unsafe fn reborrow_shared(&self) -> &'a T {
        // SAFETY: our own safety conditions imply this reference is again unique.
        unsafe { &*self.ptr.as_ptr() }
    }
}

#[cfg(test)]
mod tests;
