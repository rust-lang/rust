use crate::marker::{Destruct, PhantomData};
use crate::mem::{ManuallyDrop, SizedTypeProperties, conjure_zst};
use crate::ptr::{drop_in_place, slice_from_raw_parts_mut};

impl<'l, 'f, T, U, const N: usize, F: FnMut(T) -> U> Drain<'l, 'f, T, N, F> {
    /// This function returns a function that lets you index the given array in const.
    /// As implemented it can optimize better than iterators, and can be constified.
    /// It acts like a sort of guard (owns the array) and iterator combined, which can be implemented
    /// as it is a struct that implements const fn;
    /// in that regard it is somewhat similar to an array::Iter implementing `UncheckedIterator`.
    /// The only method you're really allowed to call is `next()`,
    /// anything else is more or less UB, hence this function being unsafe.
    /// Moved elements will not be dropped.
    /// This will also not actually store the array.
    ///
    /// SAFETY: must only be called `N` times. Thou shalt not drop the array either.
    // FIXME(const-hack): this is a hack for `let guard = Guard(array); |i| f(guard[i])`.
    #[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
    pub(super) const unsafe fn new(array: &'l mut ManuallyDrop<[T; N]>, f: &'f mut F) -> Self {
        // dont drop the array, transfers "ownership" to Self
        let end = array.as_mut_ptr_range().end;
        Self { end, remaining: N, f, l: PhantomData }
    }
}

/// See [`Drain::new`]; this is our fake iterator.
#[unstable(feature = "array_try_map", issue = "79711")]
pub(super) struct Drain<'l, 'f, T, const N: usize, F> {
    // FIXME(const-hack): This is essentially a slice::IterMut<'static>, replace when possible.
    /// Pointer to the past-the-end element.
    /// As we "own" this array, we dont need to store any lifetime.
    end: *mut T,
    /// The number of elements still to be drained.
    remaining: usize,

    f: &'f mut F,
    l: PhantomData<&'l mut [T; N]>,
}

impl<T, const N: usize, F> Drain<'_, '_, T, N, F> {
    /// Returns a pointer to the next element to be drained, or the past-the-end element if there
    /// are no remaining elements to be drained.
    const fn ptr(&mut self) -> *mut T {
        // SAFETY: By the type invariants, self.remaining is always the number of elements prior to
        //         self.end that are still to be drained.
        unsafe { self.end.sub(self.remaining) }
    }
}

#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T, U, const N: usize, F> const FnOnce<(usize,)> for &mut Drain<'_, '_, T, N, F>
where
    F: [const] FnMut(T) -> U,
{
    type Output = U;

    /// This implementation is useless.
    extern "rust-call" fn call_once(mut self, args: (usize,)) -> Self::Output {
        self.call_mut(args)
    }
}
#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T, U, const N: usize, F> const FnMut<(usize,)> for &mut Drain<'_, '_, T, N, F>
where
    F: [const] FnMut(T) -> U,
{
    // FIXME(const-hack): ideally this would be an unsafe fn `next()`, and to use it you would instead `|_| unsafe { drain.next() }`.
    extern "rust-call" fn call_mut(
        &mut self,
        (_ /* ignore argument */,): (usize,),
    ) -> Self::Output {
        let p = self.ptr();
        // decrement before moving; if `f` panics, we drop the rest.
        self.remaining -= 1;
        if T::IS_ZST {
            // its UB to call this more than N times, so returning more ZSTs is valid.
            // SAFETY: its a ZST? we conjur.
            (self.f)(unsafe { conjure_zst::<T>() })
        } else {
            // SAFETY: we are allowed to move this.
            (self.f)(unsafe { p.read() })
        }
    }
}
#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T: [const] Destruct, const N: usize, F> const Drop for Drain<'_, '_, T, N, F> {
    fn drop(&mut self) {
        let slice = slice_from_raw_parts_mut(self.ptr(), self.remaining);

        // SAFETY: By the type invariant, we're allowed to drop all these. (we own it, after all)
        unsafe { drop_in_place(slice) }
    }
}
