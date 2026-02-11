use crate::marker::{Destruct, PhantomData};
use crate::mem::{ManuallyDrop, SizedTypeProperties, conjure_zst, transmute};
use crate::ptr::{NonNull, drop_in_place, from_raw_parts_mut, without_provenance_mut};

impl<'l, 'f, T, U, F: FnMut(T) -> U> Drain<'l, 'f, T, F> {
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
    #[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
    pub(super) const unsafe fn new<const N: usize>(
        array: &'l mut ManuallyDrop<[T; N]>,
        f: &'f mut F,
    ) -> Self {
        // dont drop the array, transfers "ownership" to Self
        let ptr: NonNull<T> = NonNull::from_mut(array).cast();
        // SAFETY:
        // Adding `slice.len()` to the starting pointer gives a pointer
        // at the end of `slice`. `end` will never be dereferenced, only checked
        // for direct pointer equality with `ptr` to check if the drainer is done.
        unsafe {
            let end_or_len =
                if T::IS_ZST { without_provenance_mut(N) } else { ptr.as_ptr().add(N) };
            Self { ptr, end_or_len, f, l: PhantomData }
        }
    }
}

/// See [`Drain::new`]; this is our fake iterator.
#[unstable(feature = "array_try_map", issue = "79711")]
pub(super) struct Drain<'l, 'f, T, F> {
    // FIXME(const-hack): This is a slice::IterMut<'l>, replace when possible.
    /// The pointer to the next element to return, or the past-the-end location
    /// if the drainer is empty.
    ///
    /// This address will be used for all ZST elements, never changed.
    /// As we "own" this array, we dont need to store any lifetime.
    ptr: NonNull<T>,
    /// For non-ZSTs, the non-null pointer to the past-the-end element.
    /// For ZSTs, this is the number of unprocessed items.
    end_or_len: *mut T,

    f: &'f mut F,
    l: PhantomData<&'l mut [T]>,
}

#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T, U, F> const FnOnce<(usize,)> for &mut Drain<'_, '_, T, F>
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
impl<T, U, F> const FnMut<(usize,)> for &mut Drain<'_, '_, T, F>
where
    F: [const] FnMut(T) -> U,
{
    // FIXME(const-hack): ideally this would be an unsafe fn `next()`, and to use it you would instead `|_| unsafe { drain.next() }`.
    extern "rust-call" fn call_mut(
        &mut self,
        (_ /* ignore argument */,): (usize,),
    ) -> Self::Output {
        if T::IS_ZST {
            #[expect(ptr_to_integer_transmute_in_consts)]
            // SAFETY:
            // This is equivalent to `self.end_or_len.addr`, but that's not
            // available in `const`. `self.end_or_len` doesn't have provenance,
            // so transmuting is fine.
            let len = unsafe { transmute::<*mut T, usize>(self.end_or_len) };
            // SAFETY:
            // The caller guarantees that this is never called more than N times
            // (see `Drain::new`), hence this cannot underflow.
            self.end_or_len = without_provenance_mut(unsafe { len.unchecked_sub(1) });
            // its UB to call this more than N times, so returning more ZSTs is valid.
            // SAFETY: its a ZST? we conjur.
            (self.f)(unsafe { conjure_zst::<T>() })
        } else {
            // increment before moving; if `f` panics, we drop the rest.
            let p = self.ptr;
            // SAFETY: caller guarantees never called more than N times (see `Drain::new`)
            self.ptr = unsafe { self.ptr.add(1) };
            // SAFETY: we are allowed to move this.
            (self.f)(unsafe { p.read() })
        }
    }
}
#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T: [const] Destruct, F> const Drop for Drain<'_, '_, T, F> {
    fn drop(&mut self) {
        let slice = if T::IS_ZST {
            from_raw_parts_mut::<[T]>(
                self.ptr.as_ptr(),
                #[expect(ptr_to_integer_transmute_in_consts)]
                // SAFETY:
                // This is equivalent to `self.end_or_len.addr`, but that's not
                // available in `const`. `self.end_or_len` doesn't have provenance,
                // so transmuting is fine.
                unsafe {
                    transmute::<*mut T, usize>(self.end_or_len)
                },
            )
        } else {
            // SAFETY: we cant read more than N elements
            unsafe {
                from_raw_parts_mut::<[T]>(
                    self.ptr.as_ptr(),
                    // SAFETY: `start <= end`
                    self.end_or_len.offset_from_unsigned(self.ptr.as_ptr()),
                )
            }
        };

        // SAFETY: By the type invariant, we're allowed to drop all these. (we own it, after all)
        unsafe { drop_in_place(slice) }
    }
}
