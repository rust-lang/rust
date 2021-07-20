use crate::{
    fmt,
    mem::{self, MaybeUninit},
    ptr,
};

/// An iterator over the mapped windows of another iterator.
///
/// This `struct` is created by the [`Iterator::map_windows`]. See its
/// documentation for more information.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "87155")]
pub struct MapWindows<I: Iterator, F, const N: usize> {
    iter: I,
    f: F,

    // The buffer is semantically `[MaybeUninit<I::Item>; 2 * N]`. However, due
    // to limitations of const generics, we use this different type. Note that
    // it has the same underlying memory layout.
    //
    // Invariant: if `buffer` is `Some`, `buffer[self.start..self.start + N]` is
    // initialized, with all other elements being uninitialized. This also
    // implies that `start <= N`.
    buffer: Option<[[MaybeUninit<I::Item>; N]; 2]>,
    start: usize,
}

impl<I: Iterator, F, const N: usize> MapWindows<I, F, N> {
    pub(in crate::iter) fn new(mut iter: I, f: F) -> Self {
        assert!(N > 0, "array in `Iterator::map_windows` must contain more than 0 elements");

        let buffer = crate::array::collect_into_array(&mut iter).map(|first_half: [_; N]| {
            // SAFETY: `MaybeUninit` is `repr(transparent)` and going from `T` to
            // `MaybeUninit<T>` is always safe.
            let first_half = unsafe {
                // FIXME(LukasKalbertodt): use `mem::transmute` once it works with arrays.
                let copy: [MaybeUninit<I::Item>; N] = mem::transmute_copy(&first_half);
                mem::forget(first_half);
                copy
            };
            [first_half, MaybeUninit::uninit_array()]
        });

        Self { iter, f, buffer, start: 0 }
    }
}

#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "87155")]
impl<I, F, R, const N: usize> Iterator for MapWindows<I, F, N>
where
    I: Iterator,
    F: FnMut(&[I::Item; N]) -> R,
{
    type Item = R;
    fn next(&mut self) -> Option<Self::Item> {
        let buffer_ptr = self.buffer.as_mut()?.as_mut_ptr().cast::<MaybeUninit<I::Item>>();

        let out = {
            debug_assert!(self.start + N <= 2 * N);

            // SAFETY: our invariant guarantees these elements are initialized.
            let initialized_part = unsafe {
                let ptr = buffer_ptr.add(self.start) as *const [I::Item; N];
                &*ptr
            };
            (self.f)(initialized_part)
        };

        // Advance iterator. We first call `next` before changing our buffer at
        // all. This means that if `next` panics, our invariant is upheld and
        // our `Drop` impl drops the correct elements.
        if let Some(next) = self.iter.next() {
            if self.start == N {
                // We have reached the end of our buffer and have to copy
                // everything to the start. Example layout for N = 3.
                //
                //    0   1   2   3   4   5            0   1   2   3   4   5
                //  ┌───┬───┬───┬───┬───┬───┐        ┌───┬───┬───┬───┬───┬───┐
                //  │ - │ - │ - │ a │ b │ c │   ->   │ b │ c │ n │ - │ - │ - │
                //  └───┴───┴───┴───┴───┴───┘        └───┴───┴───┴───┴───┴───┘
                //                ↑                    ↑
                //              start                start

                // SAFETY: the two pointers are valid for reads/writes of N -1
                // elements because our array's size is semantically 2 * N. The
                // regions also don't overlap for the same reason.
                //
                // We leave the old elements in place. As soon as `start` is set
                // to 0, we treat them as uninitialized and treat their copies
                // as initialized.
                unsafe {
                    ptr::copy_nonoverlapping(buffer_ptr.add(N), buffer_ptr, N - 1);
                    (*buffer_ptr.add(N - 1)).write(next);
                }
                self.start = 0;

                // SAFETY: the index is valid and this is element `a` in the
                // diagram above and has not been dropped yet.
                unsafe { (*buffer_ptr.add(N)).assume_init_drop() };
            } else {
                // SAFETY: `self.start` is < N as guaranteed by the invariant
                // plus the check above. Even if the drop at the end panics,
                // the invariant is upheld.
                //
                // Example layout for N = 3:
                //
                //    0   1   2   3   4   5            0   1   2   3   4   5
                //  ┌───┬───┬───┬───┬───┬───┐        ┌───┬───┬───┬───┬───┬───┐
                //  │ - │ a │ b │ c │ - │ - │   ->   │ - │ - │ b │ c │ n │ - │
                //  └───┴───┴───┴───┴───┴───┘        └───┴───┴───┴───┴───┴───┘
                //        ↑                                    ↑
                //      start                                start
                //
                unsafe {
                    (*buffer_ptr.add(self.start + N)).write(next);
                    self.start += 1;
                    (*buffer_ptr.add(self.start - 1)).assume_init_drop();
                }
            }
        } else {
            // SAFETY: our invariant guarantees that N elements starting from
            // `self.start` are initialized. We drop them here.
            unsafe {
                let initialized_part = crate::ptr::slice_from_raw_parts_mut(
                    buffer_ptr.add(self.start) as *mut I::Item,
                    N,
                );
                crate::ptr::drop_in_place(initialized_part);
            }
            self.buffer = None;
        }

        Some(out)
    }
}

#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "87155")]
impl<I: Iterator, F, const N: usize> Drop for MapWindows<I, F, N> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.as_mut() {
            // SAFETY: our invariant guarantees that N elements starting from
            // `self.start` are initialized. We drop them here.
            unsafe {
                let initialized_part = crate::ptr::slice_from_raw_parts_mut(
                    buffer.as_mut_ptr().cast::<I::Item>().add(self.start),
                    N,
                );
                crate::ptr::drop_in_place(initialized_part);
            }
        }
    }
}

#[unstable(feature = "iter_map_windows", reason = "recently added", issue = "87155")]
impl<I: Iterator + fmt::Debug, F, const N: usize> fmt::Debug for MapWindows<I, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapWindows").field("iter", &self.iter).finish()
    }
}
