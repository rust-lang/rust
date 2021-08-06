use core::iter::FusedIterator;
use core::mem::{self, MaybeUninit};
use core::ops::{ControlFlow, Try};
use core::ptr;

#[derive(Debug)]
struct Buffer<T, const N: usize> {
    array: [MaybeUninit<T>; N],
    init: usize,
}

impl<T, const N: usize> Buffer<T, N> {
    fn new() -> Self {
        Self { array: MaybeUninit::uninit_array(), init: 0 }
    }
}

impl<T: Clone, const N: usize> Clone for Buffer<T, N> {
    fn clone(&self) -> Self {
        let mut new = Self::new();
        // SAFETY: this raw slice contains only the initialized objects.
        let src = unsafe { MaybeUninit::slice_assume_init_ref(&self.array[..self.init]) };
        MaybeUninit::write_slice_cloned(&mut new.array[..self.init], src);
        new.init = self.init;
        new
    }
}

impl<T, const N: usize> Drop for Buffer<T, N> {
    fn drop(&mut self) {
        debug_assert!(self.init <= N);

        let initialized_part = &mut self.array[..self.init];

        // SAFETY: this raw slice will contain only the initialized objects
        // that have not been dropped or moved.
        unsafe {
            ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(initialized_part));
        }
    }
}

// FIXME: Combine with `Guard` in `collect_into_array`.
struct Guard<T, const N: usize> {
    ptr: *mut T,
    init: usize,
}

impl<T, const N: usize> Drop for Guard<T, N> {
    fn drop(&mut self) {
        debug_assert!(self.init <= N);

        let initialized_part = crate::ptr::slice_from_raw_parts_mut(self.ptr, self.init);

        // SAFETY: this raw slice will contain only initialized objects.
        unsafe {
            crate::ptr::drop_in_place(initialized_part);
        }
    }
}

impl<T, const N: usize> Guard<T, N> {
    fn new(array: &mut [MaybeUninit<T>; N]) -> Self {
        Self { ptr: MaybeUninit::slice_as_mut_ptr(array), init: 0 }
    }

    fn with<R, F>(buffer: &mut Buffer<T, N>, f: F) -> R
    where
        F: FnOnce(&mut [MaybeUninit<T>; N], &mut usize) -> R,
    {
        let mut array = MaybeUninit::uninit_array();
        let mut guard = Self::new(&mut array);
        if buffer.init > 0 {
            array = mem::replace(&mut buffer.array, MaybeUninit::uninit_array());
            guard.init = mem::replace(&mut buffer.init, 0);
        }
        let res = f(&mut array, &mut guard.init);
        if guard.init > 0 {
            buffer.array = array;
            buffer.init = guard.init;
            mem::forget(guard);
        }
        res
    }
}

/// An iterator that yields the elements of another iterator in
/// chunks of size `N`.
///
/// This `struct` is created by the [`array_chunks`] method on [`Iterator`]. See
/// its documentation for more.
///
/// [`array_chunks`]: Iterator::array_chunks
#[unstable(feature = "iter_array_chunks", issue = "none")]
#[derive(Debug, Clone)]
pub struct ArrayChunks<I: Iterator, const N: usize> {
    iter: I,
    buffer: Buffer<I::Item, N>,
}

impl<I: Iterator, const N: usize> ArrayChunks<I, N> {
    pub(in crate::iter) fn new(iter: I) -> Self {
        Self { iter, buffer: Buffer::new() }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder(&self) -> &[I::Item] {
        // SAFETY: We know that all elements before `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buffer.array[..self.buffer.init]) }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder_mut(&mut self) -> &mut [I::Item] {
        // SAFETY: We know that all elements before `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buffer.array[..self.buffer.init]) }
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: Iterator, const N: usize> Iterator for ArrayChunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        let iter = &mut self.iter;
        Guard::with(&mut self.buffer, |array, init| {
            for slot in &mut array[*init..] {
                slot.write(iter.next()?);
                *init += 1;
            }
            *init = 0;
            // SAFETY: The entire array has just been initialized.
            unsafe {
                Some(MaybeUninit::array_assume_init(mem::replace(
                    array,
                    MaybeUninit::uninit_array(),
                )))
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        (lower / N, upper.map(|x| x / N))
    }

    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        let res = match n.checked_mul(N) {
            Some(n) => self.iter.advance_by(n),
            None => {
                let n = (usize::MAX / N) * N;
                self.iter.advance_by(n).and(Err(n))
            }
        };
        res.map_err(|k| k / N)
    }

    fn try_fold<Acc, Fold, R>(&mut self, acc: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        let iter = &mut self.iter;
        Guard::with(&mut self.buffer, |array, init| {
            let result = iter.try_fold(acc, |mut acc, x| {
                // SAFETY: `init` starts at 0, is increased by one each iteration
                // until it equals N (which is `array.len()`) and is reset to 0.
                unsafe {
                    array.get_unchecked_mut(*init).write(x);
                }
                *init += 1;

                if *init == N {
                    *init = 0;
                    // SAFETY: The entire array has just been initialized.
                    let array = unsafe {
                        MaybeUninit::array_assume_init(mem::replace(
                            array,
                            MaybeUninit::uninit_array(),
                        ))
                    };
                    acc = fold(acc, array).branch()?;
                }
                ControlFlow::Continue(acc)
            });

            match result {
                ControlFlow::Continue(acc) => R::from_output(acc),
                ControlFlow::Break(res) => R::from_residual(res),
            }
        })
    }

    fn fold<Acc, Fold>(self, acc: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let Self { iter, mut buffer } = self;
        Guard::with(&mut buffer, |array, init| {
            iter.fold(acc, |mut acc, x| {
                // SAFETY: `init` starts at 0, is increased by one each iteration
                // until it equals N (which is `array.len()`) and is reset to 0.
                unsafe {
                    array.get_unchecked_mut(*init).write(x);
                }
                *init += 1;

                if *init == N {
                    *init = 0;
                    // SAFETY: The entire array has just been initialized.
                    let array = unsafe {
                        MaybeUninit::array_assume_init(mem::replace(
                            array,
                            MaybeUninit::uninit_array(),
                        ))
                    };
                    acc = fold(acc, array);
                }
                acc
            })
        })
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: FusedIterator, const N: usize> FusedIterator for ArrayChunks<I, N> {}
