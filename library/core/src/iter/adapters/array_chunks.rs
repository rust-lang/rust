use core::array::Guard;
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
    done: bool,
}

impl<I: Iterator, const N: usize> ArrayChunks<I, N> {
    pub(in crate::iter) fn new(iter: I) -> Self {
        Self { iter, buffer: Buffer::new(), done: false }
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
        if self.done {
            return None;
        }

        let mut array = MaybeUninit::uninit_array();
        // SAFETY: `guard` is always either forgotten or dropped before the array
        // is moved/dropped and `guard.init` properly tracks the initialized
        // members of the array.
        let mut guard = unsafe { Guard::new(&mut array) };
        for slot in &mut array {
            let next = match self.iter.next() {
                Some(n) => n,
                None => {
                    self.done = true;
                    if guard.init > 0 {
                        self.buffer.init = guard.init;
                        mem::forget(guard);
                        self.buffer.array = array;
                    }
                    return None;
                }
            };
            slot.write(next);
            guard.init += 1;
        }
        mem::forget(guard);
        // SAFETY: The entire array has just been initialized.
        unsafe { Some(MaybeUninit::array_assume_init(array)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        let (lower, upper) = self.iter.size_hint();
        (lower / N, upper.map(|x| x / N))
    }

    fn try_fold<Acc, Fold, R>(&mut self, acc: Acc, mut fold: Fold) -> R
    where
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if self.done {
            return R::from_output(acc);
        }

        let mut array = MaybeUninit::uninit_array();
        // SAFETY: `guard` is always either forgotten or dropped before the array
        // is moved/dropped and `guard.init` properly tracks the initialized
        // members of the array.
        let mut guard = unsafe { Guard::new(&mut array) };
        let result = self.iter.try_fold(acc, |mut acc, x| {
            // SAFETY: `guard.init` starts at 0, is increased by one each iteration
            // until it equals N (which is `array.len()`) and is reset to 0.
            unsafe {
                array.get_unchecked_mut(guard.init).write(x);
            }
            guard.init += 1;

            if guard.init == N {
                guard.init = 0;
                // SAFETY: The entire array has just been initialized.
                let array = unsafe {
                    MaybeUninit::array_assume_init(mem::replace(
                        &mut array,
                        MaybeUninit::uninit_array(),
                    ))
                };
                acc = fold(acc, array)?;
            }
            R::from_output(acc)
        });
        match result.branch() {
            ControlFlow::Continue(x) => {
                self.done = true;
                if guard.init > 0 {
                    self.buffer.init = guard.init;
                    mem::forget(guard);
                    self.buffer.array = array;
                }
                R::from_output(x)
            }
            ControlFlow::Break(x) => R::from_residual(x),
        }
    }

    fn fold<Acc, Fold>(self, acc: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if self.done {
            return acc;
        }

        let mut array = MaybeUninit::uninit_array();
        // SAFETY: `guard` is always either forgotten or dropped before the array
        // is moved/dropped and `guard.init` properly tracks the initialized
        // members of the array.
        let mut guard = unsafe { Guard::new(&mut array) };
        self.iter.fold(acc, |mut acc, x| {
            // SAFETY: `guard.init` starts at 0, is increased by one each iteration
            // until it equals N (which is `array.len()`) and is reset to 0.
            unsafe {
                array.get_unchecked_mut(guard.init).write(x);
            }
            guard.init += 1;

            if guard.init == N {
                guard.init = 0;
                // SAFETY: The entire array has just been initialized.
                let array = unsafe {
                    MaybeUninit::array_assume_init(mem::replace(
                        &mut array,
                        MaybeUninit::uninit_array(),
                    ))
                };
                acc = fold(acc, array);
            }
            acc
        })
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: DoubleEndedIterator, const N: usize> DoubleEndedIterator for ArrayChunks<I, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mut array = MaybeUninit::uninit_array();
        // SAFETY: `guard` is always either forgotten or dropped before the array
        // is moved/dropped and `guard.init` properly tracks the initialized
        // members of the array.
        let mut guard = unsafe { Guard::new(&mut array) };
        for slot in &mut array {
            let next = match self.iter.next_back() {
                Some(n) => n,
                None => {
                    self.done = true;
                    if guard.init > 0 {
                        (&mut array[..guard.init]).reverse();
                        self.buffer.init = guard.init;
                        mem::forget(guard);
                        self.buffer.array = array;
                    }
                    return None;
                }
            };
            slot.write(next);
            guard.init += 1;
        }
        guard.init = 0;
        array.reverse();
        // SAFETY: The entire array has just been initialized.
        unsafe { Some(MaybeUninit::array_assume_init(array)) }
    }

    fn try_rfold<Acc, Fold, R>(&mut self, acc: Acc, mut fold: Fold) -> R
    where
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        if self.done {
            return R::from_output(acc);
        }

        let mut array = MaybeUninit::uninit_array();
        // SAFETY: `guard` is always either forgotten or dropped before the array
        // is moved/dropped and `guard.init` properly tracks the initialized
        // members of the array.
        let mut guard = unsafe { Guard::new(&mut array) };
        let result = self.iter.try_rfold(acc, |mut acc, x| {
            // SAFETY: `guard.init` starts at 0, is increased by one each iteration
            // until it equals N (which is `array.len()`) and is reset to 0.
            unsafe {
                array.get_unchecked_mut(guard.init).write(x);
            }
            guard.init += 1;

            if guard.init == N {
                guard.init = 0;
                array.reverse();
                // SAFETY: The entire array has just been initialized.
                let array = unsafe {
                    MaybeUninit::array_assume_init(mem::replace(
                        &mut array,
                        MaybeUninit::uninit_array(),
                    ))
                };
                acc = fold(acc, array)?;
            }
            R::from_output(acc)
        });
        match result.branch() {
            ControlFlow::Continue(x) => {
                self.done = true;
                if guard.init > 0 {
                    (&mut array[..guard.init]).reverse();
                    self.buffer.init = guard.init;
                    mem::forget(guard);
                    self.buffer.array = array;
                }
                R::from_output(x)
            }
            ControlFlow::Break(x) => R::from_residual(x),
        }
    }

    fn rfold<Acc, Fold>(self, acc: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        if self.done {
            return acc;
        }

        let mut array = MaybeUninit::uninit_array();
        // SAFETY: `guard` is always either forgotten or dropped before the array
        // is moved/dropped and `guard.init` properly tracks the initialized
        // members of the array.
        let mut guard = unsafe { Guard::new(&mut array) };
        self.iter.rfold(acc, |mut acc, x| {
            // SAFETY: `guard.init` starts at 0, is increased by one each iteration
            // until it equals N (which is `array.len()`) and is reset to 0.
            unsafe {
                array.get_unchecked_mut(guard.init).write(x);
            }
            guard.init += 1;

            if guard.init == N {
                guard.init = 0;
                array.reverse();
                // SAFETY: The entire array has just been initialized.
                let array = unsafe {
                    MaybeUninit::array_assume_init(mem::replace(
                        &mut array,
                        MaybeUninit::uninit_array(),
                    ))
                };
                acc = fold(acc, array);
            }
            acc
        })
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: Iterator, const N: usize> FusedIterator for ArrayChunks<I, N> {}
