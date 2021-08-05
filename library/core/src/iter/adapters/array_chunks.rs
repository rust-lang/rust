use core::iter::FusedIterator;
use core::mem::{self, MaybeUninit};
use core::ops::{ControlFlow, Try};
use core::ptr;

#[derive(Debug)]
struct Guard<T, const N: usize> {
    arr: [MaybeUninit<T>; N],
    init: usize,
}

impl<T, const N: usize> Guard<T, N> {
    fn new() -> Self {
        Self { arr: MaybeUninit::uninit_array(), init: 0 }
    }

    // SAFETY: The entire array must have been properly initialized.
    unsafe fn assume_init_read(&mut self) -> [T; N] {
        debug_assert_eq!(self.init, N);
        // Reset the tracked initialization state.
        self.init = 0;
        // SAFETY: This is safe: `MaybeUninit<T>` is guaranteed to have the same layout
        // as `T` and per the function's contract the entire array has been initialized.
        unsafe { mem::transmute_copy(&self.arr) }
    }
}

impl<T: Clone, const N: usize> Clone for Guard<T, N> {
    fn clone(&self) -> Self {
        let mut new = Self::new();
        // SAFETY: this raw slice contains only the initialized objects.
        let src = unsafe { MaybeUninit::slice_assume_init_ref(&self.arr[..self.init]) };
        MaybeUninit::write_slice_cloned(&mut new.arr[..self.init], src);
        new.init = self.init;
        new
    }
}

impl<T, const N: usize> Drop for Guard<T, N> {
    fn drop(&mut self) {
        debug_assert!(self.init <= N);

        let init_part = &mut self.arr[..self.init];
        // SAFETY: this raw slice will contain only the initialized objects
        // that have not been dropped or moved.
        unsafe {
            ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(init_part));
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
    guard: Guard<I::Item, N>,
}

impl<I: Iterator, const N: usize> ArrayChunks<I, N> {
    pub(in crate::iter) fn new(iter: I) -> Self {
        Self { iter, guard: Guard::new() }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder(&self) -> &[I::Item] {
        // SAFETY: We know that all elements before `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_ref(&self.guard.arr[..self.guard.init]) }
    }

    /// Returns the remainder of the elements yielded by the original
    /// iterator that were insufficient to fill another chunk. The
    /// returned slice has at most `N-1` elements.
    #[unstable(feature = "iter_array_chunks", issue = "none")]
    pub fn remainder_mut(&mut self) -> &mut [I::Item] {
        // SAFETY: We know that all elements before `init` are properly initialized.
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.guard.arr[..self.guard.init]) }
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: Iterator, const N: usize> Iterator for ArrayChunks<I, N> {
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        while self.guard.init < N {
            self.guard.arr[self.guard.init] = MaybeUninit::new(self.iter.next()?);
            self.guard.init += 1;
        }
        // SAFETY: The entire array has just been initialized.
        unsafe { Some(self.guard.assume_init_read()) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        (lower / N, upper.map(|x| x / N))
    }

    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        let res = match n.checked_mul(N) {
            Some(n) => self.iter.advance_by(n),
            None => self.iter.advance_by(usize::MAX).and(Err(usize::MAX)),
        };
        res.map_err(|k| k / N)
    }

    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, mut fold: Fold) -> R
    where
        Self: Sized,
        Fold: FnMut(Acc, Self::Item) -> R,
        R: Try<Output = Acc>,
    {
        let mut guard = mem::replace(&mut self.guard, Guard::new());
        let result = self.iter.try_fold(init, |mut acc, x| {
            guard.arr[guard.init] = MaybeUninit::new(x);
            guard.init += 1;

            if guard.init == N {
                // SAFETY: The entire array has just been initialized.
                let arr = unsafe { guard.assume_init_read() };
                acc = fold(acc, arr).branch()?;
            }
            ControlFlow::Continue(acc)
        });

        if guard.init != 0 {
            self.guard = guard;
        }

        match result {
            ControlFlow::Continue(acc) => R::from_output(acc),
            ControlFlow::Break(res) => R::from_residual(res),
        }
    }

    fn fold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut guard = Guard::<_, N>::new();
        self.iter.fold(init, |mut acc, x| {
            guard.arr[guard.init] = MaybeUninit::new(x);
            guard.init += 1;

            if guard.init == N {
                // SAFETY: The entire array has just been initialized.
                let arr = unsafe { guard.assume_init_read() };
                acc = fold(acc, arr);
            }
            acc
        })
    }
}

#[unstable(feature = "iter_array_chunks", issue = "none")]
impl<I: FusedIterator, const N: usize> FusedIterator for ArrayChunks<I, N> {}
