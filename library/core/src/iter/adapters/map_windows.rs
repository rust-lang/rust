use crate::iter::FusedIterator;
use crate::mem::MaybeUninit;
use crate::{fmt, ptr};

/// An iterator over the mapped windows of another iterator.
///
/// This `struct` is created by the [`Iterator::map_windows`]. See its
/// documentation for more information.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[unstable(feature = "iter_map_windows", issue = "87155")]
pub struct MapWindows<I: Iterator, F, const N: usize> {
    f: F,
    inner: MapWindowsInner<I, N>,
}

struct MapWindowsInner<I: Iterator, const N: usize> {
    iter: I,
    // Since iterators are assumed lazy, i.e. it only yields an item when
    // `Iterator::next()` is called, and `MapWindows` is not an exception.
    //
    // Before the first iteration, we keep the buffer `None`. When the user
    // first call `next` or other methods that makes the iterator advance,
    // we collect the first `N` items yielded from the inner iterator and
    // put it into the buffer.
    //
    // When the inner iterator has returned a `None`, we take
    // away this `buffer` and leave it `None` to reclaim its resources.
    //
    // FIXME: should we shrink the size of `buffer` using niche optimization?
    buffer: Option<Buffer<I::Item, N>>,
}

// `Buffer` uses two times of space to reduce moves among the iterations.
struct Buffer<T, const N: usize> {
    // Invariant: N elements starting from `self.start` must be initialized at
    // with all other elements left uninitialized.
    buffer: RawBuffer<T, N>,
    // Invariant: `self.start <= N`
    start: usize,
}

/// Internal storage for `Buffer<T, N>`.
///
/// Has no storage invariants, but has access invariants.
/// See the unsafe method contracts for more information.
///
/// This type does not implement Drop, it will leak any data stored within.
//
// `Buffer<T, N>` is semantically `[MaybeUninit<T>; 2 * N]`. However, due
// to limitations of const generics, we use this different type. Note that
// it has the same underlying memory layout.
#[repr(transparent)]
struct RawBuffer<T, const N: usize> {
    data: [[MaybeUninit<T>; N]; 2],
}

impl<I: Iterator, F, const N: usize> MapWindows<I, F, N> {
    pub(in crate::iter) const fn new(iter: I, f: F) -> Self {
        assert!(N != 0, "array in `Iterator::map_windows` must contain more than 0 elements");

        // Only ZST arrays' length can be so large.
        if size_of::<I::Item>() == 0 {
            assert!(
                N.checked_mul(2).is_some(),
                "array size of `Iterator::map_windows` is too large"
            );
        }

        Self { inner: MapWindowsInner::new(iter), f }
    }
}

impl<I: Iterator, const N: usize> MapWindowsInner<I, N> {
    #[inline]
    const fn new(iter: I) -> Self {
        Self { iter, buffer: None }
    }

    fn next_window(&mut self) -> Option<&[I::Item; N]> {
        match self.buffer {
            // It is the first time to advance. We collect
            // the first `N` items from `self.iter` to initialize `self.buffer`.
            None => self.buffer = Buffer::try_from_iter(&mut self.iter),
            Some(ref mut buffer) => match self.iter.next() {
                None => {
                    self.buffer.take();
                }
                // Advance the iterator. We first call `next` before changing our buffer
                // at all. This means that if `next` panics, our invariant is upheld and
                // our `Drop` impl drops the correct elements.
                Some(item) => buffer.push(item),
            },
        }
        self.buffer.as_ref().map(|buf|
            // SAFETY:
            // - if this was the first time to advance, this is a new well-formed `Buffer`.
            //
            // - if we already had a buffer, and `iter.next()` was Some;
            //   `Buffer::push` is responsible for upholding the invariant before we reach this.
            //
            // - if we already had a buffer, and `iter.next()` was None;
            //   this closure is unreachable, as the buffer was taken beforehand.
            unsafe { buf.buffer.as_array_ref(buf.start) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo, hi) = self.iter.size_hint();
        if self.buffer.is_some() {
            // If the first `N` items are already yielded by the inner iterator,
            // the size hint is then equal to the that of the inner iterator's.
            (lo, hi)
        } else {
            // If the first `N` items are not yet yielded by the inner iterator,
            // the first `N` elements should be counted as one window, so both bounds
            // should subtract `N - 1`.
            (lo.saturating_sub(N - 1), hi.map(|hi| hi.saturating_sub(N - 1)))
        }
    }
}

impl<T, const N: usize> Buffer<T, N> {
    /// # Safety
    ///
    /// This type implements `Drop`, and it has invariants that must be upheld:
    /// - `start` must be within the bounds `0..=N`.
    /// - `raw[start..start + N]` must be initialized.
    #[inline]
    unsafe fn new(raw: RawBuffer<T, N>, start: usize) -> Self {
        Self { buffer: raw, start }
    }

    fn try_from_iter(iter: &mut impl Iterator<Item = T>) -> Option<Self> {
        let first_half: [T; N] = crate::array::iter_next_chunk(iter).ok()?;
        let raw = RawBuffer {
            data: [MaybeUninit::new(first_half).transpose(), [const { MaybeUninit::uninit() }; N]],
        };
        // SAFETY: buffer has the first `first_half.len()` items initialized, which upholds the
        // internal invariant for the start offset 0, which is also within bounds.
        Some(unsafe { Self::new(raw, 0) })
    }

    /// Pushes a new item `next` to the back, and pops the front-most one.
    ///
    /// All the elements will be shifted to the front end when pushing reaches
    /// the back end.
    fn push(&mut self, next: T) {
        let buffer_mut_ptr = self.buffer.as_mut_ptr();
        debug_assert!(self.start + N <= 2 * N);

        let to_drop = if self.start == N {
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
            let to_drop = unsafe {
                ptr::copy_nonoverlapping(buffer_mut_ptr.add(self.start + 1), buffer_mut_ptr, N - 1);
                (*buffer_mut_ptr.add(N - 1)).write(next);
                buffer_mut_ptr.add(self.start)
            };
            self.start = 0;
            to_drop
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
            let to_drop = unsafe {
                (*buffer_mut_ptr.add(self.start + N)).write(next);
                buffer_mut_ptr.add(self.start)
            };
            self.start += 1;
            to_drop
        };

        // SAFETY: the index is valid and this is element `a` in the
        // diagram above and has not been dropped yet.
        unsafe { to_drop.cast_init().drop_in_place() };
    }
}

impl<T, const N: usize> RawBuffer<T, N> {
    fn as_ptr(&self) -> *const MaybeUninit<T> {
        self.data.as_ptr().cast()
    }

    fn as_mut_ptr(&mut self) -> *mut MaybeUninit<T> {
        self.data.as_mut_ptr().cast()
    }

    /// # Safety
    ///
    /// `self.data` must uphold the internal invariants of `Buffer`:
    /// - `start` must be within the bounds `0..=N`
    /// - `self.data[start..start + N]` must be initialized.
    unsafe fn as_array_ref(&self, start: usize) -> &[T; N] {
        debug_assert!(start + N <= 2 * N);

        // SAFETY: caller satisfies that `start` is within bounds.
        unsafe { &*self.as_ptr().add(start).cast() }
    }

    /// # Safety
    ///
    /// `self.data` must uphold the internal invariants of `Buffer`:
    /// - `start` must be within the bounds `0..=N`
    /// - `self.data[start..start + N]` must not overlap with the initialized part.
    #[inline]
    unsafe fn as_uninit_array_mut(&mut self, start: usize) -> &mut MaybeUninit<[T; N]> {
        debug_assert!(start + N <= 2 * N);

        // SAFETY: caller satisfies that `start` is within bounds.
        unsafe { &mut *self.as_mut_ptr().add(start).cast() }
    }
}

impl<T: Clone, const N: usize> Clone for Buffer<T, N> {
    fn clone(&self) -> Self {
        let mut new_raw = RawBuffer {
            data: [
                [const { MaybeUninit::<T>::uninit() }; N],
                [const { MaybeUninit::<T>::uninit() }; N],
            ],
        };

        // SAFETY: invariants of a well-formed `Buffer` guarantee N elements
        // starting at `start` are initialized.
        let init = unsafe { self.buffer.as_array_ref(self.start) };
        let cloned = init.clone();

        // SAFETY: new_raw is currently fully uninitialized, which does not
        // overlap with any initialized part.
        unsafe { new_raw.as_uninit_array_mut(self.start).write(cloned) };

        // SAFETY: new_raw has just been initialized above at the offset `self.start`,
        // and the invariants of a well-formed `Buffer` guarantee `self.start` is within
        // the bounds `0..=N`.
        unsafe { Self::new(new_raw, self.start) }
    }
}

impl<I, const N: usize> Clone for MapWindowsInner<I, N>
where
    I: Iterator + Clone,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        Self { iter: self.iter.clone(), buffer: self.buffer.clone() }
    }
}

impl<T, const N: usize> Drop for Buffer<T, N> {
    fn drop(&mut self) {
        // SAFETY: our invariant guarantees that `self.start` is within bounds
        let init_ptr = unsafe { self.buffer.as_mut_ptr().add(self.start).cast_init() };

        // SAFETY: our invariant guarantees N elements starting from
        // `init_ptr` are initialized. We drop them here.
        unsafe { init_ptr.cast_slice(N).drop_in_place() };
    }
}

#[unstable(feature = "iter_map_windows", issue = "87155")]
impl<I, F, R, const N: usize> Iterator for MapWindows<I, F, N>
where
    I: Iterator,
    F: FnMut(&[I::Item; N]) -> R,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        let window = self.inner.next_window()?;
        let out = (self.f)(window);
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[unstable(feature = "iter_map_windows", issue = "87155")]
impl<I, F, R, const N: usize> FusedIterator for MapWindows<I, F, N>
where
    I: FusedIterator,
    F: FnMut(&[I::Item; N]) -> R,
{
}

#[unstable(feature = "iter_map_windows", issue = "87155")]
impl<I, F, R, const N: usize> ExactSizeIterator for MapWindows<I, F, N>
where
    I: ExactSizeIterator,
    F: FnMut(&[I::Item; N]) -> R,
{
}

#[unstable(feature = "iter_map_windows", issue = "87155")]
impl<I: Iterator + fmt::Debug, F, const N: usize> fmt::Debug for MapWindows<I, F, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapWindows").field("iter", &self.inner.iter).finish()
    }
}

#[unstable(feature = "iter_map_windows", issue = "87155")]
impl<I, F, const N: usize> Clone for MapWindows<I, F, N>
where
    I: Iterator + Clone,
    F: Clone,
    I::Item: Clone,
{
    fn clone(&self) -> Self {
        Self { f: self.f.clone(), inner: self.inner.clone() }
    }
}
