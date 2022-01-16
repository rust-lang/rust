use core::mem::{transmute_copy, ManuallyDrop, MaybeUninit};
use core::ptr::drop_in_place;

/// Like [`Iter`], but traverses 3 arrays at once
pub struct ZipMapIter<T, U, O, const N: usize> {
    lhs: [MaybeUninit<T>; N],
    rhs: [MaybeUninit<U>; N],
    output: [MaybeUninit<O>; N],
    i: usize,
}

impl<T, U, O, const N: usize> Drop for ZipMapIter<T, U, O, N> {
    fn drop(&mut self) {
        let i = self.i;
        // SAFETY:
        // `i` defines how many elements have been processed from the arrays.
        // Caveat, the only potential panic would happen *before* the write to the output,
        // so the `i`th output is not initialised as one would assume.
        unsafe {
            drop_in_place((&mut self.lhs[i..]) as *mut [_] as *mut [T]);
            drop_in_place((&mut self.rhs[i..]) as *mut [_] as *mut [T]);
            drop_in_place(&mut self.output[..i - 1] as *mut [_] as *mut [O]);
        }
    }
}

impl<T, U, O, const N: usize> ZipMapIter<T, U, O, N> {
    pub fn new(lhs: [T; N], rhs: [U; N]) -> Self {
        Self { lhs: mu_array(lhs), rhs: mu_array(rhs), output: MaybeUninit::uninit_array(), i: 0 }
    }

    /// # Safety
    /// All values of output must be initialised, and all values in the inputs must be consumed
    pub unsafe fn output(self) -> [O; N] {
        debug_assert_eq!(self.i, N);

        let md = ManuallyDrop::new(self);
        // SAFETY:
        // caller is responsible for ensuring the output is fully initialised
        unsafe { assume_array_init_copy(&md.output) }
    }

    /// # Safety
    /// Must be called no more than `N` times.
    pub unsafe fn step(&mut self, f: impl FnOnce(T, U) -> O) {
        debug_assert!(self.i < N);

        // SAFETY:
        // Since `self.i` is stricty-monotonic, we will only
        // take each element only once from each of lhs/rhs/out
        unsafe {
            let lhs = take(self.lhs.get_unchecked_mut(self.i));
            let rhs = take(self.rhs.get_unchecked_mut(self.i));
            let out = self.output.get_unchecked_mut(self.i);
            self.i += 1;
            out.write(f(lhs, rhs));
        }
    }
}

/// For sake of optimisation, it's a simplified version of [`array::IntoIter`]
/// that can only go forward, and can only be accessed through unsafe (to avoid bounds checks)
pub struct ForwardIter<U, const N: usize> {
    rhs: [MaybeUninit<U>; N],
    i: usize,
}

impl<U, const N: usize> Drop for ForwardIter<U, N> {
    fn drop(&mut self) {
        let i = self.i;
        // SAFETY:
        // `i` defines how many elements have been processed from the array,
        // meaning that theres `i..` elements left to process (and therefore, drop)
        unsafe {
            drop_in_place((&mut self.rhs[i..]) as *mut [_] as *mut [U]);
        }
    }
}

impl<U, const N: usize> ForwardIter<U, N> {
    pub fn index(&self) -> usize {
        self.i
    }

    pub fn new(rhs: [U; N]) -> Self {
        Self { rhs: mu_array(rhs), i: 0 }
    }

    /// # Safety
    /// Must be called no more than `N` times.
    #[inline(always)]
    pub unsafe fn next_unchecked(&mut self) -> U {
        debug_assert!(self.i < N);

        // SAFETY:
        // Caller ensures that next is not called more than `N` times, so self.i must be
        // smaller than N at this point
        let rhs = unsafe { self.rhs.get_unchecked_mut(self.i) };

        // SAFETY:
        // Since `dc.i` is stricty-monotonic, we will only
        // take each element only once from each of lhs/rhs
        let rhs = unsafe { take(rhs) };

        self.i += 1;
        rhs
    }
}

pub unsafe fn take<T>(slot: &mut MaybeUninit<T>) -> T {
    // SAFETY: we are reading from a reference, which is guaranteed
    // to be valid for reads.
    unsafe { core::ptr::read(slot.assume_init_mut()) }
}

/// Create a new `[ManuallyDrop<U>; N]` from the initialised array
fn mu_array<T, const N: usize>(a: [T; N]) -> [MaybeUninit<T>; N] {
    a.map(MaybeUninit::new)
}

/// # Safety
/// the array must be fully initialised, and it must not be used after this call.
pub unsafe fn assume_array_init_copy<T, const N: usize>(a: &[MaybeUninit<T>; N]) -> [T; N] {
    // SAFETY: MaybeUninit is guaranteed to have the same layout
    unsafe { transmute_copy(a) }
}
