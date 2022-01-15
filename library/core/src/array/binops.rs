use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use core::ptr::{drop_in_place, read};


macro_rules! binop {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.59.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N]
        where
            T: $trait<U>,
        {
            type Output = [T::Output; N];

            fn $method(self, rhs: [U; N]) -> Self::Output {
                let mut dc = Iter3::new(self, rhs);

                for _ in 0..N {
                    // SAFETY:
                    // Will only be called a maximum of N times
                    unsafe { dc.step(T::$method) }
                }

                // SAFETY:
                // By this point, we are certain we have initialised all N elements
                unsafe { dc.output() }
            }
        }
    };
}

macro_rules! binop_assign {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.59.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N]
        where
            T: $trait<U>,
        {
            fn $method(&mut self, rhs: [U; N]) {
                let mut dc = Iter::new(rhs);

                for _ in 0..N {
                    // SAFETY:
                    // Will only be called a maximum of N times
                    unsafe { self.get_unchecked_mut(dc.i).$method(dc.next_unchecked()) }
                }
            }
        }
    };
}

binop!(Add, add);
binop!(Mul, mul);
binop!(Div, div);
binop!(Sub, sub);

binop_assign!(AddAssign, add_assign);
binop_assign!(MulAssign, mul_assign);
binop_assign!(DivAssign, div_assign);
binop_assign!(SubAssign, sub_assign);

/// Like [`Iter`], but traverses 3 arrays at once
struct Iter3<T, U, O, const N: usize> {
    rhs: Iter<U, N>,
    lhs: [ManuallyDrop<T>; N],
    output: [MaybeUninit<O>; N],
}

impl<T, U, O, const N: usize> Drop for Iter3<T, U, O, N> {
    fn drop(&mut self) {
        let i = self.rhs.i;
        // SAFETY:
        // `i` defines how many elements are valid at this point.
        // The only possible panic point is in the `f` passed to `step`,
        // so it means that `i..` elements are currently live in lhs
        // and `..i-1` elements are live in the output.
        unsafe {
            drop_in_place((&mut self.lhs[i..]) as *mut [_] as *mut [T]);
            drop_in_place(&mut self.output[..i - 1] as *mut [_] as *mut [O]);
        }
    }
}

impl<T, U, O, const N: usize> Iter3<T, U, O, N> {
    fn new(lhs: [T; N], rhs: [U; N]) -> Self {
        Self {
            rhs: Iter::new(rhs),
            lhs: md_array(lhs),
            output: MaybeUninit::uninit_array(),
        }
    }

    /// # Safety
    /// All values of output must be initialised, and all values in the inputs must be consumed
    unsafe fn output(self) -> [O; N] {
        debug_assert_eq!(self.rhs.i, N);

        let md = ManuallyDrop::new(self);
        // SAFETY:
        // Since it's wrapped in a ManuallyDrop, the contents will not be dropped/accessed again
        // so it's safe to perform the copy
        unsafe { read(&md.output as *const [MaybeUninit<O>; N] as *const [O; N]) }
    }

    /// # Safety
    /// Must be called no more than `N` times.
    #[inline(always)]
    unsafe fn step(&mut self, f: impl FnOnce(T, U) -> O) {
        // SAFETY:
        // Since `dc.i` is stricty-monotonic, we will only
        // take each element only once from each of lhs/rhs
        unsafe {
            let i = self.rhs.i;
            let rhs = self.rhs.next_unchecked();
            let lhs = ManuallyDrop::take(self.lhs.get_unchecked_mut(i));
            let out = self.output.get_unchecked_mut(i);
            out.write(f(lhs, rhs));
        }
    }
}

/// For sake of optimisation, it's a simplified version of [`array::IntoIter`]
/// that can only go forward, and can only be accessed through unsafe (to avoid bounds checks)
struct Iter<U, const N: usize> {
    rhs: [ManuallyDrop<U>; N],
    i: usize,
}

impl<U, const N: usize> Drop for Iter<U, N> {
    fn drop(&mut self) {
        let i = self.i;
        // SAFETY:
        // `i` defines how many elements are valid at this point.
        // The only possible panic point is the element-wise `$method` op,
        // so it means that `i+1..` elements are currently live in rhs
        unsafe {
            drop_in_place((&mut self.rhs[i..]) as *mut [_] as *mut [U]);
        }
    }
}

impl<U, const N: usize> Iter<U, N> {
    fn new(rhs: [U; N]) -> Self {
        Self { rhs: md_array(rhs), i: 0 }
    }

    /// # Safety
    /// Must be called no more than `N` times.
    #[inline(always)]
    unsafe fn next_unchecked(&mut self) -> U {
        debug_assert!(self.i < N);

        // SAFETY:
        // Caller ensures that next is not called more than `N` times, so self.i must be
        // smaller than N at this point
        let rhs = unsafe { self.rhs.get_unchecked_mut(self.i) };

        // SAFETY:
        // Since `dc.i` is stricty-monotonic, we will only
        // take each element only once from each of lhs/rhs
        let rhs = unsafe { ManuallyDrop::take(rhs) };

        self.i += 1;
        rhs
    }
}

/// Create a new `[ManuallyDrop<U>; N]` from the initialised array
fn md_array<T, const N: usize>(a: [T; N]) -> [ManuallyDrop<T>; N] {
    // SAFETY:
    // This is safe since `ManuallyDrop` guarantees the same layout as `T`.
    unsafe { read(&a as *const [T; N] as *const [ManuallyDrop<T>; N]) }
}
