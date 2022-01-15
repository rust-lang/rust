use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use core::ptr::{drop_in_place, read};

fn md_array<T, const N: usize>(a: [T; N]) -> [ManuallyDrop<T>; N] {
    // SAFETY:
    // This is safe since `ManuallyDrop` guarantees the same layout as `T`.
    unsafe { read(&a as *const [T; N] as *const [ManuallyDrop<T>; N]) }
}

/// Drop checked to handle dropping the values properly in case of a panic
struct DropCheck<T, U, O, const N: usize> {
    lhs: [ManuallyDrop<T>; N],
    assign: DropCheckAssign<U, N>,
    output: [MaybeUninit<O>; N],
}

impl<T, U, O, const N: usize> Drop for DropCheck<T, U, O, N> {
    fn drop(&mut self) {
        let i = self.assign.i;
        // SAFETY:
        // `i` defines how many elements are valid at this point.
        // The only possible panic point is the element-wise `$method` op,
        // so it means that `i+1..` elements are currently live in lhs
        // and `..i` elements are live in the output.
        unsafe {
            drop_in_place((&mut self.lhs[i..]) as *mut [_] as *mut [T]);
            drop_in_place(&mut self.output[..i - 1] as *mut [_] as *mut [O]);
        }
    }
}

impl<T, U, O, const N: usize> DropCheck<T, U, O, N> {
    fn new(lhs: [T; N], rhs: [U; N]) -> Self {
        Self {
            lhs: md_array(lhs),
            assign: DropCheckAssign::new(rhs),
            output: MaybeUninit::uninit_array(),
        }
    }

    /// # Safety
    /// All values of output must be initialised, and all values in the inputs must be consumed
    unsafe fn output(self) -> [O; N] {
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
            let lhs = ManuallyDrop::take(&mut self.lhs[self.assign.i]);
            let out = &mut self.output[self.assign.i];
            out.write(f(lhs, self.assign.next_unchecked()));
        }
    }
}

macro_rules! binop {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.59.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N]
        where
            T: $trait<U>,
        {
            type Output = [T::Output; N];

            fn $method(self, rhs: [U; N]) -> Self::Output {
                let mut dc = DropCheck::new(self, rhs);

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

/// Drop checked to handle dropping the values properly in case of a panic
struct DropCheckAssign<U, const N: usize> {
    rhs: [ManuallyDrop<U>; N],
    i: usize,
}

impl<U, const N: usize> Drop for DropCheckAssign<U, N> {
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

impl<U, const N: usize> DropCheckAssign<U, N> {
    fn new(rhs: [U; N]) -> Self {
        Self { rhs: md_array(rhs), i: 0 }
    }

    /// # Safety
    /// Must be called no more than `N` times.
    #[inline(always)]
    unsafe fn next_unchecked(&mut self) -> U {
        // SAFETY:
        // Since `dc.i` is stricty-monotonic, we will only
        // take each element only once from each of lhs/rhs
        unsafe {
            let rhs = ManuallyDrop::take(&mut self.rhs[self.i]);
            self.i += 1;
            rhs
        }
    }
}

macro_rules! binop_assign {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.59.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N]
        where
            T: $trait<U>,
        {
            fn $method(&mut self, rhs: [U; N]) {
                let mut dc = DropCheckAssign::new(rhs);

                for _ in 0..N {
                    // SAFETY:
                    // Will only be called a maximum of N times
                    unsafe { self[dc.i].$method(dc.next_unchecked()) }
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
