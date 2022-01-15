use core::ops::{Add, AddAssign, Mul, MulAssign, Div, DivAssign, Sub, SubAssign};
use core::mem::{MaybeUninit, ManuallyDrop};
use core::ptr::{read, drop_in_place};

fn md_array<T, const N: usize>(a: [T; N]) -> [ManuallyDrop<T>; N] {
    // SAFETY:
    // This is safe since `ManuallyDrop` guarantees the same layout as `T`.
    unsafe { read(&a as *const [T; N] as *const [ManuallyDrop<T>; N]) }
}

macro_rules! binop {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.59.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N] where T: $trait<U> {
            type Output = [T::Output; N];

            fn $method(self, rhs: [U; N]) -> Self::Output {
                /// Drop checked to handle dropping the values properly in case of a panic
                struct DropCheck<T: $trait<U>, U, const N: usize> {
                    lhs: [ManuallyDrop<T>; N],
                    rhs: [ManuallyDrop<U>; N],
                    output: [MaybeUninit<T::Output>; N],
                    i: usize,
                }
                impl<T: $trait<U>, U, const N: usize> Drop for DropCheck<T, U, N> {
                    fn drop(&mut self) {
                        let i = self.i;
                        // SAFETY:
                        // `i` defines how many elements are valid at this point.
                        // The only possible panic point is the element-wise `$method` op,
                        // so it means that `i+1..` elements are currently live in lhs/rhs
                        // and `..i` elements are live in the output.
                        unsafe {
                            drop_in_place((&mut self.lhs[i+1..]) as *mut [_] as *mut [T]);
                            drop_in_place((&mut self.rhs[i+1..]) as *mut [_] as *mut [U]);
                            drop_in_place(&mut self.output[..i] as *mut [_] as *mut [T::Output]);
                        }
                    }
                }
                let mut dc = DropCheck {
                    lhs: md_array(self),
                    rhs: md_array(rhs),
                    output: MaybeUninit::uninit_array(),
                    i: 0,
                };

                while dc.i < N {
                    // SAFETY:
                    // Since `dc.i` is stricty-monotonic, we will only
                    // take each element only once from each of lhs/rhs
                    unsafe {
                        let lhs = ManuallyDrop::take(&mut dc.lhs[dc.i]);
                        let rhs = ManuallyDrop::take(&mut dc.rhs[dc.i]);
                        dc.output[dc.i].write(T::$method(lhs, rhs));
                        dc.i += 1;
                    }
                }

                // SAFETY:
                // By this point, we are certain we have initialised all N elements
                let output = unsafe { read(&dc.output as *const [_; N] as *const [T::Output; N]) };

                // No more panics can occur after this point, so we 'forget'
                // the dc value to skip dropping the contents
                core::mem::forget(dc);

                output
            }
        }
    }
}

macro_rules! binop_assign {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.59.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N] where T: $trait<U> {
            fn $method(&mut self, rhs: [U; N]) {
                let mut rhs = IntoIterator::into_iter(rhs);
                for i in 0..N {
                    // SAFETY:
                    // Since this is a constant size array, we know there's spare elements
                    self[i].$method(unsafe { rhs.next().unwrap_unchecked() })
                }
            }
        }
    }
}

binop!(Add, add);
binop!(Mul, mul);
binop!(Div, div);
binop!(Sub, sub);

binop_assign!(AddAssign, add_assign);
binop_assign!(MulAssign, mul_assign);
binop_assign!(DivAssign, div_assign);
binop_assign!(SubAssign, sub_assign);
