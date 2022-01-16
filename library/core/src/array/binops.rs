use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

macro_rules! binop {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.60.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N]
        where
            T: $trait<U>,
        {
            type Output = [T::Output; N];

            fn $method(self, rhs: [U; N]) -> Self::Output {
                self.zip_map(rhs, T::$method)
            }
        }
    };
}

macro_rules! binop_assign {
    ($trait:ident, $method:ident) => {
        #[stable(feature = "array_bin_ops", since = "1.60.0")]
        impl<T, U, const N: usize> $trait<[U; N]> for [T; N]
        where
            T: $trait<U>,
        {
            fn $method(&mut self, rhs: [U; N]) {
                self.zip_map_assign(rhs, T::$method)
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
