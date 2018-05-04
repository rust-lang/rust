//! Used by the wrapping_sum and wrapping_product algorithms for AArch64.

pub(crate) trait Wrapping {
    fn add(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
}

macro_rules! int_impl {
    ($id:ident) => {
        impl Wrapping for $id {
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            fn mul(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
        }
    };
}
int_impl!(i8);
int_impl!(i16);
int_impl!(i32);
int_impl!(i64);
int_impl!(u8);
int_impl!(u16);
int_impl!(u32);
int_impl!(u64);

macro_rules! float_impl {
    ($id:ident) => {
        impl Wrapping for $id {
            fn add(self, other: Self) -> Self {
                self + other
            }
            fn mul(self, other: Self) -> Self {
                self * other
            }
        }
    };
}
float_impl!(f32);
float_impl!(f64);
