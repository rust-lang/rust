use num::{Zero, One};
use ops::Neg;
use sys::error;

pub fn cvt_zero<T: PartialEq + Zero>(t: T) -> error::Result<T> {
    if t == T::zero() {
        error::expect_last_result()
    } else {
        Ok(t)
    }
}

pub fn cvt_neg1<T: One + PartialEq + Neg<Output=T>>(t: T) -> error::Result<T> {
    if t == -T::one() {
        error::expect_last_result()
    } else {
        Ok(t)
    }
}
