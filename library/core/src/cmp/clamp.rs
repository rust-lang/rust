use crate::marker::Destruct;
use crate::ops::{RangeFrom, RangeFull, RangeInclusive, RangeToInclusive};

/// Trait for ranges supported by [`Ord::clamp_to`].
#[unstable(feature = "clamp_bounds", issue = "147781")]
#[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
pub const trait ClampBounds<T>: Sized {
    /// The implementation of [`Ord::clamp_to`].
    fn clamp(self, value: T) -> T
    where
        T: [const] Destruct;
}

#[unstable(feature = "clamp_bounds", issue = "147781")]
#[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
impl<T> const ClampBounds<T> for RangeFrom<T>
where
    T: [const] Ord,
{
    fn clamp(self, value: T) -> T
    where
        T: [const] Destruct,
    {
        value.max(self.start)
    }
}

#[unstable(feature = "clamp_bounds", issue = "147781")]
#[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
impl<T> const ClampBounds<T> for RangeToInclusive<T>
where
    T: [const] Ord,
{
    fn clamp(self, value: T) -> T
    where
        T: [const] Destruct,
    {
        value.min(self.end)
    }
}

#[unstable(feature = "clamp_bounds", issue = "147781")]
#[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
impl<T> const ClampBounds<T> for RangeInclusive<T>
where
    T: [const] Ord,
{
    fn clamp(self, value: T) -> T
    where
        T: [const] Destruct,
    {
        let (start, end) = self.into_inner();
        value.clamp(start, end)
    }
}

#[unstable(feature = "clamp_bounds", issue = "147781")]
#[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
impl<T> const ClampBounds<T> for RangeFull {
    fn clamp(self, value: T) -> T {
        value
    }
}

macro impl_for_float($t:ty) {
    #[unstable(feature = "clamp_bounds", issue = "147781")]
    #[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
    impl const ClampBounds<$t> for RangeFrom<$t> {
        fn clamp(self, value: $t) -> $t {
            value.max(self.start)
        }
    }

    #[unstable(feature = "clamp_bounds", issue = "147781")]
    #[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
    impl const ClampBounds<$t> for RangeToInclusive<$t> {
        fn clamp(self, value: $t) -> $t {
            value.min(self.end)
        }
    }

    #[unstable(feature = "clamp_bounds", issue = "147781")]
    #[rustc_const_unstable(feature = "clamp_bounds", issue = "147781")]
    impl const ClampBounds<$t> for RangeInclusive<$t> {
        fn clamp(self, value: $t) -> $t {
            let (start, end) = self.into_inner();
            // Deliberately avoid using `clamp` to handle NaN consistently
            value.max(start).min(end)
        }
    }
}

// #[unstable(feature = "f16", issue = "116909")]
impl_for_float!(f16);
impl_for_float!(f32);
impl_for_float!(f64);
// #[unstable(feature = "f128", issue = "116909")]
impl_for_float!(f128);
