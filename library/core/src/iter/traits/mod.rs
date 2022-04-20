mod accum;
mod collect;
mod double_ended;
mod exact_size;
mod iterator;
mod marker;
pub(crate) mod trusted_random_access;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::{
    accum::{Product, Sum},
    collect::{Extend, FromIterator, IntoIterator},
    double_ended::DoubleEndedIterator,
    exact_size::ExactSizeIterator,
    iterator::Iterator,
    marker::{FusedIterator, TrustedLen},
};

#[unstable(issue = "none", feature = "inplace_iteration")]
pub use self::marker::InPlaceIterable;
#[unstable(feature = "trusted_step", issue = "85731")]
pub use self::marker::TrustedStep;
#[unstable(feature = "trusted_random_access", issue = "none")]
pub use self::trusted_random_access::{
    TrustedRandomAccess, TrustedRandomAccessNeedsCleanup, TrustedRandomAccessNeedsForwardSetup,
    TrustedRandomAccessNeedsReverseSetup,
};
