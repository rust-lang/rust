mod accum;
mod collect;
mod double_ended;
mod exact_size;
mod iterator;
mod marker;
mod unchecked_iterator;

#[unstable(issue = "none", feature = "inplace_iteration")]
pub use self::marker::InPlaceIterable;
#[unstable(issue = "none", feature = "trusted_fused")]
pub use self::marker::TrustedFused;
#[unstable(feature = "trusted_step", issue = "85731")]
pub use self::marker::TrustedStep;
pub(crate) use self::unchecked_iterator::UncheckedIterator;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::{
    accum::{Product, Sum},
    collect::{Extend, FromIterator, IntoIterator},
    double_ended::DoubleEndedIterator,
    exact_size::ExactSizeIterator,
    iterator::Iterator,
    marker::{FusedIterator, TrustedLen},
};
