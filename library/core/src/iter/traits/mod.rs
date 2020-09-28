mod accum;
mod collect;
mod double_ended;
mod exact_size;
mod iterator;
mod marker;

#[allow(missing_docs)]
pub use self::accum::{Product, Sum};
#[allow(missing_docs)]
pub use self::collect::{Extend, FromIterator, IntoIterator};
#[allow(missing_docs)]
pub use self::double_ended::DoubleEndedIterator;
#[allow(missing_docs)]
pub use self::exact_size::ExactSizeIterator;
#[allow(missing_docs)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::iterator::Iterator;
#[allow(missing_docs)]
#[unstable(issue = "none", feature = "inplace_iteration")]
pub use self::marker::InPlaceIterable;
#[allow(missing_docs)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::marker::{FusedIterator, TrustedLen};
