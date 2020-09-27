mod accum;
mod collect;
mod double_ended;
mod exact_size;
mod iterator;
mod marker;

pub use self::accum::{Product, Sum};
pub use self::collect::{Extend, FromIterator, IntoIterator};
pub use self::double_ended::DoubleEndedIterator;
pub use self::exact_size::ExactSizeIterator;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::iterator::Iterator;
#[unstable(issue = "none", feature = "inplace_iteration")]
pub use self::marker::InPlaceIterable;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::marker::{FusedIterator, TrustedLen};
