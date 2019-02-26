mod iterator;
mod double_ended;
mod exact_size;
mod collect;
mod accum;
mod marker;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::iterator::Iterator;
pub use self::double_ended::DoubleEndedIterator;
pub use self::exact_size::ExactSizeIterator;
pub use self::collect::{FromIterator, IntoIterator, Extend};
pub use self::accum::{Sum, Product};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::marker::{FusedIterator, TrustedLen};
