mod iterator;
mod double_ended;
mod exact_size;
mod collect;
mod accum;
mod marker;
mod optimistic_collect;

pub use self::iterator::Iterator;
pub use self::double_ended::DoubleEndedIterator;
pub use self::exact_size::ExactSizeIterator;
pub use self::collect::{FromIterator, IntoIterator, Extend};
pub use self::accum::{Sum, Product};
pub use self::marker::{FusedIterator, TrustedLen};
pub use self::optimistic_collect::OptimisticCollect;
