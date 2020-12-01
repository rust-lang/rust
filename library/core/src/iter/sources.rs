mod empty;
mod from_fn;
mod once;
mod once_with;
mod repeat;
mod repeat_with;
mod successors;

pub use self::repeat::{repeat, Repeat};

#[stable(feature = "iter_empty", since = "1.2.0")]
pub use self::empty::{empty, Empty};

#[stable(feature = "iter_once", since = "1.2.0")]
pub use self::once::{once, Once};

#[stable(feature = "iterator_repeat_with", since = "1.28.0")]
pub use self::repeat_with::{repeat_with, RepeatWith};

#[stable(feature = "iter_from_fn", since = "1.34.0")]
pub use self::from_fn::{from_fn, FromFn};

#[stable(feature = "iter_successors", since = "1.34.0")]
pub use self::successors::{successors, Successors};

#[stable(feature = "iter_once_with", since = "1.43.0")]
pub use self::once_with::{once_with, OnceWith};
