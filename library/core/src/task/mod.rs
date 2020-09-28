#![stable(feature = "futures_api", since = "1.36.0")]

//! Types and Traits for working with asynchronous tasks.

mod poll;
/// Reexporting `Poll` type.
#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::poll::Poll;

mod wake;
/// Reexporting `wake` items.
#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::wake::{Context, RawWaker, RawWakerVTable, Waker};

mod ready;
/// Reexporting `ready` module.
#[unstable(feature = "ready_macro", issue = "70922")]
pub use ready::ready;
