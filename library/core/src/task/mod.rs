#![stable(feature = "futures_api", since = "1.36.0")]

//! Types and Traits for working with asynchronous tasks.

mod poll;
#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::poll::Poll;

mod wake;
#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::wake::{Context, RawWaker, RawWakerVTable, Waker};

mod ready;
#[stable(feature = "ready_macro", since = "1.64.0")]
pub use ready::ready;
#[unstable(feature = "poll_ready", issue = "89780")]
pub use ready::Ready;
