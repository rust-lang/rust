#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

//! Types and Traits for working with asynchronous tasks.

mod poll;
pub use self::poll::Poll;

mod wake;
pub use self::wake::{Waker, LocalWaker, UnsafeWake};
