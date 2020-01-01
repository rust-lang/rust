#![stable(feature = "futures_api", since = "1.36.0")]

//! Asynchronous values.

mod future;
#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::future::Future;

#[unstable(feature = "into_future", issue = "67644")]
pub use self::future::IntoFuture;
