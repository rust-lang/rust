#![stable(feature = "futures_api", since = "1.36.0")]

//! Asynchronous values.

mod future;
#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::future::Future;
