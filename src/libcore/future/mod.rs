#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

//! Asynchronous values.

mod future;
pub use self::future::Future;

mod future_obj;
pub use self::future_obj::{FutureObj, LocalFutureObj, UnsafeFutureObj};
