//! Asynchronous values.

#[doc(inline)]
#[stable(feature = "futures_api", since = "1.36.0")]
pub use core::future::Future;

#[doc(inline)]
#[unstable(feature = "gen_future", issue = "50547")]
pub use core::future::{from_generator, get_context, ResumeTy};

#[doc(inline)]
#[stable(feature = "future_readiness_fns", since = "1.47.0")]
pub use core::future::{pending, ready, Pending, Ready};

#[doc(inline)]
#[unstable(feature = "into_future", issue = "67644")]
pub use core::future::IntoFuture;
