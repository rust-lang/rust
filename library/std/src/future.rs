//! Asynchronous values.

#[allow(missing_docs)]
#[doc(inline)]
#[stable(feature = "futures_api", since = "1.36.0")]
pub use core::future::Future;

#[allow(missing_docs)]
#[doc(inline)]
#[unstable(feature = "gen_future", issue = "50547")]
pub use core::future::{from_generator, get_context, ResumeTy};

#[allow(missing_docs)]
#[doc(inline)]
#[stable(feature = "future_readiness_fns", since = "1.48.0")]
pub use core::future::{pending, ready, Pending, Ready};

#[allow(missing_docs)]
#[doc(inline)]
#[unstable(feature = "into_future", issue = "67644")]
pub use core::future::IntoFuture;
