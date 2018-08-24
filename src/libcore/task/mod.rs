#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

//! Types and Traits for working with asynchronous tasks.

mod context;
pub use self::context::Context;

mod spawn;
pub use self::spawn::{Spawn, SpawnErrorKind, SpawnObjError, SpawnLocalObjError};

mod poll;
pub use self::poll::Poll;

mod wake;
pub use self::wake::{Waker, LocalWaker, UnsafeWake};
