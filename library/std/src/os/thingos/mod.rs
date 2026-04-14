//! ThingOS-specific definitions.

#![stable(feature = "os_thingos", since = "1.0.0")]
#![doc(cfg(target_os = "thingos"))]

pub mod fd;
pub mod fs;
pub mod net;
pub mod process;
pub mod raw;

/// A prelude for conveniently writing platform-specific code.
#[stable(feature = "os_thingos", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    #[stable(feature = "os_thingos", since = "1.0.0")]
    pub use super::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
}
