//! Unix-specific networking functionality.

#![allow(irrefutable_let_patterns)]
#![stable(feature = "unix_socket", since = "1.10.0")]

mod addr;
#[doc(cfg(any(target_os = "android", target_os = "linux")))]
#[cfg(any(doc, target_os = "android", target_os = "linux"))]
mod ancillary;
mod datagram;
mod listener;
mod stream;
#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::addr::*;
#[cfg(any(doc, target_os = "android", target_os = "linux"))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub use self::ancillary::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::datagram::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::listener::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::stream::*;
