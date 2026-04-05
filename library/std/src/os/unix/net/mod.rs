//! Unix-specific networking functionality.

#![allow(irrefutable_let_patterns)]
#![stable(feature = "unix_socket", since = "1.10.0")]

#[cfg(not(target_os = "qurt"))]
mod addr;
#[doc(cfg(any(target_os = "android", target_os = "linux", target_os = "cygwin")))]
#[cfg(any(doc, target_os = "android", target_os = "linux", target_os = "cygwin"))]
mod ancillary;
#[cfg(not(target_os = "qurt"))]
mod datagram;
#[cfg(not(target_os = "qurt"))]
mod listener;
#[cfg(not(target_os = "qurt"))]
mod stream;
#[cfg(all(test, not(any(target_os = "emscripten", target_os = "qurt"))))]
mod tests;
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "nto",
    target_vendor = "apple",
    target_os = "cygwin"
))]
mod ucred;

#[cfg(not(target_os = "qurt"))]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::addr::*;
#[cfg(any(doc, target_os = "android", target_os = "linux", target_os = "cygwin"))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub use self::ancillary::*;
#[cfg(not(target_os = "qurt"))]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::datagram::*;
#[cfg(not(target_os = "qurt"))]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::listener::*;
#[cfg(not(target_os = "qurt"))]
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::stream::*;
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "nto",
    target_vendor = "apple",
    target_os = "cygwin",
))]
#[unstable(feature = "peer_credentials_unix_socket", issue = "42839", reason = "unstable")]
pub use self::ucred::*;
