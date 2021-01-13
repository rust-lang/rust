//! Unix-specific networking functionality

#![stable(feature = "unix_socket", since = "1.10.0")]

mod addr;
#[doc(cfg(any(
    target_os = "android",
    target_os = "dragonfly",
    target_os = "emscripten",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_os = "openbsd",
)))]
#[cfg(any(
    doc,
    target_os = "android",
    target_os = "dragonfly",
    target_os = "emscripten",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_os = "openbsd",
))]
mod ancillary;
mod datagram;
mod listener;
mod raw_fd;
mod stream;
#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::addr::*;
#[cfg(any(
    doc,
    target_os = "android",
    target_os = "dragonfly",
    target_os = "emscripten",
    target_os = "freebsd",
    target_os = "linux",
    target_os = "netbsd",
    target_os = "openbsd",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub use self::ancillary::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::datagram::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::listener::*;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::raw_fd::*;
#[stable(feature = "unix_socket", since = "1.10.0")]
pub use self::stream::*;
