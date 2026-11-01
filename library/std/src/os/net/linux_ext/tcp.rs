//! Linux and Android-specific tcp extensions to primitives in the [`std::net`] module.
//!
//! [`std::net`]: crate::net

use crate::macros::rustdoc_skip_body;
use crate::sys::AsInner;
#[cfg(target_os = "linux")]
use crate::time::Duration;
use crate::{io, net};

/// Os-specific extensions for [`TcpStream`]
///
/// [`TcpStream`]: net::TcpStream
#[stable(feature = "tcp_quickack", since = "1.89.0")]
pub impl(self) trait TcpStreamExt {
    /// Enable or disable `TCP_QUICKACK`.
    ///
    /// This flag causes Linux to eagerly send ACKs rather than delaying them.
    /// Linux may reset this flag after further operations on the socket.
    ///
    /// See [`man 7 tcp`](https://man7.org/linux/man-pages/man7/tcp.7.html) and
    /// [TCP delayed acknowledgement](https://en.wikipedia.org/wiki/TCP_delayed_acknowledgment)
    /// for more information.
    ///
    /// # Examples
    ///
    #[cfg_attr(
        any(target_os = "linux", target_os = "android", target_os = "cygwin"),
        doc = "```no_run"
    )]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_os = "cygwin")),
        doc = "```ignore (needs linux)"
    )]
    /// use std::net::TcpStream;
    /// #[cfg(target_os = "linux")]
    /// use std::os::linux::net::TcpStreamExt;
    /// #[cfg(target_os = "android")]
    /// use std::os::android::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_quickack(true).expect("set_quickack call failed");
    /// ```
    #[stable(feature = "tcp_quickack", since = "1.89.0")]
    fn set_quickack(&self, quickack: bool) -> io::Result<()>;

    /// Gets the value of the `TCP_QUICKACK` option on this socket.
    ///
    /// For more information about this option, see [`TcpStreamExt::set_quickack`].
    ///
    /// # Examples
    ///
    #[cfg_attr(
        any(target_os = "linux", target_os = "android", target_os = "cygwin"),
        doc = "```no_run"
    )]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_os = "cygwin")),
        doc = "```ignore (needs linux)"
    )]
    /// use std::net::TcpStream;
    /// #[cfg(target_os = "linux")]
    /// use std::os::linux::net::TcpStreamExt;
    /// #[cfg(target_os = "android")]
    /// use std::os::android::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_quickack(true).expect("set_quickack call failed");
    /// assert_eq!(stream.quickack().unwrap_or(false), true);
    /// ```
    #[stable(feature = "tcp_quickack", since = "1.89.0")]
    fn quickack(&self) -> io::Result<bool>;

    /// Configures the `TCP_DEFER_ACCEPT` option on the socket.
    ///
    /// This defers accepting an incoming connection that has not yet received data.
    /// The `accept` argument specifies an approximate delay.
    ///
    /// Currently, the standard library rounds the duration down to whole seconds,
    /// capped at `c_int::MAX` seconds. This may change if the underlying API becomes
    /// more expressive. Passing [`Duration::ZERO`] disables this option.
    ///
    /// This option can also limit the number of attempts TCP makes to complete a
    /// connection, so connections may be dropped if more retransmissions are needed.
    ///
    /// Unlike the equivalent `SO_ACCEPTFILTER` option on other platforms, this option
    /// can be set before calling `listen`.
    ///
    /// See [`man 7 tcp`](https://man7.org/linux/man-pages/man7/tcp.7.html) for more information.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_deferaccept)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    /// use std::time::Duration;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_defer_accept(Duration::from_secs(1u64)).expect("set_defer_accept call failed");
    /// ```
    #[unstable(feature = "tcp_deferaccept", issue = "119639")]
    #[cfg(target_os = "linux")]
    fn set_defer_accept(&self, accept: Duration) -> io::Result<()>;

    /// Gets the accept delay value of the `TCP_DEFER_ACCEPT` option.
    ///
    /// Returns [`None`] if this option is disabled, or [`Some`] containing the delay
    /// otherwise. The returned duration may differ from the requested duration due
    /// to rounding.
    ///
    /// For more information about this option, see [`TcpStreamExt::set_defer_accept`].
    ///
    /// # Examples
    ///
    #[cfg_attr(
        any(target_os = "linux", target_os = "android", target_os = "cygwin"),
        doc = "```no_run"
    )]
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_os = "cygwin")),
        doc = "```ignore (needs linux)"
    )]
    /// #![feature(tcp_deferaccept)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    /// use std::time::Duration;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_defer_accept(Duration::from_secs(1u64)).expect("set_defer_accept call failed");
    /// assert_eq!(stream.defer_accept().unwrap(), Some(Duration::from_secs(1u64)));
    /// stream.set_defer_accept(Duration::ZERO).expect("set_defer_accept call failed");
    /// assert_eq!(stream.defer_accept().unwrap(), None);
    /// ```
    #[unstable(feature = "tcp_deferaccept", issue = "119639")]
    #[cfg(target_os = "linux")]
    fn defer_accept(&self) -> io::Result<Option<Duration>>;
}

#[stable(feature = "tcp_quickack", since = "1.89.0")]
impl TcpStreamExt for net::TcpStream {
    #[rustdoc_skip_body]
    fn set_quickack(&self, quickack: bool) -> io::Result<()> {
        self.as_inner().as_inner().set_quickack(quickack)
    }

    #[rustdoc_skip_body]
    fn quickack(&self) -> io::Result<bool> {
        self.as_inner().as_inner().quickack()
    }

    #[cfg(target_os = "linux")]
    fn set_defer_accept(&self, accept: Duration) -> io::Result<()> {
        self.as_inner().as_inner().set_defer_accept(accept)
    }

    #[cfg(target_os = "linux")]
    fn defer_accept(&self) -> io::Result<Option<Duration>> {
        self.as_inner().as_inner().defer_accept()
    }
}
