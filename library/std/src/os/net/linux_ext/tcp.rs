//! Linux and Android-specific tcp extensions to primitives in the [`std::net`] module.
//!
//! [`std::net`]: crate::net

use crate::sealed::Sealed;
use crate::sys_common::AsInner;
use crate::{io, net};

/// Os-specific extensions for [`TcpStream`]
///
/// [`TcpStream`]: net::TcpStream
#[stable(feature = "tcp_quickack", since = "CURRENT_RUSTC_VERSION")]
pub trait TcpStreamExt: Sealed {
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
    /// ```no_run
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
    #[stable(feature = "tcp_quickack", since = "CURRENT_RUSTC_VERSION")]
    fn set_quickack(&self, quickack: bool) -> io::Result<()>;

    /// Gets the value of the `TCP_QUICKACK` option on this socket.
    ///
    /// For more information about this option, see [`TcpStreamExt::set_quickack`].
    ///
    /// # Examples
    ///
    /// ```no_run
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
    #[stable(feature = "tcp_quickack", since = "CURRENT_RUSTC_VERSION")]
    fn quickack(&self) -> io::Result<bool>;

    /// A socket listener will be awakened solely when data arrives.
    ///
    /// The `accept` argument set the delay in seconds until the
    /// data is available to read, reducing the number of short lived
    /// connections without data to process.
    /// Contrary to other platforms `SO_ACCEPTFILTER` feature equivalent, there is
    /// no necessity to set it after the `listen` call.
    ///
    /// See [`man 7 tcp`](https://man7.org/linux/man-pages/man7/tcp.7.html)
    ///
    /// # Examples
    ///
    /// ```no run
    /// #![feature(tcp_deferaccept)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_deferaccept(1).expect("set_deferaccept call failed");
    /// ```
    #[unstable(feature = "tcp_deferaccept", issue = "119639")]
    #[cfg(target_os = "linux")]
    fn set_deferaccept(&self, accept: u32) -> io::Result<()>;

    /// Gets the accept delay value (in seconds) of the `TCP_DEFER_ACCEPT` option.
    ///
    /// For more information about this option, see [`TcpStreamExt::set_deferaccept`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_deferaccept)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_deferaccept(1).expect("set_deferaccept call failed");
    /// assert_eq!(stream.deferaccept().unwrap_or(0), 1);
    /// ```
    #[unstable(feature = "tcp_deferaccept", issue = "119639")]
    #[cfg(target_os = "linux")]
    fn deferaccept(&self) -> io::Result<u32>;
}

#[stable(feature = "tcp_quickack", since = "CURRENT_RUSTC_VERSION")]
impl Sealed for net::TcpStream {}

#[stable(feature = "tcp_quickack", since = "CURRENT_RUSTC_VERSION")]
impl TcpStreamExt for net::TcpStream {
    fn set_quickack(&self, quickack: bool) -> io::Result<()> {
        self.as_inner().as_inner().set_quickack(quickack)
    }

    fn quickack(&self) -> io::Result<bool> {
        self.as_inner().as_inner().quickack()
    }

    #[cfg(target_os = "linux")]
    fn set_deferaccept(&self, accept: u32) -> io::Result<()> {
        self.as_inner().as_inner().set_deferaccept(accept)
    }

    #[cfg(target_os = "linux")]
    fn deferaccept(&self) -> io::Result<u32> {
        self.as_inner().as_inner().deferaccept()
    }
}
