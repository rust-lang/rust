//! Linux and Android-specific tcp extensions to primitives in the [`std::net`] module.
//!
//! [`std::net`]: crate::net

use crate::io;
use crate::net;
use crate::sealed::Sealed;
use crate::sys_common::AsInner;

/// Os-specific extensions for [`TcpStream`]
///
/// [`TcpStream`]: net::TcpStream
#[unstable(feature = "tcp_quickack", issue = "96256")]
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
    /// #![feature(tcp_quickack)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_quickack(true).expect("set_quickack call failed");
    /// ```
    #[unstable(feature = "tcp_quickack", issue = "96256")]
    fn set_quickack(&self, quickack: bool) -> io::Result<()>;

    /// Gets the value of the `TCP_QUICKACK` option on this socket.
    ///
    /// For more information about this option, see [`TcpStreamExt::set_quickack`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_quickack)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_quickack(true).expect("set_quickack call failed");
    /// assert_eq!(stream.quickack().unwrap_or(false), true);
    /// ```
    #[unstable(feature = "tcp_quickack", issue = "96256")]
    fn quickack(&self) -> io::Result<bool>;
}

#[unstable(feature = "tcp_quickack", issue = "96256")]
impl Sealed for net::TcpStream {}

#[unstable(feature = "tcp_quickack", issue = "96256")]
impl TcpStreamExt for net::TcpStream {
    fn set_quickack(&self, quickack: bool) -> io::Result<()> {
        self.as_inner().as_inner().set_quickack(quickack)
    }

    fn quickack(&self) -> io::Result<bool> {
        self.as_inner().as_inner().quickack()
    }
}
