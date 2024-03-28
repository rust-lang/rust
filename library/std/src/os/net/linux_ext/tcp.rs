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

    /// Set the number of `SYN` packets to send before giving up establishing a connection.
    ///
    /// In case the server does not repond, a `SYN` packet is sent by the client.
    /// This option controls the number of attempts, the default system value
    /// can be seen via the `net.ipv4.tcp_syn_retries` sysctl's OID (usually 5 or 6).
    /// The maximum valid value is 255.
    ///
    /// See [`man 7 tcp`](https://man7.org/linux/man-pages/man7/tcp.7.html)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_syncnt)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_syncnt(3).expect("set_setcnt call failed");
    #[unstable(feature = "tcp_syncnt", issue = "123112")]
    fn set_syncnt(&self, count: u8) -> io::Result<()>;

    /// Get the number of `SYN` packets to send before giving up establishing a connection.
    ///
    /// For more information about this option, see [`TcpStreamExt::set_syncnt`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(tcp_syncnt)]
    /// use std::net::TcpStream;
    /// use std::os::linux::net::TcpStreamExt;
    ///
    /// let stream = TcpStream::connect("127.0.0.1:8080")
    ///         .expect("Couldn't connect to the server...");
    /// stream.set_syncnt(3).expect("set_syncnt call failed");
    /// assert_eq!(stream.syncnt().unwrap_or(0), 3);
    /// ```
    #[unstable(feature = "tcp_syncnt", issue = "123112")]
    fn syncnt(&self) -> io::Result<u8>;
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

    #[cfg(target_os = "linux")]
    fn set_deferaccept(&self, accept: u32) -> io::Result<()> {
        self.as_inner().as_inner().set_deferaccept(accept)
    }

    #[cfg(target_os = "linux")]
    fn deferaccept(&self) -> io::Result<u32> {
        self.as_inner().as_inner().deferaccept()
    }

    fn set_syncnt(&self, count: u8) -> io::Result<()> {
        self.as_inner().as_inner().set_syncnt(count)
    }

    fn syncnt(&self) -> io::Result<u8> {
        self.as_inner().as_inner().syncnt()
    }
}
