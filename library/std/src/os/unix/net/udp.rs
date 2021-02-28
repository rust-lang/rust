use super::IpAncillary;
use crate::{
    io::{self, IoSlice, IoSliceMut},
    net::{SocketAddr, ToSocketAddrs, UdpSocket},
    sys_common::AsInner,
};

impl UdpSocket {
    /// Sets the value of the `IP_RECVTTL` option for this socket.
    ///
    /// If enabled, received packets will come with ancillary data ([`IpAncillary`]) providing
    /// the time-to-live (TTL) value of the packet.
    ///
    /// # Examples
    ///
    ///```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::net::UdpSocket;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    ///     socket.set_recvttl(true).expect("set_recvttl function failed");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn set_recvttl(&self, recvttl: bool) -> io::Result<()> {
        self.as_inner().set_recvttl(recvttl)
    }

    /// Get the current value of the socket for receiving TTL in [`IpAncillary`].
    /// This value can be change by [`set_recvttl`].
    ///
    /// Get the socket option `IP_RECVTTL`.
    ///
    /// [`set_recvttl`]: UdpSocket::set_recvttl
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn recvttl(&self) -> io::Result<bool> {
        self.as_inner().recvttl()
    }

    /// Receives data and ancillary data from socket.
    ///
    /// On success, returns the number of bytes read.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::io::IoSliceMut;
    /// use std::net::UdpSocket;
    /// use std::os::unix::net::{IpAncillary, IpAncillaryData};
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    ///     socket.set_recvttl(true).expect("set_recvttl function failed");
    ///     let mut buf1 = [1; 8];
    ///     let mut buf2 = [2; 16];
    ///     let mut buf3 = [3; 8];
    ///     let mut bufs = &mut [
    ///         IoSliceMut::new(&mut buf1),
    ///         IoSliceMut::new(&mut buf2),
    ///         IoSliceMut::new(&mut buf3),
    ///     ][..];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
    ///     let (size, truncated) = socket.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     println!("received {} truncated {}", size, truncated);
    ///     for ancillary_result in ancillary.messages() {
    ///         if let IpAncillaryData::Ttl(ttl) = ancillary_result.unwrap() {
    ///             println!("UDP packet has a TTL of {}", ttl);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn recv_vectored_with_ancillary(
        &self,
        bufs: &mut [IoSliceMut<'_>],
        ancillary: &mut IpAncillary<'_>,
    ) -> io::Result<(usize, bool)> {
        self.as_inner().recv_vectored_with_ancillary(bufs, ancillary)
    }

    /// Receives data and ancillary data from socket.
    ///
    /// On success, returns the number of bytes read, if the data was truncated and the address
    /// from whence the msg came.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::io::IoSliceMut;
    /// use std::net::UdpSocket;
    /// use std::os::unix::net::{IpAncillary, IpAncillaryData};
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    ///     socket.set_recvttl(true).expect("set_recvttl function failed");
    ///     let mut buf1 = [1; 8];
    ///     let mut buf2 = [2; 16];
    ///     let mut buf3 = [3; 8];
    ///     let mut bufs = &mut [
    ///         IoSliceMut::new(&mut buf1),
    ///         IoSliceMut::new(&mut buf2),
    ///         IoSliceMut::new(&mut buf3),
    ///     ][..];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
    ///     let (size, truncated, addr) =
    ///         socket.recv_vectored_with_ancillary_from(bufs, &mut ancillary)?;
    ///     println!("received {} truncated {} from {}", size, truncated, addr);
    ///     for ancillary_result in ancillary.messages() {
    ///         if let IpAncillaryData::Ttl(ttl) = ancillary_result.unwrap() {
    ///             println!("UDP packet has a TTL of {}", ttl);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn recv_vectored_with_ancillary_from(
        &self,
        bufs: &mut [IoSliceMut<'_>],
        ancillary: &mut IpAncillary<'_>,
    ) -> io::Result<(usize, bool, SocketAddr)> {
        self.as_inner().recv_vectored_with_ancillary_from(bufs, ancillary)
    }

    /// Sends data and ancillary data on the socket.
    ///
    /// On success, returns the number of bytes written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::io::IoSlice;
    /// use std::net::UdpSocket;
    /// use std::os::unix::net::IpAncillary;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    ///     socket.connect("127.0.0.1:41203").expect("couldn't connect to address");
    ///     let buf1 = [1; 8];
    ///     let buf2 = [2; 16];
    ///     let buf3 = [3; 8];
    ///     let bufs = &[
    ///         IoSlice::new(&buf1),
    ///         IoSlice::new(&buf2),
    ///         IoSlice::new(&buf3),
    ///     ][..];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
    ///     ancillary.add_ttl(20);
    ///     socket.send_vectored_with_ancillary(bufs, &mut ancillary)
    ///         .expect("send_vectored_with_ancillary function failed");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn send_vectored_with_ancillary(
        &self,
        bufs: &[IoSlice<'_>],
        ancillary: &mut IpAncillary<'_>,
    ) -> io::Result<usize> {
        self.as_inner().send_vectored_with_ancillary(bufs, ancillary)
    }

    /// Sends data and ancillary data on the socket to the specified address.
    ///
    /// On success, returns the number of bytes written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::io::IoSlice;
    /// use std::net::UdpSocket;
    /// use std::os::unix::net::IpAncillary;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let socket = UdpSocket::bind("127.0.0.1:34254").expect("couldn't bind to address");
    ///     let buf1 = [1; 8];
    ///     let buf2 = [2; 16];
    ///     let buf3 = [3; 8];
    ///     let bufs = &[
    ///         IoSlice::new(&buf1),
    ///         IoSlice::new(&buf2),
    ///         IoSlice::new(&buf3),
    ///     ][..];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
    ///     ancillary.add_ttl(20);
    ///     socket.send_vectored_with_ancillary_to(bufs, &mut ancillary, "127.0.0.1:4242")
    ///         .expect("send_vectored_with_ancillary function failed");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn send_vectored_with_ancillary_to<A: ToSocketAddrs>(
        &self,
        bufs: &[IoSlice<'_>],
        ancillary: &mut IpAncillary<'_>,
        addr: A,
    ) -> io::Result<usize> {
        match addr.to_socket_addrs()?.next() {
            Some(addr) => self.as_inner().send_vectored_with_ancillary_to(bufs, ancillary, &addr),
            None => {
                Err(io::Error::new(io::ErrorKind::InvalidInput, "no addresses to send data to"))
            }
        }
    }
}
