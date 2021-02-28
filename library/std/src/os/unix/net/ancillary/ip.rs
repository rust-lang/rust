use super::inner::{get_data_from_cmsghdr, Ancillary, AncillaryDataIter, AncillaryError, Messages};
use crate::{
    io::{self, IoSlice, IoSliceMut},
    mem::{size_of, zeroed},
    net::SocketAddr,
    slice::from_ref,
    sys::net::Socket,
    sys_common::{net::sockaddr_to_addr, IntoInner},
};

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(all(doc, not(target_os = "linux"), not(target_os = "android")))]
#[allow(non_camel_case_types)]
mod libc {
    pub use libc::c_int;
    pub struct cmsghdr;
}

impl Socket {
    pub(crate) fn recv_vectored_with_ancillary_from_udp(
        &self,
        bufs: &mut [IoSliceMut<'_>],
        ancillary: &mut IpAncillary<'_>,
    ) -> io::Result<(usize, bool, io::Result<SocketAddr>)> {
        unsafe {
            let mut msg_name: libc::sockaddr_storage = zeroed();
            let (count, truncated, msg_namelen) = self.recv_vectored_with_ancillary_from(
                &mut msg_name as *mut _ as *mut libc::c_void,
                size_of::<libc::sockaddr_storage>() as libc::socklen_t,
                bufs,
                &mut ancillary.inner,
            )?;
            let addr = sockaddr_to_addr(&msg_name, msg_namelen as usize);

            Ok((count, truncated, addr))
        }
    }

    pub(crate) fn send_vectored_with_ancillary_to_udp(
        &self,
        dst: Option<&SocketAddr>,
        bufs: &[IoSlice<'_>],
        ancillary: &mut IpAncillary<'_>,
    ) -> io::Result<usize> {
        unsafe {
            let (msg_name, msg_namelen) =
                if let Some(dst) = dst { dst.into_inner() } else { (zeroed(), 0) };
            self.send_vectored_with_ancillary_to(
                msg_name as *mut libc::c_void,
                msg_namelen,
                bufs,
                &mut ancillary.inner,
            )
        }
    }
}

/// This enum represent one control message of variable type for `IPPROTO_IP`.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[non_exhaustive]
pub enum IpAncillaryData {
    Ttl(u8),
}

impl IpAncillaryData {
    /// Convert the a `libc::c_int` to `u8`
    ///
    /// # Safety
    ///
    /// `data` must contain at least one `libc::c_int`.
    unsafe fn as_u8(data: &[u8]) -> u8 {
        let mut ancillary_data_iter = AncillaryDataIter::<libc::c_int>::new(data);
        if let Some(u) = ancillary_data_iter.next() { u as u8 } else { 0 }
    }

    /// Create a `AncillaryData::Ttl` variant.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message and the control message must be type of
    /// `IPPROTO_IP` and level of `IP_TTL`.
    unsafe fn as_ttl(data: &[u8]) -> Self {
        let ttl = IpAncillaryData::as_u8(data);
        IpAncillaryData::Ttl(ttl)
    }

    unsafe fn try_from(cmsg: &libc::cmsghdr) -> Result<Self, AncillaryError> {
        let data = get_data_from_cmsghdr(cmsg);

        match (*cmsg).cmsg_level {
            libc::IPPROTO_IP => match (*cmsg).cmsg_type {
                libc::IP_TTL => Ok(IpAncillaryData::as_ttl(data)),
                cmsg_type => {
                    Err(AncillaryError::Unknown { cmsg_level: libc::IPPROTO_IP, cmsg_type })
                }
            },
            cmsg_level => Err(AncillaryError::Unknown { cmsg_level, cmsg_type: (*cmsg).cmsg_type }),
        }
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for Messages<'a, IpAncillaryData> {
    type Item = Result<IpAncillaryData, AncillaryError>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let cmsg = self.next_cmsghdr()?;
            let ancillary_result = IpAncillaryData::try_from(cmsg);
            Some(ancillary_result)
        }
    }
}

/// A net IP Ancillary data struct.
///
/// # Example
/// ```no_run
/// #![feature(unix_socket_ancillary_data)]
/// use std::net::UdpSocket;
/// use std::os::unix::net::{IpAncillary, IpAncillaryData};
/// use std::io::IoSliceMut;
///
/// fn main() -> std::io::Result<()> {
///     let sock = UdpSocket::bind("127.0.0.1:34254")?;
///
///     let mut ancillary_buffer = [0; 128];
///     let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
///
///     let mut buf = [1; 8];
///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
///
///     for ancillary_result in ancillary.messages() {
///         if let IpAncillaryData::Ttl(ttl) = ancillary_result.unwrap() {
///             println!("receive packet with TTL: {}", ttl);
///         }
///     }
///     Ok(())
/// }
/// ```
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[derive(Debug)]
pub struct IpAncillary<'a> {
    pub(crate) inner: Ancillary<'a>,
}

impl<'a> IpAncillary<'a> {
    /// Create an ancillary data with the given buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(unused_mut)]
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::IpAncillary;
    /// let mut ancillary_buffer = [0; 128];
    /// let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        IpAncillary { inner: Ancillary::new(buffer) }
    }

    /// Returns the capacity of the buffer.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Returns `true` if the ancillary data is empty.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of used bytes.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns the iterator of the control messages.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn messages(&self) -> Messages<'_, IpAncillaryData> {
        self.inner.messages()
    }

    /// Add TTL to the ancillary data.
    ///
    /// The function returns `true` if there was enough space in the buffer.
    /// If there was not enough space then no file descriptors was appended.
    /// Technically, that means this operation adds a control message with the level `IPPROTO_IP`
    /// and type `IP_TTL`.
    ///
    /// # Example
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::io::IoSlice;
    /// use std::net::UdpSocket;
    /// use std::os::unix::net::IpAncillary;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UdpSocket::bind("127.0.0.1:34254")?;
    ///     sock.connect("127.0.0.1:41203")?;
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
    ///     sock.send_vectored_with_ancillary(bufs, &mut ancillary)
    ///         .expect("send_vectored_with_ancillary function failed");
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn add_ttl(&mut self, ttl: u8) -> bool {
        let ttl: libc::c_int = ttl as libc::c_int;
        self.inner.add_to_ancillary_data(from_ref(&ttl), libc::IPPROTO_IP, libc::IP_TTL)
    }

    /// Clears the ancillary data, removing all values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::net::UdpSocket;
    /// use std::os::unix::net::{IpAncillary, IpAncillaryData};
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UdpSocket::bind("127.0.0.1:34254")?;
    ///
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = IpAncillary::new(&mut ancillary_buffer[..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
    ///
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     for ancillary_result in ancillary.messages() {
    ///         if let IpAncillaryData::Ttl(ttl) = ancillary_result.unwrap() {
    ///             println!("receive packet with TTL: {}", ttl);
    ///         }
    ///     }
    ///
    ///     ancillary.clear();
    ///
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     for ancillary_result in ancillary.messages() {
    ///         if let IpAncillaryData::Ttl(ttl) = ancillary_result.unwrap() {
    ///             println!("receive packet with TTL: {}", ttl);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}
