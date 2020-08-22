use crate::io::{self, IoSliceMut};
use crate::mem;
use crate::os::unix::io::RawFd;
use crate::path::Path;
use crate::ptr::null_mut;
use crate::slice::from_raw_parts;
use crate::sys::unix::ext::net::addr::{sockaddr_un, SocketAddr};
use crate::sys::unix::net::{add_to_ancillary_data, AncillaryDataIter, Socket};

pub(super) fn recv_vectored_with_ancillary_from(
    socket: &Socket,
    bufs: &mut [IoSliceMut<'_>],
    ancillary: &mut SocketAncillary<'_>,
) -> io::Result<(usize, bool, io::Result<SocketAddr>)> {
    unsafe {
        let mut msg_name: libc::sockaddr_un = mem::zeroed();

        let mut msg = libc::msghdr {
            msg_name: &mut msg_name as *mut _ as *mut _,
            msg_namelen: mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
            msg_iov: bufs.as_mut_ptr().cast(),
            msg_iovlen: bufs.len(),
            msg_control: ancillary.buffer.as_mut_ptr().cast(),
            msg_controllen: ancillary.buffer.len(),
            msg_flags: 0,
        };

        let count = socket.recv_msg(&mut msg)?;

        ancillary.length = msg.msg_controllen;
        ancillary.truncated = msg.msg_flags & libc::MSG_CTRUNC == libc::MSG_CTRUNC;

        let truncated = msg.msg_flags & libc::MSG_TRUNC == libc::MSG_TRUNC;
        let addr = SocketAddr::from_parts(msg_name, msg.msg_namelen);

        Ok((count, truncated, addr))
    }
}

pub(super) fn send_vectored_with_ancillary_to(
    socket: &Socket,
    path: Option<&Path>,
    bufs: &mut [IoSliceMut<'_>],
    ancillary: &mut SocketAncillary<'_>,
) -> io::Result<usize> {
    unsafe {
        let (mut msg_name, msg_namelen) =
            if let Some(path) = path { sockaddr_un(path)? } else { (mem::zeroed(), 0) };

        let mut msg = libc::msghdr {
            msg_name: &mut msg_name as *mut _ as *mut _,
            msg_namelen,
            msg_iov: bufs.as_mut_ptr().cast(),
            msg_iovlen: bufs.len(),
            msg_control: ancillary.buffer.as_mut_ptr().cast(),
            msg_controllen: ancillary.length,
            msg_flags: 0,
        };

        ancillary.truncated = false;

        socket.send_msg(&mut msg)
    }
}

#[cfg(any(
    target_os = "haiku",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
pub struct ScmRights<'a>(AncillaryDataIter<'a, RawFd>);

#[cfg(any(
    target_os = "haiku",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
impl<'a> Iterator for ScmRights<'a> {
    type Item = RawFd;

    fn next(&mut self) -> Option<RawFd> {
        self.0.next()
    }
}

#[cfg(any(
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
pub struct ScmCredentials<'a>(AncillaryDataIter<'a, libc::ucred>);

#[cfg(any(
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
impl<'a> Iterator for ScmCredentials<'a> {
    type Item = libc::ucred;

    fn next(&mut self) -> Option<libc::ucred> {
        self.0.next()
    }
}

#[cfg(any(
    target_os = "haiku",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
pub enum AncillaryData<'a> {
    ScmRights(ScmRights<'a>),
    #[cfg(any(
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_env = "uclibc",
    ))]
    ScmCredentials(ScmCredentials<'a>),
}

impl<'a> AncillaryData<'a> {
    #[cfg(any(
        target_os = "haiku",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "linux",
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_env = "uclibc",
    ))]
    fn as_rights(data: &'a [u8]) -> Self {
        let ancillary_data_iter = AncillaryDataIter::new(data);
        let scm_rights = ScmRights(ancillary_data_iter);
        AncillaryData::ScmRights(scm_rights)
    }

    #[cfg(any(
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_env = "uclibc",
    ))]
    fn as_credentials(data: &'a [u8]) -> Self {
        let ancillary_data_iter = AncillaryDataIter::new(data);
        let scm_credentials = ScmCredentials(ancillary_data_iter);
        AncillaryData::ScmCredentials(scm_credentials)
    }
}

#[cfg(any(
    target_os = "haiku",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
impl<'a> AncillaryData<'a> {
    fn from(cmsg: &'a libc::cmsghdr) -> Self {
        unsafe {
            let cmsg_len_zero = libc::CMSG_LEN(0) as usize;
            let data_len = (*cmsg).cmsg_len - cmsg_len_zero;
            let data = libc::CMSG_DATA(cmsg).cast();
            let data = from_raw_parts(data, data_len);

            if (*cmsg).cmsg_level == libc::SOL_SOCKET {
                match (*cmsg).cmsg_type {
                    libc::SCM_RIGHTS => AncillaryData::as_rights(data),
                    #[cfg(any(
                        target_os = "linux",
                        target_os = "android",
                        target_os = "emscripten",
                        target_os = "fuchsia",
                        target_env = "uclibc",
                    ))]
                    libc::SCM_CREDENTIALS => AncillaryData::as_credentials(data),
                    #[cfg(any(
                        target_os = "netbsd",
                        target_os = "openbsd",
                        target_os = "freebsd",
                        target_os = "dragonfly",
                        target_os = "macos",
                        target_os = "ios",
                    ))]
                    libc::SCM_CREDS => AncillaryData::as_credentials(data),
                    _ => panic!("Unknown cmsg type"),
                }
            } else {
                panic!("Unknown cmsg level");
            }
        }
    }
}

#[cfg(any(
    target_os = "haiku",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
pub struct Messages<'a> {
    buffer: &'a [u8],
    current: Option<&'a libc::cmsghdr>,
}

#[cfg(any(
    target_os = "haiku",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "linux",
    target_os = "android",
    target_os = "emscripten",
    target_os = "fuchsia",
    target_env = "uclibc",
))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
impl<'a> Iterator for Messages<'a> {
    type Item = AncillaryData<'a>;

    fn next(&mut self) -> Option<AncillaryData<'a>> {
        unsafe {
            let msg = libc::msghdr {
                msg_name: null_mut(),
                msg_namelen: 0,
                msg_iov: null_mut(),
                msg_iovlen: 0,
                msg_control: self.buffer.as_ptr() as *mut _,
                msg_controllen: self.buffer.len(),
                msg_flags: 0,
            };

            let cmsg = if let Some(current) = self.current {
                libc::CMSG_NXTHDR(&msg, current)
            } else {
                libc::CMSG_FIRSTHDR(&msg)
            };

            let cmsg = cmsg.as_ref()?;
            self.current = Some(cmsg);
            let ancillary_data = AncillaryData::from(cmsg);
            Some(ancillary_data)
        }
    }
}

/// A Unix socket Ancillary data struct.
///
/// # Example
/// ```no_run
/// #![feature(unix_socket_ancillary_data)]
/// use std::os::unix::net::{UnixStream, SocketAncillary, AncillaryData};
/// use std::io::IoSliceMut;
///
/// fn main() -> std::io::Result<()> {
///     let sock = UnixStream::connect("/tmp/sock")?;
///
///     let mut fds = [0; 8];
///     let mut ancillary_buffer = [0; 128];
///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
///
///     let mut buf = [1; 8];
///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
///
///     for ancillary_data in ancillary.messages() {
///         if let AncillaryData::ScmRights(scm_rights) = ancillary_data {
///             for fd in scm_rights {
///                 println!("receive file descriptor: {}", fd);
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
#[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
#[derive(Debug)]
pub struct SocketAncillary<'a> {
    buffer: &'a mut [u8],
    length: usize,
    truncated: bool,
}

impl<'a> SocketAncillary<'a> {
    /// Create an ancillary data with the given buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(unused_mut)]
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::SocketAncillary;
    /// let mut ancillary_buffer = [0; 128];
    /// let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        SocketAncillary { buffer, length: 0, truncated: false }
    }

    /// Returns the capacity of the buffer.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the number of used bytes.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn len(&self) -> usize {
        self.length
    }

    #[cfg(any(
        target_os = "haiku",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "linux",
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_env = "uclibc",
    ))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn messages(&'a self) -> Messages<'a> {
        Messages { buffer: &self.buffer[..self.length], current: None }
    }

    /// Is `true` if during a recv operation the ancillary was truncated.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, SocketAncillary};
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///
    ///     println!("Is truncated: {}", ancillary.truncated());
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Add file descriptors to the ancillary data.
    ///
    /// The function returns `true` if there was enough space in the buffer.
    /// If there was not enough space then no file descriptors was appended.
    /// Technically, that means this operation adds a control message with the level `SOL_SOCKET`
    /// and type `SCM_RIGHTS`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, SocketAncillary};
    /// use std::os::unix::io::AsRawFd;
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ///     ancillary.add_fds(&[sock.as_raw_fd()][..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
    ///     sock.send_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     Ok(())
    /// }
    /// ```
    #[cfg(any(
        target_os = "haiku",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "linux",
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_env = "uclibc",
    ))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn add_fds(&mut self, fds: &[RawFd]) -> bool {
        self.truncated = false;
        add_to_ancillary_data(
            &mut self.buffer,
            &mut self.length,
            fds,
            libc::SOL_SOCKET,
            libc::SCM_RIGHTS,
        )
    }

    /// Add credentials to the ancillary data.
    ///
    /// The function returns `true` if there was enough space in the buffer.
    /// If there was not enough space then no credentials was appended.
    /// Technically, that means this operation adds a control message with the level `SOL_SOCKET`
    /// and type `SCM_CREDENTIALS`.
    ///
    #[cfg(any(
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_env = "uclibc",
    ))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn add_creds(&mut self, creds: &[libc::ucred]) -> bool {
        self.truncated = false;
        add_to_ancillary_data(
            &mut self.buffer,
            &mut self.length,
            creds,
            libc::SOL_SOCKET,
            #[cfg(any(
                target_os = "linux",
                target_os = "android",
                target_os = "emscripten",
                target_os = "fuchsia",
                target_env = "uclibc",
            ))]
            libc::SCM_CREDENTIALS,
            #[cfg(any(
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "macos",
                target_os = "ios",
            ))]
            libc::SCM_CREDS,
        )
    }

    /// Clears the ancillary data, removing all values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, SocketAncillary, AncillaryData};
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut fds1 = [0; 8];
    ///     let mut fds2 = [0; 8];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
    ///
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     for ancillary_data in ancillary.messages() {
    ///         if let AncillaryData::ScmRights(scm_rights) = ancillary_data {
    ///             for fd in scm_rights {
    ///                 println!("receive file descriptor: {}", fd);
    ///             }
    ///         }
    ///     }
    ///
    ///     ancillary.clear();
    ///
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     for ancillary_data in ancillary.messages() {
    ///         if let AncillaryData::ScmRights(scm_rights) = ancillary_data {
    ///             for fd in scm_rights {
    ///                 println!("receive file descriptor: {}", fd);
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "none")]
    pub fn clear(&mut self) {
        self.length = 0;
        self.truncated = false;
    }
}
