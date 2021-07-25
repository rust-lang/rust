use super::{
    super::{sockaddr_un, SocketAddr},
    inner::{get_data_from_cmsghdr, Ancillary, AncillaryDataIter, AncillaryError, Messages},
};
use crate::{
    io::{self, IoSlice, IoSliceMut},
    mem::{size_of, zeroed},
    os::unix::io::RawFd,
    path::Path,
    sys::net::Socket,
};

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(all(doc, not(target_os = "linux"), not(target_os = "android")))]
#[allow(non_camel_case_types)]
mod libc {
    pub struct ucred;
    pub struct cmsghdr;
    pub type pid_t = i32;
    pub type gid_t = u32;
    pub type uid_t = u32;
}

impl Socket {
    pub(crate) fn recv_vectored_with_ancillary_from_unix(
        &self,
        bufs: &mut [IoSliceMut<'_>],
        ancillary: &mut UnixAncillary<'_>,
    ) -> io::Result<(usize, bool, io::Result<SocketAddr>)> {
        unsafe {
            let mut msg_name: libc::sockaddr_un = zeroed();
            let (count, truncated, msg_namelen) = self.recv_vectored_with_ancillary_from(
                &mut msg_name as *mut _ as *mut libc::c_void,
                size_of::<libc::sockaddr_un>() as libc::socklen_t,
                bufs,
                &mut ancillary.inner,
            )?;
            let addr = SocketAddr::from_parts(msg_name, msg_namelen);

            Ok((count, truncated, addr))
        }
    }

    pub(crate) fn send_vectored_with_ancillary_to_unix(
        &self,
        path: Option<&Path>,
        bufs: &[IoSlice<'_>],
        ancillary: &mut UnixAncillary<'_>,
    ) -> io::Result<usize> {
        unsafe {
            let (mut msg_name, msg_namelen) =
                if let Some(path) = path { sockaddr_un(path)? } else { (zeroed(), 0) };
            self.send_vectored_with_ancillary_to(
                &mut msg_name as *mut _ as *mut libc::c_void,
                msg_namelen,
                bufs,
                &mut ancillary.inner,
            )
        }
    }
}

/// Unix credential.
#[doc(cfg(any(target_os = "android", target_os = "linux",)))]
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[derive(Clone)]
pub struct SocketCred(libc::ucred);

#[doc(cfg(any(target_os = "android", target_os = "linux",)))]
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
impl SocketCred {
    /// Create a Unix credential struct.
    ///
    /// PID, UID and GID is set to 0.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn new() -> SocketCred {
        SocketCred(libc::ucred { pid: 0, uid: 0, gid: 0 })
    }

    /// Set the PID.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn set_pid(&mut self, pid: libc::pid_t) {
        self.0.pid = pid;
    }

    /// Get the current PID.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn get_pid(&self) -> libc::pid_t {
        self.0.pid
    }

    /// Set the UID.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn set_uid(&mut self, uid: libc::uid_t) {
        self.0.uid = uid;
    }

    /// Get the current UID.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn get_uid(&self) -> libc::uid_t {
        self.0.uid
    }

    /// Set the GID.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn set_gid(&mut self, gid: libc::gid_t) {
        self.0.gid = gid;
    }

    /// Get the current GID.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn get_gid(&self) -> libc::gid_t {
        self.0.gid
    }
}

/// This control message contains file descriptors.
///
/// The level is equal to `SOL_SOCKET` and the type is equal to `SCM_RIGHTS`.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct ScmRights<'a>(AncillaryDataIter<'a, RawFd>);

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for ScmRights<'a> {
    type Item = RawFd;

    fn next(&mut self) -> Option<RawFd> {
        self.0.next()
    }
}

/// This control message contains unix credentials.
///
/// The level is equal to `SOL_SOCKET` and the type is equal to `SCM_CREDENTIALS` or `SCM_CREDS`.
#[doc(cfg(any(target_os = "android", target_os = "linux",)))]
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct ScmCredentials<'a>(AncillaryDataIter<'a, libc::ucred>);

#[doc(cfg(any(target_os = "android", target_os = "linux",)))]
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for ScmCredentials<'a> {
    type Item = SocketCred;

    fn next(&mut self) -> Option<SocketCred> {
        Some(SocketCred(self.0.next()?))
    }
}

/// This enum represent one control message of variable type for `SOL_SOCKET`.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[non_exhaustive]
pub enum UnixAncillaryData<'a> {
    ScmRights(ScmRights<'a>),
    #[doc(cfg(any(target_os = "android", target_os = "linux",)))]
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    ScmCredentials(ScmCredentials<'a>),
}

impl<'a> UnixAncillaryData<'a> {
    /// Create a `UnixAncillaryData::ScmRights` variant.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message and the control message must be type of
    /// `SOL_SOCKET` and level of `SCM_RIGHTS`.
    unsafe fn as_rights(data: &'a [u8]) -> Self {
        let ancillary_data_iter = AncillaryDataIter::new(data);
        let scm_rights = ScmRights(ancillary_data_iter);
        UnixAncillaryData::ScmRights(scm_rights)
    }

    /// Create a `UnixAncillaryData::ScmCredentials` variant.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message and the control message must be type of
    /// `SOL_SOCKET` and level of `SCM_CREDENTIALS` or `SCM_CREDENTIALS`.
    #[cfg(any(target_os = "android", target_os = "linux",))]
    unsafe fn as_credentials(data: &'a [u8]) -> Self {
        let ancillary_data_iter = AncillaryDataIter::new(data);
        let scm_credentials = ScmCredentials(ancillary_data_iter);
        UnixAncillaryData::ScmCredentials(scm_credentials)
    }

    unsafe fn try_from(cmsg: &'a libc::cmsghdr) -> Result<Self, AncillaryError> {
        let data = get_data_from_cmsghdr(cmsg);

        match (*cmsg).cmsg_level {
            libc::SOL_SOCKET => match (*cmsg).cmsg_type {
                libc::SCM_RIGHTS => Ok(UnixAncillaryData::as_rights(data)),
                #[cfg(any(target_os = "android", target_os = "linux",))]
                libc::SCM_CREDENTIALS => Ok(UnixAncillaryData::as_credentials(data)),
                cmsg_type => {
                    Err(AncillaryError::Unknown { cmsg_level: libc::SOL_SOCKET, cmsg_type })
                }
            },
            cmsg_level => Err(AncillaryError::Unknown { cmsg_level, cmsg_type: (*cmsg).cmsg_type }),
        }
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for Messages<'a, UnixAncillaryData<'a>> {
    type Item = Result<UnixAncillaryData<'a>, AncillaryError>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let cmsg = self.next_cmsghdr()?;
            let ancillary_result = UnixAncillaryData::try_from(cmsg);
            Some(ancillary_result)
        }
    }
}

/// A Unix socket Ancillary data struct.
///
/// # Example
/// ```no_run
/// #![feature(unix_socket_ancillary_data)]
/// use std::os::unix::net::{UnixStream, UnixAncillary, UnixAncillaryData};
/// use std::io::IoSliceMut;
///
/// fn main() -> std::io::Result<()> {
///     let sock = UnixStream::connect("/tmp/sock")?;
///
///     let mut fds = [0; 8];
///     let mut ancillary_buffer = [0; 128];
///     let mut ancillary = UnixAncillary::new(&mut ancillary_buffer[..]);
///
///     let mut buf = [1; 8];
///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
///
///     for ancillary_result in ancillary.messages() {
///         if let UnixAncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
///             for fd in scm_rights {
///                 println!("receive file descriptor: {}", fd);
///             }
///         }
///     }
///     Ok(())
/// }
/// ```
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[derive(Debug)]
pub struct UnixAncillary<'a> {
    pub(crate) inner: Ancillary<'a>,
}

impl<'a> UnixAncillary<'a> {
    /// Create an ancillary data with the given buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(unused_mut)]
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::UnixAncillary;
    /// let mut ancillary_buffer = [0; 128];
    /// let mut ancillary = UnixAncillary::new(&mut ancillary_buffer[..]);
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        UnixAncillary { inner: Ancillary::new(buffer) }
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
    pub fn messages(&self) -> Messages<'_, UnixAncillaryData<'_>> {
        self.inner.messages()
    }

    /// Is `true` if during a recv operation the ancillary was truncated.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, UnixAncillary};
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = UnixAncillary::new(&mut ancillary_buffer[..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///
    ///     println!("Is truncated: {}", ancillary.truncated());
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn truncated(&self) -> bool {
        self.inner.truncated()
    }

    /// Add file descriptors to the ancillary data.
    ///
    /// The function returns `true` if there was enough space in the buffer.
    /// If there was not enough space then no file descriptors was appended.
    /// This adds a control message with the level `SOL_SOCKET` and type `SCM_RIGHTS`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, UnixAncillary};
    /// use std::os::unix::io::AsRawFd;
    /// use std::io::IoSlice;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = UnixAncillary::new(&mut ancillary_buffer[..]);
    ///     ancillary.add_fds(&[sock.as_raw_fd()][..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSlice::new(&mut buf[..])][..];
    ///     sock.send_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn add_fds(&mut self, fds: &[RawFd]) -> bool {
        self.inner.add_to_ancillary_data(fds, libc::SOL_SOCKET, libc::SCM_RIGHTS)
    }

    /// Add credentials to the ancillary data.
    ///
    /// The function returns `true` if there was enough space in the buffer.
    /// If there was not enough space then no credentials was appended.
    /// This adds a control message with the level `SOL_SOCKET` and type `SCM_CREDENTIALS`
    /// or `SCM_CREDS`.
    ///
    #[doc(cfg(any(target_os = "android", target_os = "linux",)))]
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn add_creds(&mut self, creds: &[SocketCred]) -> bool {
        self.inner.add_to_ancillary_data(creds, libc::SOL_SOCKET, libc::SCM_CREDENTIALS)
    }

    /// Clears the ancillary data, removing all values.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, UnixAncillary, UnixAncillaryData};
    /// use std::io::IoSliceMut;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut fds1 = [0; 8];
    ///     let mut fds2 = [0; 8];
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = UnixAncillary::new(&mut ancillary_buffer[..]);
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSliceMut::new(&mut buf[..])][..];
    ///
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     for ancillary_result in ancillary.messages() {
    ///         if let UnixAncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
    ///             for fd in scm_rights {
    ///                 println!("receive file descriptor: {}", fd);
    ///             }
    ///         }
    ///     }
    ///
    ///     ancillary.clear();
    ///
    ///     sock.recv_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     for ancillary_result in ancillary.messages() {
    ///         if let UnixAncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
    ///             for fd in scm_rights {
    ///                 println!("receive file descriptor: {}", fd);
    ///             }
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
