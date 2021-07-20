use super::{sockaddr_un, SocketAddr};
use crate::convert::TryFrom;
use crate::fmt;
use crate::io::{self, IoSlice, IoSliceMut};
use crate::marker::PhantomData;
use crate::mem::{size_of, zeroed, MaybeUninit};
use crate::os::unix::io::RawFd;
use crate::path::Path;
use crate::ptr::{eq, read_unaligned};
use crate::slice;
use crate::sys::net::Socket;

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(all(doc, not(target_os = "linux"), not(target_os = "android")))]
#[allow(non_camel_case_types)]
mod libc {
    pub use libc::c_int;
    pub struct ucred;
    pub struct cmsghdr;
    pub type pid_t = i32;
    pub type gid_t = u32;
    pub type uid_t = u32;
}

pub(super) fn recv_vectored_with_ancillary_from(
    socket: &Socket,
    bufs: &mut [IoSliceMut<'_>],
    ancillary: &mut SocketAncillary<'_>,
) -> io::Result<(usize, bool, io::Result<SocketAddr>)> {
    unsafe {
        let mut msg_name: libc::sockaddr_un = zeroed();
        let mut msg: libc::msghdr = zeroed();
        msg.msg_name = &mut msg_name as *mut _ as *mut _;
        msg.msg_namelen = size_of::<libc::sockaddr_un>() as libc::socklen_t;
        msg.msg_iov = bufs.as_mut_ptr().cast();
        msg.msg_iovlen = bufs.len() as _;
        msg.msg_controllen = ancillary.buffer.len() as _;
        // macos requires that the control pointer is null when the len is 0.
        if msg.msg_controllen > 0 {
            msg.msg_control = ancillary.buffer.as_mut_ptr().cast();
        }

        let count = socket.recv_msg(&mut msg)?;

        ancillary.length = msg.msg_controllen as usize;
        ancillary.truncated = msg.msg_flags & libc::MSG_CTRUNC == libc::MSG_CTRUNC;

        let truncated = msg.msg_flags & libc::MSG_TRUNC == libc::MSG_TRUNC;
        let addr = SocketAddr::from_parts(msg_name, msg.msg_namelen);

        Ok((count, truncated, addr))
    }
}

pub(super) fn send_vectored_with_ancillary_to(
    socket: &Socket,
    path: Option<&Path>,
    bufs: &[IoSlice<'_>],
    ancillary: &mut SocketAncillary<'_>,
) -> io::Result<usize> {
    unsafe {
        let (mut msg_name, msg_namelen) =
            if let Some(path) = path { sockaddr_un(path)? } else { (zeroed(), 0) };

        let mut msg: libc::msghdr = zeroed();
        msg.msg_name = &mut msg_name as *mut _ as *mut _;
        msg.msg_namelen = msg_namelen;
        msg.msg_iov = bufs.as_ptr() as *mut _;
        msg.msg_iovlen = bufs.len() as _;
        msg.msg_controllen = ancillary.length as _;
        // macos requires that the control pointer is null when the len is 0.
        if msg.msg_controllen > 0 {
            msg.msg_control = ancillary.buffer.as_mut_ptr().cast();
        }

        ancillary.truncated = false;

        socket.send_msg(&mut msg)
    }
}

fn add_to_ancillary_data<T>(
    buffer: &mut [MaybeUninit<u8>],
    length: &mut usize,
    source: &[T],
    cmsg_level: libc::c_int,
    cmsg_type: libc::c_int,
) -> Result<(), AddAncillaryError> {
    let source_len =
        source.len().checked_mul(size_of::<T>()).ok_or_else(|| AddAncillaryError::new())?;
    let source_len = u32::try_from(source_len).map_err(|_| AddAncillaryError::new())?;

    unsafe {
        let additional_space = libc::CMSG_SPACE(source_len) as usize;

        let new_length = if let Some(new_length) = additional_space.checked_add(*length) {
            new_length
        } else {
            return Err(AddAncillaryError::new());
        };

        if new_length > buffer.len() {
            return Err(AddAncillaryError::new());
        }

        buffer[*length..new_length].fill(MaybeUninit::new(0));

        *length = new_length;

        let mut msg: libc::msghdr = zeroed();
        msg.msg_control = buffer.as_mut_ptr().cast();
        msg.msg_controllen = *length as _;

        let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
        let mut previous_cmsg = cmsg;
        while !cmsg.is_null() {
            previous_cmsg = cmsg;
            cmsg = libc::CMSG_NXTHDR(&msg, cmsg);

            // Most operating systems, but not Linux or emscripten, return the previous pointer
            // when its length is zero. Therefore, check if the previous pointer is the same as
            // the current one.
            if eq(cmsg, previous_cmsg) {
                break;
            }
        }

        if previous_cmsg.is_null() {
            return Err(AddAncillaryError::new());
        }

        (*previous_cmsg).cmsg_level = cmsg_level;
        (*previous_cmsg).cmsg_type = cmsg_type;
        (*previous_cmsg).cmsg_len = libc::CMSG_LEN(source_len) as _;

        let data = libc::CMSG_DATA(previous_cmsg).cast();

        libc::memcpy(data, source.as_ptr().cast(), source_len as usize);
    }
    Ok(())
}

struct AncillaryDataIter<'a, T> {
    data: &'a [u8],
    phantom: PhantomData<T>,
}

impl<'a, T> AncillaryDataIter<'a, T> {
    /// Create `AncillaryDataIter` struct to iterate through the data unit in the control message.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message.
    unsafe fn new(data: &'a [u8]) -> AncillaryDataIter<'a, T> {
        AncillaryDataIter { data, phantom: PhantomData }
    }
}

impl<'a, T> Iterator for AncillaryDataIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if size_of::<T>() <= self.data.len() {
            unsafe {
                let unit = read_unaligned(self.data.as_ptr().cast());
                self.data = &self.data[size_of::<T>()..];
                Some(unit)
            }
        } else {
            None
        }
    }
}

/// Unix credential.
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[derive(Clone)]
pub struct SocketCred(libc::ucred);

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
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct ScmCredentials<'a>(AncillaryDataIter<'a, libc::ucred>);

#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for ScmCredentials<'a> {
    type Item = SocketCred;

    fn next(&mut self) -> Option<SocketCred> {
        Some(SocketCred(self.0.next()?))
    }
}

/// The error type which is returned from parsing the type a control message.
#[non_exhaustive]
#[derive(Debug)]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub enum AncillaryError {
    Unknown { cmsg_level: i32, cmsg_type: i32 },
}

/// This enum represent one control message of variable type.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub enum AncillaryData<'a> {
    ScmRights(ScmRights<'a>),
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    ScmCredentials(ScmCredentials<'a>),
}

impl<'a> AncillaryData<'a> {
    /// Create a `AncillaryData::ScmRights` variant.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message and the control message must be type of
    /// `SOL_SOCKET` and level of `SCM_RIGHTS`.
    unsafe fn as_rights(data: &'a [u8]) -> Self {
        let ancillary_data_iter = AncillaryDataIter::new(data);
        let scm_rights = ScmRights(ancillary_data_iter);
        AncillaryData::ScmRights(scm_rights)
    }

    /// Create a `AncillaryData::ScmCredentials` variant.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message and the control message must be type of
    /// `SOL_SOCKET` and level of `SCM_CREDENTIALS` or `SCM_CREDENTIALS`.
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    unsafe fn as_credentials(data: &'a [u8]) -> Self {
        let ancillary_data_iter = AncillaryDataIter::new(data);
        let scm_credentials = ScmCredentials(ancillary_data_iter);
        AncillaryData::ScmCredentials(scm_credentials)
    }

    fn try_from_cmsghdr(cmsg: &'a libc::cmsghdr) -> Result<Self, AncillaryError> {
        unsafe {
            let cmsg_len_zero = libc::CMSG_LEN(0) as usize;
            let data_len = (*cmsg).cmsg_len as usize - cmsg_len_zero;
            let data = libc::CMSG_DATA(cmsg).cast();
            let data = slice::from_raw_parts(data, data_len);

            match (*cmsg).cmsg_level {
                libc::SOL_SOCKET => match (*cmsg).cmsg_type {
                    libc::SCM_RIGHTS => Ok(AncillaryData::as_rights(data)),
                    #[cfg(any(target_os = "android", target_os = "linux",))]
                    libc::SCM_CREDENTIALS => Ok(AncillaryData::as_credentials(data)),
                    cmsg_type => {
                        Err(AncillaryError::Unknown { cmsg_level: libc::SOL_SOCKET, cmsg_type })
                    }
                },
                cmsg_level => {
                    Err(AncillaryError::Unknown { cmsg_level, cmsg_type: (*cmsg).cmsg_type })
                }
            }
        }
    }
}

/// This struct is used to iterate through the control messages.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct Messages<'a> {
    buffer: &'a [u8],
    current: Option<&'a libc::cmsghdr>,
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for Messages<'a> {
    type Item = Result<AncillaryData<'a>, AncillaryError>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let mut msg: libc::msghdr = zeroed();
            msg.msg_control = self.buffer.as_ptr() as *mut _;
            msg.msg_controllen = self.buffer.len() as _;

            let cmsg = if let Some(current) = self.current {
                libc::CMSG_NXTHDR(&msg, current)
            } else {
                libc::CMSG_FIRSTHDR(&msg)
            };

            let cmsg = cmsg.as_ref()?;

            // Most operating systems, but not Linux or emscripten, return the previous pointer
            // when its length is zero. Therefore, check if the previous pointer is the same as
            // the current one.
            if let Some(current) = self.current {
                if eq(current, cmsg) {
                    return None;
                }
            }

            self.current = Some(cmsg);
            let ancillary_result = AncillaryData::try_from_cmsghdr(cmsg);
            Some(ancillary_result)
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
///     for ancillary_result in ancillary.messages() {
///         if let AncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
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
pub struct SocketAncillary<'a> {
    buffer: &'a mut [MaybeUninit<u8>],
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
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn new(buffer: &'a mut [u8]) -> Self {
        let buffer = unsafe { slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(), buffer.len()) };
        Self::new_uninit(buffer)
    }

    /// Create an ancillary data with an uninitialized buffer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #![allow(unused_mut)]
    /// #![feature(unix_socket_ancillary_data, new_uninit)]
    /// use std::os::unix::net::SocketAncillary;
    /// let mut ancillary_buffer = Box::new_uninit_slice(128);
    /// let mut ancillary = SocketAncillary::new_uninit(&mut ancillary_buffer[..]);
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn new_uninit(buffer: &'a mut [MaybeUninit<u8>]) -> Self {
        SocketAncillary { buffer, length: 0, truncated: false }
    }

    /// Returns the capacity of the buffer.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the raw ancillary data as byte slice.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn data(&self) -> &[u8] {
        unsafe { MaybeUninit::slice_assume_init_ref(&self.buffer[..self.length]) }
    }

    /// Returns the entire buffer, including unused capacity.
    ///
    /// Use [`data()`](Self::data) if you are only interested in the used portion of the buffer.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn buffer(&self) -> &[MaybeUninit<u8>] {
        self.buffer
    }

    /// Returns the entire buffer as mutable slice, including unused capacity.
    ///
    /// You should normally call [`set_len()`](Self::set_len)
    /// after changing the contents of the buffer.
    ///
    /// # Safety
    /// All data written to the buffer must be valid ancillary data for the target platform,
    /// and you must call [`set_len()`](Self::set_len) after changing
    /// the buffer contents to update the internal bookkeeping.
    ///
    /// Make sure to zero-initialize the buffer if you are going to use it with [`libc::CMSG_NXTHDR()`].
    /// That function requires the buffer to be zero-initialized in order to work correctly
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub unsafe fn buffer_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.buffer
    }

    /// Returns `true` if the ancillary data is empty.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Returns the number of used bytes.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Set the number of valid ancillary data bytes and the truncated flag.
    ///
    /// This can be used with [`buffer_mut()`](Self::buffer_mut)
    /// to manually write ancillary data into the buffer.
    ///
    /// # Safety
    /// - The length may not exceed [`capacity()`](Self::capacity).
    /// - The data in the buffer at `0..length` must be valid ancillary data.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub unsafe fn set_len(&mut self, length: usize, truncated: bool) {
        self.length = length;
        self.truncated = truncated;
    }

    /// Returns the iterator of the control messages.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn messages(&self) -> Messages<'_> {
        Messages { buffer: self.data(), current: None }
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
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Add file descriptors to the ancillary data.
    ///
    /// This operation adds a control message with the level `SOL_SOCKET` and type `SCM_RIGHTS`.
    /// If there is not enough space in the buffer for all file descriptors,
    /// an error is returned and no file descriptors are added.
    ///
    /// # Safety
    /// This function copies raw file descriptors into the ancillary data buffer without a lifetime.
    /// You must guarantee that the file descriptor remains open until the ancillary data is no longer used.
    ///
    /// Additionally, you must ensure that duplicating the file descriptor and sending it to a remote process
    /// does not violate the safety requirements of the object managing the file descriptor.
    ///
    /// # Example
    ///
    /// ```no_run
    /// #![feature(unix_socket_ancillary_data)]
    /// use std::os::unix::net::{UnixStream, SocketAncillary};
    /// use std::os::unix::io::AsRawFd;
    /// use std::io::IoSlice;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let sock = UnixStream::connect("/tmp/sock")?;
    ///
    ///     let mut ancillary_buffer = [0; 128];
    ///     let mut ancillary = SocketAncillary::new(&mut ancillary_buffer[..]);
    ///     unsafe {
    ///         ancillary.add_fds(&[sock.as_raw_fd()][..])?;
    ///     }
    ///
    ///     let mut buf = [1; 8];
    ///     let mut bufs = &mut [IoSlice::new(&mut buf[..])][..];
    ///     sock.send_vectored_with_ancillary(bufs, &mut ancillary)?;
    ///     Ok(())
    /// }
    /// ```
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub unsafe fn add_fds(&mut self, fds: &[RawFd]) -> Result<(), AddAncillaryError> {
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
    /// This function adds a control message with the level `SOL_SOCKET`
    /// and type `SCM_CREDENTIALS` or `SCM_CREDS` (depending on the platform).
    /// If there is not enough space in the buffer for all credentials,
    /// an error is returned and no credentials are added.
    #[cfg(any(doc, target_os = "android", target_os = "linux",))]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn add_creds(&mut self, creds: &[SocketCred]) -> Result<(), AddAncillaryError> {
        self.truncated = false;
        add_to_ancillary_data(
            &mut self.buffer,
            &mut self.length,
            creds,
            libc::SOL_SOCKET,
            libc::SCM_CREDENTIALS,
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
    ///     for ancillary_result in ancillary.messages() {
    ///         if let AncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
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
    ///         if let AncillaryData::ScmRights(scm_rights) = ancillary_result.unwrap() {
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
        self.length = 0;
        self.truncated = false;
    }
}

/// An error returned when trying to add anciallary data that exceeds the buffer capacity.
#[cfg(any(doc, target_os = "android", target_os = "linux",))]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct AddAncillaryError {
    _priv: (),
}

impl AddAncillaryError {
    fn new() -> Self {
        Self { _priv: () }
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl fmt::Debug for AddAncillaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AddAncillaryError").finish()
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl fmt::Display for AddAncillaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "could not add data to anciallary buffer")
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl crate::error::Error for AddAncillaryError {}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl From<AddAncillaryError> for io::Error {
    fn from(other: AddAncillaryError) -> Self {
        Self::new(io::ErrorKind::Other, other)
    }
}
