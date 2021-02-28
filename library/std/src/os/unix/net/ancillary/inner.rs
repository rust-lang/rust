use crate::{
    convert::TryFrom,
    io::{self, IoSlice, IoSliceMut},
    marker::PhantomData,
    mem::{size_of, zeroed},
    ptr::{eq, read_unaligned},
    slice::from_raw_parts,
    sys::net::Socket,
};

// FIXME(#43348): Make libc adapt #[doc(cfg(...))] so we don't need these fake definitions here?
#[cfg(all(doc, not(target_os = "linux"), not(target_os = "android")))]
#[allow(non_camel_case_types)]
mod libc {
    pub use libc::c_int;
    pub struct cmsghdr;
}

impl Socket {
    pub(super) unsafe fn recv_vectored_with_ancillary_from(
        &self,
        msg_name: *mut libc::c_void,
        msg_namelen: libc::socklen_t,
        bufs: &mut [IoSliceMut<'_>],
        ancillary: &mut Ancillary<'_>,
    ) -> io::Result<(usize, bool, libc::socklen_t)> {
        let mut msg: libc::msghdr = zeroed();
        msg.msg_name = msg_name;
        msg.msg_namelen = msg_namelen;
        msg.msg_iov = bufs.as_mut_ptr().cast();
        msg.msg_iovlen = bufs.len() as _;
        msg.msg_controllen = ancillary.buffer.len() as _;
        // macos requires that the control pointer is null when the len is 0.
        if msg.msg_controllen > 0 {
            msg.msg_control = ancillary.buffer.as_mut_ptr().cast();
        }

        let count = self.recv_msg(&mut msg)?;

        ancillary.length = msg.msg_controllen as usize;
        ancillary.truncated = msg.msg_flags & libc::MSG_CTRUNC == libc::MSG_CTRUNC;

        let truncated = msg.msg_flags & libc::MSG_TRUNC == libc::MSG_TRUNC;

        Ok((count, truncated, msg.msg_namelen))
    }

    pub(super) unsafe fn send_vectored_with_ancillary_to(
        &self,
        msg_name: *mut libc::c_void,
        msg_namelen: libc::socklen_t,
        bufs: &[IoSlice<'_>],
        ancillary: &mut Ancillary<'_>,
    ) -> io::Result<usize> {
        let mut msg: libc::msghdr = zeroed();
        msg.msg_name = msg_name;
        msg.msg_namelen = msg_namelen;
        msg.msg_iov = bufs.as_ptr() as *mut _;
        msg.msg_iovlen = bufs.len() as _;
        msg.msg_controllen = ancillary.length as _;
        // macos requires that the control pointer is null when the len is 0.
        if msg.msg_controllen > 0 {
            msg.msg_control = ancillary.buffer.as_mut_ptr().cast();
        }

        ancillary.truncated = false;

        self.send_msg(&mut msg)
    }
}

#[derive(Debug)]
pub(crate) struct Ancillary<'a> {
    buffer: &'a mut [u8],
    length: usize,
    truncated: bool,
}

impl<'a> Ancillary<'a> {
    pub(super) fn new(buffer: &'a mut [u8]) -> Self {
        Ancillary { buffer, length: 0, truncated: false }
    }
}

impl Ancillary<'_> {
    pub(super) fn add_to_ancillary_data<T>(
        &mut self,
        source: &[T],
        cmsg_level: libc::c_int,
        cmsg_type: libc::c_int,
    ) -> bool {
        self.truncated = false;

        let source_len = if let Some(source_len) = source.len().checked_mul(size_of::<T>()) {
            if let Ok(source_len) = u32::try_from(source_len) {
                source_len
            } else {
                return false;
            }
        } else {
            return false;
        };

        unsafe {
            let additional_space = libc::CMSG_SPACE(source_len) as usize;

            let new_length = if let Some(new_length) = additional_space.checked_add(self.length) {
                new_length
            } else {
                return false;
            };

            if new_length > self.buffer.len() {
                return false;
            }

            self.buffer[self.length..new_length].fill(0);

            self.length = new_length;

            let mut msg: libc::msghdr = zeroed();
            msg.msg_control = self.buffer.as_mut_ptr().cast();
            msg.msg_controllen = self.length as _;

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
                return false;
            }

            (*previous_cmsg).cmsg_level = cmsg_level;
            (*previous_cmsg).cmsg_type = cmsg_type;
            (*previous_cmsg).cmsg_len = libc::CMSG_LEN(source_len) as _;

            let data = libc::CMSG_DATA(previous_cmsg).cast();

            libc::memcpy(data, source.as_ptr().cast(), source_len as usize);
        }
        true
    }

    pub(super) fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub(super) fn len(&self) -> usize {
        self.length
    }

    pub(super) fn messages<T>(&self) -> Messages<'_, T> {
        Messages { buffer: &self.buffer[..self.length], current: None, phantom: PhantomData {} }
    }

    pub(super) fn truncated(&self) -> bool {
        self.truncated
    }

    pub(super) fn clear(&mut self) {
        self.length = 0;
        self.truncated = false;
    }
}

pub(super) struct AncillaryDataIter<'a, T> {
    data: &'a [u8],
    phantom: PhantomData<T>,
}

impl<'a, T> AncillaryDataIter<'a, T> {
    /// Create `AncillaryDataIter` struct to iterate through the data unit in the control message.
    ///
    /// # Safety
    ///
    /// `data` must contain a valid control message.
    pub(super) unsafe fn new(data: &'a [u8]) -> AncillaryDataIter<'a, T> {
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

/// The error type which is returned from parsing the type a control message.
#[non_exhaustive]
#[derive(Debug)]
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub enum AncillaryError {
    Unknown { cmsg_level: i32, cmsg_type: i32 },
}

/// Return the data of `cmsghdr` as a `u8` slice.
pub(super) unsafe fn get_data_from_cmsghdr(cmsg: &libc::cmsghdr) -> &[u8] {
    let cmsg_len_zero = libc::CMSG_LEN(0) as usize;
    let data_len = (*cmsg).cmsg_len as usize - cmsg_len_zero;
    let data = libc::CMSG_DATA(cmsg).cast();
    from_raw_parts(data, data_len)
}

/// This struct is used to iterate through the control messages.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct Messages<'a, T> {
    buffer: &'a [u8],
    current: Option<&'a libc::cmsghdr>,
    phantom: PhantomData<T>,
}

impl<'a, T> Messages<'a, T> {
    pub(super) unsafe fn next_cmsghdr(&mut self) -> Option<&'a libc::cmsghdr> {
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
        Some(cmsg)
    }
}
