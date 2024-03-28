use crate::ffi::c_int;
use crate::mem::{size_of, MaybeUninit};

// Wrapper around `libc::CMSG_LEN` to safely decouple from OS-specific ints.
//
// https://github.com/rust-lang/libc/issues/3240
#[inline]
const fn CMSG_LEN(len: usize) -> usize {
    let c_len = len & 0x7FFFFFFF;
    let padding = (unsafe { libc::CMSG_LEN(c_len as _) } as usize) - c_len;
    len + padding
}

// Wrapper around `libc::CMSG_SPACE` to safely decouple from OS-specific ints.
//
// https://github.com/rust-lang/libc/issues/3240
#[inline]
const fn CMSG_SPACE(len: usize) -> usize {
    let c_len = len & 0x7FFFFFFF;
    let padding = (unsafe { libc::CMSG_SPACE(c_len as _) } as usize) - c_len;
    len + padding
}

/// A socket control message with borrowed data.
///
/// This type is semantically equivalent to POSIX `struct cmsghdr`, but is
/// not guaranteed to have the same internal representation.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ControlMessage<'a> {
    cmsg_len: usize,
    cmsg_level: c_int,
    cmsg_type: c_int,
    data: &'a [u8],
}

impl<'a> ControlMessage<'a> {
    /// Creates a `ControlMessage` with the given level, type, and data.
    ///
    /// The semantics of a control message "level" and "type" are OS-specific,
    /// but generally the level is a sort of general category of socket and the
    /// type identifies a specific control message data layout.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn new(cmsg_level: c_int, cmsg_type: c_int, data: &'a [u8]) -> ControlMessage<'a> {
        let cmsg_len = CMSG_LEN(data.len());
        ControlMessage { cmsg_len, cmsg_level, cmsg_type, data }
    }
}

impl ControlMessage<'_> {
    /// Returns the control message's level, an OS-specific value.
    ///
    /// POSIX describes this field as the "originating protocol".
    #[inline]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn cmsg_level(&self) -> c_int {
        self.cmsg_level
    }

    /// Returns the control message's type, an OS-specific value.
    ///
    /// POSIX describes this field as the "protocol-specific type".
    #[inline]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn cmsg_type(&self) -> c_int {
        self.cmsg_type
    }

    /// Returns the control message's type-specific data.
    ///
    /// The returned slice is equivalent to the result of C macro `CMSG_DATA()`.
    /// Control message data is not guaranteed to be aligned, so code that needs
    /// to inspect it should first copy the data to a properly-aligned location.
    #[inline]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Returns `true` if the control message data is truncated.
    ///
    /// The kernel may truncate a control message if its data is too large to
    /// fit into the capacity of the userspace buffer.
    ///
    /// The semantics of truncated control messages are OS- and type-specific.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn truncated(&self) -> bool {
        self.cmsg_len > CMSG_LEN(self.data.len())
    }

    #[inline]
    pub(super) fn cmsg_space(&self) -> usize {
        CMSG_SPACE(self.data.len())
    }

    #[allow(dead_code)] // currently the only use is in the test suite
    pub(super) fn copy_to_slice<'a>(&self, dst: &'a mut [MaybeUninit<u8>]) -> &'a [u8] {
        assert_eq!(dst.len(), self.cmsg_space());

        // SAFETY: C type `struct cmsghdr` is safe to zero-initialize.
        let mut hdr: libc::cmsghdr = unsafe { core::mem::zeroed() };

        // Write `cmsg.cmsg_len` instead of `CMSG_LEN(data.len())` so that
        // truncated control messages are preserved as-is.
        hdr.cmsg_len = self.cmsg_len as _;
        hdr.cmsg_level = self.cmsg_level;
        hdr.cmsg_type = self.cmsg_type;

        #[inline]
        unsafe fn sized_to_slice<T: Sized>(t: &T) -> &[u8] {
            let t_ptr = (t as *const T).cast::<u8>();
            crate::slice::from_raw_parts(t_ptr, size_of::<T>())
        }

        let (hdr_dst, after_hdr) = dst.split_at_mut(size_of::<libc::cmsghdr>());
        let (data_dst, padding_dst) = after_hdr.split_at_mut(self.data.len());

        // SAFETY: C type `struct cmsghdr` is safe to bitwise-copy from.
        MaybeUninit::write_slice(hdr_dst, unsafe { sized_to_slice(&hdr) });

        // See comment in `ControlMessagesIter` regarding `CMSG_DATA()`.
        MaybeUninit::write_slice(data_dst, self.data());

        if padding_dst.len() > 0 {
            for byte in padding_dst.iter_mut() {
                byte.write(0);
            }
        }

        // SAFETY: Every byte in `dst` has been initialized.
        unsafe { MaybeUninit::slice_assume_init_ref(dst) }
    }
}

/// A borrowed reference to a `&[u8]` slice containing control messages.
///
/// Note that this type does not guarantee the control messages are valid, or
/// even well-formed. Code that uses control messages to implement (for example)
/// access control or file descriptor passing should maintain a chain of custody
/// to verify that the `&ControlMessages` came from a trusted source, such as
/// a syscall.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct ControlMessages {
    bytes: [u8],
}

impl ControlMessages {
    /// Creates a `ControlMessages` wrapper from a `&[u8]` slice containing
    /// encoded control messages.
    ///
    /// This method does not attempt to verify that the provided bytes represent
    /// valid control messages.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn from_bytes(bytes: &[u8]) -> &ControlMessages {
        // SAFETY: casting `&[u8]` to `&ControlMessages` is safe because its
        // internal representation is `[u8]`.
        unsafe { &*(bytes as *const [u8] as *const ControlMessages) }
    }

    /// Returns a `&[u8]` slice containing encoded control messages.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns `true` if `self.as_bytes()` is an empty slice.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns an iterator over the control messages.
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn iter(&self) -> ControlMessagesIter<'_> {
        ControlMessagesIter { bytes: &self.bytes }
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> IntoIterator for &'a ControlMessages {
    type Item = ControlMessage<'a>;
    type IntoIter = ControlMessagesIter<'a>;

    fn into_iter(self) -> ControlMessagesIter<'a> {
        self.iter()
    }
}

/// An iterator over the content of a [`ControlMessages`].
///
/// Each control message starts with a header describing its own length. This
/// iterator is safe even if the header lengths are incorrect, but the returned
/// control messages may contain incorrect data.
///
/// Iteration ends when the remaining data is smaller than the size of a single
/// control message header.
#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
pub struct ControlMessagesIter<'a> {
    bytes: &'a [u8],
}

impl<'a> ControlMessagesIter<'a> {
    /// Returns a `&[u8]` slice containing any remaining data.
    ///
    /// Even if `next()` returns `None`, this method may return a non-empty
    /// slice if the original `ControlMessages` was truncated in the middle
    /// of a control message header.
    #[inline]
    #[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
    pub fn into_bytes(self) -> &'a [u8] {
        self.bytes
    }
}

#[unstable(feature = "unix_socket_ancillary_data", issue = "76915")]
impl<'a> Iterator for ControlMessagesIter<'a> {
    type Item = ControlMessage<'a>;

    fn next(&mut self) -> Option<ControlMessage<'a>> {
        const CMSGHDR_SIZE: usize = size_of::<libc::cmsghdr>();

        if CMSGHDR_SIZE > self.bytes.len() {
            return None;
        }

        // SAFETY: C type `struct cmsghdr` is safe to bitwise-copy from.
        let hdr = unsafe {
            let mut hdr = MaybeUninit::<libc::cmsghdr>::uninit();
            hdr.as_mut_ptr().cast::<u8>().copy_from(self.bytes.as_ptr(), CMSGHDR_SIZE);
            hdr.assume_init()
        };

        // `cmsg_bytes` contains the full content of the control message,
        // which may have been truncated if there was insufficient capacity.
        let cmsg_bytes;
        let hdr_cmsg_len = hdr.cmsg_len as usize;
        if hdr_cmsg_len >= self.bytes.len() {
            cmsg_bytes = self.bytes;
        } else {
            cmsg_bytes = &self.bytes[..hdr_cmsg_len];
        }

        // `cmsg_data` is the portion of the control message that contains
        // type-specific content (file descriptors, etc).
        //
        // POSIX specifies that a pointer to this data should be obtained with
        // macro `CMSG_DATA()`, but its definition is problematic for Rust:
        //
        //   1. The macro may in principle read fields of `cmsghdr`. To avoid
        //      unaligned reads this code would call it as `CMSG_DATA(&hdr)`.
        //      But the resulting pointer would be relative to the stack value
        //      `hdr`, not the actual message data contained in `cmsg_bytes`.
        //
        //   2. `CMSG_DATA()` is implemented with `pointer::offset()`, which
        //      causes undefined behavior if its result is outside the original
        //      allocated object. The POSIX spec allows control messages to
        //      have padding between the header and data, in which case
        //      `CMSG_DATA(&hdr)` is UB.
        //
        //   3. The control message may have been truncated. We know there's
        //      at least `CMSGHDR_SIZE` bytes available, but anything past that
        //      isn't guaranteed. Again, possible UB in the presence of padding.
        //
        // Therefore, this code obtains `cmsg_data` by assuming it directly
        // follows the header (with no padding, and no header field dependency).
        // This is true on all target OSes currently supported by Rust.
        //
        // If in the future support is added for an OS with cmsg data padding,
        // then this implementation will cause unit test failures rather than
        // risking silent UB.
        let cmsg_data = &cmsg_bytes[CMSGHDR_SIZE..];

        // `cmsg_space` is the length of the control message plus any padding
        // necessary to align the next message.
        let cmsg_space = CMSG_SPACE(cmsg_data.len());
        if cmsg_space >= self.bytes.len() {
            self.bytes = &[];
        } else {
            self.bytes = &self.bytes[cmsg_space..];
        }

        Some(ControlMessage {
            cmsg_len: hdr_cmsg_len,
            cmsg_level: hdr.cmsg_level,
            cmsg_type: hdr.cmsg_type,
            data: cmsg_data,
        })
    }
}
