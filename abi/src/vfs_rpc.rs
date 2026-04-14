//! VFS RPC wire format — shared between kernel and userland providers.
//!
//! When a userland process calls `SYS_FS_MOUNT(port_write_handle, path)`, the
//! kernel registers a [`ProviderFs`][kernel-side] at the given path.  From that
//! point on, every VFS operation that touches a path under that mount point is
//! serialised into one of the messages below and sent to the provider's port.
//!
//! ## Wire layout
//!
//! Every **request** starts with a 5-byte header:
//! ```text
//! [resp_port: u32 LE] [op: u8]
//! ```
//! followed by op-specific payload.
//!
//! Every **response** starts with a 1-byte status:
//! ```text
//! [status: u8]   0 = OK, non-zero = errno value
//! ```
//! followed by op-specific payload (only when status == 0).
//!
//! All multi-byte integers are **little-endian**.

/// VFS RPC operation codes sent from the kernel to a userland provider.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VfsRpcOp {
    /// Look up a path relative to the mount point and return an opaque handle.
    ///
    /// Payload: `[path_len: u32][path bytes (UTF-8)]`
    /// Response payload (on OK): `[handle: u64]`
    Lookup = 1,
    /// Read bytes from an open handle.
    ///
    /// Payload: `[handle: u64][offset: u64][len: u32]`
    /// Response payload (on OK): `[bytes_read: u32][data bytes...]`
    Read = 2,
    /// Write bytes to an open handle.
    ///
    /// Payload: `[handle: u64][offset: u64][data_len: u32][data bytes...]`
    /// Response payload (on OK): `[bytes_written: u32]`
    Write = 3,
    /// Read directory entries from a directory handle.
    ///
    /// The response data is a sequence of packed [`DirentWire`] entries.
    ///
    /// Payload: `[handle: u64][offset: u64][len: u32]`
    /// Response payload (on OK): `[bytes_read: u32][dirent data...]`
    Readdir = 4,
    /// Stat an open handle.
    ///
    /// Payload: `[handle: u64]`
    /// Response payload (on OK): `[mode: u32][size: u64][ino: u64]`
    Stat = 5,
    /// Close an open handle, allowing the provider to free resources.
    ///
    /// Payload: `[handle: u64]`
    /// Response payload (on OK): (empty)
    Close = 6,
    /// Poll readiness bits for an open handle (non-blocking check).
    ///
    /// Payload: `[handle: u64][events: u32]`
    /// Response payload (on OK): `[revents: u32]`
    Poll = 7,
    /// Device-specific control call (ioctl).
    ///
    /// Payload: `[handle: u64][DeviceCall struct]`
    /// Response payload (on OK): `[u32 return value]`
    DeviceCall = 8,
    /// Subscribe to readiness notifications for an open handle.
    ///
    /// Payload: `[handle: u64][events: u32]`
    /// Response payload (on OK): (empty)
    SubscribeReady = 9,
    /// Unsubscribe from readiness notifications for an open handle.
    ///
    /// Payload: `[handle: u64]`
    /// Response payload (on OK): (empty)
    UnsubscribeReady = 10,
    /// Rename a path.
    ///
    /// Payload: `[old_path_len: u32][old_path bytes][new_path_len: u32][new_path bytes]`
    /// Response payload (on OK): (empty)
    Rename = 11,
}

impl VfsRpcOp {
    /// Convert a raw byte to a `VfsRpcOp`, returning `None` for unknown codes.
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Lookup),
            2 => Some(Self::Read),
            3 => Some(Self::Write),
            4 => Some(Self::Readdir),
            5 => Some(Self::Stat),
            6 => Some(Self::Close),
            7 => Some(Self::Poll),
            8 => Some(Self::DeviceCall),
            9 => Some(Self::SubscribeReady),
            10 => Some(Self::UnsubscribeReady),
            11 => Some(Self::Rename),
            _ => None,
        }
    }
}

/// Header prepended to every request sent by the kernel to a provider port.
///
/// Layout (7 bytes):
/// ```text
/// [resp_port: u32 LE][op: u8][_pad: u8][_pad: u8]
/// ```
/// The `resp_port` is the write-handle of the kernel's private response port.
/// After processing the request, the provider **must** send its response to
/// that handle using `SYS_channel_send`.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct VfsRpcReqHeader {
    /// Port write-handle the provider should send the response back to.
    pub resp_port: u32,
    /// Operation code (one of [`VfsRpcOp`]).
    pub op: u8,
    pub _pad: [u8; 2],
}

/// Packed directory entry as returned by the provider in a `Readdir` response.
///
/// Layout (fixed prefix followed by variable-length name):
/// ```text
/// [ino: u64][file_type: u8][name_len: u8][name bytes (UTF-8, no NUL)]
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct DirentWire {
    /// Inode-like unique identifier.
    pub ino: u64,
    /// File type bits, same encoding as `VfsStat::mode & S_IFMT`.
    pub file_type: u8,
    /// Length of the name that follows immediately after this struct.
    pub name_len: u8,
}

/// Maximum path length accepted in a `Lookup` request.
pub const VFS_RPC_MAX_PATH: usize = 4096;
/// Maximum data length for a single `Read` or `Write` payload.
pub const VFS_RPC_MAX_DATA: usize = 65536;
/// Maximum response buffer size a provider should allocate.
pub const VFS_RPC_MAX_RESP: usize = VFS_RPC_MAX_DATA + 64;
/// Maximum size of a VFS RPC request buffer (header + path or data).
pub const VFS_RPC_MAX_REQ: usize = core::mem::size_of::<VfsRpcReqHeader>() + VFS_RPC_MAX_DATA + 64;
