//! Userspace syscall wrappers for Unix domain sockets.
//!
//! These are thin wrappers around the socket syscalls introduced in
//! the Thing-OS IPC overhaul.  Only `AF_UNIX + SOCK_STREAM` is supported for
//! now; `SOCK_DGRAM` and other domains will follow.

use abi::errors::SysResult;
use abi::syscall::{
    SYS_ACCEPT, SYS_BIND, SYS_CONNECT, SYS_LISTEN, SYS_SHUTDOWN, SYS_SOCKET, SYS_SOCKETPAIR,
};

use super::arch::raw_syscall6;

/// Create a new Unix domain stream socket.
///
/// Returns the file descriptor on success.
///
/// # Example
/// ```no_run
/// use stem::syscall::socket::socket;
/// use stem::syscall::socket_domain::AF_UNIX;
/// use stem::syscall::socket_type::SOCK_STREAM;
/// let fd = socket(AF_UNIX, SOCK_STREAM, 0).unwrap();
/// ```
pub fn socket(domain: u32, type_: u32, protocol: u32) -> SysResult<u32> {
    let ret = unsafe {
        raw_syscall6(
            SYS_SOCKET,
            domain as usize,
            type_ as usize,
            protocol as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Bind a socket fd to a filesystem path.
///
/// The path should be absolute (e.g. `/run/my.sock`).
pub fn bind(fd: u32, path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_BIND,
            fd as usize,
            path.as_ptr() as usize,
            path.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Mark a socket as listening for incoming connections.
///
/// `backlog` is the maximum number of queued connections; 0 uses the
/// kernel default (currently clamped to 1..128).
pub fn listen(fd: u32, backlog: usize) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(SYS_LISTEN, fd as usize, backlog, 0, 0, 0, 0)
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Accept one incoming connection on a listening socket.
///
/// Blocks until a connection is available.
/// Returns the new file descriptor for the accepted connection.
pub fn accept(fd: u32) -> SysResult<u32> {
    let ret = unsafe { raw_syscall6(SYS_ACCEPT, fd as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Connect a socket to a listening socket at `path`.
///
/// Blocks until the connection is established.
pub fn connect(fd: u32, path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CONNECT,
            fd as usize,
            path.as_ptr() as usize,
            path.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Shut down part or all of a connected socket.
///
/// `how`:
/// - `0` (`SHUT_RD`)   — stop receiving
/// - `1` (`SHUT_WR`)   — stop sending (sends EOF to the peer)
/// - `2` (`SHUT_RDWR`) — both directions
pub fn shutdown(fd: u32, how: u32) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_SHUTDOWN, fd as usize, how as usize, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Create a connected pair of anonymous Unix domain stream sockets.
///
/// The returned `(fd_a, fd_b)` are both fully connected: writes to one
/// appear as reads on the other, with no filesystem path required.
///
/// This is the simplest way to obtain bidirectional IPC between a parent
/// and child process (before exec) or between two threads in the same
/// process.
///
/// # Example
/// ```no_run
/// use stem::syscall::socket::socketpair;
/// use stem::syscall::socket_domain::AF_UNIX;
/// use stem::syscall::socket_type::SOCK_STREAM;
/// use stem::syscall::{vfs_write, vfs_read};
///
/// let (a, b) = socketpair(AF_UNIX, SOCK_STREAM, 0).unwrap();
/// let mut buf = [0u8; 32];
/// vfs_write(a, b"hello").unwrap();
/// let n = vfs_read(b, &mut buf).unwrap();
/// ```
pub fn socketpair(domain: u32, type_: u32, protocol: u32) -> SysResult<(u32, u32)> {
    let mut fds_bytes = [0u8; 8];
    let ret = unsafe {
        raw_syscall6(
            SYS_SOCKETPAIR,
            domain as usize,
            type_ as usize,
            protocol as usize,
            fds_bytes.as_mut_ptr() as usize,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| {
        let fd_a = u32::from_le_bytes(fds_bytes[..4].try_into().unwrap());
        let fd_b = u32::from_le_bytes(fds_bytes[4..].try_into().unwrap());
        (fd_a, fd_b)
    })
}
