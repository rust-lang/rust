//! Unix domain socket syscall handlers.
//!
//! Implements:
//! - [`sys_socket`]    — create a new unbound socket fd
//! - [`sys_bind`]      — bind the socket to a filesystem path
//! - [`sys_listen`]    — mark the socket as listening
//! - [`sys_accept`]    — block and accept one incoming connection
//! - [`sys_connect`]   — connect to a listening socket by path
//! - [`sys_shutdown`]  — shut down one or both directions of a connection
//! - [`sys_socketpair`]— create an anonymous connected socket pair

use alloc::sync::Arc;
use alloc::vec;

use abi::errors::{Errno, SysResult};
use abi::syscall::{socket_domain, socket_type};

use crate::syscall::validate::{copyin, copyout, validate_user_range};
use crate::vfs::OpenFlags;

// ---------------------------------------------------------------------------
// Helper: copy a path string from userspace
// ---------------------------------------------------------------------------

fn copy_path(path_ptr: usize, path_len: usize) -> SysResult<alloc::string::String> {
    if path_len == 0 || path_len > 4096 {
        return Err(Errno::EINVAL);
    }
    validate_user_range(path_ptr, path_len, false)?;
    let mut buf = vec![0u8; path_len];
    unsafe { copyin(&mut buf, path_ptr)? };
    alloc::string::String::from_utf8(buf).map_err(|_| Errno::EINVAL)
}

// ---------------------------------------------------------------------------
// sys_socket
// ---------------------------------------------------------------------------

/// Create a new Unix domain socket and return a file descriptor.
///
/// Only `AF_UNIX + SOCK_STREAM + protocol=0` is currently supported.
/// Returns `EAFNOSUPPORT` for unknown domains, `EPROTOTYPE` for unknown types.
pub fn sys_socket(domain: usize, type_: usize, _protocol: usize) -> SysResult<usize> {
    if domain as u32 != socket_domain::AF_UNIX {
        return Err(Errno::EAFNOSUPPORT);
    }
    if type_ as u32 != socket_type::SOCK_STREAM {
        return Err(Errno::EPROTOTYPE);
    }

    let node = crate::ipc::unix_socket::UnixSocketNode::new();
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let fd = pinfo_arc.lock().fd_table.open(
        node as Arc<dyn crate::vfs::VfsNode>,
        OpenFlags::read_write(),
        alloc::string::String::from("socket:[unix]"),
    )?;
    Ok(fd as usize)
}

// ---------------------------------------------------------------------------
// sys_bind
// ---------------------------------------------------------------------------

/// Bind a socket fd to a filesystem path.
///
/// Args: `(fd, path_ptr, path_len)`
pub fn sys_bind(fd: usize, path_ptr: usize, path_len: usize) -> SysResult<usize> {
    let path = copy_path(path_ptr, path_len)?;

    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let node = {
        let lock = pinfo_arc.lock();
        lock.fd_table.get(fd as u32)?.node.clone()
    };
    node.sock_bind(&path)?;
    Ok(0)
}

// ---------------------------------------------------------------------------
// sys_listen
// ---------------------------------------------------------------------------

/// Mark a socket as listening.
///
/// Args: `(fd, backlog)`
pub fn sys_listen(fd: usize, backlog: usize) -> SysResult<usize> {
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let node = {
        let lock = pinfo_arc.lock();
        lock.fd_table.get(fd as u32)?.node.clone()
    };
    node.sock_listen(backlog)?;
    Ok(0)
}

// ---------------------------------------------------------------------------
// sys_accept
// ---------------------------------------------------------------------------

/// Accept one incoming connection on a listening socket.
///
/// Args: `(fd)`
/// Returns the new file descriptor for the accepted connection.
pub fn sys_accept(fd: usize) -> SysResult<usize> {
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let node = {
        let lock = pinfo_arc.lock();
        lock.fd_table.get(fd as u32)?.node.clone()
    };
    let accepted = node.sock_accept()?;
    let new_fd = pinfo_arc.lock().fd_table.open(
        accepted,
        OpenFlags::read_write(),
        alloc::string::String::from("socket:[unix/accepted]"),
    )?;
    Ok(new_fd as usize)
}

// ---------------------------------------------------------------------------
// sys_connect
// ---------------------------------------------------------------------------

/// Connect a socket to a listening socket at the given path.
///
/// Args: `(fd, path_ptr, path_len)`
pub fn sys_connect(fd: usize, path_ptr: usize, path_len: usize) -> SysResult<usize> {
    let path = copy_path(path_ptr, path_len)?;

    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let node = {
        let lock = pinfo_arc.lock();
        lock.fd_table.get(fd as u32)?.node.clone()
    };
    node.sock_connect(&path)?;
    Ok(0)
}

// ---------------------------------------------------------------------------
// sys_shutdown
// ---------------------------------------------------------------------------

/// Shut down part or all of a socket connection.
///
/// Args: `(fd, how)` — how: 0=SHUT_RD, 1=SHUT_WR, 2=SHUT_RDWR
pub fn sys_shutdown(fd: usize, how: usize) -> SysResult<usize> {
    if how > 2 {
        return Err(Errno::EINVAL);
    }
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let node = {
        let lock = pinfo_arc.lock();
        lock.fd_table.get(fd as u32)?.node.clone()
    };
    node.sock_shutdown(how as u32)?;
    Ok(0)
}

// ---------------------------------------------------------------------------
// sys_socketpair
// ---------------------------------------------------------------------------

/// Create a connected pair of sockets (anonymous, no filesystem binding).
///
/// Args: `(domain, type, protocol, fds_ptr)`
/// `fds_ptr` must point to a `[u32; 2]` writable user buffer.
/// On success writes `[fd_a, fd_b]` and returns 0.
pub fn sys_socketpair(
    domain: usize,
    type_: usize,
    _protocol: usize,
    fds_ptr: usize,
) -> SysResult<usize> {
    if domain as u32 != socket_domain::AF_UNIX {
        return Err(Errno::EAFNOSUPPORT);
    }
    if type_ as u32 != socket_type::SOCK_STREAM {
        return Err(Errno::EPROTOTYPE);
    }
    validate_user_range(fds_ptr, 8, true)?;

    let (a, b) = crate::ipc::unix_socket::UnixSocketNode::new_pair();

    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let (fd_a, fd_b) = {
        let mut lock = pinfo_arc.lock();
        let fa = lock.fd_table.open(
            a as Arc<dyn crate::vfs::VfsNode>,
            OpenFlags::read_write(),
            alloc::string::String::from("socketpair:[unix/a]"),
        )?;
        match lock.fd_table.open(
            b as Arc<dyn crate::vfs::VfsNode>,
            OpenFlags::read_write(),
            alloc::string::String::from("socketpair:[unix/b]"),
        ) {
            Ok(fb) => (fa, fb),
            Err(e) => {
                let _ = lock.fd_table.close(fa);
                return Err(e);
            }
        }
    };

    // Write [fd_a, fd_b] to userspace as two consecutive u32 values (8 bytes,
    // little-endian to keep ABI stable across host endianness).
    let mut fds_bytes = [0u8; 8];
    fds_bytes[..4].copy_from_slice(&fd_a.to_le_bytes());
    fds_bytes[4..].copy_from_slice(&fd_b.to_le_bytes());
    unsafe { copyout(fds_ptr, &fds_bytes)? };
    Ok(0)
}

