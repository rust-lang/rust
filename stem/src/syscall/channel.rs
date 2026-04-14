//! Channel IPC syscall wrappers for userspace

use crate::syscall::arch::raw_syscall6;
use abi::errors::Errno;
use abi::syscall::*;

/// A handle to a channel endpoint for IPC.
pub type ChannelHandle = u32;

/// Create a new channel pair (returns packed read/write handles).
/// Result: (write_handle << 16) | read_handle
pub fn channel_create(capacity: usize) -> Result<(ChannelHandle, ChannelHandle), Errno> {
    let ret = unsafe { raw_syscall6(SYS_CHANNEL_CREATE, capacity, 0, 0, 0, 0, 0) };
    let val = abi::errors::errno(ret)?;
    let write_handle = ((val >> 16) & 0xFFFF) as ChannelHandle;
    let read_handle = (val & 0xFFFF) as ChannelHandle;
    Ok((write_handle, read_handle))
}

/// Create a new channel pair and immediately expose both ends as file descriptors.
///
/// This is the preferred FD-first entry point for new code.  The returned
/// `(write_fd, read_fd)` can be used directly with `vfs_write`, `vfs_read`,
/// `vfs_poll`, and `vfs_close` without ever touching the underlying handles.
pub fn channel_create_fds(capacity: usize) -> Result<(u32, u32), Errno> {
    let (write_handle, read_handle) = channel_create(capacity)?;
    let write_fd = super::vfs::vfs_fd_from_handle(write_handle)?;
    let read_fd = super::vfs::vfs_fd_from_handle(read_handle)?;
    Ok((write_fd, read_fd))
}

pub fn channel_send(handle: ChannelHandle, data: &[u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_SEND,
            handle as usize,
            data.as_ptr() as usize,
            data.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

pub fn channel_send_all(handle: ChannelHandle, data: &[u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_SEND_ALL,
            handle as usize,
            data.as_ptr() as usize,
            data.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

pub fn channel_recv(handle: ChannelHandle, buf: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_RECV,
            handle as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

pub fn channel_try_recv(handle: ChannelHandle, buf: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_TRY_RECV,
            handle as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

pub fn channel_close(handle: ChannelHandle) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_CHANNEL_CLOSE, handle as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Wait for one of the given channel handles to become readable or writable.
///
/// # Deprecated
///
/// Use [`crate::syscall::vfs::vfs_fd_from_handle`] to convert each handle to an FD,
/// then call [`crate::syscall::vfs::vfs_poll`] with a [`PollFd`] slice.
///
/// ```no_run
/// use abi::syscall::{PollFd, poll_flags};
/// use stem::syscall::vfs::{vfs_fd_from_handle, vfs_poll};
///
/// let channel_handle: u32 = 1; // obtained from channel_create
/// let fd = vfs_fd_from_handle(channel_handle).expect("bridge");
/// let mut pollfds = [PollFd { fd: fd as i32, events: poll_flags::POLLIN, revents: 0 }];
/// vfs_poll(&mut pollfds, u64::MAX).expect("poll");
/// ```
#[deprecated(
    note = "Convert handles to FDs with `vfs_fd_from_handle` and use `vfs_poll` instead"
)]
pub fn channel_wait(handles: &[ChannelHandle], flags: u32) -> Result<ChannelHandle, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_WAIT,
            handles.as_ptr() as usize,
            handles.len(),
            flags as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as ChannelHandle)
}

pub fn channel_len(handle: ChannelHandle) -> Result<usize, Errno> {
    let ret = unsafe { raw_syscall6(SYS_CHANNEL_INFO, handle as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| (v & 0xFFFFFFFF) as usize)
}

pub fn channel_capacity(handle: ChannelHandle) -> Result<usize, Errno> {
    let ret = unsafe { raw_syscall6(SYS_CHANNEL_INFO, handle as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| (v >> 32) as usize)
}

#[deprecated(
    note = "Use `channel_send_msg` instead, which bundles data and FDs atomically. \
            Example: channel_send_msg(channel, &[], &[fd])"
)]
pub fn channel_send_handle(channel: ChannelHandle, handle: u32) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_SEND_HANDLE,
            channel as usize,
            handle as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

#[deprecated(
    note = "Use `channel_recv_msg` instead, which receives data and FDs atomically. \
            Example: let mut fds=[0u32;1]; channel_recv_msg(ch, &mut[], &mut fds)"
)]
pub fn channel_recv_handle(channel: ChannelHandle) -> Result<u32, Errno> {
    let mut out_fd: u32 = 0;
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_RECV_HANDLE,
            channel as usize,
            &mut out_fd as *mut u32 as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| out_fd)
}

/// Send a message with zero or more attached handles over a channel.
///
/// `data` may be empty (handle-only message).  `handles` is a slice of
/// fd/handle numbers from the calling process's table; the kernel resolves
/// each number and attaches the underlying capability to the message.
///
/// Transfer semantics: **duplicate** — the caller retains its own fd/handle.
pub fn channel_send_msg(
    channel: ChannelHandle,
    data: &[u8],
    handles: &[u32],
) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_SEND_MSG,
            channel as usize,
            data.as_ptr() as usize,
            data.len(),
            handles.as_ptr() as usize,
            handles.len(),
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Receive a message with zero or more attached handles from a channel.
///
/// `data_buf` receives the payload bytes (truncated if the buffer is too small).
/// `handles_buf` receives the new fd numbers assigned in the calling process for
/// each transferred capability (truncated if the buffer is too small).
///
/// Returns `(actual_data_len, actual_handles_count)` on success, or
/// `Err(Errno::EAGAIN)` when the message queue is empty.
pub fn channel_recv_msg(
    channel: ChannelHandle,
    data_buf: &mut [u8],
    handles_buf: &mut [u32],
) -> Result<(usize, usize), Errno> {
    let mut out_lens = [0usize; 2];
    let ret = unsafe {
        raw_syscall6(
            SYS_CHANNEL_RECV_MSG,
            channel as usize,
            data_buf.as_mut_ptr() as usize,
            data_buf.len(),
            handles_buf.as_mut_ptr() as usize,
            handles_buf.len(),
            out_lens.as_mut_ptr() as usize,
        )
    };
    abi::errors::errno(ret).map(|_| (out_lens[0], out_lens[1]))
}
