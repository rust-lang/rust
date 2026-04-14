//! Userspace VFS syscall wrappers.
//!
//! These are thin wrappers around the raw VFS syscalls introduced in the
//! janix de-graphing migration (Act III – Birth of the VFS, Act IV – Kernel
//! Filesystems).

use abi::errors::SysResult;
use abi::syscall::{
    PollFd, SYS_FD_FROM_HANDLE, SYS_FS_CHDIR, SYS_FS_CHMOD, SYS_FS_CLOSE, SYS_FS_DEVICE_CALL,
    SYS_FS_DUP, SYS_FS_DUP2, SYS_FS_FCHMOD, SYS_FS_FCNTL, SYS_FS_FLOCK, SYS_FS_FTRUNCATE,
    SYS_FS_FUTIMES, SYS_FS_GETCWD, SYS_FS_ISATTY, SYS_FS_LINK, SYS_FS_LSTAT, SYS_FS_MKDIR,
    SYS_FS_MOUNT, SYS_FS_NOTIFY, SYS_FS_OPEN, SYS_FS_POLL, SYS_FS_READ, SYS_FS_READDIR,
    SYS_FS_READLINK, SYS_FS_READV, SYS_FS_REALPATH, SYS_FS_RENAME, SYS_FS_SEEK, SYS_FS_STAT,
    SYS_FS_SYMLINK, SYS_FS_SYNC, SYS_FS_UMOUNT, SYS_FS_UNLINK, SYS_FS_UTIMES, SYS_FS_WATCH_FD,
    SYS_FS_WATCH_PATH, SYS_FS_WRITE, SYS_FS_WRITEV, SYS_PIPE,
};

use super::arch::raw_syscall6;

/// Open a file at the given absolute path.
///
/// `flags` follows the same encoding as [`abi::syscall::vfs_flags`].
/// Returns a file descriptor on success, or an [`Errno`] on failure.
pub fn vfs_open(path: &str, flags: u32) -> SysResult<u32> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_OPEN,
            path.as_ptr() as usize,
            path.len(),
            flags as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Close a VFS file descriptor previously returned by [`vfs_open`].
pub fn vfs_close(fd: u32) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_FS_CLOSE, fd as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Read up to `buf.len()` bytes from `fd` into `buf`.
///
/// Returns the number of bytes actually read (may be 0 at EOF).
pub fn vfs_read(fd: u32, buf: &mut [u8]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_READ,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Read directory entries from `fd` into `buf`.
///
/// Returns the number of bytes actually read (0 at EOF).
pub fn vfs_readdir(fd: u32, buf: &mut [u8]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_READDIR,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Write `buf` to `fd`.
///
/// Returns the number of bytes written.
pub fn vfs_write(fd: u32, buf: &[u8]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_WRITE,
            fd as usize,
            buf.as_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Read into a scatter-gather buffer list (vectored read).
///
/// Each element of `iovecs` describes one buffer: `base` is the buffer pointer
/// (as `usize`) and `len` is the buffer length.  The kernel reads sequentially
/// into each buffer and stops on a short read or EOF.
///
/// Returns the total number of bytes read across all buffers.
pub fn vfs_readv(fd: u32, iovecs: &[abi::syscall::IoVec]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_READV,
            fd as usize,
            iovecs.as_ptr() as usize,
            iovecs.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Write from a scatter-gather buffer list (vectored write).
///
/// Each element of `iovecs` describes one buffer.  The kernel writes
/// sequentially from each buffer and stops on a short write.
///
/// Returns the total number of bytes written across all buffers.
pub fn vfs_writev(fd: u32, iovecs: &[abi::syscall::IoVec]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_WRITEV,
            fd as usize,
            iovecs.as_ptr() as usize,
            iovecs.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Seek to `offset` relative to `whence`.
///
/// `whence` is: 0 (SEEK_SET), 1 (SEEK_CUR), 2 (SEEK_END).
/// Returns the new absolute offset from the start of the file.
pub fn vfs_seek(fd: u32, offset: i64, whence: u32) -> SysResult<u64> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_SEEK,
            fd as usize,
            offset as usize,
            whence as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u64)
}

/// Stat an open file descriptor.
///
/// Returns a [`abi::fs::FileStat`] containing mode, size, inode, ownership
/// (`uid`/`gid`), link count (`nlink`), device number (`rdev`), block
/// accounting (`blksize`/`blocks`), and the three standard timestamps
/// (`atime`, `mtime`, `ctime`).
pub fn vfs_stat(fd: u32) -> SysResult<abi::fs::FileStat> {
    let mut stat = abi::fs::FileStat::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_STAT,
            fd as usize,
            &mut stat as *mut abi::fs::FileStat as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| stat)
}

/// Stat a path without following the final symlink (`lstat` semantics).
///
/// Unlike [`vfs_stat`] which operates on an open fd, this takes a path and
/// resolves it without following the last path component if it is a symlink.
/// Returns the symlink node's own metadata (mode `S_IFLNK`, size = target
/// length, etc.) rather than the target's metadata.
///
/// Returns a [`abi::fs::FileStat`] on success, or an [`Errno`] on failure.
pub fn vfs_lstat(path: &str) -> SysResult<abi::fs::FileStat> {
    let mut stat = abi::fs::FileStat::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_LSTAT,
            path.as_ptr() as usize,
            path.len(),
            &mut stat as *mut abi::fs::FileStat as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| stat)
}

/// Check if an open file descriptor refers to a terminal/TTY device.
pub fn vfs_isatty(fd: u32) -> SysResult<bool> {
    let ret = unsafe { raw_syscall6(SYS_FS_ISATTY, fd as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v != 0)
}

/// Remove a file or empty directory at `path`.
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_unlink(path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_UNLINK,
            path.as_ptr() as usize,
            path.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Create a directory at `path`.
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_mkdir(path: &str) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_FS_MKDIR, path.as_ptr() as usize, path.len(), 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Mount a userland VFS provider at `path`.
///
/// `provider_write_handle` is the write end of a port pair that the provider
/// owns.  The kernel will send [`abi::vfs_rpc`] messages to that port whenever
/// a VFS operation touches a path under `path`.
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_mount(provider_write_handle: u32, path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_MOUNT,
            provider_write_handle as usize,
            path.as_ptr() as usize,
            path.len(),
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Unmount the userland VFS provider previously mounted at `path`.
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_umount(path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_UMOUNT,
            path.as_ptr() as usize,
            path.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Duplicate `old_fd` to the lowest available file descriptor.
///
/// Returns the new file descriptor on success.
pub fn dup(old_fd: u32) -> SysResult<u32> {
    let ret = unsafe { raw_syscall6(SYS_FS_DUP, old_fd as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Duplicate `old_fd` to `new_fd`, closing `new_fd` first if it is open.
///
/// Returns `new_fd` on success.
pub fn dup2(old_fd: u32, new_fd: u32) -> SysResult<u32> {
    let ret = unsafe { raw_syscall6(SYS_FS_DUP2, old_fd as usize, new_fd as usize, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// File-descriptor control.
///
/// Supports `F_GETFL`, `F_SETFL`, `F_GETFD`, and `F_SETFD`.
pub fn vfs_fcntl(fd: u32, cmd: u32, arg: u32) -> SysResult<u32> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_FCNTL,
            fd as usize,
            cmd as usize,
            arg as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Create an anonymous VFS pipe, writing the read and write file descriptors
/// into `pipefd[0]` and `pipefd[1]` respectively.
///
/// Returns `Ok(())` on success.
pub fn pipe(pipefd: &mut [u32; 2]) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_PIPE, pipefd.as_mut_ptr() as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Poll a set of VFS file descriptors for I/O readiness.
///
/// Fills in `pollfds[i].revents` for each entry and returns the number of
/// entries with non-zero `revents`.  `timeout_ms` is the maximum number of
/// milliseconds to wait; pass `-1i64 as u64` to wait indefinitely.
///
/// # Example
/// ```no_run
/// use abi::syscall::{PollFd, poll_flags};
/// use stem::syscall::vfs_poll;
/// let mut fds = [PollFd { fd: 0, events: poll_flags::POLLIN, revents: 0 }];
/// let n = vfs_poll(&mut fds, u64::MAX).unwrap();
/// if n > 0 { /* fd 0 is readable */ }
/// ```
pub fn vfs_poll(pollfds: &mut [PollFd], timeout_ms: u64) -> SysResult<usize> {
    if pollfds.is_empty() {
        return Ok(0);
    }
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_POLL,
            pollfds.as_mut_ptr() as usize,
            pollfds.len(),
            timeout_ms as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Watch a file descriptor for changes.
///
/// `mask` is a bitmask of [`abi::vfs_watch::mask`] events.
/// `flags` is a bitmask of [`abi::vfs_watch::flags`].
/// Returns a new watch file descriptor.
pub fn vfs_watch_fd(fd: u32, mask: u32, flags: u32) -> SysResult<u32> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_WATCH_FD,
            fd as usize,
            mask as usize,
            flags as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Watch a path for changes.
///
/// `mask` is a bitmask of [`abi::vfs_watch::mask`] events.
/// `flags` is a bitmask of [`abi::vfs_watch::flags`].
/// Returns a new watch file descriptor.
pub fn vfs_watch_path(path: &str, mask: u32, flags: u32) -> SysResult<u32> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_WATCH_PATH,
            path.as_ptr() as usize,
            path.len(),
            mask as usize,
            flags as usize,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Rename a file or directory from `old_path` to `new_path`.
pub fn vfs_rename(old_path: &str, new_path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_RENAME,
            old_path.as_ptr() as usize,
            old_path.len(),
            new_path.as_ptr() as usize,
            new_path.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Issue a device-specific call (ioctl) to a VFS file descriptor.
/// Issue a device-specific call (ioctl) to a VFS file descriptor.
pub fn vfs_device_call(
    fd: u32,
    kind: abi::device::DeviceKind,
    op: u32,
    arg: u64,
) -> SysResult<u64> {
    let call = abi::device::DeviceCall {
        kind,
        op,
        in_ptr: arg,
        in_len: 0,
        out_ptr: 0,
        out_len: 0,
    };
    vfs_device_call_raw(fd, &call)
}

/// Issue a raw device-specific call using a pre-filled DeviceCall struct.
pub fn vfs_device_call_raw(fd: u32, call: &abi::device::DeviceCall) -> SysResult<u64> {
    let ret = unsafe {
        super::arch::raw_syscall6(
            SYS_FS_DEVICE_CALL,
            fd as usize,
            call as *const _ as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u64)
}

/// Change the current working directory of the process.
pub fn vfs_chdir(path: &str) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_FS_CHDIR, path.as_ptr() as usize, path.len(), 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Get the current working directory of the process.
/// Writes the path into `buf` and returns the number of bytes written.
pub fn vfs_getcwd(buf: &mut [u8]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_GETCWD,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Bridge an IPC handle into the VFS as a file descriptor.
pub fn vfs_fd_from_handle(handle: u32) -> SysResult<u32> {
    let ret = unsafe { raw_syscall6(SYS_FD_FROM_HANDLE, handle as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u32)
}

/// Notify the kernel that a provider-backed node is ready.
pub fn vfs_notify(req_handle: u32, node_handle: u64, revents: u16) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_NOTIFY,
            req_handle as usize,
            node_handle as usize,
            revents as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Resolve `path` (relative or absolute) to its canonical absolute form.
///
/// The kernel normalises `.` and `..` components and prepends the process
/// working directory when `path` is relative.  The result is written into
/// `buf` as raw UTF-8 bytes **without** a NUL terminator; callers that need
/// a C-style string must append `\0` themselves.
///
/// Returns the number of bytes of the canonical path.  If the return value
/// is greater than `buf.len()`, the buffer was too small and nothing was
/// written; the caller should retry with a larger buffer.
///
/// # Errors
/// - [`Errno::EINVAL`]  — `path` is empty or not valid UTF-8.
/// - [`Errno::ENOENT`]  — no process context (relative path with missing cwd).
pub fn vfs_realpath(path: &str, buf: &mut [u8]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_REALPATH,
            path.as_ptr() as usize,
            path.len(),
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as usize)
}

/// Flush all pending writes on `fd` to the backing store (fsync).
///
/// For RAM-backed filesystems this is a no-op that always succeeds.
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_fsync(fd: u32) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_FS_SYNC, fd as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Truncate the file associated with `fd` to exactly `size` bytes.
///
/// If `size` is greater than the current file length the file is extended with
/// zero bytes.  If `size` is smaller, the excess data is discarded.
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_ftruncate(fd: u32, size: u64) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_FS_FTRUNCATE, fd as usize, size as usize, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Create a symbolic link at `link_path` that points to `target`.
///
/// `target` is the content of the symlink (not validated for existence).
/// `link_path` is the absolute path at which the symlink entry is created.
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_symlink(target: &str, link_path: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_SYMLINK,
            target.as_ptr() as usize,
            target.len(),
            link_path.as_ptr() as usize,
            link_path.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Read the target of the symbolic link at `path`.
///
/// Writes the symlink target bytes (without a NUL terminator) into `buf`
/// and returns the number of bytes in the target.  If the return value is
/// greater than `buf.len()`, the buffer was too small; the caller should
/// retry with a larger buffer.
///
/// # Errors
/// - [`Errno::EINVAL`] — `path` does not refer to a symlink.
/// - [`Errno::ENOENT`] — `path` does not exist.
pub fn vfs_readlink(path: &str, buf: &mut [u8]) -> SysResult<usize> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_READLINK,
            path.as_ptr() as usize,
            path.len(),
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as usize)
}

/// Create a hard link at `dst` that refers to the same file as `src`.
///
/// `src` and `dst` must be on the same filesystem (mount point).  Only
/// regular files may be hard-linked; attempting to link a directory or
/// symlink returns [`abi::errors::Errno::EPERM`].
///
/// Returns `Ok(())` on success, or an [`Errno`] on failure.
pub fn vfs_link(src: &str, dst: &str) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_LINK,
            src.as_ptr() as usize,
            src.len(),
            dst.as_ptr() as usize,
            dst.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

// ── Terminal I/O control (termios) ────────────────────────────────────────────

/// Query the termios settings for the terminal device on `fd`.
///
/// On success the current [`abi::termios::Termios`] is written into `termios`
/// and `Ok(())` is returned.  If `fd` is not a terminal device the kernel
/// returns [`abi::errors::Errno::ENOSYS`].
///
/// Equivalent to POSIX `tcgetattr(fd, termios)`.
pub fn tcgetattr(fd: u32, termios: &mut abi::termios::Termios) -> SysResult<()> {
    let size = core::mem::size_of::<abi::termios::Termios>();
    let call = abi::device::DeviceCall {
        kind: abi::device::DeviceKind::Terminal,
        op: abi::termios::TERMINAL_OP_TCGETS,
        in_ptr: 0,
        in_len: 0,
        out_ptr: termios as *mut abi::termios::Termios as u64,
        out_len: size as u32,
    };
    vfs_device_call_raw(fd, &call).map(|_| ())
}

/// Set the termios settings for the terminal device on `fd`.
///
/// The new settings are applied immediately (equivalent to `TCSANOW`).
/// Returns `Ok(())` on success; [`abi::errors::Errno::ENOSYS`] if `fd` is not
/// a terminal device.
///
/// Equivalent to POSIX `tcsetattr(fd, TCSANOW, termios)`.
pub fn tcsetattr(fd: u32, termios: &abi::termios::Termios) -> SysResult<()> {
    let size = core::mem::size_of::<abi::termios::Termios>();
    let call = abi::device::DeviceCall {
        kind: abi::device::DeviceKind::Terminal,
        op: abi::termios::TERMINAL_OP_TCSETS,
        in_ptr: termios as *const abi::termios::Termios as u64,
        in_len: size as u32,
        out_ptr: 0,
        out_len: 0,
    };
    vfs_device_call_raw(fd, &call).map(|_| ())
}

/// Query the foreground process group ID for the controlling terminal on `fd`.
///
/// Equivalent to POSIX `tcgetpgrp(fd)`.
pub fn tcgetpgrp(fd: u32) -> SysResult<u32> {
    let mut pgid: u32 = 0;
    let size = core::mem::size_of::<u32>();
    let call = abi::device::DeviceCall {
        kind: abi::device::DeviceKind::Terminal,
        op: abi::termios::TERMINAL_OP_TCGETPGRP,
        in_ptr: 0,
        in_len: 0,
        out_ptr: &mut pgid as *mut u32 as u64,
        out_len: size as u32,
    };
    vfs_device_call_raw(fd, &call).map(|_| pgid)
}

/// Set the foreground process group ID for the controlling terminal on `fd`.
///
/// Equivalent to POSIX `tcsetpgrp(fd, pgrp)`.
pub fn tcsetpgrp(fd: u32, pgrp: u32) -> SysResult<()> {
    let size = core::mem::size_of::<u32>();
    let call = abi::device::DeviceCall {
        kind: abi::device::DeviceKind::Terminal,
        op: abi::termios::TERMINAL_OP_TCSETPGRP,
        in_ptr: &pgrp as *const u32 as u64,
        in_len: size as u32,
        out_ptr: 0,
        out_len: 0,
    };
    vfs_device_call_raw(fd, &call).map(|_| ())
}

// ── chmod / fchmod ────────────────────────────────────────────────────────────

/// Change the permission bits of the file at `path` (chmod).
///
/// `mode` contains the lower 12 bits of the POSIX permission mask (`0o7777`);
/// the file-type bits are ignored by the kernel.
///
/// Returns `Ok(())` on success, [`Errno::ENOTSUP`] if the filesystem does
/// not support permission mutation, or another errno on failure.
pub fn vfs_chmod(path: &str, mode: u32) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_CHMOD,
            path.as_ptr() as usize,
            path.len(),
            (mode & 0o7777) as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Change the permission bits of the file associated with `fd` (fchmod).
///
/// `mode` contains the lower 12 bits of the POSIX permission mask (`0o7777`).
///
/// Returns `Ok(())` on success, [`Errno::ENOTSUP`] if the filesystem does
/// not support permission mutation, or another errno on failure.
pub fn vfs_fchmod(fd: u32, mode: u32) -> SysResult<()> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_FCHMOD,
            fd as usize,
            (mode & 0o7777) as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

// ── utimes / futimes ──────────────────────────────────────────────────────────

/// Set the access and/or modification timestamps of the file at `path`.
///
/// Pass `Some(ts)` to update a timestamp, or `None` to leave it unchanged.
///
/// Returns `Ok(())` on success, [`Errno::ENOTSUP`] if the filesystem does
/// not support timestamp mutation, or another errno on failure.
pub fn vfs_utimes(
    path: &str,
    atime: Option<abi::fs::Timespec>,
    mtime: Option<abi::fs::Timespec>,
) -> SysResult<()> {
    let req = abi::fs::UtimesRequest {
        atime_sec: atime.map_or(abi::fs::UtimesRequest::OMIT, |t| t.sec),
        atime_nsec: atime.map_or(0, |t| t.nsec),
        _pad1: 0,
        mtime_sec: mtime.map_or(abi::fs::UtimesRequest::OMIT, |t| t.sec),
        mtime_nsec: mtime.map_or(0, |t| t.nsec),
        _pad2: 0,
    };
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_UTIMES,
            path.as_ptr() as usize,
            path.len(),
            &req as *const abi::fs::UtimesRequest as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Set the access and/or modification timestamps of the file associated with `fd`.
///
/// Pass `Some(ts)` to update a timestamp, or `None` to leave it unchanged.
///
/// Returns `Ok(())` on success, [`Errno::ENOTSUP`] if the filesystem does
/// not support timestamp mutation, or another errno on failure.
pub fn vfs_futimes(
    fd: u32,
    atime: Option<abi::fs::Timespec>,
    mtime: Option<abi::fs::Timespec>,
) -> SysResult<()> {
    let req = abi::fs::UtimesRequest {
        atime_sec: atime.map_or(abi::fs::UtimesRequest::OMIT, |t| t.sec),
        atime_nsec: atime.map_or(0, |t| t.nsec),
        _pad1: 0,
        mtime_sec: mtime.map_or(abi::fs::UtimesRequest::OMIT, |t| t.sec),
        mtime_nsec: mtime.map_or(0, |t| t.nsec),
        _pad2: 0,
    };
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_FUTIMES,
            fd as usize,
            &req as *const abi::fs::UtimesRequest as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

// ── flock ─────────────────────────────────────────────────────────────────────

/// Apply or remove an advisory file lock on the open file `fd`.
///
/// `how` is a combination of [`abi::syscall::flock_flags`] constants:
/// * `LOCK_SH` (1) — acquire a shared (read) lock
/// * `LOCK_EX` (2) — acquire an exclusive (write) lock
/// * `LOCK_NB` (4) — non-blocking; return `EWOULDBLOCK` instead of blocking
/// * `LOCK_UN` (8) — release any lock held on the file
///
/// Returns `Ok(())` on success, [`abi::errors::Errno::EWOULDBLOCK`] if the
/// lock is held and `LOCK_NB` was specified, or another errno on failure.
pub fn vfs_flock(fd: u32, how: u32) -> SysResult<()> {
    let ret = unsafe { raw_syscall6(SYS_FS_FLOCK, fd as usize, how as usize, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}
