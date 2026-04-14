use crate::sched;
use crate::syscall::validate::{copyin, copyout, validate_user_range};
use abi::errors::{Errno, SysResult};
use alloc::vec;

/// Read from an open file descriptor.
///
/// Works for all fds including 0 (stdin), 1 (stdout), 2 (stderr) which are
/// now backed by `VfsNode` entries in the per-process `FdTable`.
pub fn sys_read(fd: usize, buf_ptr: usize, buf_len: usize) -> SysResult<usize> {
    validate_user_range(buf_ptr, buf_len, true)?;
    if buf_len == 0 {
        return Ok(0);
    }

    // Clone the node Arc and shared offset so we don't hold the process lock
    // during the (potentially blocking) read.
    let (node, offset_cell, status_flags) = {
        let pinfo_arc = sched::process_info_current().ok_or(Errno::ENOENT)?;
        let lock = pinfo_arc.lock();
        let file = lock.fd_table.get(fd as u32)?;
        let status_flags = *file.status_flags.lock();
        if !status_flags.is_readable() {
            return Err(Errno::EBADF);
        }
        (file.node.clone(), file.offset.clone(), status_flags)
    };

    if status_flags.read_would_block(node.poll()) {
        return Err(Errno::EAGAIN);
    }

    let offset = *offset_cell.lock();
    let mut kbuf = vec![0u8; buf_len];
    let n = node.read(offset, &mut kbuf)?;

    if n > 0 {
        *offset_cell.lock() = offset.saturating_add(n as u64);
    }

    unsafe { copyout(buf_ptr, &kbuf[..n])? };
    Ok(n)
}

/// Write to an open file descriptor.
///
/// Works for all fds including 0 (stdin), 1 (stdout), 2 (stderr).
pub fn sys_write(fd: usize, buf_ptr: usize, buf_len: usize) -> SysResult<usize> {
    validate_user_range(buf_ptr, buf_len, false)?;
    if buf_len == 0 {
        return Ok(0);
    }

    let mut kbuf = vec![0u8; buf_len];
    unsafe { copyin(&mut kbuf, buf_ptr)? };

    let (node, offset_cell, status_flags) = {
        let pinfo_arc = sched::process_info_current().ok_or(Errno::ENOENT)?;
        let lock = pinfo_arc.lock();
        let file = lock.fd_table.get(fd as u32)?;
        let status_flags = *file.status_flags.lock();
        if !status_flags.is_writable() {
            return Err(Errno::EBADF);
        }
        (file.node.clone(), file.offset.clone(), status_flags)
    };

    if status_flags.write_would_block(node.poll()) {
        return Err(Errno::EAGAIN);
    }

    let write_offset = status_flags.effective_write_offset(*offset_cell.lock(), node.stat()?.size);
    let n = node.write(write_offset, &kbuf)?;

    if n > 0 {
        *offset_cell.lock() = write_offset.saturating_add(n as u64);
    }

    Ok(n)
}
