//! Port IPC syscalls

use super::{copyin, copyout};
use crate::syscall::validate::validate_user_range;
use abi::errors::{Errno, SysResult};
use alloc::sync::Arc;

// Lines 7-9 are duplicates of 3-5

pub fn sys_channel_create(capacity: usize) -> SysResult<usize> {
    let capacity = capacity.min(65536).max(64);
    let port_id = crate::ipc::create_port(capacity);

    let mut table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
    let write_handle = table
        .alloc(port_id, crate::ipc::HandleMode::Write)
        .ok_or(Errno::ENOMEM)?;
    let read_handle = table
        .alloc(port_id, crate::ipc::HandleMode::Read)
        .ok_or(Errno::ENOMEM)?;

    let packed = ((write_handle.0 as usize) << 16) | (read_handle.0 as usize);
    Ok(packed)
}

pub fn sys_channel_send(handle: usize, ptr: usize, len: usize) -> SysResult<usize> {
    let len = len.min(4096);
    if len == 0 {
        return Ok(0);
    }

    validate_user_range(ptr, len, false)?;

    let handle = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(handle, crate::ipc::HandleMode::Write)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    if !port.has_readers() {
        crate::ipc::diag::CHANNEL_PEER_DEATHS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        return Err(Errno::EPIPE);
    }

    let mut buf = [0u8; 4096];
    unsafe {
        copyin(&mut buf[..len], ptr)?;
    }

    let written = port.send(&buf[..len]);
    if written > 0 {
        crate::ipc::diag::CHANNEL_SENDS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        crate::ipc::diag::CHANNEL_BYTES_SENT
            .fetch_add(written as u64, core::sync::atomic::Ordering::Relaxed);
    }
    Ok(written)
}

pub fn sys_channel_send_all(handle: usize, ptr: usize, len: usize) -> SysResult<usize> {
    let len = len.min(4096);
    if len == 0 {
        return Ok(0);
    }

    validate_user_range(ptr, len, false)?;

    let handle = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(handle, crate::ipc::HandleMode::Write)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    if !port.has_readers() {
        crate::ipc::diag::CHANNEL_PEER_DEATHS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        return Err(Errno::EPIPE);
    }

    let mut buf = [0u8; 4096];
    unsafe {
        copyin(&mut buf[..len], ptr)?;
    }

    if port.send_all(&buf[..len]) {
        crate::ktrace!(
            "sys_channel_send_all: wrote {} bytes to port {}",
            len,
            entry.port_id.0
        );
        crate::ipc::diag::CHANNEL_SENDS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        crate::ipc::diag::CHANNEL_BYTES_SENT
            .fetch_add(len as u64, core::sync::atomic::Ordering::Relaxed);
        Ok(len)
    } else {
        crate::ktrace!(
            "sys_channel_send_all: port {} FULL, returning EAGAIN",
            entry.port_id.0
        );
        crate::ipc::diag::CHANNEL_FULL_EVENTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Err(Errno::EAGAIN)
    }
}

fn sys_channel_recv_impl(
    handle: usize,
    ptr: usize,
    len: usize,
    blocking: bool,
) -> SysResult<usize> {
    let len = len.min(4096);
    if len == 0 {
        return Ok(0);
    }

    validate_user_range(ptr, len, true)?;

    let handle = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(handle, crate::ipc::HandleMode::Read)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    let mut buf = [0u8; 4096];
    let tid = unsafe { crate::sched::current_tid_current() };

    loop {
        let read = port.try_recv(&mut buf[..len]);
        if read > 0 {
            unsafe {
                copyout(ptr, &buf[..read])?;
            }
            crate::ktrace!(
                "sys_channel_recv: read {} bytes from port {}",
                read,
                entry.port_id.0
            );
            crate::ipc::diag::CHANNEL_RECVS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            crate::ipc::diag::CHANNEL_BYTES_RECV
                .fetch_add(read as u64, core::sync::atomic::Ordering::Relaxed);
            return Ok(read);
        }

        if !port.has_writers() {
            crate::ipc::diag::CHANNEL_PEER_DEATHS
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            return Err(Errno::EPIPE);
        }

        if !blocking {
            return Err(Errno::EAGAIN);
        }

        port.add_waiter_read(tid);

        let read = port.try_recv(&mut buf[..len]);
        if read > 0 {
            port.remove_waiter_read(tid);
            unsafe {
                copyout(ptr, &buf[..read])?;
            }
            crate::ktrace!(
                "sys_channel_recv: read {} bytes from port {} after wait registration",
                read,
                entry.port_id.0
            );
            crate::ipc::diag::CHANNEL_RECVS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            crate::ipc::diag::CHANNEL_BYTES_RECV
                .fetch_add(read as u64, core::sync::atomic::Ordering::Relaxed);
            return Ok(read);
        }

        if !port.has_writers() {
            port.remove_waiter_read(tid);
            crate::ipc::diag::CHANNEL_PEER_DEATHS
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            return Err(Errno::EPIPE);
        }

        unsafe {
            crate::sched::block_current_erased();
        }
    }
}

pub fn sys_channel_recv(handle: usize, ptr: usize, len: usize) -> SysResult<usize> {
    sys_channel_recv_impl(handle, ptr, len, true)
}

pub fn sys_channel_try_recv(handle: usize, ptr: usize, len: usize) -> SysResult<usize> {
    sys_channel_recv_impl(handle, ptr, len, false)
}

pub fn sys_channel_close(handle: usize) -> SysResult<usize> {
    let handle = crate::ipc::Handle(handle as u32);
    let mut table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
    if let Some(entry) = table.close(handle) {
        drop(table);

        let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
        let destroy = match entry.mode {
            crate::ipc::HandleMode::Read => port.close_reader(),
            crate::ipc::HandleMode::Write => port.close_writer(),
        };
        if destroy {
            crate::ipc::close_port(entry.port_id);
        }
        Ok(0)
    } else {
        Err(Errno::EBADF)
    }
}

/// `SYS_CHANNEL_WAIT` — wait for one of the given handles to become ready.
///
/// # Deprecated
///
/// Prefer converting handles to FDs with `SYS_FD_FROM_HANDLE` and using
/// `SYS_FS_POLL` instead.  This syscall operates on raw IPC handle numbers
/// rather than file descriptors and is retained only for backward compatibility.
pub fn sys_channel_wait(handles_ptr: usize, count: usize, flags: usize) -> SysResult<usize> {
    if count == 0 || count > 64 {
        return Err(Errno::EINVAL);
    }
    validate_user_range(handles_ptr, count * 4, false)?;

    let mut handles = [0u32; 64];
    unsafe {
        let dest = core::slice::from_raw_parts_mut(handles.as_mut_ptr() as *mut u8, count * 4);
        copyin(dest, handles_ptr)?;
    }

    let tid = unsafe { crate::sched::current_tid_current() };
    let flags = flags as u32;

    // Cleanup helper to ensure we don't leave stale entries in any port's wait queue
    let cleanup = |handles: &[u32], table: &mut crate::ipc::HandleTable| {
        for &h in handles {
            let h_ipc = crate::ipc::Handle(h);
            if let Some(entry) = table.get(h_ipc, crate::ipc::HandleMode::Read) {
                if let Some(port) = crate::ipc::get_port(entry.port_id) {
                    port.remove_waiter_read(tid);
                }
            }
            if let Some(entry) = table.get(h_ipc, crate::ipc::HandleMode::Write) {
                if let Some(port) = crate::ipc::get_port(entry.port_id) {
                    port.remove_waiter_write(tid);
                }
            }
        }
    };

    loop {
        // 1. Register as waiter BEFORE checking (prevents race)
        {
            let mut table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
            for i in 0..count {
                let h = handles[i];
                let h_ipc = crate::ipc::Handle(h);
                if (flags & abi::syscall::channel_wait::READABLE) != 0 {
                    if let Some(entry) = table.get(h_ipc, crate::ipc::HandleMode::Read) {
                        if let Some(port) = crate::ipc::get_port(entry.port_id) {
                            port.add_waiter_read(tid);
                        }
                    }
                }
                if (flags & abi::syscall::channel_wait::WRITABLE) != 0 {
                    if let Some(entry) = table.get(h_ipc, crate::ipc::HandleMode::Write) {
                        if let Some(port) = crate::ipc::get_port(entry.port_id) {
                            port.add_waiter_write(tid);
                        }
                    }
                }
            }
        }

        // 2. Check if any port is ready
        {
            let mut table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
            for i in 0..count {
                let h = handles[i];
                let h_ipc = crate::ipc::Handle(h);
                if let Some(entry) = table.get(h_ipc, crate::ipc::HandleMode::Read) {
                    if (flags & abi::syscall::channel_wait::READABLE) != 0 {
                        if let Some(port) = crate::ipc::get_port(entry.port_id) {
                            if !port.is_empty() {
                                crate::ktrace!(
                                    "sys_port_wait: port {} is NOT empty, returning Ok({})",
                                    entry.port_id.0,
                                    h
                                );
                                cleanup(&handles[..count], &mut table);
                                return Ok(h as usize);
                            }
                        }
                    }
                }
                if let Some(entry) = table.get(h_ipc, crate::ipc::HandleMode::Write) {
                    if (flags & abi::syscall::channel_wait::WRITABLE) != 0 {
                        if let Some(port) = crate::ipc::get_port(entry.port_id) {
                            if !port.is_full() {
                                cleanup(&handles[..count], &mut table);
                                return Ok(h as usize);
                            }
                        }
                    }
                }
            }
        }

        // 3. Block current task
        unsafe {
            crate::ktrace!("sys_port_wait: blocking task {}", tid);
            crate::sched::block_current_erased();
            crate::ktrace!("sys_port_wait: unblocked task {}", tid);
        }
    }
}

pub fn sys_channel_info(handle: usize) -> SysResult<usize> {
    let handle = crate::ipc::Handle(handle as u32);
    let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
    let entry = table
        .get(handle, crate::ipc::HandleMode::Read)
        .or_else(|| table.get(handle, crate::ipc::HandleMode::Write))
        .ok_or(Errno::EBADF)?;

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;

    let len = port.len();
    let cap = port.capacity(); // Need to expose capacity

    // Return packed: top 32 bits capacity, bottom 32 bits length
    Ok((cap << 32) | (len & 0xFFFFFFFF))
}

/// `SYS_CHANNEL_SEND_HANDLE` — send a single capability over a channel handle.
///
/// # Deprecated
///
/// Prefer `SYS_CHANNEL_SEND_MSG` which bundles data bytes and FDs atomically
/// in a single message and operates without a separate capability queue.
pub fn sys_channel_send_handle(handle: usize, fd: usize) -> SysResult<usize> {
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;

    // First, try to get it as a VFS FD
    let vfs_node = {
        let lock = pinfo_arc.lock();
        if let Ok(file) = lock.fd_table.get(fd as u32) {
            Some(file.node.clone())
        } else {
            None
        }
    };

    let node_to_send = if let Some(node) = vfs_node {
        node
    } else {
        // Not a VFS FD, try to get it as an IPC handle
        let h = crate::ipc::Handle(fd as u32);
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        // We look for either Read or Write mode
        let entry = table
            .get(h, crate::ipc::HandleMode::Write)
            .or_else(|| table.get(h, crate::ipc::HandleMode::Read))
            .ok_or(Errno::EBADF)?; // If not in either table, it's a bad handle

        let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
        Arc::new(crate::vfs::port_node::PortNode::new(port, entry.mode))
    };

    let handle = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(handle, crate::ipc::HandleMode::Write)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    crate::kinfo!(
        "kernel: sys_channel_send_handle handle={} port_id={:?}",
        handle.0,
        entry.port_id
    );

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    // Compatibility wrapper: send as a handle-only message (no data bytes).
    port.send_cap(node_to_send);
    crate::ipc::diag::CHANNEL_HANDLES_SENT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Ok(0)
}

/// `SYS_CHANNEL_RECV_HANDLE` — receive a single capability from a channel handle.
///
/// # Deprecated
///
/// Prefer `SYS_CHANNEL_RECV_MSG` which receives data bytes and FDs atomically.
pub fn sys_channel_recv_handle(handle: usize, out_fd_ptr: usize) -> SysResult<usize> {
    validate_user_range(out_fd_ptr, 4, true)?;

    let handle = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(handle, crate::ipc::HandleMode::Read)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    // Compatibility wrapper: receive the first cap from the message queue.
    let cap = port.try_recv_cap().ok_or(Errno::EAGAIN)?;

    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let new_fd = pinfo_arc.lock().fd_table.open(
        cap,
        crate::vfs::OpenFlags::read_write(),
        "recv_handle".into(),
    )?;

    let new_fd_bytes = new_fd.to_ne_bytes();
    unsafe { super::copyout(out_fd_ptr, &new_fd_bytes)? };

    crate::ipc::diag::CHANNEL_HANDLES_RECV.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Ok(0)
}

/// Resolve a userspace handle/fd number to a `VfsNode`.
///
/// Tries FD table first, then IPC handle table (as a `PortNode` wrapper).
fn resolve_handle_to_node(
    pinfo_arc: &Arc<spin::Mutex<crate::task::ProcessInfo>>,
    raw: u32,
) -> SysResult<Arc<dyn crate::vfs::VfsNode>> {
    // Try VFS fd table
    {
        let lock = pinfo_arc.lock();
        if let Ok(file) = lock.fd_table.get(raw) {
            return Ok(file.node.clone());
        }
    }
    // Try IPC handle table
    let h = crate::ipc::Handle(raw);
    let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
    let entry = table
        .get(h, crate::ipc::HandleMode::Write)
        .or_else(|| table.get(h, crate::ipc::HandleMode::Read))
        .ok_or(Errno::EBADF)?;
    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    Ok(Arc::new(crate::vfs::port_node::PortNode::new(
        port,
        entry.mode,
    )))
}

/// `SYS_CHANNEL_SEND_MSG` — send a message with attached handles over a channel.
///
/// Syscall args:
///   0: channel write handle
///   1: data pointer (may be null when data_len == 0)
///   2: data length (capped at 4096)
///   3: handles pointer — array of u32 fd/handle numbers (may be null when count == 0)
///   4: handles count (capped at 64)
///   5: (reserved, must be 0)
///
/// Transfer semantics: **duplicate** — the sender retains its own fd/handle;
/// a clone of the underlying `Arc<dyn VfsNode>` is placed in the message.
pub fn sys_channel_send_msg(
    handle: usize,
    data_ptr: usize,
    data_len: usize,
    handles_ptr: usize,
    handles_count: usize,
) -> SysResult<usize> {
    const MAX_MSG_DATA: usize = 4096;
    const MAX_MSG_HANDLES: usize = 64;

    // Enforce hard limits rather than silently truncating.
    if handles_count > MAX_MSG_HANDLES {
        return Err(Errno::EINVAL);
    }
    let data_len = data_len.min(MAX_MSG_DATA);

    // Validate userspace ranges up-front
    if data_len > 0 {
        validate_user_range(data_ptr, data_len, false)?;
    }
    if handles_count > 0 {
        validate_user_range(handles_ptr, handles_count * 4, false)?;
    }

    // Read data bytes
    let mut data_buf = alloc::vec![0u8; data_len];
    if data_len > 0 {
        unsafe {
            super::copyin(&mut data_buf, data_ptr)?;
        }
    }

    // Read handle numbers and resolve to VFS nodes
    let mut handle_nums = alloc::vec![0u32; handles_count];
    if handles_count > 0 {
        unsafe {
            super::copyin(
                core::slice::from_raw_parts_mut(handle_nums.as_mut_ptr() as *mut u8, handles_count * 4),
                handles_ptr,
            )?;
        }
    }

    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let mut caps = alloc::vec::Vec::with_capacity(handles_count);
    for &raw in &handle_nums {
        let node = resolve_handle_to_node(&pinfo_arc, raw)?;
        caps.push(node);
    }

    // Look up the destination channel
    let ch = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(ch, crate::ipc::HandleMode::Write)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    if !port.has_readers() {
        crate::ipc::diag::CHANNEL_PEER_DEATHS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        return Err(Errno::EPIPE);
    }

    port.send_msg(data_buf, caps);

    if handles_count > 0 {
        crate::ipc::diag::CHANNEL_HANDLES_SENT
            .fetch_add(handles_count as u64, core::sync::atomic::Ordering::Relaxed);
    }
    crate::ipc::diag::CHANNEL_SENDS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Ok(0)
}

/// `SYS_CHANNEL_RECV_MSG` — receive a message with attached handles from a channel.
///
/// Syscall args:
///   0: channel read handle
///   1: data buffer pointer
///   2: data buffer capacity (max bytes to copy)
///   3: handles buffer pointer — array of u32 (output FD numbers written here)
///   4: handles buffer capacity (max handles to install)
///   5: out_lens pointer — points to a `[usize; 2]` filled with
///        `[actual_data_len, actual_handles_count]`
///
/// Returns `EAGAIN` if the message queue is empty.
/// If the message data exceeds `data_cap` the excess is silently truncated.
/// If the message has more caps than `handles_cap` the excess are dropped.
pub fn sys_channel_recv_msg(
    handle: usize,
    data_ptr: usize,
    data_cap: usize,
    handles_ptr: usize,
    handles_cap: usize,
    out_lens_ptr: usize,
) -> SysResult<usize> {
    validate_user_range(out_lens_ptr, core::mem::size_of::<usize>() * 2, true)?;
    if data_cap > 0 {
        validate_user_range(data_ptr, data_cap, true)?;
    }
    if handles_cap > 0 {
        validate_user_range(handles_ptr, handles_cap * 4, true)?;
    }

    let ch = crate::ipc::Handle(handle as u32);
    let entry = {
        let table = crate::ipc::GLOBAL_HANDLE_TABLE.lock();
        table
            .get(ch, crate::ipc::HandleMode::Read)
            .copied()
            .ok_or(Errno::EBADF)?
    };

    let port = crate::ipc::get_port(entry.port_id).ok_or(Errno::EBADF)?;
    let msg = port.try_recv_msg().ok_or(Errno::EAGAIN)?;

    // Copy data bytes to userspace
    let copy_len = msg.data.len().min(data_cap);
    if copy_len > 0 {
        unsafe {
            super::copyout(data_ptr, &msg.data[..copy_len])?;
        }
    }

    // Install caps as FDs in the receiving process
    let pinfo_arc = crate::sched::process_info_current().ok_or(Errno::ENOENT)?;
    let install_count = msg.caps.len().min(handles_cap);
    let mut out_fds = alloc::vec![0u32; install_count];
    {
        let mut pinfo = pinfo_arc.lock();
        for (i, cap) in msg.caps.into_iter().take(install_count).enumerate() {
            let fd = pinfo.fd_table.open(cap, crate::vfs::OpenFlags::read_write(), "recv_msg".into())?;
            out_fds[i] = fd;
        }
    }

    // Write the installed FD numbers back to userspace
    if install_count > 0 {
        unsafe {
            super::copyout(
                handles_ptr,
                core::slice::from_raw_parts(out_fds.as_ptr() as *const u8, install_count * 4),
            )?;
        }
    }

    // Write [actual_data_len, actual_handles_count] to out_lens_ptr
    let out_lens: [usize; 2] = [copy_len, install_count];
    unsafe {
        super::copyout(
            out_lens_ptr,
            core::slice::from_raw_parts(
                out_lens.as_ptr() as *const u8,
                core::mem::size_of::<usize>() * 2,
            ),
        )?;
    }

    if install_count > 0 {
        crate::ipc::diag::CHANNEL_HANDLES_RECV
            .fetch_add(install_count as u64, core::sync::atomic::Ordering::Relaxed);
    }
    crate::ipc::diag::CHANNEL_RECVS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    Ok(0)
}
