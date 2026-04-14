//! Userland VFS provider channel — kernel side.
//!
//! When a userland process calls `SYS_FS_MOUNT`, the kernel instantiates a
//! [`ProviderFs`] and registers it in the global mount table.  From that point
//! on every VFS operation whose path falls under the mount point is serialised
//! into a [`VfsRpcOp`] message and forwarded to the provider process via the
//! IPC port it supplied.

use alloc::collections::BTreeMap;
use alloc::sync::{Arc, Weak};
use alloc::vec;
use spin::Mutex;

use crate::sched::wait_queue::WaitQueue;
use crate::syscall::validate::{copyin, copyout};
use abi::{
    errors::{Errno, SysResult},
    vfs_rpc::{VFS_RPC_MAX_DATA, VFS_RPC_MAX_RESP, VfsRpcOp, VfsRpcReqHeader},
};

use super::{VfsDriver, VfsNode, VfsStat};

// ── ProviderChannel ──────────────────────────────────────────────────────────

/// Inner state protected by the serialisation lock.
struct ProviderChannel {
    /// The provider's request port (kernel → provider).
    req: Arc<crate::ipc::Port>,
    /// The kernel's private response port (provider → kernel).
    resp: Arc<crate::ipc::Port>,
    /// Write handle for the response port.  Sent to the provider in every
    /// request header so the provider knows where to send its reply.
    resp_write_handle: u32,
    /// Map of provider-assigned handles to kernel-side wait queues.
    waiters: BTreeMap<u64, Arc<WaitQueue>>,
}

impl ProviderChannel {
    /// Perform a blocking round-trip RPC with the provider.
    fn rpc(&self, op: VfsRpcOp, payload: &[u8]) -> SysResult<alloc::vec::Vec<u8>> {
        let hdr = VfsRpcReqHeader {
            resp_port: self.resp_write_handle,
            op: op as u8,
            _pad: [0, 0],
        };
        let hdr_size = core::mem::size_of::<VfsRpcReqHeader>();
        let mut msg = vec![0u8; hdr_size + payload.len()];
        unsafe {
            core::ptr::copy_nonoverlapping(
                &hdr as *const VfsRpcReqHeader as *const u8,
                msg.as_mut_ptr(),
                hdr_size,
            );
        }
        msg[hdr_size..].copy_from_slice(payload);

        crate::ipc::diag::VFS_RPC_REQUESTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        let written = self.req.send(&msg);
        if written < msg.len() {
            crate::ipc::diag::record_dead_provider_error();
            return Err(Errno::EIO);
        }

        let mut resp_buf = vec![0u8; VFS_RPC_MAX_RESP];
        let n = self.recv_response(&mut resp_buf)?;
        resp_buf.truncate(n);
        Ok(resp_buf)
    }

    fn recv_response(&self, buf: &mut [u8]) -> SysResult<usize> {
        let tid = unsafe { crate::sched::current_tid_current() };
        loop {
            let n = self.resp.try_recv(buf);
            if n > 0 {
                return Ok(n);
            }
            if !self.resp.has_writers() {
                crate::ipc::diag::record_dead_provider_error();
                return Err(Errno::EPIPE);
            }
            self.resp.add_waiter_read(tid);
            let n = self.resp.try_recv(buf);
            if n > 0 {
                self.resp.remove_waiter_read(tid);
                return Ok(n);
            }
            if !self.resp.has_writers() {
                self.resp.remove_waiter_read(tid);
                crate::ipc::diag::record_dead_provider_error();
                return Err(Errno::EPIPE);
            }
            unsafe {
                crate::sched::block_current_erased();
            }
        }
    }

    fn get_wait_queue(&mut self, handle: u64) -> Arc<WaitQueue> {
        self.waiters
            .entry(handle)
            .or_insert_with(|| Arc::new(WaitQueue::new()))
            .clone()
    }
}

// ── ProviderFs ───────────────────────────────────────────────────────────────

/// A [`VfsDriver`] that forwards all operations to a userland provider via IPC.
pub struct ProviderFs {
    channel: Mutex<ProviderChannel>,
}

static PROVIDER_MAP: Mutex<BTreeMap<u32, Weak<ProviderFs>>> = Mutex::new(BTreeMap::new());

impl ProviderFs {
    pub fn new(
        req_port: Arc<crate::ipc::Port>,
        resp_port: Arc<crate::ipc::Port>,
        resp_write_handle: u32,
        req_port_id: u32,
    ) -> Arc<Self> {
        let this = Arc::new(Self {
            channel: Mutex::new(ProviderChannel {
                req: req_port,
                resp: resp_port,
                resp_write_handle,
                waiters: BTreeMap::new(),
            }),
        });
        PROVIDER_MAP
            .lock()
            .insert(req_port_id, Arc::downgrade(&this));
        this
    }

    pub fn notify(&self, handle: u64, _revents: u16) {
        let mut chan = self.channel.lock();
        if let Some(wq) = chan.waiters.get(&handle) {
            wq.wake_all();
        }
    }
}

impl VfsDriver for ProviderFs {
    fn lookup(&self, path: &str) -> SysResult<Arc<dyn VfsNode>> {
        let path_bytes = path.as_bytes();
        let path_len = path_bytes.len() as u32;

        let mut payload = vec![0u8; 4 + path_bytes.len()];
        payload[..4].copy_from_slice(&path_len.to_le_bytes());
        payload[4..].copy_from_slice(path_bytes);

        let resp = self.channel.lock().rpc(VfsRpcOp::Lookup, &payload)?;
        parse_response_handle(&resp).map(|handle| {
            let wq = self.channel.lock().get_wait_queue(handle);
            Arc::new(ProviderNode {
                handle,
                channel: Arc::new(Mutex::new(ProviderChannelRef {
                    req: self.channel.lock().req.clone(),
                    resp: self.channel.lock().resp.clone(),
                    resp_write_handle: self.channel.lock().resp_write_handle,
                })),
                wait_queue: wq,
            }) as Arc<dyn VfsNode>
        })
    }

    fn rename(&self, old_path: &str, new_path: &str) -> SysResult<()> {
        let old_bytes = old_path.as_bytes();
        let new_bytes = new_path.as_bytes();
        let mut payload = vec![0u8; 4 + old_bytes.len() + 4 + new_bytes.len()];
        payload[0..4].copy_from_slice(&(old_bytes.len() as u32).to_le_bytes());
        payload[4..4 + old_bytes.len()].copy_from_slice(old_bytes);
        let off = 4 + old_bytes.len();
        payload[off..off + 4].copy_from_slice(&(new_bytes.len() as u32).to_le_bytes());
        payload[off + 4..].copy_from_slice(new_bytes);

        let resp = self.channel.lock().rpc(VfsRpcOp::Rename, &payload)?;
        if resp.is_empty() {
            return Err(Errno::EIO);
        }
        if resp[0] != 0 {
            return Err(errno_from_u8(resp[0]));
        }
        Ok(())
    }
}

fn parse_response_handle(resp: &[u8]) -> SysResult<u64> {
    if resp.is_empty() {
        return Err(Errno::EIO);
    }
    if resp[0] != 0 {
        return Err(errno_from_u8(resp[0]));
    }
    if resp.len() < 9 {
        return Err(Errno::EIO);
    }
    Ok(u64::from_le_bytes([
        resp[1], resp[2], resp[3], resp[4], resp[5], resp[6], resp[7], resp[8],
    ]))
}

// ── ProviderChannelRef ────────────────────────────────────────────────────────

struct ProviderChannelRef {
    req: Arc<crate::ipc::Port>,
    resp: Arc<crate::ipc::Port>,
    resp_write_handle: u32,
}

impl ProviderChannelRef {
    fn rpc(&self, op: VfsRpcOp, payload: &[u8]) -> SysResult<alloc::vec::Vec<u8>> {
        let hdr = VfsRpcReqHeader {
            resp_port: self.resp_write_handle,
            op: op as u8,
            _pad: [0, 0],
        };
        let hdr_size = core::mem::size_of::<VfsRpcReqHeader>();
        let mut msg = vec![0u8; hdr_size + payload.len()];
        unsafe {
            core::ptr::copy_nonoverlapping(
                &hdr as *const _ as *const u8,
                msg.as_mut_ptr(),
                hdr_size,
            );
        }
        msg[hdr_size..].copy_from_slice(payload);

        crate::ipc::diag::VFS_RPC_REQUESTS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        let written = self.req.send(&msg);
        if written < msg.len() {
            crate::ipc::diag::record_dead_provider_error();
            return Err(Errno::EIO);
        }

        let mut resp_buf = vec![0u8; VFS_RPC_MAX_RESP];
        let n = self.recv_response(&mut resp_buf)?;
        resp_buf.truncate(n);
        Ok(resp_buf)
    }

    fn recv_response(&self, buf: &mut [u8]) -> SysResult<usize> {
        let tid = unsafe { crate::sched::current_tid_current() };
        loop {
            let n = self.resp.try_recv(buf);
            if n > 0 {
                return Ok(n);
            }
            if !self.resp.has_writers() {
                crate::ipc::diag::record_dead_provider_error();
                return Err(Errno::EPIPE);
            }
            self.resp.add_waiter_read(tid);
            let n = self.resp.try_recv(buf);
            if n > 0 {
                self.resp.remove_waiter_read(tid);
                return Ok(n);
            }
            if !self.resp.has_writers() {
                self.resp.remove_waiter_read(tid);
                crate::ipc::diag::record_dead_provider_error();
                return Err(Errno::EPIPE);
            }
            unsafe {
                crate::sched::block_current_erased();
            }
        }
    }
}

// ── ProviderNode ─────────────────────────────────────────────────────────────

pub struct ProviderNode {
    handle: u64,
    channel: Arc<Mutex<ProviderChannelRef>>,
    wait_queue: Arc<WaitQueue>,
}

unsafe impl Send for ProviderNode {}
unsafe impl Sync for ProviderNode {}

impl VfsNode for ProviderNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let len = buf.len().min(abi::vfs_rpc::VFS_RPC_MAX_DATA) as u32;
        let mut payload = [0u8; 20];
        payload[..8].copy_from_slice(&self.handle.to_le_bytes());
        payload[8..16].copy_from_slice(&offset.to_le_bytes());
        payload[16..20].copy_from_slice(&len.to_le_bytes());
        let resp = self.channel.lock().rpc(VfsRpcOp::Read, &payload)?;
        parse_response_read(&resp, buf)
    }

    fn write(&self, offset: u64, data: &[u8]) -> SysResult<usize> {
        let data_len = data.len().min(abi::vfs_rpc::VFS_RPC_MAX_DATA) as u32;
        let mut payload = vec![0u8; 20 + data_len as usize];
        payload[..8].copy_from_slice(&self.handle.to_le_bytes());
        payload[8..16].copy_from_slice(&offset.to_le_bytes());
        payload[16..20].copy_from_slice(&data_len.to_le_bytes());
        payload[20..].copy_from_slice(&data[..data_len as usize]);
        let resp = self.channel.lock().rpc(VfsRpcOp::Write, &payload)?;
        parse_response_u32(&resp).map(|n| n as usize)
    }

    fn stat(&self) -> SysResult<VfsStat> {
        let payload = self.handle.to_le_bytes();
        let resp = self.channel.lock().rpc(VfsRpcOp::Stat, &payload)?;
        parse_response_stat(&resp)
    }

    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let len = buf.len().min(abi::vfs_rpc::VFS_RPC_MAX_DATA) as u32;
        let mut payload = [0u8; 20];
        payload[..8].copy_from_slice(&self.handle.to_le_bytes());
        payload[8..16].copy_from_slice(&offset.to_le_bytes());
        payload[16..20].copy_from_slice(&len.to_le_bytes());
        let resp = self.channel.lock().rpc(VfsRpcOp::Readdir, &payload)?;
        parse_response_read(&resp, buf)
    }

    fn close(&self) {
        let payload = self.handle.to_le_bytes();
        let _ = self.channel.lock().rpc(VfsRpcOp::Close, &payload);
    }

    fn device_call(&self, call: &abi::device::DeviceCall) -> SysResult<usize> {
        let in_len = call.in_len as usize;
        let out_len = call.out_len as usize;
        if in_len > VFS_RPC_MAX_DATA || out_len > VFS_RPC_MAX_DATA {
            return Err(Errno::EINVAL);
        }
        let mut payload = vec![0u8; 8 + core::mem::size_of::<abi::device::DeviceCall>() + in_len];
        payload[..8].copy_from_slice(&self.handle.to_le_bytes());
        unsafe {
            core::ptr::copy_nonoverlapping(
                call as *const _ as *const u8,
                payload[8..].as_mut_ptr(),
                core::mem::size_of::<abi::device::DeviceCall>(),
            );
        }
        if in_len > 0 {
            unsafe {
                copyin(
                    &mut payload[8 + core::mem::size_of::<abi::device::DeviceCall>()..],
                    call.in_ptr as usize,
                )?;
            }
        }
        let resp = self.channel.lock().rpc(VfsRpcOp::DeviceCall, &payload)?;
        if resp.is_empty() {
            return Err(Errno::EIO);
        }
        if resp[0] != 0 {
            return Err(errno_from_u8(resp[0]));
        }
        if resp.len() < 9 {
            return Err(Errno::EIO);
        }
        let ret_val = u32::from_le_bytes([resp[1], resp[2], resp[3], resp[4]]);
        let actual_out_len = u32::from_le_bytes([resp[5], resp[6], resp[7], resp[8]]) as usize;
        if actual_out_len > 0 && out_len > 0 {
            let copy_n = actual_out_len.min(out_len).min(resp.len() - 9);
            unsafe {
                copyout(call.out_ptr as usize, &resp[9..9 + copy_n])?;
            }
        }
        Ok(ret_val as usize)
    }

    fn poll(&self) -> u16 {
        let mut payload = [0u8; 12];
        payload[..8].copy_from_slice(&self.handle.to_le_bytes());
        let events = abi::syscall::poll_flags::POLLIN | abi::syscall::poll_flags::POLLOUT;
        payload[8..12].copy_from_slice(&(events as u32).to_le_bytes());
        let resp = self.channel.lock().rpc(VfsRpcOp::Poll, &payload);
        match resp {
            Ok(r) => parse_response_u32(&r).unwrap_or(0) as u16,
            Err(_) => abi::syscall::poll_flags::POLLERR,
        }
    }

    fn add_waiter(&self, tid: u64) {
        self.wait_queue.push_back(tid);
        let mut payload = [0u8; 12];
        payload[..8].copy_from_slice(&self.handle.to_le_bytes());
        let events = abi::syscall::poll_flags::POLLIN | abi::syscall::poll_flags::POLLOUT;
        payload[8..12].copy_from_slice(&(events as u32).to_le_bytes());
        let _ = self.channel.lock().rpc(VfsRpcOp::SubscribeReady, &payload);
    }

    fn remove_waiter(&self, tid: u64) {
        self.wait_queue.remove(tid);
        if self.wait_queue.is_empty() {
            let payload = self.handle.to_le_bytes();
            let _ = self
                .channel
                .lock()
                .rpc(VfsRpcOp::UnsubscribeReady, &payload);
        }
    }
}

pub fn notify_by_port(port_id: u32, handle: u64, revents: u16) -> SysResult<()> {
    let weak = PROVIDER_MAP
        .lock()
        .get(&port_id)
        .cloned()
        .ok_or(Errno::ENOENT)?;
    if let Some(fs) = weak.upgrade() {
        fs.notify(handle, revents);
        Ok(())
    } else {
        Err(Errno::ENOENT)
    }
}

fn parse_response_read(resp: &[u8], buf: &mut [u8]) -> SysResult<usize> {
    if resp.is_empty() {
        return Err(Errno::EIO);
    }
    if resp[0] != 0 {
        return Err(errno_from_u8(resp[0]));
    }
    if resp.len() < 5 {
        return Err(Errno::EIO);
    }
    let n = u32::from_le_bytes([resp[1], resp[2], resp[3], resp[4]]) as usize;
    let data = &resp[5..];
    let copy_n = n.min(data.len()).min(buf.len());
    buf[..copy_n].copy_from_slice(&data[..copy_n]);
    Ok(copy_n)
}

fn parse_response_u32(resp: &[u8]) -> SysResult<u32> {
    if resp.is_empty() {
        return Err(Errno::EIO);
    }
    if resp[0] != 0 {
        return Err(errno_from_u8(resp[0]));
    }
    if resp.len() < 5 {
        return Err(Errno::EIO);
    }
    Ok(u32::from_le_bytes([resp[1], resp[2], resp[3], resp[4]]))
}

fn parse_response_stat(resp: &[u8]) -> SysResult<VfsStat> {
    if resp.is_empty() {
        return Err(Errno::EIO);
    }
    if resp[0] != 0 {
        return Err(errno_from_u8(resp[0]));
    }
    if resp.len() < 21 {
        return Err(Errno::EIO);
    }
    let mode = u32::from_le_bytes([resp[1], resp[2], resp[3], resp[4]]);
    let size = u64::from_le_bytes([
        resp[5], resp[6], resp[7], resp[8], resp[9], resp[10], resp[11], resp[12],
    ]);
    let ino = u64::from_le_bytes([
        resp[13], resp[14], resp[15], resp[16], resp[17], resp[18], resp[19], resp[20],
    ]);
    Ok(VfsStat {
        mode,
        size,
        ino,
        ..Default::default()
    })
}

fn errno_from_u8(v: u8) -> Errno {
    match v {
        1 => Errno::EPERM,
        2 => Errno::ENOENT,
        5 => Errno::EIO,
        9 => Errno::EBADF,
        11 => Errno::EAGAIN,
        12 => Errno::ENOMEM,
        13 => Errno::EACCES,
        17 => Errno::EEXIST,
        20 => Errno::ENOTDIR,
        21 => Errno::EISDIR,
        22 => Errno::EINVAL,
        28 => Errno::ENOSPC,
        32 => Errno::EPIPE,
        38 => Errno::ENOSYS,
        _ => Errno::EIO,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use alloc::vec;

    fn make_port(cap: usize) -> Arc<crate::ipc::Port> {
        Arc::new(crate::ipc::Port::new(cap))
    }

    // ── errno_from_u8 ────────────────────────────────────────────────────────

    #[test]
    fn errno_from_u8_known_values() {
        assert!(matches!(errno_from_u8(1), Errno::EPERM));
        assert!(matches!(errno_from_u8(2), Errno::ENOENT));
        assert!(matches!(errno_from_u8(5), Errno::EIO));
        assert!(matches!(errno_from_u8(9), Errno::EBADF));
        assert!(matches!(errno_from_u8(11), Errno::EAGAIN));
        assert!(matches!(errno_from_u8(12), Errno::ENOMEM));
        assert!(matches!(errno_from_u8(13), Errno::EACCES));
        assert!(matches!(errno_from_u8(17), Errno::EEXIST));
        assert!(matches!(errno_from_u8(20), Errno::ENOTDIR));
        assert!(matches!(errno_from_u8(21), Errno::EISDIR));
        assert!(matches!(errno_from_u8(22), Errno::EINVAL));
        assert!(matches!(errno_from_u8(28), Errno::ENOSPC));
        assert!(matches!(errno_from_u8(32), Errno::EPIPE));
        assert!(matches!(errno_from_u8(38), Errno::ENOSYS));
    }

    #[test]
    fn errno_from_u8_unknown_falls_back_to_eio() {
        assert!(matches!(errno_from_u8(200), Errno::EIO));
        assert!(matches!(errno_from_u8(255), Errno::EIO));
        assert!(matches!(errno_from_u8(0), Errno::EIO)); // 0 is "OK", not an errno
    }

    // ── parse_response_handle ────────────────────────────────────────────────

    #[test]
    fn parse_response_handle_ok() {
        let mut resp = vec![0u8; 9];
        resp[0] = 0; // status OK
        resp[1..9].copy_from_slice(&42u64.to_le_bytes());
        assert_eq!(parse_response_handle(&resp).unwrap(), 42u64);
    }

    #[test]
    fn parse_response_handle_error_enoent() {
        let resp = vec![2u8]; // ENOENT
        assert!(matches!(parse_response_handle(&resp), Err(Errno::ENOENT)));
    }

    #[test]
    fn parse_response_handle_too_short() {
        // Status is OK (0) but only 2 payload bytes — need 8
        let resp = vec![0u8, 0u8, 1u8];
        assert!(matches!(parse_response_handle(&resp), Err(Errno::EIO)));
    }

    #[test]
    fn parse_response_handle_empty() {
        assert!(matches!(parse_response_handle(&[]), Err(Errno::EIO)));
    }

    // ── parse_response_u32 ───────────────────────────────────────────────────

    #[test]
    fn parse_response_u32_ok() {
        let mut resp = vec![0u8; 5];
        resp[0] = 0;
        resp[1..5].copy_from_slice(&1024u32.to_le_bytes());
        assert_eq!(parse_response_u32(&resp).unwrap(), 1024u32);
    }

    #[test]
    fn parse_response_u32_error_eagain() {
        let resp = vec![11u8]; // EAGAIN
        assert!(matches!(parse_response_u32(&resp), Err(Errno::EAGAIN)));
    }

    #[test]
    fn parse_response_u32_empty() {
        assert!(matches!(parse_response_u32(&[]), Err(Errno::EIO)));
    }

    #[test]
    fn parse_response_u32_too_short() {
        let resp = vec![0u8, 1u8]; // status OK, only 1 byte payload
        assert!(matches!(parse_response_u32(&resp), Err(Errno::EIO)));
    }

    // ── parse_response_stat ──────────────────────────────────────────────────

    #[test]
    fn parse_response_stat_ok() {
        let mut resp = vec![0u8; 21]; // 1 (status) + 4 (mode) + 8 (size) + 8 (ino)
        resp[0] = 0;
        resp[1..5].copy_from_slice(&0o100644u32.to_le_bytes());
        resp[5..13].copy_from_slice(&4096u64.to_le_bytes());
        resp[13..21].copy_from_slice(&7u64.to_le_bytes());
        let stat = parse_response_stat(&resp).unwrap();
        assert_eq!(stat.mode, 0o100644);
        assert_eq!(stat.size, 4096);
        assert_eq!(stat.ino, 7);
    }

    #[test]
    fn parse_response_stat_error_eacces() {
        let resp = vec![13u8]; // EACCES
        assert!(matches!(parse_response_stat(&resp), Err(Errno::EACCES)));
    }

    #[test]
    fn parse_response_stat_too_short() {
        let resp = vec![0u8; 10]; // status OK but only 9 payload bytes (need 20)
        assert!(matches!(parse_response_stat(&resp), Err(Errno::EIO)));
    }

    // ── parse_response_read ──────────────────────────────────────────────────

    #[test]
    fn parse_response_read_ok() {
        let data = b"hello";
        let mut resp = vec![0u8; 1 + 4 + data.len()];
        resp[0] = 0;
        resp[1..5].copy_from_slice(&(data.len() as u32).to_le_bytes());
        resp[5..].copy_from_slice(data);
        let mut buf = [0u8; 16];
        let n = parse_response_read(&resp, &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf[..5], b"hello");
    }

    #[test]
    fn parse_response_read_buf_smaller_than_data() {
        // Provider reports 10 bytes, but caller only has a 4-byte buf
        let data = b"0123456789";
        let mut resp = vec![0u8; 1 + 4 + data.len()];
        resp[0] = 0;
        resp[1..5].copy_from_slice(&(data.len() as u32).to_le_bytes());
        resp[5..].copy_from_slice(data);
        let mut buf = [0u8; 4];
        let n = parse_response_read(&resp, &mut buf).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf, b"0123");
    }

    #[test]
    fn parse_response_read_error_eio() {
        let resp = vec![5u8]; // EIO
        let mut buf = [0u8; 8];
        assert!(matches!(parse_response_read(&resp, &mut buf), Err(Errno::EIO)));
    }

    // ── Dead provider: request ring full → EIO ───────────────────────────────

    /// When the kernel cannot write to the provider's request port (ring full),
    /// `rpc()` must return `Err(EIO)` immediately — this is the
    /// "provider back-pressure / dead" path.
    #[test]
    fn rpc_returns_eio_when_request_ring_full() {
        let req_port = make_port(16); // tiny ring so it fills quickly
        let resp_port = make_port(256);

        // Flood the ring buffer so the next send() will return 0.
        let fill = vec![0xABu8; req_port.capacity()];
        req_port.send(&fill);

        let ch = ProviderChannelRef {
            req: req_port,
            resp: resp_port,
            resp_write_handle: 99,
        };

        // A Stat request payload is 8 bytes (handle: u64); combined with the
        // 7-byte header the message is 15 bytes and won't fit the full ring.
        let result = ch.rpc(VfsRpcOp::Stat, &[0u8; 8]);
        assert!(
            matches!(result, Err(Errno::EIO)),
            "expected EIO when request ring is full, got {:?}",
            result
        );
    }

    // ── Dead provider: response writer gone → EPIPE ──────────────────────────

    /// When the provider dies after the kernel sends a request but before it
    /// sends a response, the response port's writer count drops to zero.
    /// `recv_response()` must detect this and return `Err(EPIPE)` without
    /// blocking forever.
    #[test]
    fn rpc_returns_epipe_when_response_writer_gone() {
        let req_port = make_port(4096);
        let resp_port = make_port(256);

        // Simulate provider death: drop the write end of the response port.
        // (In production the write handle is in the provider's handle table;
        // when the process exits the handle table drops all handles.)
        resp_port.close_writer();

        let ch = ProviderChannelRef {
            req: req_port,
            resp: resp_port,
            resp_write_handle: 99,
        };

        // Send succeeds (data lands in the ring), but response never arrives.
        let payload = b"\x05\x00\x00\x00hello"; // Lookup "hello"
        let result = ch.rpc(VfsRpcOp::Lookup, payload);
        assert!(
            matches!(result, Err(Errno::EPIPE)),
            "expected EPIPE when response writer is gone, got {:?}",
            result
        );
    }

    // ── Dead provider: diagnostics counter ───────────────────────────────────

    /// Every dead-provider event must increment `VFS_RPC_DEAD_PROVIDER`.
    #[test]
    fn dead_provider_increments_counter() {
        use core::sync::atomic::Ordering;

        let before = crate::ipc::diag::VFS_RPC_DEAD_PROVIDER.load(Ordering::Relaxed);

        let req_port = make_port(4096);
        let resp_port = make_port(256);
        resp_port.close_writer();

        let ch = ProviderChannelRef {
            req: req_port,
            resp: resp_port,
            resp_write_handle: 0,
        };
        // Ignore the result; we only care about the counter.
        let _ = ch.rpc(VfsRpcOp::Stat, &[0u8; 8]);

        let after = crate::ipc::diag::VFS_RPC_DEAD_PROVIDER.load(Ordering::Relaxed);
        assert!(
            after > before,
            "VFS_RPC_DEAD_PROVIDER should have been incremented"
        );
    }

    /// Every dead-provider event must also increment the generic
    /// `VFS_RPC_ERRORS` counter.
    #[test]
    fn dead_provider_increments_error_counter() {
        use core::sync::atomic::Ordering;

        let before = crate::ipc::diag::VFS_RPC_ERRORS.load(Ordering::Relaxed);

        let req_port = make_port(4096);
        let resp_port = make_port(256);
        resp_port.close_writer();

        let ch = ProviderChannelRef {
            req: req_port,
            resp: resp_port,
            resp_write_handle: 0,
        };
        let _ = ch.rpc(VfsRpcOp::Stat, &[0u8; 8]);

        let after = crate::ipc::diag::VFS_RPC_ERRORS.load(Ordering::Relaxed);
        assert!(after > before);
    }

    // ── Successful round-trip (response in ring before recv) ─────────────────

    /// When a response is already waiting in the ring before `rpc()` is called,
    /// the call must return successfully without ever blocking.
    #[test]
    fn rpc_ok_when_response_preloaded() {
        let req_port = make_port(4096);
        let resp_port = make_port(4096);

        // Pre-load a valid Stat response into the response ring.
        // Format: [status=0][mode: u32 LE][size: u64 LE][ino: u64 LE]
        let mut preloaded = vec![0u8; 21];
        preloaded[0] = 0; // OK
        preloaded[1..5].copy_from_slice(&0o040755u32.to_le_bytes()); // mode: dir
        preloaded[5..13].copy_from_slice(&0u64.to_le_bytes()); // size: 0
        preloaded[13..21].copy_from_slice(&1u64.to_le_bytes()); // ino: 1
        resp_port.send(&preloaded);

        let ch = ProviderChannelRef {
            req: req_port,
            resp: resp_port,
            resp_write_handle: 0,
        };

        // The Stat RPC should complete without blocking.
        let raw = ch.rpc(VfsRpcOp::Stat, &[0u8; 8]).unwrap();
        let stat = parse_response_stat(&raw).unwrap();
        assert_eq!(stat.mode, 0o040755);
        assert_eq!(stat.ino, 1);
    }
}
