//! VFS-backed smoltcp Device implementation.
//!
//! Frames are exchanged over `/dev/net/virtio0/{rx,tx,events}` using a
//! length-prefixed wire format:
//! `[4 bytes little-endian frame length][raw Ethernet frame bytes...]`
extern crate alloc;
use alloc::string::ToString;
use core::default::Default;

use abi::syscall::vfs_flags;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use smoltcp::phy::{self, Device, DeviceCapabilities, Medium};
use smoltcp::time::Instant;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_write};

const MAX_FRAME_LEN: usize = 2048;
const RX_STAGING_BUF_LEN: usize = 65536;

/// Network device backed by the virtio_netd VFS tree.
pub struct VfsNicDevice {
    rx_fd: u32,
    tx_fd: u32,
    events_fd: u32,
    #[allow(dead_code)]
    mac: [u8; 6],
    mtu: usize,
    link_up: bool,
    rx_staging: [u8; RX_STAGING_BUF_LEN],
    rx_staging_len: usize,
    rx_queue: VecDeque<([u8; MAX_FRAME_LEN], usize)>,
}

impl VfsNicDevice {
    /// Open the virtio-net VFS tree, retrying until it is present.
    #[allow(dead_code)]
    pub fn open() -> Self {
        loop {
            if let Some(dev) = Self::try_open() {
                return dev;
            }
            stem::info!("VfsNicDevice: /dev/net/virtio0 not ready, retrying...");
            stem::time::sleep_ms(100);
        }
    }

    fn try_open() -> Option<Self> {
        let rx_path = "/dev/net/virtio0/rx";
        let tx_path = "/dev/net/virtio0/tx";
        let events_path = "/dev/net/virtio0/events";
        let mac_path = "/dev/net/virtio0/mac";
        let mtu_path = "/dev/net/virtio0/mtu";

        let rx_fd = vfs_open(rx_path, vfs_flags::O_RDONLY | vfs_flags::O_NONBLOCK).ok()?;
        let tx_fd = match vfs_open(tx_path, vfs_flags::O_WRONLY) {
            Ok(fd) => fd,
            Err(e) => {
                stem::warn!("VfsNicDevice: failed to open tx: {:?}", e);
                let _ = vfs_close(rx_fd);
                return None;
            }
        };
        let events_fd = match vfs_open(events_path, vfs_flags::O_RDONLY | vfs_flags::O_NONBLOCK) {
            Ok(fd) => fd,
            Err(_) => u32::MAX,
        };

        let mac = Self::read_mac(mac_path).unwrap_or([0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        let mtu = Self::read_mtu(mtu_path).unwrap_or(1500);

        Some(Self::new(rx_fd, tx_fd, events_fd, mac, mtu, true))
    }

    pub fn new(
        rx_fd: u32,
        tx_fd: u32,
        events_fd: u32,
        mac: [u8; 6],
        mtu: usize,
        link_up: bool,
    ) -> Self {
        Self {
            rx_fd,
            tx_fd,
            events_fd,
            mac,
            mtu,
            link_up,
            rx_staging: [0u8; RX_STAGING_BUF_LEN],
            rx_staging_len: 0,
            rx_queue: VecDeque::new(),
        }
    }

    fn read_mac(path: &str) -> Option<[u8; 6]> {
        let fd = vfs_open(path, vfs_flags::O_RDONLY).ok()?;
        let mut buf = [0u8; 32];
        let n = vfs_read(fd, &mut buf).ok()?;
        let _ = vfs_close(fd);
        let s = core::str::from_utf8(&buf[..n]).ok()?.trim();
        Self::parse_mac(s)
    }

    fn parse_mac(s: &str) -> Option<[u8; 6]> {
        let mut mac = [0u8; 6];
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 6 {
            return None;
        }
        for (idx, part) in parts.iter().enumerate() {
            mac[idx] = u8::from_str_radix(part.trim(), 16).ok()?;
        }
        Some(mac)
    }

    fn read_mtu(path: &str) -> Option<usize> {
        let fd = vfs_open(path, vfs_flags::O_RDONLY).ok()?;
        let mut buf = [0u8; 16];
        let n = vfs_read(fd, &mut buf).ok()?;
        let _ = vfs_close(fd);
        let s = core::str::from_utf8(&buf[..n]).ok()?.trim();
        s.parse::<usize>().ok()
    }

    pub fn now() -> Instant {
        Instant::from_millis(stem::time::now().as_millis() as i64)
    }

    #[allow(dead_code)]
    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    #[allow(dead_code)]
    pub fn mtu(&self) -> usize {
        self.mtu
    }

    pub fn link_up(&self) -> bool {
        self.link_up
    }

    fn poll_events(&mut self) {
        if self.events_fd == u32::MAX {
            return;
        }

        let mut buf = [0u8; 64];
        loop {
            match vfs_read(self.events_fd, &mut buf) {
                Ok(n) if n > 0 => {
                    for &byte in &buf[..n] {
                        match byte {
                            0x01 => self.link_up = true,
                            0x00 => self.link_up = false,
                            _ => {}
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn poll_rx(&mut self) {
        self.poll_events();

        loop {
            let free = self.rx_staging.len().saturating_sub(self.rx_staging_len);
            if free == 0 {
                break;
            }

            match vfs_read(self.rx_fd, &mut self.rx_staging[self.rx_staging_len..]) {
                Ok(n) if n > 0 => self.rx_staging_len += n,
                _ => break,
            }
        }

        let mut offset = 0usize;
        while self.rx_staging_len.saturating_sub(offset) >= 4 {
            let remaining = &self.rx_staging[offset..self.rx_staging_len];
            let frame_len =
                u32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]])
                    as usize;

            if frame_len == 0 || frame_len > MAX_FRAME_LEN {
                stem::warn!(
                    "VfsNicDevice: invalid frame length {} in RX stream, resyncing",
                    frame_len
                );
                offset += 1;
                continue;
            }

            if remaining.len() < 4 + frame_len {
                break;
            }

            let mut frame = [0u8; MAX_FRAME_LEN];
            frame[..frame_len].copy_from_slice(&remaining[4..4 + frame_len]);
            self.rx_queue.push_back((frame, frame_len));
            offset += 4 + frame_len;
        }

        if offset > 0 {
            if offset < self.rx_staging_len {
                self.rx_staging.copy_within(offset..self.rx_staging_len, 0);
                self.rx_staging_len -= offset;
            } else {
                self.rx_staging_len = 0;
            }
        }
    }

    fn send_frame(&mut self, data: &[u8]) {
        let frame_len = data.len() as u32;
        let mut msg = Vec::with_capacity(4 + data.len());
        msg.extend_from_slice(&frame_len.to_le_bytes());
        msg.extend_from_slice(data);

        if let Err(e) = vfs_write(self.tx_fd, &msg) {
            stem::warn!("VfsNicDevice: TX write failed: {:?}", e);
        }
    }
}

impl Drop for VfsNicDevice {
    fn drop(&mut self) {
        let _ = vfs_close(self.rx_fd);
        let _ = vfs_close(self.tx_fd);
        if self.events_fd != u32::MAX {
            let _ = vfs_close(self.events_fd);
        }
    }
}

impl Device for VfsNicDevice {
    type RxToken<'a>
        = VfsRxToken
    where
        Self: 'a;
    type TxToken<'a>
        = VfsTxToken<'a>
    where
        Self: 'a;

    fn receive(&mut self, _timestamp: Instant) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        self.poll_rx();
        self.rx_queue
            .pop_front()
            .map(|(frame, len)| (VfsRxToken { frame, len }, VfsTxToken { device: self }))
    }

    fn transmit(&mut self, _timestamp: Instant) -> Option<Self::TxToken<'_>> {
        Some(VfsTxToken { device: self })
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.max_transmission_unit = self.mtu;
        caps.max_burst_size = Some(1);
        caps.medium = Medium::Ethernet;
        caps
    }
}

pub struct VfsRxToken {
    frame: [u8; MAX_FRAME_LEN],
    len: usize,
}

impl phy::RxToken for VfsRxToken {
    fn consume<R, F>(self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut frame = self.frame;
        f(&mut frame[..self.len])
    }
}

pub struct VfsTxToken<'a> {
    device: &'a mut VfsNicDevice,
}

impl<'a> phy::TxToken for VfsTxToken<'a> {
    fn consume<R, F>(self, len: usize, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut frame = [0u8; MAX_FRAME_LEN];
        let result = f(&mut frame[..len]);
        self.device.send_frame(&frame[..len]);
        result
    }
}
