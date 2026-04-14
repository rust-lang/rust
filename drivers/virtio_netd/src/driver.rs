extern crate alloc;

use abi::errors::Errno;
use stem::syscall::{device_alloc_dma, device_dma_phys};
use stem::warn;
use virtio::VirtioDevice;

/// VirtIO network device header (prepended to each frame)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct VirtioNetHeader {
    pub flags: u8,
    pub gso_type: u8,
    pub hdr_len: u16,
    pub gso_size: u16,
    pub csum_start: u16,
    pub csum_offset: u16,
    pub num_buffers: u16,
}

impl VirtioNetHeader {
    pub fn zeroed() -> Self {
        Self {
            flags: 0,
            gso_type: 0,
            hdr_len: 0,
            gso_size: 0,
            csum_start: 0,
            csum_offset: 0,
            num_buffers: 0,
        }
    }
}

/// VirtIO-NET feature bits
pub const VIRTIO_NET_F_MAC: u32 = 1 << 5;
pub const VIRTIO_NET_F_STATUS: u32 = 1 << 16;
#[allow(dead_code)]
pub const VIRTIO_NET_F_MRG_RXBUF: u32 = 1 << 15;

const QUEUE_SIZE: u16 = 32;
const RX_BUFFER_SIZE: usize = 2048;
const TX_BUFFER_SIZE: usize = 2048;
const NET_HEADER_SIZE: usize = 10; // Without num_buffers (no MRG_RXBUF feature)
#[allow(dead_code)]
const NET_HEADER_SIZE_MRG: usize = 12; // With num_buffers (requires MRG_RXBUF)

/// VirtIO-NET device driver
pub struct VirtioNetDriver {
    device: VirtioDevice,
    mac: [u8; 6],
    link_up: bool,
    /// RX buffers (virtual addresses)
    rx_buffers_virt: [u64; QUEUE_SIZE as usize],
    /// RX buffers (physical addresses)
    rx_buffers_phys: [u64; QUEUE_SIZE as usize],
    /// Active RX buffer count
    rx_active: u16,
    /// TX buffer virtual
    tx_buffer_virt: u64,
    /// TX buffer physical
    tx_buffer_phys: u64,
    /// Last received frame
    last_rx: Option<(u64, usize)>, // (buffer virt, len)
}

impl VirtioNetDriver {
    /// Find and claim a VirtIO-NET device, then initialize it
    pub fn find_and_claim() -> Result<Self, Errno> {
        stem::debug!("VirtIO-NET: Searching for NIC device...");

        // Find a dev.net.nic node
        let nic_path = find_nic_device()?;
        Self::claim_device(&nic_path)
    }

    pub fn claim_device(nic_path: &str) -> Result<Self, Errno> {
        stem::debug!("VirtIO-NET: Claiming NIC device at {}", nic_path);

        // Create VirtIO device wrapper
        let mut device = VirtioDevice::new(nic_path)?;

        // Initialize with NET features
        let desired_features = VIRTIO_NET_F_MAC | VIRTIO_NET_F_STATUS;
        device
            .init(desired_features)
            .map_err(|_| Errno::NotSupported)?;

        stem::debug!(
            "VirtIO-NET: Device features 0x{:08x}",
            device.device_features()
        );

        // Read MAC address from device config
        let mac = if device.has_feature(5) {
            // VIRTIO_NET_F_MAC
            [
                device.read_device_config_u8(0).unwrap_or(0x52),
                device.read_device_config_u8(1).unwrap_or(0x54),
                device.read_device_config_u8(2).unwrap_or(0x00),
                device.read_device_config_u8(3).unwrap_or(0x12),
                device.read_device_config_u8(4).unwrap_or(0x34),
                device.read_device_config_u8(5).unwrap_or(0x56),
            ]
        } else {
            [0x52, 0x54, 0x00, 0x12, 0x34, 0x56] // Default
        };

        stem::debug!(
            "VirtIO-NET: MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
            mac[0],
            mac[1],
            mac[2],
            mac[3],
            mac[4],
            mac[5]
        );

        // Read link status
        let link_up = if device.has_feature(16) {
            // VIRTIO_NET_F_STATUS
            let status = device.read_device_config(6).unwrap_or(1) as u16;
            (status & 1) != 0
        } else {
            true
        };

        stem::debug!("VirtIO-NET: Link {}", if link_up { "UP" } else { "DOWN" });

        // Setup RX queue (queue 0)
        device
            .setup_queue(0, QUEUE_SIZE)
            .map_err(|_| Errno::ENOMEM)?;
        stem::debug!("VirtIO-NET: RX queue 0 setup (size={})", QUEUE_SIZE);

        // Setup TX queue (queue 1)
        device
            .setup_queue(1, QUEUE_SIZE)
            .map_err(|_| Errno::ENOMEM)?;
        stem::debug!("VirtIO-NET: TX queue 1 setup (size={})", QUEUE_SIZE);

        // Allocate RX buffers in bulk (32 pages = 128KB for 64 x 2KB buffers)
        // This dramatically reduces syscall overhead compared to 64 separate allocations
        let claim_handle = device.claim_handle();
        let mut rx_buffers_virt = [0u64; QUEUE_SIZE as usize];
        let mut rx_buffers_phys = [0u64; QUEUE_SIZE as usize];

        // Calculate pages needed: 64 buffers * 2KB = 128KB = 32 pages (4KB each)
        // Each page holds 2 buffers (4KB / 2KB = 2)
        const BUFFERS_PER_PAGE: usize = 4096 / RX_BUFFER_SIZE; // 2
        const RX_PAGES_NEEDED: usize =
            (QUEUE_SIZE as usize + BUFFERS_PER_PAGE - 1) / BUFFERS_PER_PAGE; // 32

        let rx_pool_virt =
            device_alloc_dma(claim_handle, RX_PAGES_NEEDED).map_err(|_| Errno::ENOMEM)?;
        let rx_pool_phys = device_dma_phys(rx_pool_virt).map_err(|_| Errno::EFAULT)?;
        stem::debug!(
            "VirtIO-NET: Allocated RX pool ({} pages, {} buffers)",
            RX_PAGES_NEEDED,
            QUEUE_SIZE
        );

        // Sub-allocate individual buffers from the pool
        for i in 0..QUEUE_SIZE as usize {
            let offset = (i * RX_BUFFER_SIZE) as u64;
            rx_buffers_virt[i] = rx_pool_virt + offset;
            rx_buffers_phys[i] = rx_pool_phys + offset;
        }

        // Allocate TX buffer (single page is fine for 1 buffer)
        let tx_buffer_virt = device_alloc_dma(claim_handle, 1).map_err(|_| Errno::ENOMEM)?;
        let tx_buffer_phys = device_dma_phys(tx_buffer_virt).map_err(|_| Errno::EFAULT)?;
        stem::debug!("VirtIO-NET: Allocated TX buffer (1 page)");

        let mut driver = Self {
            device,
            mac,
            link_up,
            rx_buffers_virt,
            rx_buffers_phys,
            rx_active: 0,
            tx_buffer_virt,
            tx_buffer_phys,
            last_rx: None,
        };

        // Fill RX queue with buffers BEFORE setting DRIVER_OK
        // This is critical: device won't receive until buffers are posted
        driver.refill_rx_queue();
        stem::debug!(
            "VirtIO-NET: RX queue filled with {} buffers",
            driver.rx_active
        );

        // NOW mark device ready - it will start receiving
        driver.device.driver_ok();
        stem::debug!("VirtIO-NET: DRIVER_OK set, device is live");

        // Kick RX queue again to be sure device notices our buffers
        driver.device.notify_queue(0);

        stem::debug!("VirtIO-NET: Driver initialized successfully");
        Ok(driver)
    }

    /// Fill RX queue with available buffers
    fn refill_rx_queue(&mut self) {
        if let Some(rxq) = self.device.queue_mut(0) {
            for i in self.rx_active..QUEUE_SIZE {
                let phys = self.rx_buffers_phys[i as usize];
                // Add buffer as device-writable
                if rxq
                    .add_buffer(&[(phys, RX_BUFFER_SIZE as u32, true)])
                    .is_none()
                {
                    break;
                }
                self.rx_active += 1;
            }
            // Notify device
            self.device.notify_queue(0);
        }
    }

    /// Get MAC address
    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    /// Check if link is up
    pub fn link_up(&self) -> bool {
        self.link_up
    }

    /// Return the negotiated virtio feature flags.
    pub fn device_features(&self) -> u32 {
        self.device.device_features()
    }

    /// Poll device config and report link-state changes.
    pub fn poll_link_change(&mut self) -> Option<bool> {
        let next = if self.device.has_feature(16) {
            let status = self
                .device
                .read_device_config(6)
                .unwrap_or(if self.link_up { 1 } else { 0 }) as u16;
            (status & 1) != 0
        } else {
            true
        };

        if next != self.link_up {
            self.link_up = next;
            Some(next)
        } else {
            None
        }
    }

    /// Log debug stats for diagnosis
    #[allow(dead_code)]
    pub fn log_stats(&self, label: &str) {
        stem::debug!("VirtIO-NET [{}]: rx_active={}", label, self.rx_active);
    }

    /// Poll for received frames
    pub fn poll_rx(&mut self) -> Option<&[u8]> {
        let rxq = self.device.queue_mut(0)?;

        if let Some((desc_id, len)) = rxq.poll_used() {
            stem::debug!("VirtIO-NET: RX frame! desc={} len={}", desc_id, len);

            let buf_virt = self.rx_buffers_virt[desc_id as usize];

            // Skip virtio-net header
            let header_size = NET_HEADER_SIZE;
            let data_len = (len as usize).saturating_sub(header_size);

            if data_len > 0 {
                self.last_rx = Some((buf_virt + header_size as u64, data_len));
                self.rx_active -= 1;

                // Re-add buffer to RX queue
                let phys = self.rx_buffers_phys[desc_id as usize];
                rxq.add_buffer(&[(phys, RX_BUFFER_SIZE as u32, true)]);
                self.rx_active += 1;
                self.device.notify_queue(0);

                return self.last_rx.map(|(virt, len)| unsafe {
                    core::slice::from_raw_parts(virt as *const u8, len)
                });
            }
        }
        None
    }

    /// Transmit a frame
    pub fn tx(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if data.len() > TX_BUFFER_SIZE - NET_HEADER_SIZE {
            return Err("Frame too large");
        }

        stem::debug!("VirtIO-NET: TX {} bytes", data.len());

        // Write header
        let header = VirtioNetHeader::zeroed();
        let header_ptr = self.tx_buffer_virt as *mut VirtioNetHeader;
        unsafe {
            core::ptr::write_volatile(header_ptr, header);
        }

        // Copy data after header
        let data_ptr = (self.tx_buffer_virt + NET_HEADER_SIZE as u64) as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        }

        let total_len = NET_HEADER_SIZE + data.len();

        // Add to TX queue (scoped to release borrow before notify)
        {
            let txq = self.device.queue_mut(1).ok_or("No TX queue")?;
            txq.add_buffer(&[(self.tx_buffer_phys, total_len as u32, false)])
                .ok_or("TX queue full")?;
        }

        // Notify device
        self.device.notify_queue(1);

        // Wait for completion with yield-based backoff
        // Spin briefly (10 iterations), then yield to scheduler
        for i in 0..1000 {
            if let Some(txq) = self.device.queue_mut(1) {
                if txq.poll_used().is_some() {
                    if i >= 1000 {
                        stem::debug!("VirtIO-NET: TX complete after {} iterations (yielded)", i);
                    }
                    return Ok(());
                }
            }
            if i < 10 {
                core::hint::spin_loop();
            } else {
                stem::time::sleep_ms(1); // Yield to scheduler instead of burning CPU
            }
        }

        warn!("VirtIO-NET: TX timeout!");
        Err("TX timeout")
    }
}

/// Find a VirtIO-NET device in sysfs
pub fn find_nic_device() -> Result<alloc::string::String, Errno> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_readdir};

    let fd = vfs_open("/sys/devices", O_RDONLY).map_err(|_| Errno::ENODEV)?;
    let mut buf = [0u8; 4096];
    let n = vfs_readdir(fd, &mut buf).map_err(|_| Errno::ENODEV)?;
    let _ = vfs_close(fd);

    let mut offset = 0;
    while offset < n {
        let mut end = offset;
        while end < n && buf[end] != 0 {
            end += 1;
        }
        if end > offset {
            if let Ok(name) = core::str::from_utf8(&buf[offset..end]) {
                let path = alloc::format!("/sys/devices/{}", name);
                let vendor = read_sys_u32(&alloc::format!("{}/vendor", path)).unwrap_or(0);
                let class = read_sys_u32(&alloc::format!("{}/class", path)).unwrap_or(0);

                // VirtIO Vendor = 0x1af4, Network Class = 0x0200xx
                if vendor == 0x1af4 && (class >> 8) == 0x0200 {
                    return Ok(path);
                }
            }
        }
        offset = end + 1;
    }
    Err(Errno::ENODEV)
}

fn read_sys_u32(path: &str) -> Option<u32> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

    let fd = vfs_open(path, O_RDONLY).ok()?;
    let mut buf = [0u8; 32];
    let n = vfs_read(fd, &mut buf).ok()?;
    let _ = vfs_close(fd);

    let s = core::str::from_utf8(&buf[..n]).ok()?;
    let trimmed = s.trim();
    if trimmed.starts_with("0x") {
        u32::from_str_radix(&trimmed[2..], 16).ok()
    } else {
        trimmed.parse::<u32>().ok()
    }
}
