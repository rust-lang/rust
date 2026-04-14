//! VirtIO device abstraction
//!
//! Provides a generic VirtioDevice struct that handles:
//! - Device claiming and BAR mapping
//! - Feature negotiation  
//! - Virtqueue setup
//! - Command/response communication
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use abi::errors::Errno;
use abi::ids::HandleId;
use abi::schema::keys;
use core::ptr::{read_volatile, write_volatile};
use stem::info;
use stem::syscall::{device_alloc_dma, device_claim, device_dma_phys, device_map_mmio};

use crate::constants::*;
use crate::virtqueue::Virtqueue;

/// A generic VirtIO PCI device
pub struct VirtioDevice {
    /// Device claim handle for syscalls
    claim_handle: usize,
    /// Common config virtual address
    common_cfg: u64,
    /// Notify config virtual address
    notify_cfg: u64,
    /// Device-specific config virtual address (if available)
    device_cfg: Option<u64>,
    /// Notify offset multiplier
    notify_off_multiplier: u32,
    /// Virtqueues (indexed by queue number)
    queues: [Option<Virtqueue>; 8],
    /// Number of queues supported by device
    num_queues: u16,
    /// Command buffer virtual address
    cmd_buf: u64,
    /// Command buffer physical address
    cmd_buf_phys: u64,
    /// Device features (lower 32 bits)
    device_features: u32,
    /// Driver-negotiated features
    driver_features: u32,
}

impl VirtioDevice {
    /// Create a new VirtioDevice by claiming and mapping a device node
    pub fn new(sys_path: &str) -> Result<Self, Errno> {
        stem::debug!("VirtIO: device::new({})", sys_path);

        // Claim the device using its sysfs path as the primary key.
        stem::debug!("VirtIO: claiming '{}'...", sys_path);
        let claim_handle = device_claim(sys_path)?;
        stem::debug!("VirtIO: claimed, handle={}", claim_handle);

        // Read VirtIO capability offsets from sysfs
        let common_bar = read_sys_u32(&alloc::format!("{}/virtio/common_bar", sys_path))? as usize;
        let common_offset = read_sys_u32(&alloc::format!("{}/virtio/common_offset", sys_path))? as u64;
        let notify_bar = read_sys_u32(&alloc::format!("{}/virtio/notify_bar", sys_path))? as usize;
        let notify_offset = read_sys_u32(&alloc::format!("{}/virtio/notify_offset", sys_path))? as u64;
        let notify_multiplier = read_sys_u32(&alloc::format!("{}/virtio/notify_multiplier", sys_path))?;

        stem::debug!(
            "VirtIO: common_bar={} common_off=0x{:x} notify_bar={} notify_off=0x{:x} mult={}",
            common_bar,
            common_offset,
            notify_bar,
            notify_offset,
            notify_multiplier
        );

        // Device config is optional (0xFF if not present)
        let device_bar = read_sys_u32(&alloc::format!("{}/virtio/device_bar", sys_path)).unwrap_or(0xFF);
        let device_offset =
            read_sys_u32(&alloc::format!("{}/virtio/device_offset", sys_path)).unwrap_or(0);

        // Map the BAR containing common config
        stem::debug!("VirtIO: mapping common BAR{}...", common_bar);
        let common_bar_base = device_map_mmio(claim_handle, common_bar)?;
        let common_cfg = common_bar_base + common_offset;
        stem::debug!("VirtIO: common_cfg at 0x{:x}", common_cfg);

        // Map notify BAR (may be same as common BAR)
        let notify_cfg = if notify_bar == common_bar {
            common_bar_base + notify_offset
        } else {
            stem::info!("VirtIO: mapping notify BAR{}...", notify_bar);
            let notify_bar_base = device_map_mmio(claim_handle, notify_bar)?;
            notify_bar_base + notify_offset
        };
        stem::debug!("VirtIO: notify_cfg at 0x{:x}", notify_cfg);

        // Map device config BAR if available
        let device_cfg = if device_bar != 0xFF {
            let bar_idx = device_bar as usize;
            let bar_base = if bar_idx == common_bar {
                common_bar_base
            } else if bar_idx == notify_bar {
                notify_cfg - notify_offset // Already mapped
            } else {
                device_map_mmio(claim_handle, bar_idx)?
            };
            Some(bar_base + device_offset as u64)
        } else {
            None
        };
        stem::debug!("VirtIO: device_cfg = {:?}", device_cfg);

        // Allocate command buffer (1 page for commands + responses)
        stem::debug!("VirtIO: allocating DMA command buffer...");
        let cmd_buf = device_alloc_dma(claim_handle, 1).map_err(|_| Errno::ENOMEM)?;
        let cmd_buf_phys = device_dma_phys(cmd_buf).map_err(|_| Errno::EFAULT)?;
        stem::info!(
            "VirtIO: cmd_buf virt=0x{:x} phys=0x{:x}",
            cmd_buf,
            cmd_buf_phys
        );

        stem::debug!("VirtIO: device::new complete");
        Ok(Self {
            claim_handle,
            common_cfg,
            notify_cfg,
            device_cfg,
            notify_off_multiplier: notify_multiplier,
            queues: [None, None, None, None, None, None, None, None],
            num_queues: 0,
            cmd_buf,
            cmd_buf_phys,
            device_features: 0,
            driver_features: 0,
        })
    }
}

fn read_sys_u32(path: &str) -> Result<u32, Errno> {
    use abi::syscall::vfs_flags::O_RDONLY;
    use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read};

    let fd = vfs_open(path, O_RDONLY)?;
    let mut buf = [0u8; 32];
    let n = vfs_read(fd, &mut buf).map_err(|e| {
        stem::info!("READ_SYS: {} read failed: {:?}", path, e);
        e
    })?;
    let _ = vfs_close(fd);

    let s = core::str::from_utf8(&buf[..n]).map_err(|_| Errno::EIO)?;
    let trimmed = s.trim();
    stem::debug!("READ_SYS: {} -> '{}' (n={})", path, trimmed, n);
    if trimmed.starts_with("0x") {
        u32::from_str_radix(&trimmed[2..], 16).map_err(|_| Errno::EIO)
    } else {
        trimmed.parse::<u32>().map_err(|_| Errno::EIO)
    }
}

impl VirtioDevice {
    /// Initialize the VirtIO device with feature negotiation
    ///
    /// `desired_features` - Features the driver wants to use (device-specific bits)
    pub fn init(&mut self, desired_features: u32) -> Result<(), &'static str> {
        // 1. Reset device
        self.write_common(VIRTIO_COMMON_STATUS, 0);

        // 2. Set ACKNOWLEDGE status
        self.write_common(VIRTIO_COMMON_STATUS, VIRTIO_STATUS_ACKNOWLEDGE);

        // 3. Set DRIVER status
        let status = self.read_common(VIRTIO_COMMON_STATUS);
        self.write_common(VIRTIO_COMMON_STATUS, status | VIRTIO_STATUS_DRIVER);

        // 4. Read device features (bank 0 for device-specific features)
        self.write_common(VIRTIO_COMMON_DEVICE_FEATURE_SELECT, 0);
        self.device_features = self.read_common(VIRTIO_COMMON_DEVICE_FEATURE);

        // 5. Negotiate features (intersection of device and desired)
        self.driver_features = self.device_features & desired_features;
        self.write_common(VIRTIO_COMMON_DRIVER_FEATURE_SELECT, 0);
        self.write_common(VIRTIO_COMMON_DRIVER_FEATURE, self.driver_features);

        // 6. Set FEATURES_OK
        let status = self.read_common(VIRTIO_COMMON_STATUS);
        self.write_common(VIRTIO_COMMON_STATUS, status | VIRTIO_STATUS_FEATURES_OK);

        // 7. Verify FEATURES_OK
        let status = self.read_common(VIRTIO_COMMON_STATUS);
        if (status & VIRTIO_STATUS_FEATURES_OK) == 0 {
            return Err("Features not accepted");
        }

        // Read number of queues
        self.num_queues = self.read_common_u16(VIRTIO_COMMON_NUM_QUEUES);

        Ok(())
    }

    /// Setup a virtqueue
    pub fn setup_queue(&mut self, queue_idx: u16, size: u16) -> Result<(), &'static str> {
        if queue_idx >= 8 {
            return Err("Queue index too large");
        }

        let vq_virt =
            device_alloc_dma(self.claim_handle, 4).map_err(|_| "Failed to alloc virtqueue")?;
        let vq_phys = device_dma_phys(vq_virt).map_err(|_| "Failed to get vq phys")?;

        let vq = Virtqueue::new(vq_virt, vq_phys, size);

        // Configure the queue in device
        self.write_common(VIRTIO_COMMON_QUEUE_SELECT, queue_idx as u32);
        self.write_common(VIRTIO_COMMON_QUEUE_SIZE, size as u32);

        // Write queue addresses
        self.write_common(VIRTIO_COMMON_QUEUE_DESC_LO, (vq_phys & 0xFFFFFFFF) as u32);
        self.write_common(VIRTIO_COMMON_QUEUE_DESC_HI, (vq_phys >> 32) as u32);

        let avail_offset = (size as u64) * 16;
        let avail_phys = vq_phys + avail_offset;
        self.write_common(
            VIRTIO_COMMON_QUEUE_AVAIL_LO,
            (avail_phys & 0xFFFFFFFF) as u32,
        );
        self.write_common(VIRTIO_COMMON_QUEUE_AVAIL_HI, (avail_phys >> 32) as u32);

        // Used ring must be 4-byte aligned (VirtIO 1.0 spec)
        let used_unaligned = avail_offset + 6 + (size as u64) * 2;
        let used_offset = (used_unaligned + 3) & !3; // Align up to 4 bytes
        let used_phys = vq_phys + used_offset;
        self.write_common(VIRTIO_COMMON_QUEUE_USED_LO, (used_phys & 0xFFFFFFFF) as u32);
        self.write_common(VIRTIO_COMMON_QUEUE_USED_HI, (used_phys >> 32) as u32);

        // Enable the queue
        self.write_common(VIRTIO_COMMON_QUEUE_ENABLE, 1);

        self.queues[queue_idx as usize] = Some(vq);
        Ok(())
    }

    /// Mark device as ready (call after setting up all queues)
    pub fn driver_ok(&mut self) {
        let status = self.read_common(VIRTIO_COMMON_STATUS);
        self.write_common(VIRTIO_COMMON_STATUS, status | VIRTIO_STATUS_DRIVER_OK);
    }

    /// Get a mutable reference to a virtqueue
    pub fn queue_mut(&mut self, idx: u16) -> Option<&mut Virtqueue> {
        self.queues.get_mut(idx as usize)?.as_mut()
    }

    /// Notify the device about a queue update
    pub fn notify_queue(&self, queue_idx: u16) {
        let notify_addr = self.notify_cfg + (queue_idx as u64 * self.notify_off_multiplier as u64);
        unsafe { write_volatile(notify_addr as *mut u16, queue_idx) }
    }

    /// Read from common config space (u32)
    pub fn read_common(&self, offset: u32) -> u32 {
        unsafe { read_volatile((self.common_cfg + offset as u64) as *const u32) }
    }

    /// Write to common config space (u32)
    pub fn write_common(&self, offset: u32, value: u32) {
        unsafe { write_volatile((self.common_cfg + offset as u64) as *mut u32, value) }
    }

    /// Read from common config space (u16)
    pub fn read_common_u16(&self, offset: u32) -> u16 {
        unsafe { read_volatile((self.common_cfg + offset as u64) as *const u16) }
    }

    /// Write to common config space (u16)
    pub fn write_common_u16(&self, offset: u32, value: u16) {
        unsafe { write_volatile((self.common_cfg + offset as u64) as *mut u16, value) }
    }

    /// Read from device-specific config space
    pub fn read_device_config(&self, offset: u32) -> Option<u32> {
        self.device_cfg
            .map(|base| unsafe { read_volatile((base + offset as u64) as *const u32) })
    }

    /// Read u8 from device-specific config space
    pub fn read_device_config_u8(&self, offset: u32) -> Option<u8> {
        self.device_cfg
            .map(|base| unsafe { read_volatile((base + offset as u64) as *const u8) })
    }

    /// Get claim handle for additional device operations  
    pub fn claim_handle(&self) -> usize {
        self.claim_handle
    }

    /// Get negotiated device features
    pub fn device_features(&self) -> u32 {
        self.device_features
    }

    /// Check if a feature was negotiated
    pub fn has_feature(&self, bit: u32) -> bool {
        (self.driver_features & (1 << bit)) != 0
    }

    /// Get command buffer virtual address
    pub fn cmd_buf(&self) -> u64 {
        self.cmd_buf
    }

    /// Get command buffer physical address
    pub fn cmd_buf_phys(&self) -> u64 {
        self.cmd_buf_phys
    }

    /// Get number of queues supported
    pub fn num_queues(&self) -> u16 {
        self.num_queues
    }
}
