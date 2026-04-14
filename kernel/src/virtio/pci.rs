//! VirtIO PCI device support (modern devices)
//!
//! Implements VirtIO 1.0+ "modern" PCI device discovery and configuration.

use core::ptr::{read_volatile, write_volatile};

/// VirtIO PCI capability types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioCapabilityType {
    /// Common configuration
    CommonCfg = 1,
    /// Notifications
    NotifyCfg = 2,
    /// ISR status
    IsrCfg = 3,
    /// Device-specific configuration
    DeviceCfg = 4,
    /// PCI configuration access
    PciCfg = 5,
}

/// A parsed VirtIO PCI capability
#[derive(Debug, Clone, Copy)]
pub struct VirtioCapability {
    pub cap_type: u8,
    pub bar: u8,
    pub offset: u32,
    pub length: u32,
    /// Only for notify capability: multiplier for queue notify offset
    pub notify_off_multiplier: u32,
}

/// VirtIO PCI device abstraction
pub struct VirtioPciDevice {
    /// Physical base address of the selected BAR
    pub bar_phys: u64,
    /// Virtual base address (BAR phys + HHDM offset)
    pub bar_virt: u64,
    /// Common config capability
    pub common_cfg: Option<VirtioCapability>,
    /// Notify config capability
    pub notify_cfg: Option<VirtioCapability>,
    /// ISR config capability
    pub isr_cfg: Option<VirtioCapability>,
    /// Device-specific config capability
    pub device_cfg: Option<VirtioCapability>,
}

impl VirtioPciDevice {
    /// Create a new VirtIO PCI device from parsed capabilities
    pub fn new(
        bar_phys: u64,
        hhdm_offset: u64,
        common_cfg: Option<VirtioCapability>,
        notify_cfg: Option<VirtioCapability>,
        isr_cfg: Option<VirtioCapability>,
        device_cfg: Option<VirtioCapability>,
    ) -> Self {
        Self {
            bar_phys,
            bar_virt: bar_phys + hhdm_offset,
            common_cfg,
            notify_cfg,
            isr_cfg,
            device_cfg,
        }
    }

    /// Get pointer to common config register
    fn common_ptr(&self, offset: u32) -> *mut u32 {
        let cap = self.common_cfg.expect("No common config capability");
        let addr = self.bar_virt + cap.offset as u64 + offset as u64;
        addr as *mut u32
    }

    /// Get pointer to device-specific config register
    fn device_ptr(&self, offset: u32) -> *const u8 {
        let cap = self.device_cfg.expect("No device config capability");
        let addr = self.bar_virt + cap.offset as u64 + offset as u64;
        addr as *const u8
    }

    /// Read from common config space (u32)
    pub fn read_common_u32(&self, offset: u32) -> u32 {
        unsafe { read_volatile(self.common_ptr(offset)) }
    }

    /// Write to common config space (u32)
    pub fn write_common_u32(&self, offset: u32, value: u32) {
        unsafe { write_volatile(self.common_ptr(offset), value) }
    }

    /// Read from common config space (u8)
    pub fn read_common_u8(&self, offset: u32) -> u8 {
        let aligned = offset & !0x3;
        let shift = (offset & 0x3) * 8;
        let val = self.read_common_u32(aligned);
        ((val >> shift) & 0xFF) as u8
    }

    /// Write to common config space (u8)
    pub fn write_common_u8(&self, offset: u32, value: u8) {
        let aligned = offset & !0x3;
        let shift = (offset & 0x3) * 8;
        let mut val = self.read_common_u32(aligned);
        val &= !(0xFF << shift);
        val |= (value as u32) << shift;
        self.write_common_u32(aligned, val);
    }

    /// Read from device-specific config (u8)
    pub fn read_device_u8(&self, offset: u32) -> u8 {
        unsafe { read_volatile(self.device_ptr(offset)) }
    }

    /// Read from device-specific config (u32)
    pub fn read_device_u32(&self, offset: u32) -> u32 {
        let ptr = self.device_ptr(offset) as *const u32;
        unsafe { read_volatile(ptr) }
    }

    /// Read device status
    pub fn read_status(&self) -> u8 {
        self.read_common_u8(COMMON_CFG_DEVICE_STATUS)
    }

    /// Write device status
    pub fn write_status(&self, status: u8) {
        self.write_common_u8(COMMON_CFG_DEVICE_STATUS, status);
    }

    /// Reset the device
    pub fn reset(&self) {
        self.write_status(0);
        // Wait for reset to complete
        while self.read_status() != 0 {
            core::hint::spin_loop();
        }
    }

    /// Read device features (low 32 bits or high 32 bits)
    pub fn read_device_features(&self, select: u32) -> u32 {
        self.write_common_u32(COMMON_CFG_DEVICE_FEATURE_SELECT, select);
        self.read_common_u32(COMMON_CFG_DEVICE_FEATURE)
    }

    /// Read all 64-bit device features
    pub fn read_device_features_64(&self) -> u64 {
        let low = self.read_device_features(0) as u64;
        let high = self.read_device_features(1) as u64;
        low | (high << 32)
    }

    /// Write driver features (low 32 bits or high 32 bits)
    pub fn write_driver_features(&self, select: u32, features: u32) {
        self.write_common_u32(COMMON_CFG_DRIVER_FEATURE_SELECT, select);
        self.write_common_u32(COMMON_CFG_DRIVER_FEATURE, features);
    }

    /// Write all 64-bit driver features
    pub fn write_driver_features_64(&self, features: u64) {
        self.write_driver_features(0, features as u32);
        self.write_driver_features(1, (features >> 32) as u32);
    }

    /// Select a queue for configuration
    pub fn select_queue(&self, queue_index: u16) {
        self.write_common_u32(COMMON_CFG_QUEUE_SELECT, queue_index as u32);
    }

    /// Read max queue size
    pub fn read_queue_size(&self) -> u16 {
        self.read_common_u32(COMMON_CFG_QUEUE_SIZE) as u16
    }

    /// Write queue size
    pub fn write_queue_size(&self, size: u16) {
        self.write_common_u32(COMMON_CFG_QUEUE_SIZE, size as u32);
    }

    /// Enable queue
    pub fn enable_queue(&self) {
        self.write_common_u32(COMMON_CFG_QUEUE_ENABLE, 1);
    }

    /// Write queue descriptor table address
    pub fn write_queue_desc(&self, addr: u64) {
        self.write_common_u32(COMMON_CFG_QUEUE_DESC_LOW, addr as u32);
        self.write_common_u32(COMMON_CFG_QUEUE_DESC_HIGH, (addr >> 32) as u32);
    }

    /// Write queue available ring address
    pub fn write_queue_avail(&self, addr: u64) {
        self.write_common_u32(COMMON_CFG_QUEUE_AVAIL_LOW, addr as u32);
        self.write_common_u32(COMMON_CFG_QUEUE_AVAIL_HIGH, (addr >> 32) as u32);
    }

    /// Write queue used ring address
    pub fn write_queue_used(&self, addr: u64) {
        self.write_common_u32(COMMON_CFG_QUEUE_USED_LOW, addr as u32);
        self.write_common_u32(COMMON_CFG_QUEUE_USED_HIGH, (addr >> 32) as u32);
    }

    /// Notify device about available descriptors
    pub fn notify_queue(&self, queue_index: u16) {
        let cap = self.notify_cfg.expect("No notify config capability");
        let queue_notify_off = {
            self.select_queue(queue_index);
            self.read_common_u32(COMMON_CFG_QUEUE_NOTIFY_OFF) as u64
        };
        let notify_addr =
            self.bar_virt + cap.offset as u64 + queue_notify_off * cap.notify_off_multiplier as u64;
        unsafe {
            write_volatile(notify_addr as *mut u16, queue_index);
        }
    }

    /// Read ISR status (clears on read)
    pub fn read_isr_status(&self) -> u8 {
        let cap = self.isr_cfg.expect("No ISR config capability");
        let addr = self.bar_virt + cap.offset as u64;
        unsafe { read_volatile(addr as *const u8) }
    }
}

// Common configuration layout offsets (VirtIO 1.0 spec)
const COMMON_CFG_DEVICE_FEATURE_SELECT: u32 = 0x00;
const COMMON_CFG_DEVICE_FEATURE: u32 = 0x04;
const COMMON_CFG_DRIVER_FEATURE_SELECT: u32 = 0x08;
const COMMON_CFG_DRIVER_FEATURE: u32 = 0x0C;
const COMMON_CFG_MSIX_CONFIG: u32 = 0x10;
const COMMON_CFG_NUM_QUEUES: u32 = 0x12;
const COMMON_CFG_DEVICE_STATUS: u32 = 0x14;
const COMMON_CFG_CONFIG_GENERATION: u32 = 0x15;
const COMMON_CFG_QUEUE_SELECT: u32 = 0x16;
const COMMON_CFG_QUEUE_SIZE: u32 = 0x18;
const COMMON_CFG_QUEUE_MSIX_VECTOR: u32 = 0x1A;
const COMMON_CFG_QUEUE_ENABLE: u32 = 0x1C;
const COMMON_CFG_QUEUE_NOTIFY_OFF: u32 = 0x1E;
const COMMON_CFG_QUEUE_DESC_LOW: u32 = 0x20;
const COMMON_CFG_QUEUE_DESC_HIGH: u32 = 0x24;
const COMMON_CFG_QUEUE_AVAIL_LOW: u32 = 0x28;
const COMMON_CFG_QUEUE_AVAIL_HIGH: u32 = 0x2C;
const COMMON_CFG_QUEUE_USED_LOW: u32 = 0x30;
const COMMON_CFG_QUEUE_USED_HIGH: u32 = 0x34;
