//! VirtIO device support for Thing-OS
//!
//! This module provides core VirtIO infrastructure including:
//! - PCI capability parsing for modern VirtIO devices
//! - Virtqueue management (descriptor rings, available/used rings)
//! - Common configuration access
//! - Device-specific implementations (virtio-net, etc.)

pub mod pci;
pub mod queue;

pub use pci::{VirtioCapability, VirtioPciDevice};
pub use queue::{Virtqueue, VirtqueueDescriptor};

/// VirtIO device status bits
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Reset - device is in reset state
    Reset = 0,
    /// Acknowledge - guest OS has found the device
    Acknowledge = 1,
    /// Driver - guest OS knows how to drive the device
    Driver = 2,
    /// FeaturesOk - driver has acknowledged feature bits
    FeaturesOk = 8,
    /// DriverOk - driver is set up and ready
    DriverOk = 4,
    /// Failed - something went wrong
    Failed = 128,
}

/// VirtIO feature bits (common across all device types)
pub mod features {
    /// Device provides notification suppression
    pub const VIRTIO_F_NOTIFY_ON_EMPTY: u64 = 1 << 24;
    /// Device supports used buffer notification suppression
    pub const VIRTIO_F_ANY_LAYOUT: u64 = 1 << 27;
    /// Device supports ring event indexes
    pub const VIRTIO_F_RING_EVENT_IDX: u64 = 1 << 29;
    /// Device operates in modern mode (VirtIO 1.0+)
    pub const VIRTIO_F_VERSION_1: u64 = 1 << 32;
}

/// VirtIO network device feature bits
pub mod net_features {
    /// Device has MAC address
    pub const VIRTIO_NET_F_MAC: u64 = 1 << 5;
    /// Device has status register
    pub const VIRTIO_NET_F_STATUS: u64 = 1 << 16;
    /// Device supports MTU
    pub const VIRTIO_NET_F_MTU: u64 = 1 << 3;
}

/// VirtIO device types
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtioDeviceType {
    Network = 1,
    Block = 2,
    Console = 3,
    Entropy = 4,
    Gpu = 16,
    Input = 18,
}
