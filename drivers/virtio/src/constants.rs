//! Virtio constants and register offsets
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

// Virtio PCI common configuration offsets (VirtIO 1.0+ spec)
pub const VIRTIO_COMMON_DEVICE_FEATURE_SELECT: u32 = 0x00;
pub const VIRTIO_COMMON_DEVICE_FEATURE: u32 = 0x04;
pub const VIRTIO_COMMON_DRIVER_FEATURE_SELECT: u32 = 0x08;
pub const VIRTIO_COMMON_DRIVER_FEATURE: u32 = 0x0C;
pub const VIRTIO_COMMON_MSIX_CONFIG: u32 = 0x10;
pub const VIRTIO_COMMON_NUM_QUEUES: u32 = 0x12;
pub const VIRTIO_COMMON_STATUS: u32 = 0x14;
pub const VIRTIO_COMMON_CFG_GENERATION: u32 = 0x15;

// Queue config
pub const VIRTIO_COMMON_QUEUE_SELECT: u32 = 0x16;
pub const VIRTIO_COMMON_QUEUE_SIZE: u32 = 0x18;
pub const VIRTIO_COMMON_QUEUE_MSIX_VECTOR: u32 = 0x1A;
pub const VIRTIO_COMMON_QUEUE_ENABLE: u32 = 0x1C;
pub const VIRTIO_COMMON_QUEUE_NOTIFY_OFF: u32 = 0x1E;
pub const VIRTIO_COMMON_QUEUE_DESC_LO: u32 = 0x20;
pub const VIRTIO_COMMON_QUEUE_DESC_HI: u32 = 0x24;
pub const VIRTIO_COMMON_QUEUE_AVAIL_LO: u32 = 0x28;
pub const VIRTIO_COMMON_QUEUE_AVAIL_HI: u32 = 0x2C;
pub const VIRTIO_COMMON_QUEUE_USED_LO: u32 = 0x30;
pub const VIRTIO_COMMON_QUEUE_USED_HI: u32 = 0x34;

// Device status bits
pub const VIRTIO_STATUS_ACKNOWLEDGE: u32 = 1;
pub const VIRTIO_STATUS_DRIVER: u32 = 2;
pub const VIRTIO_STATUS_DRIVER_OK: u32 = 4;
pub const VIRTIO_STATUS_FEATURES_OK: u32 = 8;
pub const VIRTIO_STATUS_DEVICE_NEEDS_RESET: u32 = 64;
pub const VIRTIO_STATUS_FAILED: u32 = 128;

// Feature bits
pub const VIRTIO_F_VERSION_1: u64 = 1 << 32;

// Virtio desc flags
pub const VIRTQ_DESC_F_NEXT: u16 = 1;
pub const VIRTQ_DESC_F_WRITE: u16 = 2;
pub const VIRTQ_DESC_F_INDIRECT: u16 = 4;
