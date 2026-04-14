//! Block Device RPC Protocol
//!
//! Defines the port-based RPC protocol for block device drivers.
//! Drivers implement this protocol to expose block-level read/write operations.

/// Request types for block device RPC
///
/// All requests must be prefixed with a 4-byte response port (u32, little-endian).
/// Message format: [u32: response_port][u8: request_type][payload...]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDeviceRequest {
    /// Get device identification and capabilities
    Identify = 0,
    /// Read sectors from the device
    Read = 1,
    /// Write sectors to the device (optional, may not be supported)
    Write = 2,
    /// Flush any cached writes to persistent storage
    Flush = 3,
}

/// Response types for block device RPC
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDeviceResponse {
    /// Success response
    Ok = 0,
    /// Error response
    Error = 1,
}

/// Error codes for block device operations
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDeviceError {
    /// Invalid parameters
    InvalidParam = 1,
    /// I/O error
    IoError = 2,
    /// Device not ready
    NotReady = 3,
    /// LBA out of range
    OutOfRange = 4,
    /// Operation not supported
    NotSupported = 5,
}

/// Device flags returned by Identify
pub mod device_flags {
    /// Device supports 48-bit LBA addressing
    pub const LBA48: u32 = 1 << 0;
    /// Device supports write operations
    pub const WRITABLE: u32 = 1 << 1;
    /// Device supports flush operations
    pub const FLUSHABLE: u32 = 1 << 2;
    /// Device is removable media
    pub const REMOVABLE: u32 = 1 << 3;
}

/// Wire format for Identify request (0 bytes, just the request type)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IdentifyRequest {
    // Empty - just the request type in the message
}

/// Wire format for Identify response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IdentifyResponse {
    /// Sector size in bytes (typically 512 or 2048)
    pub sector_size: u32,
    /// Total number of sectors
    pub sector_count: u64,
    /// Device model string (40 bytes, ASCII, space-padded)
    pub model: [u8; 40],
    /// Device serial number (20 bytes, ASCII, space-padded)
    pub serial: [u8; 20],
    /// Device capability flags (see device_flags module)
    pub flags: u32,
}

/// Wire format for Read request
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReadRequest {
    /// Starting LBA to read from
    pub lba: u64,
    /// Number of sectors to read
    pub sector_count: u32,
}

/// Wire format for Read response header
/// Followed by the actual sector data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReadResponse {
    /// Number of bytes that follow this header
    pub data_len: u32,
}

/// Wire format for Write request header
/// Followed by the actual sector data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WriteRequest {
    /// Starting LBA to write to
    pub lba: u64,
    /// Number of sectors to write
    pub sector_count: u32,
}

/// Wire format for Write response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WriteResponse {
    /// Reserved for future use
    pub _reserved: u32,
}

/// Wire format for Flush request (0 bytes, just the request type)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlushRequest {
    // Empty - just the request type in the message
}

/// Wire format for Flush response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlushResponse {
    /// Reserved for future use
    pub _reserved: u32,
}

/// Wire format for Error response
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ErrorResponse {
    /// Error code (see BlockDeviceError)
    pub error_code: u8,
    /// Reserved for future use
    pub _reserved: [u8; 3],
}
