//! Block device abstraction for userspace drivers.
//!
//! Provides a minimal trait for reading sectors from block devices,
//! used by filesystem implementations like ISO9660.

/// Error type for block device operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockError {
    /// Invalid parameters (e.g., buffer too small).
    InvalidParam,
    /// I/O error during read.
    IoError,
    /// Device not ready.
    NotReady,
    /// LBA out of range.
    OutOfRange,
    /// Operation not supported.
    NotSupported,
}

/// Trait for block devices that support sector-based reads.
///
/// Implementations provide synchronous sector reads. The sector size
/// is device-dependent (typically 512 for ATA, 2048 for ATAPI CD-ROM).
pub trait BlockDevice {
    /// Read sectors from the device.
    ///
    /// # Arguments
    /// * `lba` - Logical block address to start reading from.
    /// * `count` - Number of sectors to read.
    /// * `buf` - Buffer to read into. Must be at least `count * sector_size()` bytes.
    ///
    /// # Returns
    /// `Ok(())` on success, `Err(BlockError)` on failure.
    fn read_sectors(&self, lba: u64, count: u64, buf: &mut [u8]) -> Result<(), BlockError>;

    /// Returns the sector size in bytes.
    fn sector_size(&self) -> u64;

    /// Returns the total number of sectors on the device, if known.
    fn sector_count(&self) -> Option<u64> {
        None
    }
}
