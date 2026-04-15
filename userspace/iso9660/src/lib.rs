//! ISO9660 Filesystem Parser
//!
//! Minimal, read-only ISO9660 parser for reading files from CD-ROM boot media.
//! Supports Level 1/2 interchange with basic Rock Ridge NM (Alternate Name) support.
//!
//! ## Performance Optimizations
//!
//! This implementation includes several optimizations to eliminate pathological
//! slowness during asset lookup:
//!
//! ### Directory Caching
//! - Each directory extent is parsed once and cached in a `BTreeMap`
//! - Cache key: `(extent_lba, data_length)` uniquely identifies a directory
//! - Subsequent lookups in the same directory reuse cached parsed entries
//! - No eviction policy (ISOs are immutable, cache lifetime = mount lifetime)
//!
//! ### Allocation-Free Filename Matching
//! - Path resolution uses `ascii_eq_ignore_case()` for case-insensitive comparison
//! - Eliminates `to_uppercase()` allocations in the hot path
//! - Properly handles ISO9660 version suffixes (e.g., `;1`)
//!
//! ### Performance Instrumentation (optional)
//! - Enable with `--features perf` to track:
//!   - `dir_parses`: Number of directory extents parsed from disk
//!   - `cache_hits`: Number of cache reuses
//!   - `path_resolution_steps`: Number of path segments traversed
//!   - `bytes_read`: Total bytes read during directory operations
//!
//! ## Expected Performance Impact
//! - O(N × M) → O(N) complexity for path lookups (N = segments, M = entries)
//! - Order-of-magnitude improvement in asset scan time
//! - Zero heap allocations during filename matching
#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use stem::block::{BlockDevice, BlockError};

/// ISO9660 sector size (logical block size).
pub const ISO_SECTOR_SIZE: u64 = 2048;

/// Primary Volume Descriptor location (sector 16).
const PVD_SECTOR: u64 = 16;

/// Volume descriptor type codes.
const VD_TYPE_PRIMARY: u8 = 1;
#[allow(dead_code)]
const VD_TYPE_TERMINATOR: u8 = 255;

/// Parsed Primary Volume Descriptor.
#[derive(Debug)]
pub struct PrimaryVolumeDescriptor {
    pub system_id: [u8; 32],
    pub volume_id: [u8; 32],
    pub volume_space_size: u32,
    pub root_dir_extent: u32,
    pub root_dir_size: u32,
    pub logical_block_size: u16,
}

/// Performance counters for ISO operations.
#[cfg(feature = "perf")]
#[derive(Debug, Default)]
pub struct PerfCounters {
    pub dir_parses: u64,
    pub cache_hits: u64,
    pub path_resolution_steps: u64,
    pub bytes_read: u64,
}

#[cfg(not(feature = "perf"))]
#[derive(Debug, Default)]
pub struct PerfCounters;

/// Directory cache key: uniquely identifies a directory extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DirCacheKey {
    extent_lba: u32,
    data_length: u32,
}

/// Cached directory index with parsed entries.
#[derive(Debug, Clone)]
struct DirIndex {
    entries: Vec<IsoDirEntry>,
}

/// Directory entry from the ISO9660 filesystem.
#[derive(Debug, Clone)]
pub struct IsoDirEntry {
    pub name: String,
    pub extent_lba: u32,
    pub size: u32,
    pub is_directory: bool,
}

/// An open file handle with extent information.
#[derive(Debug)]
pub struct IsoFile {
    pub extent_lba: u32,
    pub size: u32,
}

/// Mounted ISO9660 filesystem.
pub struct IsoFs {
    pub pvd: PrimaryVolumeDescriptor,
    dir_cache: RefCell<BTreeMap<DirCacheKey, DirIndex>>,
    #[cfg(feature = "perf")]
    pub perf: RefCell<PerfCounters>,
}

impl IsoFs {
    /// Probe a block device for ISO9660 filesystem.
    ///
    /// Returns `Some(IsoFs)` if a valid ISO9660 Primary Volume Descriptor is found.
    pub fn probe(dev: &dyn BlockDevice) -> Option<Self> {
        let mut buf = [0u8; 2048];

        // Read PVD sector
        if dev.read_sectors(PVD_SECTOR, 1, &mut buf).is_err() {
            return None;
        }

        // Check for "CD001" signature at offset 1
        if &buf[1..6] != b"CD001" {
            return None;
        }

        // Check volume descriptor type
        if buf[0] != VD_TYPE_PRIMARY {
            return None;
        }

        // Check version
        if buf[6] != 1 {
            return None;
        }

        // Parse PVD fields
        let mut system_id = [0u8; 32];
        system_id.copy_from_slice(&buf[8..40]);

        let mut volume_id = [0u8; 32];
        volume_id.copy_from_slice(&buf[40..72]);

        // Volume space size (little-endian at offset 80)
        let volume_space_size = u32::from_le_bytes([buf[80], buf[81], buf[82], buf[83]]);

        // Logical block size (little-endian at offset 128)
        let logical_block_size = u16::from_le_bytes([buf[128], buf[129]]);

        // Root directory record is at offset 156
        let root_dir_extent = u32::from_le_bytes([buf[158], buf[159], buf[160], buf[161]]);
        let root_dir_size = u32::from_le_bytes([buf[166], buf[167], buf[168], buf[169]]);

        Some(IsoFs {
            pvd: PrimaryVolumeDescriptor {
                system_id,
                volume_id,
                volume_space_size,
                root_dir_extent,
                root_dir_size,
                logical_block_size,
            },
            dir_cache: RefCell::new(BTreeMap::new()),
            #[cfg(feature = "perf")]
            perf: RefCell::new(PerfCounters::default()),
        })
    }

    /// ASCII case-insensitive string comparison without allocation.
    /// Handles ISO9660 version suffixes (e.g., ";1") in the search path `b`.
    /// `a` is the directory entry name, which is assumed to be either stripped of ISO version
    /// or a raw Rock Ridge name (which may contain semicolons).
    fn ascii_eq_ignore_case(a: &str, b: &str) -> bool {
        // Check for direct match first (handling Rock Ridge names with semicolons)
        if a.eq_ignore_ascii_case(b) {
            return true;
        }

        // If direct match failed, try stripping version suffix from search path `b`
        // (to handle looking up "FILE;1" against "FILE")
        if let Some(pos) = b.rfind(';') {
            let suffix = &b[pos + 1..];
            // Only strip if suffix is numeric (version number)
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                let b_stripped = &b[..pos];
                if a.eq_ignore_ascii_case(b_stripped) {
                    return true;
                }
            }
        }

        false
    }

    /// Parse directory entries from extent and cache them.
    /// Returns the cached directory entries.
    fn parse_and_cache_dir(
        &self,
        dev: &dyn BlockDevice,
        extent_lba: u32,
        size: u32,
    ) -> Vec<IsoDirEntry> {
        let key = DirCacheKey {
            extent_lba,
            data_length: size,
        };

        // Check if already cached
        {
            let cache = self.dir_cache.borrow();
            if let Some(index) = cache.get(&key) {
                #[cfg(feature = "perf")]
                {
                    self.perf.borrow_mut().cache_hits += 1;
                }
                return index.entries.clone();
            }
        }

        // Not cached, parse it
        #[cfg(feature = "perf")]
        {
            self.perf.borrow_mut().dir_parses += 1;
        }

        let entries = self.parse_dir_entries(dev, extent_lba, size);

        // Cache the parsed entries and return a clone from the cache
        self.dir_cache
            .borrow_mut()
            .insert(key, DirIndex { entries });

        // Return a clone of the cached entries
        self.dir_cache.borrow().get(&key).unwrap().entries.clone()
    }

    /// Parse directory entries from an extent (internal implementation).
    fn parse_dir_entries(
        &self,
        dev: &dyn BlockDevice,
        extent_lba: u32,
        size: u32,
    ) -> Vec<IsoDirEntry> {
        let mut entries = Vec::new();
        let sectors_needed = (size as u64 + ISO_SECTOR_SIZE - 1) / ISO_SECTOR_SIZE;

        // Allocate buffer for directory data
        let buf_size = (sectors_needed * ISO_SECTOR_SIZE) as usize;
        let mut buf = alloc::vec![0u8; buf_size];

        if dev
            .read_sectors(extent_lba as u64, sectors_needed, &mut buf)
            .is_err()
        {
            return entries;
        }

        // Track bytes read only after successful read
        #[cfg(feature = "perf")]
        {
            self.perf.borrow_mut().bytes_read += sectors_needed * ISO_SECTOR_SIZE;
        }

        // Parse directory records
        let mut offset = 0usize;
        while offset < size as usize {
            let record_len = buf[offset] as usize;
            if record_len == 0 {
                // Padding to next sector
                let next_sector =
                    ((offset / ISO_SECTOR_SIZE as usize) + 1) * ISO_SECTOR_SIZE as usize;
                if next_sector >= size as usize {
                    break;
                }
                offset = next_sector;
                continue;
            }

            if offset + record_len > buf.len() {
                break;
            }

            let extent = u32::from_le_bytes([
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
            ]);
            let file_size = u32::from_le_bytes([
                buf[offset + 10],
                buf[offset + 11],
                buf[offset + 12],
                buf[offset + 13],
            ]);
            let flags = buf[offset + 25];
            let name_len = buf[offset + 32] as usize;

            if name_len > 0 && offset + 33 + name_len <= buf.len() {
                let name_bytes = &buf[offset + 33..offset + 33 + name_len];

                // Skip . and .. entries
                if name_bytes == [0x00] || name_bytes == [0x01] {
                    offset += record_len;
                    continue;
                }

                // Convert name to string, strip version suffix
                let name_str = core::str::from_utf8(name_bytes).unwrap_or("");
                let mut name = if let Some(pos) = name_str.find(';') {
                    String::from(&name_str[..pos])
                } else {
                    String::from(name_str)
                };

                // Parse System Use Area for Rock Ridge NM (Alternate Name)
                let mut sys_use_offset = offset + 33 + name_len;
                if name_len % 2 == 0 {
                    sys_use_offset += 1;
                }

                let mut rock_ridge_name = String::new();
                let mut found_nm = false;

                while sys_use_offset + 4 <= offset + record_len {
                    let sig = &buf[sys_use_offset..sys_use_offset + 2];
                    let len = buf[sys_use_offset + 2] as usize;
                    let _ver = buf[sys_use_offset + 3];

                    if len < 4 || sys_use_offset + len > offset + record_len {
                        break;
                    }

                    if sig == b"NM" {
                        let _flags = buf[sys_use_offset + 4];
                        let name_start = sys_use_offset + 5;
                        let name_end = sys_use_offset + len;

                        if name_end > name_start {
                            if let Ok(nm_part) = core::str::from_utf8(&buf[name_start..name_end]) {
                                rock_ridge_name.push_str(nm_part);
                                found_nm = true;
                            }
                        }

                        // If CONTINUE bit (0) or others are not set, we might be done,
                        // but NM entries can be split. We just append them all.
                    } else if sig == b"CE" {
                        // Continuation Area entry.
                        // Format: sig(2) + len(1) + ver(1)
                        //         + block_le(4) + block_be(4)
                        //         + offset_le(4) + offset_be(4)
                        //         + length_le(4) + length_be(4) = 28 bytes total
                        if len >= 28 {
                            let ce_block = u32::from_le_bytes([
                                buf[sys_use_offset + 4],
                                buf[sys_use_offset + 5],
                                buf[sys_use_offset + 6],
                                buf[sys_use_offset + 7],
                            ]);
                            let ce_byte_offset = u32::from_le_bytes([
                                buf[sys_use_offset + 12],
                                buf[sys_use_offset + 13],
                                buf[sys_use_offset + 14],
                                buf[sys_use_offset + 15],
                            ]);
                            let ce_length = u32::from_le_bytes([
                                buf[sys_use_offset + 20],
                                buf[sys_use_offset + 21],
                                buf[sys_use_offset + 22],
                                buf[sys_use_offset + 23],
                            ]);

                            if ce_length > 0 {
                                // Sectors needed to cover ce_byte_offset + ce_length bytes
                                let sectors_needed = ((ce_byte_offset as usize
                                    + ce_length as usize)
                                    + ISO_SECTOR_SIZE as usize
                                    - 1)
                                    / ISO_SECTOR_SIZE as usize;
                                let mut ce_buf = alloc::vec![
                                    0u8;
                                    sectors_needed * ISO_SECTOR_SIZE as usize
                                ];
                                if dev
                                    .read_sectors(
                                        ce_block as u64,
                                        sectors_needed as u64,
                                        &mut ce_buf,
                                    )
                                    .is_ok()
                                {
                                    let ce_start = ce_byte_offset as usize;
                                    let ce_end = ce_start + ce_length as usize;
                                    let mut ce_pos = ce_start;
                                    while ce_pos + 4 <= ce_end
                                        && ce_pos + 4 <= ce_buf.len()
                                    {
                                        let ce_sig =
                                            &ce_buf[ce_pos..ce_pos + 2];
                                        let ce_len =
                                            ce_buf[ce_pos + 2] as usize;
                                        if ce_len < 4
                                            || ce_pos + ce_len > ce_end
                                            || ce_pos + ce_len > ce_buf.len()
                                        {
                                            break;
                                        }
                                        if ce_sig == b"NM" {
                                            let name_start = ce_pos + 5;
                                            let name_end = ce_pos + ce_len;
                                            if name_end > name_start {
                                                if let Ok(nm_part) =
                                                    core::str::from_utf8(
                                                        &ce_buf[name_start
                                                            ..name_end],
                                                    )
                                                {
                                                    rock_ridge_name
                                                        .push_str(nm_part);
                                                    found_nm = true;
                                                }
                                            }
                                        } else if ce_sig == b"ST" {
                                            break;
                                        }
                                        ce_pos += ce_len;
                                    }
                                }
                            }
                        }
                    } else if sig == b"ST" {
                        // Terminator
                        break;
                    }

                    sys_use_offset += len;
                }

                if found_nm {
                    name = rock_ridge_name;
                }

                entries.push(IsoDirEntry {
                    name,
                    extent_lba: extent,
                    size: file_size,
                    is_directory: (flags & 0x02) != 0,
                });
            }

            offset += record_len;
        }

        entries
    }

    /// List entries in a directory.
    pub fn list_dir(&self, dev: &dyn BlockDevice, extent_lba: u32, size: u32) -> Vec<IsoDirEntry> {
        // Use the cache to avoid re-parsing directories
        self.parse_and_cache_dir(dev, extent_lba, size)
    }

    /// List root directory entries.
    pub fn list_root(&self, dev: &dyn BlockDevice) -> Vec<IsoDirEntry> {
        self.list_dir(dev, self.pvd.root_dir_extent, self.pvd.root_dir_size)
    }

    /// Look up a path and return its directory entry (file or directory).
    ///
    /// `path` may be relative (no leading `/`) or absolute.  Returns `None`
    /// if any component does not exist.
    ///
    /// Unlike [`open_path`][Self::open_path] this returns the full
    /// [`IsoDirEntry`] so callers can distinguish files from directories.
    pub fn lookup_path(&self, dev: &dyn BlockDevice, path: &str) -> Option<IsoDirEntry> {
        let path = path.trim_start_matches('/');
        if path.is_empty() {
            // Root directory pseudo-entry.
            return Some(IsoDirEntry {
                name: String::from(""),
                extent_lba: self.pvd.root_dir_extent,
                size: self.pvd.root_dir_size,
                is_directory: true,
            });
        }

        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        let mut current_extent = self.pvd.root_dir_extent;
        let mut current_size = self.pvd.root_dir_size;

        for (i, part) in parts.iter().enumerate() {
            let is_last = i == parts.len() - 1;
            let entries = self.list_dir(dev, current_extent, current_size);
            let entry = entries
                .iter()
                .find(|e| Self::ascii_eq_ignore_case(&e.name, part))?
                .clone();

            if is_last {
                return Some(entry);
            } else {
                if !entry.is_directory {
                    return None;
                }
                current_extent = entry.extent_lba;
                current_size = entry.size;
            }
        }

        None
    }

    /// Resolve a path and return the directory entry.
    ///
    /// Path should be absolute, e.g., "/ASSETS/CURSOR.SVG".
    pub fn open_path(&self, dev: &dyn BlockDevice, path: &str) -> Option<IsoFile> {
        let path = path.trim_start_matches('/');
        if path.is_empty() {
            return None;
        }

        let parts: Vec<&str> = path.split('/').collect();
        let mut current_extent = self.pvd.root_dir_extent;
        let mut current_size = self.pvd.root_dir_size;

        for (i, part) in parts.iter().enumerate() {
            #[cfg(feature = "perf")]
            {
                self.perf.borrow_mut().path_resolution_steps += 1;
            }

            let is_last = i == parts.len() - 1;
            let entries = self.list_dir(dev, current_extent, current_size);

            // Use allocation-free case-insensitive comparison
            let entry = entries
                .iter()
                .find(|e| Self::ascii_eq_ignore_case(&e.name, part))?;

            if is_last {
                return Some(IsoFile {
                    extent_lba: entry.extent_lba,
                    size: entry.size,
                });
            } else {
                if !entry.is_directory {
                    return None;
                }
                current_extent = entry.extent_lba;
                current_size = entry.size;
            }
        }

        None
    }
}

impl IsoFile {
    /// Read the entire file contents.
    pub fn read_all(&self, dev: &dyn BlockDevice) -> Result<Vec<u8>, BlockError> {
        let sectors_needed = (self.size as u64 + ISO_SECTOR_SIZE - 1) / ISO_SECTOR_SIZE;
        let buf_size = (sectors_needed * ISO_SECTOR_SIZE) as usize;
        let mut buf = alloc::vec![0u8; buf_size];

        dev.read_sectors(self.extent_lba as u64, sectors_needed, &mut buf)?;

        // Truncate to actual file size
        buf.truncate(self.size as usize);
        Ok(buf)
    }

    /// Read a range of bytes from the file.
    pub fn read_range(
        &self,
        dev: &dyn BlockDevice,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, BlockError> {
        if offset >= self.size as u64 {
            return Ok(Vec::new());
        }

        let actual_len = core::cmp::min(length, (self.size as u64 - offset) as usize);
        let start_sector = offset / ISO_SECTOR_SIZE;
        let end_offset = offset + actual_len as u64;
        let end_sector = (end_offset + ISO_SECTOR_SIZE - 1) / ISO_SECTOR_SIZE;
        let sectors_to_read = end_sector - start_sector;

        let buf_size = (sectors_to_read * ISO_SECTOR_SIZE) as usize;
        let mut buf = alloc::vec![0u8; buf_size];

        dev.read_sectors(
            self.extent_lba as u64 + start_sector,
            sectors_to_read,
            &mut buf,
        )?;

        let start_in_buf = (offset % ISO_SECTOR_SIZE) as usize;
        Ok(buf[start_in_buf..start_in_buf + actual_len].to_vec())
    }
}

/// Helper to convert volume ID to a trimmed string.
pub fn volume_id_str(pvd: &PrimaryVolumeDescriptor) -> &str {
    core::str::from_utf8(&pvd.volume_id).unwrap_or("").trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_eq_ignore_case() {
        // Basic case-insensitive comparison
        assert!(IsoFs::ascii_eq_ignore_case("HELLO", "hello"));
        assert!(IsoFs::ascii_eq_ignore_case("hello", "HELLO"));
        assert!(IsoFs::ascii_eq_ignore_case("MixedCase", "mixedcase"));

        // Version suffix handling
        // 'a' is the entry name (stripped/raw), 'b' is the search path
        assert!(IsoFs::ascii_eq_ignore_case("FILE.TXT", "file.txt;1"));
        assert!(IsoFs::ascii_eq_ignore_case("FILE.TXT", "file.txt"));

        // We no longer strip 'a', so "README;1" as 'a' means literal "README;1"
        // This simulates a Rock Ridge name "README;1".
        // It should NOT match "readme;2" (which strips to "readme").
        assert!(!IsoFs::ascii_eq_ignore_case("README;1", "readme;2"));

        // But "README" (ISO stripped) matches "readme;2" (stripped)
        assert!(IsoFs::ascii_eq_ignore_case("README", "readme;2"));

        // Different strings
        assert!(!IsoFs::ascii_eq_ignore_case("hello", "world"));
        assert!(!IsoFs::ascii_eq_ignore_case("FILE", "FILES"));

        // Empty strings
        assert!(IsoFs::ascii_eq_ignore_case("", ""));
        assert!(!IsoFs::ascii_eq_ignore_case("", "hello"));
    }

    #[test]
    fn test_ascii_eq_ignore_case_rock_ridge_collision() {
        // Semicolons in names (e.g. Rock Ridge) shouldn't be treated as version separators
        // if they are part of the filename.
        // Current implementation blindly strips everything after ';'.
        assert!(!IsoFs::ascii_eq_ignore_case("foo;bar", "foo;baz"));
    }

    #[test]
    fn test_ascii_eq_ignore_case_false_positive_stripping() {
        // If we look for "foo;bar" (e.g. a Rock Ridge name), we should NOT match "foo".
        // The semicolon in the search path might be part of the name, not a version separator.
        // Currently this fails (returns true) because we blindly strip from the last semicolon.
        assert!(!IsoFs::ascii_eq_ignore_case("foo", "foo;bar"));
    }

    struct MockBlockDevice {
        pvd_data: [u8; 2048],
    }

    impl BlockDevice for MockBlockDevice {
        fn read_sectors(
            &self,
            lba: u64,
            count: u64,
            buf: &mut [u8],
        ) -> Result<(), stem::block::BlockError> {
            if lba == PVD_SECTOR && count == 1 {
                buf[0..2048].copy_from_slice(&self.pvd_data);
                Ok(())
            } else {
                Err(stem::block::BlockError::IoError)
            }
        }

        fn sector_size(&self) -> u64 {
            2048
        }
    }

    #[test]
    fn test_probe_valid_pvd() {
        let mut pvd_data = [0u8; 2048];
        pvd_data[0] = VD_TYPE_PRIMARY;
        pvd_data[1..6].copy_from_slice(b"CD001");
        pvd_data[6] = 1; // Version

        let sys_id = b"TEST_SYSTEM";
        pvd_data[8..8 + sys_id.len()].copy_from_slice(sys_id);

        let vol_id = b"TEST_VOLUME";
        pvd_data[40..40 + vol_id.len()].copy_from_slice(vol_id);

        // Root dir extent (LBA 1234)
        let root_lba = 1234u32;
        pvd_data[158..162].copy_from_slice(&root_lba.to_le_bytes());

        // Root dir size (2048 bytes)
        let root_size = 2048u32;
        pvd_data[166..170].copy_from_slice(&root_size.to_le_bytes());

        let dev = MockBlockDevice { pvd_data };

        let iso = IsoFs::probe(&dev).expect("Should probe successfully");

        assert_eq!(iso.pvd.root_dir_extent, 1234);
        assert_eq!(iso.pvd.root_dir_size, 2048);
        assert!(volume_id_str(&iso.pvd).starts_with("TEST_VOLUME"));
    }

    #[test]
    fn test_probe_invalid_signature() {
        let mut pvd_data = [0u8; 2048];
        pvd_data[0] = VD_TYPE_PRIMARY;
        pvd_data[1..6].copy_from_slice(b"WRONG"); // Invalid signature
        pvd_data[6] = 1;

        let dev = MockBlockDevice { pvd_data };
        assert!(IsoFs::probe(&dev).is_none());
    }

    struct ReadRangeMockDevice;

    impl BlockDevice for ReadRangeMockDevice {
        fn read_sectors(
            &self,
            lba: u64,
            count: u64,
            buf: &mut [u8],
        ) -> Result<(), stem::block::BlockError> {
            // Fill with a recognizable pattern based on LBA
            for i in 0..count {
                let sector_lba = lba + i;
                let start_idx = (i * 2048) as usize;
                let end_idx = start_idx + 2048;
                let sector_buf = &mut buf[start_idx..end_idx];

                // Simple pattern: byte value = (lba % 255) + offset % 255
                for (j, byte) in sector_buf.iter_mut().enumerate() {
                    *byte = ((sector_lba % 255) as u8).wrapping_add((j % 255) as u8);
                }
            }
            Ok(())
        }

        fn sector_size(&self) -> u64 {
            2048
        }
    }

    #[test]
    fn test_iso_file_read_range() {
        let dev = ReadRangeMockDevice;
        let file = IsoFile {
            extent_lba: 100,
            size: 5000, // 2048 + 2048 + 904
        };

        // 1. Read first 10 bytes
        let data = file.read_range(&dev, 0, 10).expect("read_range failed");
        assert_eq!(data.len(), 10);
        // Verify content matches pattern for LBA 100
        for (j, &byte) in data.iter().enumerate() {
            assert_eq!(byte, ((100 % 255) as u8).wrapping_add((j % 255) as u8));
        }

        // 2. Read across sector boundary (LBA 100 -> 101)
        // Sector 1 ends at 2048. Read from 2040 to 2060 (20 bytes).
        let data = file.read_range(&dev, 2040, 20).expect("read_range failed");
        assert_eq!(data.len(), 20);
        // First 8 bytes from LBA 100
        for j in 0..8 {
            let offset_in_sector = 2040 + j;
            let expected = ((100 % 255) as u8).wrapping_add((offset_in_sector % 255) as u8);
            assert_eq!(data[j], expected, "Mismatch at index {}", j);
        }
        // Next 12 bytes from LBA 101
        for j in 8..20 {
            let offset_in_sector = j - 8;
            let expected = ((101 % 255) as u8).wrapping_add((offset_in_sector % 255) as u8);
            assert_eq!(data[j], expected, "Mismatch at index {}", j);
        }

        // 3. Read past EOF
        let data = file.read_range(&dev, 4990, 20).expect("read_range failed");
        // Should return only 10 bytes (4990 to 5000)
        assert_eq!(data.len(), 10);

        // Verify content for the last part (LBA 102)
        // Offset 4990 corresponds to LBA 102, offset in sector = 4990 - 4096 = 894
        for j in 0..10 {
            let offset_in_sector = 894 + j;
            let expected = ((102 % 255) as u8).wrapping_add((offset_in_sector % 255) as u8);
            assert_eq!(data[j], expected, "Mismatch at last chunk index {}", j);
        }
    }

    struct MockDirBlockDevice {
        dir_data: Vec<u8>,
    }

    impl BlockDevice for MockDirBlockDevice {
        fn read_sectors(
            &self,
            _lba: u64,
            count: u64,
            buf: &mut [u8],
        ) -> Result<(), stem::block::BlockError> {
            let len = (count * 2048) as usize;
            if buf.len() < len {
                return Err(stem::block::BlockError::IoError);
            }
            if len > self.dir_data.len() {
                buf[..self.dir_data.len()].copy_from_slice(&self.dir_data);
                buf[self.dir_data.len()..len].fill(0);
            } else {
                buf[..len].copy_from_slice(&self.dir_data[..len]);
            }
            Ok(())
        }
        fn sector_size(&self) -> u64 {
            2048
        }
    }

    fn write_dir_record(
        buf: &mut [u8],
        offset: &mut usize,
        name: &str,
        extent: u32,
        size: u32,
        flags: u8,
        rock_ridge_nm: Option<&str>,
    ) {
        let start = *offset;
        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len();

        // Basic length: 33 fixed + name_len
        let mut record_len = 33 + name_len;
        if name_len % 2 == 0 {
            record_len += 1; // Padding byte after name if len is even
        }

        let mut system_use_len = 0;
        if let Some(rr_name) = rock_ridge_nm {
            // NM (2) + Len (1) + Ver (1) + Flags (1) + name
            system_use_len = 5 + rr_name.len();
        }
        record_len += system_use_len;

        if record_len % 2 != 0 {
            record_len += 1;
        }

        buf[start] = record_len as u8; // Length
        buf[start + 1] = 0; // Ext Attr Len

        // Extent (LE)
        buf[start + 2..start + 6].copy_from_slice(&extent.to_le_bytes());
        // Extent (BE)
        buf[start + 6..start + 10].copy_from_slice(&extent.to_be_bytes());

        // Data Length (LE)
        buf[start + 10..start + 14].copy_from_slice(&size.to_le_bytes());
        // Data Length (BE)
        buf[start + 14..start + 18].copy_from_slice(&size.to_be_bytes());

        buf[start + 25] = flags;

        // Vol seq (1) at 28
        buf[start + 28] = 1;
        buf[start + 30] = 1;

        buf[start + 32] = name_len as u8;
        buf[start + 33..start + 33 + name_len].copy_from_slice(name_bytes);

        let mut sys_use_offset = start + 33 + name_len;
        if name_len % 2 == 0 {
            sys_use_offset += 1; // Padding
        }

        if let Some(rr_name) = rock_ridge_nm {
            buf[sys_use_offset] = b'N';
            buf[sys_use_offset + 1] = b'M';
            let nm_len = 5 + rr_name.len();
            buf[sys_use_offset + 2] = nm_len as u8;
            buf[sys_use_offset + 3] = 1; // Ver
            buf[sys_use_offset + 4] = 0; // Flags
            buf[sys_use_offset + 5..sys_use_offset + 5 + rr_name.len()]
                .copy_from_slice(rr_name.as_bytes());
        }

        *offset += record_len;
    }

    #[test]
    fn test_parse_dir_entries_with_rock_ridge() {
        let mut buf = alloc::vec![0u8; 2048];
        let mut offset = 0;

        // 1. Current Directory "." (0x00)
        write_dir_record(&mut buf, &mut offset, "\x00", 100, 2048, 2, None);

        // 2. Parent Directory ".." (0x01)
        write_dir_record(&mut buf, &mut offset, "\x01", 50, 2048, 2, None);

        // 3. Regular File "TEST.TXT;1"
        write_dir_record(&mut buf, &mut offset, "TEST.TXT;1", 200, 1234, 0, None);

        // 4. Rock Ridge File "LONG_FIL.;1" with NM "LongFileName.txt"
        write_dir_record(
            &mut buf,
            &mut offset,
            "LONG_FIL.;1",
            300,
            5678,
            0,
            Some("LongFileName.txt"),
        );

        let dev = MockDirBlockDevice { dir_data: buf };

        let iso = IsoFs {
            pvd: PrimaryVolumeDescriptor {
                system_id: [0; 32],
                volume_id: [0; 32],
                volume_space_size: 0,
                root_dir_extent: 0,
                root_dir_size: 0,
                logical_block_size: 2048,
            },
            dir_cache: RefCell::new(BTreeMap::new()),
            #[cfg(feature = "perf")]
            perf: RefCell::new(PerfCounters::default()),
        };

        let entries = iso.parse_dir_entries(&dev, 0, 2048);

        // Verify results
        assert_eq!(entries.len(), 2, "Should have 2 entries (skipped . and ..)");

        // Entry 1: TEST.TXT
        let e1 = &entries[0];
        assert_eq!(e1.name, "TEST.TXT");
        assert_eq!(e1.extent_lba, 200);
        assert_eq!(e1.size, 1234);
        assert!(!e1.is_directory);

        // Entry 2: LongFileName.txt
        let e2 = &entries[1];
        assert_eq!(e2.name, "LongFileName.txt");
        assert_eq!(e2.extent_lba, 300);
        assert_eq!(e2.size, 5678);
        assert!(!e2.is_directory);
    }

    /// Build a MockIsoImage whose sector 0 is a directory with two entries:
    /// "." / ".." and one file whose Rock Ridge NM is split via a CE entry.
    ///
    /// The directory record contains:
    ///   • NM(flags=CONTINUE, "Split") — first half, continue flag set
    ///   • CE(block=ce_lba, offset=0, length=<ce_area_len>)
    ///
    /// The continuation area (sector `ce_lba`) contains:
    ///   • NM(flags=0, "File.txt") — second half
    ///
    /// Expected resolved name: "SplitFile.txt"
    struct CeMockDevice {
        /// Directory sector (LBA 0).
        dir_sector: Vec<u8>,
        /// Continuation area sector.
        ce_lba: u64,
        ce_sector: Vec<u8>,
    }

    impl BlockDevice for CeMockDevice {
        fn read_sectors(
            &self,
            lba: u64,
            count: u64,
            buf: &mut [u8],
        ) -> Result<(), stem::block::BlockError> {
            for i in 0..count {
                let sector_lba = lba + i;
                let start = (i * 2048) as usize;
                let end = start + 2048;
                if buf.len() < end {
                    return Err(stem::block::BlockError::IoError);
                }
                let src: &[u8] = if sector_lba == 0 {
                    &self.dir_sector
                } else if sector_lba == self.ce_lba {
                    &self.ce_sector
                } else {
                    return Err(stem::block::BlockError::IoError);
                };
                let copy_len = src.len().min(2048);
                buf[start..start + copy_len].copy_from_slice(&src[..copy_len]);
                buf[start + copy_len..end].fill(0);
            }
            Ok(())
        }
        fn sector_size(&self) -> u64 {
            2048
        }
    }

    #[test]
    fn test_parse_dir_entries_ce_continuation() {
        const CE_LBA: u32 = 42;

        // --- Build continuation area (sector CE_LBA) ---
        // NM(flags=0, "File.txt") — second fragment
        let ce_nm_name = b"File.txt";
        let ce_nm_len: u8 = 5 + ce_nm_name.len() as u8; // sig(2)+len(1)+ver(1)+flags(1)+name
        let mut ce_sector = alloc::vec![0u8; 2048];
        let mut cp = 0usize;
        ce_sector[cp] = b'N';
        ce_sector[cp + 1] = b'M';
        ce_sector[cp + 2] = ce_nm_len;
        ce_sector[cp + 3] = 1; // version
        ce_sector[cp + 4] = 0; // flags: no continue
        ce_sector[cp + 5..cp + 5 + ce_nm_name.len()].copy_from_slice(ce_nm_name);
        cp += ce_nm_len as usize;
        let ce_area_len = cp as u32;

        // --- Build directory sector (LBA 0) ---
        let mut dir_buf = alloc::vec![0u8; 2048];
        let mut off = 0usize;

        // "." and ".."
        write_dir_record(&mut dir_buf, &mut off, "\x00", 0, 2048, 2, None);
        write_dir_record(&mut dir_buf, &mut off, "\x01", 0, 2048, 2, None);

        // Build the file entry manually so we can embed CE + split NM.
        let iso_name = b"SPLITFIL.;1";
        let iso_name_len = iso_name.len();

        // NM entry for first fragment: "Split" with CONTINUE flag (bit 0 = 1)
        let nm1_name = b"Split";
        let nm1_len: u8 = 5 + nm1_name.len() as u8;

        // CE entry: 28 bytes fixed
        let ce_entry_len: u8 = 28;

        // System-use area size
        let sys_use_len = nm1_len as usize + ce_entry_len as usize;

        // record_len must be even
        let mut record_len = 33 + iso_name_len + sys_use_len;
        if record_len % 2 != 0 {
            record_len += 1;
        }
        // Padding after name if name_len is even
        let mut sys_use_start = 33 + iso_name_len;
        if iso_name_len % 2 == 0 {
            sys_use_start += 1;
        }

        let rec_start = off;
        dir_buf[rec_start] = record_len as u8;
        // extent (LE at +2)
        let file_extent: u32 = 300;
        let file_size: u32 = 9999;
        dir_buf[rec_start + 2..rec_start + 6].copy_from_slice(&file_extent.to_le_bytes());
        dir_buf[rec_start + 10..rec_start + 14].copy_from_slice(&file_size.to_le_bytes());
        dir_buf[rec_start + 25] = 0; // file flags
        dir_buf[rec_start + 32] = iso_name_len as u8;
        dir_buf[rec_start + 33..rec_start + 33 + iso_name_len].copy_from_slice(iso_name);

        // Write NM entry (first fragment, CONTINUE flag)
        let su = rec_start + sys_use_start;
        dir_buf[su] = b'N';
        dir_buf[su + 1] = b'M';
        dir_buf[su + 2] = nm1_len;
        dir_buf[su + 3] = 1; // version
        dir_buf[su + 4] = 0x01; // CONTINUE flag
        dir_buf[su + 5..su + 5 + nm1_name.len()].copy_from_slice(nm1_name);

        // Write CE entry immediately after NM
        let ce_su = su + nm1_len as usize;
        dir_buf[ce_su] = b'C';
        dir_buf[ce_su + 1] = b'E';
        dir_buf[ce_su + 2] = ce_entry_len;
        dir_buf[ce_su + 3] = 1; // version
        // block LE at +4
        dir_buf[ce_su + 4..ce_su + 8].copy_from_slice(&CE_LBA.to_le_bytes());
        // block BE at +8
        dir_buf[ce_su + 8..ce_su + 12].copy_from_slice(&CE_LBA.to_be_bytes());
        // offset LE at +12 (0)
        dir_buf[ce_su + 12..ce_su + 16].copy_from_slice(&0u32.to_le_bytes());
        // offset BE at +16 (0)
        dir_buf[ce_su + 16..ce_su + 20].copy_from_slice(&0u32.to_be_bytes());
        // length LE at +20
        dir_buf[ce_su + 20..ce_su + 24].copy_from_slice(&ce_area_len.to_le_bytes());
        // length BE at +24
        dir_buf[ce_su + 24..ce_su + 28].copy_from_slice(&ce_area_len.to_be_bytes());

        off += record_len;

        let dev = CeMockDevice {
            dir_sector: dir_buf,
            ce_lba: CE_LBA as u64,
            ce_sector,
        };

        let iso = IsoFs {
            pvd: PrimaryVolumeDescriptor {
                system_id: [0; 32],
                volume_id: [0; 32],
                volume_space_size: 0,
                root_dir_extent: 0,
                root_dir_size: 0,
                logical_block_size: 2048,
            },
            dir_cache: RefCell::new(BTreeMap::new()),
            #[cfg(feature = "perf")]
            perf: RefCell::new(PerfCounters::default()),
        };

        let entries = iso.parse_dir_entries(&dev, 0, off as u32);
        assert_eq!(entries.len(), 1, "Should have exactly one file entry");

        let e = &entries[0];
        assert_eq!(
            e.name, "SplitFile.txt",
            "Rock Ridge name should be stitched from CE continuation"
        );
        assert_eq!(e.extent_lba, file_extent);
        assert_eq!(e.size, file_size);
        assert!(!e.is_directory);
    }

    struct MockIsoImage {
        sectors: BTreeMap<u64, Vec<u8>>,
    }

    impl MockIsoImage {
        fn new() -> Self {
            Self {
                sectors: BTreeMap::new(),
            }
        }

        fn set_sector(&mut self, lba: u64, data: &[u8]) {
            let mut sector = alloc::vec![0u8; 2048];
            sector[..data.len()].copy_from_slice(data);
            self.sectors.insert(lba, sector);
        }
    }

    impl BlockDevice for MockIsoImage {
        fn read_sectors(
            &self,
            lba: u64,
            count: u64,
            buf: &mut [u8],
        ) -> Result<(), stem::block::BlockError> {
            for i in 0..count {
                let sector_lba = lba + i;
                if let Some(sector) = self.sectors.get(&sector_lba) {
                    let start = (i * 2048) as usize;
                    let end = start + 2048;
                    if buf.len() < end {
                        return Err(stem::block::BlockError::IoError);
                    }
                    buf[start..end].copy_from_slice(sector);
                } else {
                    return Err(stem::block::BlockError::IoError);
                }
            }
            Ok(())
        }
        fn sector_size(&self) -> u64 {
            2048
        }
    }

    #[test]
    fn test_open_path_nested_structure() {
        let mut image = MockIsoImage::new();

        // PVD at 16
        let mut pvd_data = [0u8; 2048];
        pvd_data[0] = VD_TYPE_PRIMARY;
        pvd_data[1..6].copy_from_slice(b"CD001");
        pvd_data[6] = 1;

        let root_lba = 100u32;
        let root_size = 2048u32;
        pvd_data[158..162].copy_from_slice(&root_lba.to_le_bytes());
        pvd_data[166..170].copy_from_slice(&root_size.to_le_bytes());
        image.set_sector(16, &pvd_data);

        // Root Dir (LBA 100)
        // Contains: SUBDIR (LBA 200), ROOTFILE.TXT (LBA 300)
        let mut root_buf = [0u8; 2048];
        let mut offset = 0;
        // . and ..
        write_dir_record(
            &mut root_buf,
            &mut offset,
            "\x00",
            root_lba,
            root_size,
            2,
            None,
        );
        write_dir_record(
            &mut root_buf,
            &mut offset,
            "\x01",
            root_lba,
            root_size,
            2,
            None,
        );
        // SUBDIR
        write_dir_record(&mut root_buf, &mut offset, "SUBDIR", 200, 2048, 2, None);
        // ROOTFILE.TXT;1
        write_dir_record(
            &mut root_buf,
            &mut offset,
            "ROOTFILE.TXT;1",
            300,
            100,
            0,
            None,
        );
        image.set_sector(100, &root_buf);

        // SUBDIR (LBA 200)
        // Contains: SUBFILE.TXT (LBA 400)
        let mut sub_buf = [0u8; 2048];
        let mut offset = 0;
        write_dir_record(&mut sub_buf, &mut offset, "\x00", 200, 2048, 2, None);
        write_dir_record(&mut sub_buf, &mut offset, "\x01", 100, 2048, 2, None);
        // SUBFILE.TXT;1
        write_dir_record(&mut sub_buf, &mut offset, "SUBFILE.TXT;1", 400, 50, 0, None);
        image.set_sector(200, &sub_buf);

        let iso = IsoFs::probe(&image).expect("Probe failed");

        // Test 1: Root file
        if let Some(f) = iso.open_path(&image, "/ROOTFILE.TXT") {
            assert_eq!(f.extent_lba, 300);
            assert_eq!(f.size, 100);
        } else {
            panic!("Root file not found");
        }

        // Test 2: Nested file
        if let Some(f) = iso.open_path(&image, "/SUBDIR/SUBFILE.TXT") {
            assert_eq!(f.extent_lba, 400);
            assert_eq!(f.size, 50);
        } else {
            panic!("Sub file not found");
        }

        // Test 3: Case insensitivity
        if let Some(f) = iso.open_path(&image, "/subdir/subfile.txt") {
            assert_eq!(f.extent_lba, 400);
        } else {
            panic!("Sub file case mismatch");
        }

        // Test 4: Missing file
        assert!(iso.open_path(&image, "/MISSING.TXT").is_none());

        // Test 5: Missing dir
        assert!(iso.open_path(&image, "/MISSING/FILE.TXT").is_none());

        // Test 6: File as dir (ROOTFILE.TXT is a file)
        assert!(iso.open_path(&image, "/ROOTFILE.TXT/SUBFILE").is_none());
    }

    #[test]
    fn test_parse_dir_entries_sector_boundary_padding() {
        // Allocate 2 sectors (4096 bytes)
        let mut buf = alloc::vec![0u8; 4096];
        let mut offset = 0;

        // 1. Write an entry in the first sector
        write_dir_record(&mut buf, &mut offset, "FILE1.TXT;1", 100, 1024, 0, None);

        // 2. Simulate padding by advancing to the next sector boundary.
        // The parser encounters 0 (record_len) at the current offset and skips to the next sector.
        // We ensure the rest of sector 1 is 0 (which it is by default).
        offset = 2048;

        // 3. Write an entry at the start of the second sector
        write_dir_record(&mut buf, &mut offset, "FILE2.TXT;1", 200, 2048, 0, None);

        let dev = MockDirBlockDevice { dir_data: buf };

        let iso = IsoFs {
            pvd: PrimaryVolumeDescriptor {
                system_id: [0; 32],
                volume_id: [0; 32],
                volume_space_size: 0,
                root_dir_extent: 0,
                root_dir_size: 0,
                logical_block_size: 2048,
            },
            dir_cache: RefCell::new(BTreeMap::new()),
            #[cfg(feature = "perf")]
            perf: RefCell::new(PerfCounters::default()),
        };

        // Parse 4096 bytes (2 sectors)
        let entries = iso.parse_dir_entries(&dev, 0, 4096);

        // Verify results
        assert_eq!(entries.len(), 2, "Should find 2 entries, skipping padding");

        let e1 = &entries[0];
        assert_eq!(e1.name, "FILE1.TXT");
        assert_eq!(e1.extent_lba, 100);

        let e2 = &entries[1];
        assert_eq!(e2.name, "FILE2.TXT");
        assert_eq!(e2.extent_lba, 200);
    }
}
