//! VFS file status types for the janix ABI.
//!
//! These types cross the userspace/kernel boundary and must remain stable.
//! They are used by the `SYS_FS_STAT` syscall and the `vfs_stat` stem wrapper.

/// A timestamp with nanosecond precision.
///
/// Represents a point in wall-clock time as seconds since the Unix epoch
/// (January 1, 1970 00:00:00 UTC) plus a nanosecond sub-second component.
///
/// For synthetic/virtual nodes that have no meaningful wall-clock time,
/// this is conventionally set to zero (the Unix epoch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Timespec {
    /// Seconds since the Unix epoch.
    pub sec: u64,
    /// Sub-second nanoseconds (0–999_999_999).
    pub nsec: u32,
    /// Reserved padding (must be zero).
    pub _pad: u32,
}

impl Timespec {
    /// The zero/epoch timestamp (1970-01-01 00:00:00 UTC).
    pub const ZERO: Self = Self {
        sec: 0,
        nsec: 0,
        _pad: 0,
    };

    /// Create a `Timespec` from seconds and nanoseconds.
    #[inline]
    pub const fn new(sec: u64, nsec: u32) -> Self {
        Self { sec, nsec, _pad: 0 }
    }
}

/// Request structure for the `SYS_FS_UTIMES` and `SYS_FS_FUTIMES` syscalls.
///
/// Pass to the kernel to update the access time (`atime`) and/or modification
/// time (`mtime`) of a file.  Set `atime_sec` or `mtime_sec` to
/// [`UtimesRequest::OMIT`] to leave the corresponding timestamp unchanged.
///
/// Both the kernel and the Rust stdlib PAL layer use this exact layout.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct UtimesRequest {
    /// New access time — seconds since the Unix epoch.
    /// Set to [`Self::OMIT`] to leave `atime` unchanged.
    pub atime_sec: u64,
    /// New access time — nanosecond component (0–999_999_999).
    pub atime_nsec: u32,
    /// Reserved padding; must be zero.
    pub _pad1: u32,
    /// New modification time — seconds since the Unix epoch.
    /// Set to [`Self::OMIT`] to leave `mtime` unchanged.
    pub mtime_sec: u64,
    /// New modification time — nanosecond component (0–999_999_999).
    pub mtime_nsec: u32,
    /// Reserved padding; must be zero.
    pub _pad2: u32,
}

impl UtimesRequest {
    /// Sentinel for `atime_sec`/`mtime_sec` meaning "do not update this timestamp".
    pub const OMIT: u64 = u64::MAX;

    /// Build a request that sets both timestamps.
    #[inline]
    pub const fn both(atime: Timespec, mtime: Timespec) -> Self {
        Self {
            atime_sec: atime.sec,
            atime_nsec: atime.nsec,
            _pad1: 0,
            mtime_sec: mtime.sec,
            mtime_nsec: mtime.nsec,
            _pad2: 0,
        }
    }

    /// Build a request that only updates `atime`.
    #[inline]
    pub const fn atime_only(atime: Timespec) -> Self {
        Self {
            atime_sec: atime.sec,
            atime_nsec: atime.nsec,
            _pad1: 0,
            mtime_sec: Self::OMIT,
            mtime_nsec: 0,
            _pad2: 0,
        }
    }

    /// Build a request that only updates `mtime`.
    #[inline]
    pub const fn mtime_only(mtime: Timespec) -> Self {
        Self {
            atime_sec: Self::OMIT,
            atime_nsec: 0,
            _pad1: 0,
            mtime_sec: mtime.sec,
            mtime_nsec: mtime.nsec,
            _pad2: 0,
        }
    }
}

/// File status structure returned by the `stat`/`fstat` syscall (`SYS_FS_STAT`).
///
/// All fields are stable across the userspace/kernel ABI boundary.
///
/// # Timestamp policy
/// - `atime`: updated on meaningful data read access (coarse v1 policy).
/// - `mtime`: updated when file contents change (write, truncate).
/// - `ctime`: updated when file content or metadata changes.
/// - For synthetic/virtual nodes, timestamps are set to zero (epoch).
///
/// # Ownership and extended metadata
/// - `uid`/`gid`: owner user/group IDs (0 = root for synthetic/kernel nodes).
/// - `nlink`: hard-link count (1 for most files; ≥ 2 for directories).
/// - `rdev`: device number (encoded as `(major << 8) | minor`) for character/block
///   devices; 0 for regular files and directories.
/// - `blksize`: preferred I/O block size (4096 for most nodes).
/// - `blocks`: number of 512-byte blocks allocated (computed from `size` for files).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct FileStat {
    /// File type and permissions bitmask (same encoding as POSIX `st_mode`).
    pub mode: u32,
    /// Hard-link count (1 for files, ≥ 2 for directories).
    pub nlink: u32,
    /// File size in bytes (0 for devices/directories).
    pub size: u64,
    /// Inode-like unique identifier within the filesystem.
    pub ino: u64,
    /// Last access time.
    pub atime: Timespec,
    /// Last modification time (content changed).
    pub mtime: Timespec,
    /// Last status change time (content or metadata changed).
    pub ctime: Timespec,
    /// Owner user ID (0 = root for kernel/synthetic nodes).
    pub uid: u32,
    /// Owner group ID (0 = root for kernel/synthetic nodes).
    pub gid: u32,
    /// Device number for character/block device nodes (`(major << 8) | minor`);
    /// 0 for regular files and directories.
    pub rdev: u64,
    /// Preferred I/O block size in bytes.
    pub blksize: u32,
    /// Padding to align `blocks` to an 8-byte boundary.
    pub _blksize_pad: u32,
    /// Number of 512-byte blocks allocated for this file.
    pub blocks: u64,
}
