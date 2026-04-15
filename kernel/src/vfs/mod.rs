//! VFS (Virtual Filesystem) core for thingos.
//!
//! Implements Act III of the de-graphing migration: a minimal path-based
//! kernel VFS. Provides:
//!
//! - [`VfsNode`]: trait for files/devices/directories
//! - [`VfsDriver`]: trait for filesystem backends
//! - [`VfsStat`]: file metadata
//! - [`OpenFlags`]: open(2) flags
//! - Global mount table (see [`mount`])
//! - Per-process file descriptor table (see [`fd_table`])
//! - Built-in devfs backend (see [`devfs`])
//!
//! # North Star
//! A component is *integrated* when it is reachable via a path, can be
//! opened, and can be read, written, or polled. Nothing else is required.

pub mod bootfs;
pub mod devfs;
pub mod fd_table;
pub mod flock;
pub mod memfd;
pub mod mount;
pub mod path;
pub mod port_node;
pub mod procfs;
pub mod provider;
pub mod ramfs;
pub mod sysfs;
pub mod union;
pub mod watch;

use abi::errors::{Errno, SysResult};
use alloc::sync::Arc;

// ── Open flags ─────────────────────────────────────────────────────────────

/// Subset of POSIX open(2) flags understood by the VFS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenFlags(pub u32);

impl OpenFlags {
    pub const ACCESS_MODE_MASK: u32 = 0x3;
    pub const MUTABLE_STATUS_MASK: u32 =
        abi::syscall::vfs_flags::O_APPEND | abi::syscall::vfs_flags::O_NONBLOCK;

    pub fn read_only() -> Self {
        Self(abi::syscall::vfs_flags::O_RDONLY)
    }
    pub fn write_only() -> Self {
        Self(abi::syscall::vfs_flags::O_WRONLY)
    }
    pub fn read_write() -> Self {
        Self(abi::syscall::vfs_flags::O_RDWR)
    }

    pub fn from_open_call(flags: u32) -> Self {
        Self((flags & Self::ACCESS_MODE_MASK) | (flags & Self::MUTABLE_STATUS_MASK))
    }

    pub fn access_mode_bits(self) -> u32 {
        self.0 & Self::ACCESS_MODE_MASK
    }

    pub fn with_mutable_status(self, requested: u32) -> Self {
        Self((self.0 & !Self::MUTABLE_STATUS_MASK) | (requested & Self::MUTABLE_STATUS_MASK))
    }

    pub fn is_readable(self) -> bool {
        let access = self.0 & 0x3;
        access == abi::syscall::vfs_flags::O_RDONLY || access == abi::syscall::vfs_flags::O_RDWR
    }
    pub fn is_writable(self) -> bool {
        let access = self.0 & 0x3;
        access == abi::syscall::vfs_flags::O_WRONLY || access == abi::syscall::vfs_flags::O_RDWR
    }
    pub fn is_nonblock(self) -> bool {
        self.0 & abi::syscall::vfs_flags::O_NONBLOCK != 0
    }
    pub fn is_append(self) -> bool {
        self.0 & abi::syscall::vfs_flags::O_APPEND != 0
    }

    pub fn read_would_block(self, readiness: u16) -> bool {
        self.is_nonblock()
            && readiness
                & (abi::syscall::poll_flags::POLLIN
                    | abi::syscall::poll_flags::POLLHUP
                    | abi::syscall::poll_flags::POLLERR)
                == 0
    }

    pub fn write_would_block(self, readiness: u16) -> bool {
        self.is_nonblock()
            && readiness
                & (abi::syscall::poll_flags::POLLOUT
                    | abi::syscall::poll_flags::POLLHUP
                    | abi::syscall::poll_flags::POLLERR)
                == 0
    }

    pub fn effective_write_offset(self, current_offset: u64, file_size: u64) -> u64 {
        if self.is_append() {
            file_size
        } else {
            current_offset
        }
    }
}

// ── File metadata ───────────────────────────────────────────────────────────

/// Kernel-internal file status, analogous to a subset of POSIX `struct stat`.
///
/// All VFS node implementations return this type from their `stat()` method.
/// The syscall handler converts it into the ABI-stable [`abi::fs::FileStat`]
/// before copying to userspace.
///
/// # Timestamp policy (v1)
/// - `atime`: updated on meaningful data read access (coarse policy).
/// - `mtime`: updated when file contents change (write, truncate).
/// - `ctime`: updated when file content or metadata changes.
/// - Synthetic/virtual nodes return zero (epoch) timestamps.
///
/// # Ownership and extended metadata
/// - `uid`/`gid`: owner IDs (0 = root for kernel-managed synthetic nodes).
/// - `nlink`: hard-link count (1 for files/devices; ≥ 2 for directories).
/// - `rdev`: device number `(major << 8) | minor` for char/block nodes; 0 otherwise.
/// - `blksize`: preferred I/O block size in bytes (0 → defaults to 4096).
/// - `blocks`: 512-byte block count (0 → caller may derive from `size`).
#[derive(Clone, Copy, Debug, Default)]
pub struct VfsStat {
    /// File type and permissions bitmask (same encoding as POSIX st_mode).
    pub mode: u32,
    /// Size in bytes (for regular files; 0 for devices/directories).
    pub size: u64,
    /// Inode-like unique identifier within the filesystem.
    pub ino: u64,
    /// Hard-link count (1 for files/device nodes; ≥ 2 for directories).
    pub nlink: u32,
    /// Owner user ID (0 = root for kernel/synthetic nodes).
    pub uid: u32,
    /// Owner group ID (0 = root for kernel/synthetic nodes).
    pub gid: u32,
    /// Device number for character/block device nodes (`(major << 8) | minor`);
    /// 0 for regular files and directories.
    pub rdev: u64,
    /// Preferred I/O block size in bytes (0 means caller should default to 4096).
    pub blksize: u32,
    /// Number of 512-byte blocks allocated.
    pub blocks: u64,
    /// Last access time — seconds since Unix epoch.
    pub atime_sec: u64,
    /// Last access time — nanosecond component (0–999_999_999).
    pub atime_nsec: u32,
    /// Last modification time — seconds since Unix epoch.
    pub mtime_sec: u64,
    /// Last modification time — nanosecond component (0–999_999_999).
    pub mtime_nsec: u32,
    /// Last status-change time — seconds since Unix epoch.
    pub ctime_sec: u64,
    /// Last status-change time — nanosecond component (0–999_999_999).
    pub ctime_nsec: u32,
}

impl VfsStat {
    pub const S_IFMT: u32 = 0o170000;
    pub const S_IFREG: u32 = 0o100000;
    pub const S_IFDIR: u32 = 0o040000;
    pub const S_IFCHR: u32 = 0o020000;
    pub const S_IFIFO: u32 = 0o010000;
    /// Symbolic link.
    pub const S_IFLNK: u32 = 0o120000;
    /// Unix domain socket.
    pub const S_IFSOCK: u32 = 0o140000;

    /// Encode a major/minor device number pair into a single `rdev` value.
    ///
    /// Uses the simplified Linux encoding: `(major << 8) | (minor & 0xFF)`.
    /// Sufficient for the small set of built-in devices in thingos.
    #[inline]
    pub const fn makedev(major: u32, minor: u32) -> u64 {
        ((major as u64) << 8) | ((minor as u64) & 0xFF)
    }

    pub fn is_dir(self) -> bool {
        self.mode & Self::S_IFMT == Self::S_IFDIR
    }
    pub fn is_chr(self) -> bool {
        self.mode & Self::S_IFMT == Self::S_IFCHR
    }
    pub fn is_reg(self) -> bool {
        self.mode & Self::S_IFMT == Self::S_IFREG
    }
    pub fn is_fifo(self) -> bool {
        self.mode & Self::S_IFMT == Self::S_IFIFO
    }
    pub fn is_symlink(self) -> bool {
        self.mode & Self::S_IFMT == Self::S_IFLNK
    }

    /// Convert this kernel-internal stat into the ABI-stable [`abi::fs::FileStat`]
    /// suitable for copying to userspace.
    pub fn to_abi_stat(self) -> abi::fs::FileStat {
        // Derive blksize/blocks when the node hasn't set them explicitly.
        let blksize = if self.blksize == 0 { 4096 } else { self.blksize };
        let blocks = if self.blocks == 0 && self.size > 0 {
            (self.size + 511) / 512
        } else {
            self.blocks
        };
        abi::fs::FileStat {
            mode: self.mode,
            nlink: self.nlink,
            size: self.size,
            ino: self.ino,
            atime: abi::fs::Timespec::new(self.atime_sec, self.atime_nsec),
            mtime: abi::fs::Timespec::new(self.mtime_sec, self.mtime_nsec),
            ctime: abi::fs::Timespec::new(self.ctime_sec, self.ctime_nsec),
            uid: self.uid,
            gid: self.gid,
            rdev: self.rdev,
            blksize,
            _blksize_pad: 0,
            blocks,
        }
    }
}

// ── VfsNode ─────────────────────────────────────────────────────────────────

/// A single open file or device inside the VFS.
///
/// Implementations are expected to be `Send + Sync` so they can be stored
/// in the kernel FD table (which may be accessed from any CPU).
pub trait VfsNode: Send + Sync {
    /// Read up to `buf.len()` bytes starting at `offset` into `buf`.
    /// Returns the number of bytes read, or 0 at EOF.
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize>;

    /// Write `buf` starting at `offset`.
    /// Returns the number of bytes written.
    fn write(&self, offset: u64, buf: &[u8]) -> SysResult<usize>;

    /// Return metadata for this node.
    fn stat(&self) -> SysResult<VfsStat>;

    /// Called when the last reference to an open file is dropped.
    /// Default: no-op.
    fn close(&self) {}

    /// Truncate the file to `new_size` bytes.
    ///
    /// If `new_size` is less than the current size, the extra data is discarded.
    /// If `new_size` is greater, the file is extended with zero bytes.
    /// Default: returns `EROFS` (read-only / non-truncatable).
    fn truncate(&self, _new_size: u64) -> SysResult<()> {
        Err(abi::errors::Errno::EROFS)
    }

    /// Read directory entries into `buf` starting at `offset`.
    /// Returns bytes written into `buf`, or 0 when exhausted.
    /// Only meaningful for directory nodes; regular files return `ENOTDIR`.
    fn readdir(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::ENOTDIR)
    }

    /// Return the exact physical memory backing this node, if it is directly memory-mapped.
    /// Used for zero-copy userspace memory mapping of devices and shm buffers.
    /// Returns physical base address and length in bytes.
    fn phys_region(&self) -> SysResult<(u64, usize)> {
        Err(Errno::ENOSYS)
    }

    /// Returns true if the node is a terminal/TTY device.
    fn is_tty(&self) -> bool {
        false
    }

    /// Poll this node for readiness.
    /// Returns the current readiness mask (using [`abi::syscall::poll_flags`]).
    fn poll(&self) -> u16 {
        abi::syscall::poll_flags::POLLIN | abi::syscall::poll_flags::POLLOUT
    }

    /// If this node is a port, returns the underlying port.
    fn as_port(&self) -> Option<Arc<crate::ipc::Port>> {
        None
    }

    // ── Unix domain socket operations ────────────────────────────────────────
    // Default implementations return `ENOTSOCK` so that only `UnixSocketNode`
    // needs to override them.

    /// Bind this socket to a filesystem path.
    fn sock_bind(&self, _path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::ENOTSOCK)
    }

    /// Mark this socket as listening for connections.
    fn sock_listen(&self, _backlog: usize) -> SysResult<()> {
        Err(abi::errors::Errno::ENOTSOCK)
    }

    /// Accept one incoming connection from a listening socket.
    ///
    /// Returns a new `Arc<dyn VfsNode>` for the server side of the connection.
    fn sock_accept(&self) -> SysResult<Arc<dyn VfsNode>> {
        Err(abi::errors::Errno::ENOTSOCK)
    }

    /// Connect this socket to a listening socket at the given path.
    fn sock_connect(&self, _path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::ENOTSOCK)
    }

    /// Shut down one or both directions of a connected socket.
    /// `how`: 0 = SHUT_RD, 1 = SHUT_WR, 2 = SHUT_RDWR.
    fn sock_shutdown(&self, _how: u32) -> SysResult<()> {
        Err(abi::errors::Errno::ENOTSOCK)
    }

    /// Device-specific control call (ioctl).
    fn device_call(&self, _call: &abi::device::DeviceCall) -> SysResult<usize> {
        Err(abi::errors::Errno::ENOSYS)
    }

    /// Flush any pending writes to the backing store.
    ///
    /// For RAM-backed filesystems this is a no-op that always succeeds.
    /// Drivers with real backing storage should override this to drain their
    /// write buffers and ensure durability.
    fn sync(&self) -> SysResult<()> {
        Ok(())
    }

    /// Add a task to the wait queue for this node.
    fn add_waiter(&self, _tid: u64) {}

    /// Remove a task from the wait queue for this node.
    fn remove_waiter(&self, _tid: u64) {}

    /// Read the symlink target for symlink nodes.
    ///
    /// Returns the target path string for `S_IFLNK` nodes.
    /// All other node types return `Err(EINVAL)`.
    fn readlink(&self) -> SysResult<alloc::string::String> {
        Err(abi::errors::Errno::EINVAL)
    }

    /// Change the permission bits of this node.
    ///
    /// `mode` contains the lower 12 bits of the POSIX permission mask
    /// (permission bits plus setuid/setgid/sticky: `0o7777`).
    ///
    /// Default: returns [`Errno::ENOTSUP`] for nodes that do not support
    /// permission mutation.
    fn chmod(&self, _mode: u32) -> SysResult<()> {
        Err(abi::errors::Errno::EOPNOTSUPP)
    }

    /// Update the access and/or modification timestamps of this node.
    ///
    /// `atime` — `Some((sec, nsec))` to set the access time, `None` to leave it
    /// unchanged.
    /// `mtime` — `Some((sec, nsec))` to set the modification time, `None` to
    /// leave it unchanged.
    ///
    /// Default: returns [`Errno::ENOTSUP`] for nodes that do not support
    /// timestamp mutation.
    fn utimes(&self, _atime: Option<(u64, u32)>, _mtime: Option<(u64, u32)>) -> SysResult<()> {
        Err(abi::errors::Errno::EOPNOTSUPP)
    }
}

// ── VfsDriver ───────────────────────────────────────────────────────────────

/// A filesystem backend that can resolve path components to [`VfsNode`]s.
///
/// `VfsDriver::lookup` is called with the path *relative to the mount point*.
/// For example, if `/dev` is mounted and the user opens `/dev/console`, the
/// driver receives `"console"`.
pub trait VfsDriver: Send + Sync {
    /// Look up `path` within this filesystem and return an open node.
    fn lookup(&self, path: &str) -> SysResult<Arc<dyn VfsNode>>;

    /// Create a new regular file at `path` and return it as an open node.
    ///
    /// The default implementation returns `EROFS`, indicating a read-only
    /// filesystem.  Writable filesystems (ramfs/tmpfs) should override this.
    fn create(&self, _path: &str) -> SysResult<Arc<dyn VfsNode>> {
        Err(abi::errors::Errno::EROFS)
    }

    /// Create a directory at `path`.
    ///
    /// The default implementation returns `EROFS`.
    fn mkdir(&self, _path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::EROFS)
    }

    /// Remove the file or empty directory at `path`.
    ///
    /// The default implementation returns `EROFS`.
    fn unlink(&self, _path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::EROFS)
    }

    /// Rename a file or directory from `old_path` to `new_path`.
    ///
    /// The default implementation returns `EROFS`.
    fn rename(&self, _old_path: &str, _new_path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::EROFS)
    }

    /// Create a symbolic link at `link_path` pointing to `target`.
    ///
    /// The default implementation returns `EROFS`.
    fn symlink(&self, _target: &str, _link_path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::EROFS)
    }

    /// Create a hard link at `dst_path` that refers to the same file as `src_path`.
    ///
    /// Hard links are only meaningful for regular files.  The default
    /// implementation returns `EOPNOTSUPP`; writable filesystems (ramfs)
    /// should override this.
    fn link(&self, _src_path: &str, _dst_path: &str) -> SysResult<()> {
        Err(abi::errors::Errno::EOPNOTSUPP)
    }
}

// ── Namespace ────────────────────────────────────────────────────────────────

/// A reference to the VFS namespace (mount table view) for a process.
///
/// # Current semantics (stub)
///
/// All processes share a **single global mount table**.  `NamespaceRef` is a
/// unit struct: every instance is equivalent and resolves to the same
/// underlying state.  Mounts and unmounts performed by *any* process are
/// immediately visible to *all* processes.
///
/// The field `Process.namespace` is populated at spawn time and cloned into
/// child processes, but both parent and child resolve to the same global state.
///
/// # What is intentionally NOT guaranteed today
///
/// - Mount isolation: a process cannot have a private mount table.
/// - Privilege checking: `SYS_FS_MOUNT` does not verify ownership.
/// - Snapshot-on-spawn: spawning a child does not fork the mount table.
///
/// # Roadmap
///
/// Per-process namespace divergence (sandboxing, containers) will be
/// introduced in a future milestone.  The field in [`crate::task::Process`]
/// and all call sites that call [`NamespaceRef::global()`] are already wired
/// so that adding real isolation requires only changes to this struct and
/// `vfs::mount`, without touching every spawn path again.
///
/// See `docs/concepts/namespaces.md` for the full behaviour matrix and
/// staged implementation roadmap.
#[derive(Clone, Debug, Default)]
pub struct NamespaceRef;

impl NamespaceRef {
    /// Return the shared (global) namespace reference.
    ///
    /// All current callers receive an equivalent value.  Once per-process
    /// namespace isolation is implemented this constructor will create a new
    /// namespace backed by a clone of the system-wide mount table, and this
    /// function will return the root (initial) namespace.
    pub fn global() -> Self {
        Self
    }

    /// Returns `true` once per-process namespace isolation is implemented.
    ///
    /// Currently always returns `false` because all processes share the global
    /// mount table.  Code that needs to behave differently when real isolation
    /// is active should guard on this method rather than assuming one behaviour
    /// or the other.
    pub fn is_isolated(&self) -> bool {
        false
    }
}

// ── Global init ──────────────────────────────────────────────────────────────

/// Initialise the VFS subsystem and mount built-in filesystems.
///
/// Called once from `kernel::start` during early boot, *before* any user
/// processes are spawned.
///
/// Boot mounts:
/// - `/`         ← boot filesystem (static read-only initramfs)
/// - `/`         ← root tmpfs (layered over bootfs via union)
/// - `/dev`      ← device filesystem
/// - `/proc`     ← process info (stub)
/// - `/sys`      ← kernel device discovery metadata
/// - `/tmp`      ← temporary filesystem (writable, volatile)
/// - `/run`      ← transient runtime state (tmpfs)
/// - `/services` ← populated by userland daemons (tmpfs stub for now)
pub fn init(modules: &'static [crate::BootModuleDesc]) {
    mount::init();

    // Create the root filesystem (tmpfs) — writable, volatile.
    let root_fs = Arc::new(ramfs::RamFs::new());

    // Pre-populate mount point directories in the root filesystem so they appear in readdir("/")
    let _ = root_fs.mkdir("boot");
    let _ = root_fs.mkdir("bin");
    let _ = root_fs.mkdir("etc");
    let _ = root_fs.mkdir("share");
    let _ = root_fs.mkdir("mnt");
    let _ = root_fs.mkdir("dev");
    let _ = root_fs.mkdir("dev/display");
    crate::kdebug!("VFS: Created /dev/display directory");
    let _ = root_fs.mkdir("dev/input");
    let _ = root_fs.mkdir("proc");
    let _ = root_fs.mkdir("sys");
    let _ = root_fs.mkdir("tmp");
    let _ = root_fs.mkdir("run");
    let _ = root_fs.mkdir("services");
    let _ = root_fs.mkdir("session");
    let _ = root_fs.mkdir("data");

    // Create the root union filesystem.
    let mut root_union = union::UnionFs::new_fallthrough();
    root_union.push(Arc::new(bootfs::BootFs::new(modules))); // Layer 0: Read-only boot modules
    root_union.push(root_fs); // Layer 1: Writable RAM overlay

    mount::mount("/", Arc::new(root_union));
    crate::kdebug!("vfs: mounted union filesystem at / (root)");

    // Device filesystem
    mount::mount("/dev", Arc::new(devfs::DevFs::new()));
    crate::kdebug!("vfs: mounted devfs at /dev");

    // Process info filesystem
    mount::mount("/proc", Arc::new(procfs::ProcFs::new()));
    crate::kdebug!("vfs: mounted procfs at /proc");

    // Kernel device metadata
    mount::mount("/sys", Arc::new(sysfs::SysFs::new()));
    crate::kdebug!("vfs: mounted sysfs at /sys");

    // Temporary filesystem — scratch space for userland.
    mount::mount("/tmp", Arc::new(ramfs::RamFs::new()));
    crate::kdebug!("vfs: mounted tmpfs at /tmp");

    // Transient runtime state
    mount::mount("/run", Arc::new(ramfs::RamFs::new()));
    crate::kdebug!("vfs: mounted tmpfs at /run");

    // Service namespace
    mount::mount("/services", Arc::new(ramfs::RamFs::new()));
    crate::kdebug!("vfs: mounted tmpfs at /services");

    // Session namespace — filesystem-native GUI objects live here.
    mount::mount("/session", Arc::new(ramfs::RamFs::new()));
    crate::kdebug!("vfs: mounted tmpfs at /session");

    // Persistent user data — writable scratchpad for userland programs.
    mount::mount("/data", Arc::new(ramfs::RamFs::new()));
    crate::kdebug!("vfs: mounted tmpfs at /data");
}

/// Helper for filesystem drivers to implement `readdir`.
///
/// Encodes directory entries as NUL-terminated names into `buf`.  Only entries
/// (or parts of entries) that fall within the virtual byte stream range
/// starting at `offset` are included.
pub fn write_readdir_entries<'a>(
    entries: impl IntoIterator<Item = &'a str>,
    offset: u64,
    buf: &mut [u8],
) -> SysResult<usize> {
    let mut seen = alloc::collections::BTreeSet::new();
    let mut written = 0usize;
    let mut virtual_pos = 0u64;

    for entry in entries {
        if !seen.insert(entry) {
            continue;
        }
        let name = entry.as_bytes();
        let entry_full_len = (name.len() + 1) as u64;

        let entry_start = virtual_pos;
        let entry_end = virtual_pos + entry_full_len;

        if entry_end > offset {
            // This entry (or part of it) is within the requested range.
            let start_in_entry = if offset > entry_start {
                (offset - entry_start) as usize
            } else {
                0
            };

            if start_in_entry < name.len() {
                let copy_from_entry = &name[start_in_entry..];
                let n = copy_from_entry.len().min(buf.len() - written);
                buf[written..written + n].copy_from_slice(&copy_from_entry[..n]);
                written += n;

                if n == copy_from_entry.len() && written < buf.len() {
                    buf[written] = 0;
                    written += 1;
                }
            } else if start_in_entry == name.len() && written < buf.len() {
                // Offset requested exactly the NUL byte of this entry.
                buf[written] = 0;
                written += 1;
            }

            if written == buf.len() {
                break;
            }
        }

        virtual_pos = entry_end;
    }

    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // A trivial in-memory node for unit-testing the VFS layer.
    struct MemNode {
        data: alloc::vec::Vec<u8>,
        mode: u32,
    }

    impl VfsNode for MemNode {
        fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
            let off = offset as usize;
            if off >= self.data.len() {
                return Ok(0);
            }
            let avail = &self.data[off..];
            let n = avail.len().min(buf.len());
            buf[..n].copy_from_slice(&avail[..n]);
            Ok(n)
        }
        fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
            Err(Errno::EROFS)
        }
        fn stat(&self) -> SysResult<VfsStat> {
            Ok(VfsStat {
                mode: self.mode,
                size: self.data.len() as u64,
                ino: 1,
                ..Default::default()
            })
        }
    }

    #[test]
    fn test_open_flags_access() {
        let ro = OpenFlags::read_only();
        assert!(ro.is_readable());
        assert!(!ro.is_writable());

        let wo = OpenFlags::write_only();
        assert!(!wo.is_readable());
        assert!(wo.is_writable());

        let rw = OpenFlags::read_write();
        assert!(rw.is_readable());
        assert!(rw.is_writable());
    }

    #[test]
    fn test_from_open_call_discards_creation_flags() {
        let flags = OpenFlags::from_open_call(
            abi::syscall::vfs_flags::O_WRONLY
                | abi::syscall::vfs_flags::O_CREAT
                | abi::syscall::vfs_flags::O_TRUNC
                | abi::syscall::vfs_flags::O_NONBLOCK,
        );
        assert_eq!(flags.access_mode_bits(), abi::syscall::vfs_flags::O_WRONLY);
        assert!(flags.is_nonblock());
        assert_eq!(
            flags.0 & (abi::syscall::vfs_flags::O_CREAT | abi::syscall::vfs_flags::O_TRUNC),
            0
        );
    }

    #[test]
    fn test_setfl_only_updates_mutable_status_bits() {
        let base = OpenFlags::from_open_call(
            abi::syscall::vfs_flags::O_RDWR | abi::syscall::vfs_flags::O_APPEND,
        );
        let updated = base.with_mutable_status(abi::syscall::vfs_flags::O_NONBLOCK);
        assert_eq!(updated.access_mode_bits(), abi::syscall::vfs_flags::O_RDWR);
        assert!(updated.is_nonblock());
        assert!(!updated.is_append());
    }

    #[test]
    fn test_nonblock_readiness_checks() {
        let nonblock = OpenFlags::from_open_call(
            abi::syscall::vfs_flags::O_RDONLY | abi::syscall::vfs_flags::O_NONBLOCK,
        );
        assert!(nonblock.read_would_block(0));
        assert!(!nonblock.read_would_block(abi::syscall::poll_flags::POLLIN));
        assert!(!OpenFlags::read_only().read_would_block(0));
    }

    #[test]
    fn test_effective_write_offset_uses_file_end_for_append() {
        let append = OpenFlags::from_open_call(
            abi::syscall::vfs_flags::O_WRONLY | abi::syscall::vfs_flags::O_APPEND,
        );
        assert_eq!(append.effective_write_offset(3, 12), 12);
        assert_eq!(OpenFlags::write_only().effective_write_offset(3, 12), 3);
    }

    #[test]
    fn test_vfs_stat_type_bits() {
        let dir = VfsStat {
            mode: VfsStat::S_IFDIR | 0o755,
            size: 0,
            ino: 1,
            ..Default::default()
        };
        assert!(dir.is_dir());
        assert!(!dir.is_reg());
        assert!(!dir.is_chr());

        let chr = VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 2,
            ..Default::default()
        };
        assert!(chr.is_chr());
        assert!(!chr.is_dir());

        let reg = VfsStat {
            mode: VfsStat::S_IFREG | 0o644,
            size: 42,
            ino: 3,
            ..Default::default()
        };
        assert!(reg.is_reg());
    }

    #[test]
    fn test_mem_node_read_partial() {
        let node = MemNode {
            data: vec![1, 2, 3, 4, 5],
            mode: VfsStat::S_IFREG | 0o444,
        };
        let mut buf = [0u8; 3];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&buf, &[1, 2, 3]);
    }

    #[test]
    fn test_mem_node_read_at_offset() {
        let node = MemNode {
            data: vec![10, 20, 30],
            mode: VfsStat::S_IFREG | 0o444,
        };
        let mut buf = [0u8; 2];
        let n = node.read(1, &mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&buf, &[20, 30]);
    }

    #[test]
    fn test_mem_node_read_eof() {
        let node = MemNode {
            data: vec![],
            mode: VfsStat::S_IFREG | 0o444,
        };
        let mut buf = [0u8; 4];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_vfs_node_sync_default_is_ok_for_file_like_nodes() {
        let node = MemNode {
            data: vec![1, 2, 3],
            mode: VfsStat::S_IFREG | 0o644,
        };
        assert_eq!(node.sync(), Ok(()));
    }

    #[test]
    fn test_vfs_node_sync_default_is_ok_for_dir_like_nodes() {
        let node = MemNode {
            data: vec![],
            mode: VfsStat::S_IFDIR | 0o755,
        };
        assert_eq!(node.sync(), Ok(()));
    }

    #[test]
    fn test_makedev_encodes_major_and_minor() {
        assert_eq!(VfsStat::makedev(1, 3), 0x0103);
        assert_eq!(VfsStat::makedev(29, 0), 0x1D00);
        assert_eq!(VfsStat::makedev(5, 1), 0x0501);
    }

    #[test]
    fn test_to_abi_stat_propagates_ownership_and_rdev() {
        let stat = VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 42,
            nlink: 1,
            uid: 1000,
            gid: 100,
            rdev: VfsStat::makedev(1, 3),
            ..Default::default()
        };
        let abi = stat.to_abi_stat();
        assert_eq!(abi.mode, VfsStat::S_IFCHR | 0o666);
        assert_eq!(abi.nlink, 1);
        assert_eq!(abi.uid, 1000);
        assert_eq!(abi.gid, 100);
        assert_eq!(abi.rdev, VfsStat::makedev(1, 3));
        assert_eq!(abi.ino, 42);
    }

    #[test]
    fn test_to_abi_stat_derives_blocks_from_size() {
        let stat = VfsStat {
            mode: VfsStat::S_IFREG | 0o644,
            size: 1024,
            ino: 1,
            nlink: 1,
            ..Default::default()
        };
        let abi = stat.to_abi_stat();
        // 1024 bytes → 2 × 512-byte blocks
        assert_eq!(abi.blocks, 2);
        assert_eq!(abi.blksize, 4096);
    }

    #[test]
    fn test_to_abi_stat_partial_block_rounds_up() {
        let stat = VfsStat {
            mode: VfsStat::S_IFREG | 0o644,
            size: 513,
            ino: 1,
            nlink: 1,
            ..Default::default()
        };
        let abi = stat.to_abi_stat();
        // 513 bytes → ceil(513/512) = 2 blocks
        assert_eq!(abi.blocks, 2);
    }

    #[test]
    fn test_to_abi_stat_empty_file_has_zero_blocks() {
        let stat = VfsStat {
            mode: VfsStat::S_IFREG | 0o644,
            size: 0,
            ino: 1,
            nlink: 1,
            ..Default::default()
        };
        let abi = stat.to_abi_stat();
        assert_eq!(abi.blocks, 0);
    }

    #[test]
    fn test_dir_stat_has_nlink_at_least_two() {
        let stat = VfsStat {
            mode: VfsStat::S_IFDIR | 0o755,
            size: 0,
            ino: 10,
            nlink: 2,
            ..Default::default()
        };
        let abi = stat.to_abi_stat();
        assert!(abi.nlink >= 2, "directory nlink must be >= 2, got {}", abi.nlink);
    }

    #[test]
    fn test_write_readdir_entries_deduplicates_names() {
        let entries = ["dup", "dup", "unique", "dup"];
        let mut buf = [0u8; 32];
        let n = write_readdir_entries(entries.into_iter(), 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"dup\0unique\0");
    }
}
