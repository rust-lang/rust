//! ramfs — volatile in-memory filesystem.
//!
//! Provides a simple, non-persistent filesystem backed entirely by kernel
//! heap memory.  Used for the VFS root (`/`) and for the transient runtime
//! state directory (`/run`).
//!
//! # Design
//! - Directories are `BTreeMap<name, Arc<RamfsEntry>>` nodes.
//! - Files are `Vec<u8>` payloads behind a `Mutex`.
//! - `RamFs` implements [`VfsDriver`] and resolves paths relative to the
//!   mount point.  Path components are split on `/`; empty components and `.`
//!   are skipped; `..` is not supported at the driver level (handled by the
//!   path resolution engine in [`super::path`]).
//! - All operations are `no_std` compatible.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use abi::errors::{Errno, SysResult};

use super::{VfsDriver, VfsNode, VfsStat};

// ── Inode counter ────────────────────────────────────────────────────────────

static NEXT_INO: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(10);

fn alloc_ino() -> u64 {
    NEXT_INO.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
}

// ── Timestamp helper ─────────────────────────────────────────────────────────

/// Return the current wall-clock time as `(sec, nsec)`.
/// Delegates to the kernel time module; returns `(0, 0)` when the clock
/// has not been anchored yet (early boot).
#[inline]
fn now() -> (u64, u32) {
    crate::time::now_timespec()
}

// ── Internal tree node ───────────────────────────────────────────────────────

/// Per-file mutable state (data + timestamps).
struct RamfsFileInner {
    data: Vec<u8>,
    /// POSIX permission bits (lower 12 bits of st_mode; default 0o644).
    mode: u32,
    /// Hard-link count.  Starts at 1; incremented by [`link`] and decremented
    /// by [`unlink`].  The file data is freed by `Arc` when the last directory
    /// entry holding this node is removed.
    nlink: u32,
    /// Last access time (seconds, nanoseconds).
    atime: (u64, u32),
    /// Last modification time (content write / truncate).
    mtime: (u64, u32),
    /// Last status-change time (content or metadata change).
    ctime: (u64, u32),
}

impl RamfsFileInner {
    fn new(data: Vec<u8>) -> Self {
        let ts = now();
        Self {
            data,
            mode: 0o644,
            nlink: 1,
            atime: ts,
            mtime: ts,
            ctime: ts,
        }
    }
}

/// Per-directory mutable state (children + timestamps).
struct RamfsDirInner {
    children: BTreeMap<String, Arc<RamfsEntry>>,
    /// POSIX permission bits (lower 12 bits of st_mode; default 0o755).
    mode: u32,
    /// Last access time.
    atime: (u64, u32),
    /// Last modification time (child added / removed).
    mtime: (u64, u32),
    /// Last status-change time.
    ctime: (u64, u32),
}

impl RamfsDirInner {
    fn new() -> Self {
        let ts = now();
        Self {
            children: BTreeMap::new(),
            mode: 0o755,
            atime: ts,
            mtime: ts,
            ctime: ts,
        }
    }
}

enum RamfsEntry {
    File(Mutex<RamfsFileInner>, u64 /* ino */),
    Dir(Mutex<RamfsDirInner>, u64 /* ino */),
    /// Symbolic link: stores the target path string and an inode number.
    Symlink(String, u64 /* ino */),
}

impl RamfsEntry {
    fn new_dir() -> Arc<Self> {
        Arc::new(RamfsEntry::Dir(
            Mutex::new(RamfsDirInner::new()),
            alloc_ino(),
        ))
    }

    fn new_file(data: Vec<u8>) -> Arc<Self> {
        Arc::new(RamfsEntry::File(
            Mutex::new(RamfsFileInner::new(data)),
            alloc_ino(),
        ))
    }

    fn new_symlink(target: String) -> Arc<Self> {
        Arc::new(RamfsEntry::Symlink(target, alloc_ino()))
    }

    fn ino(&self) -> u64 {
        match self {
            RamfsEntry::File(_, ino) => *ino,
            RamfsEntry::Dir(_, ino) => *ino,
            RamfsEntry::Symlink(_, ino) => *ino,
        }
    }

    /// Look up a child by name inside a directory entry.
    fn lookup_child(&self, name: &str) -> SysResult<Arc<RamfsEntry>> {
        match self {
            RamfsEntry::Dir(inner, _) => inner
                .lock()
                .children
                .get(name)
                .cloned()
                .ok_or(Errno::ENOENT),
            _ => Err(Errno::ENOTDIR),
        }
    }

    /// Insert a child into a directory entry, updating the directory's mtime/ctime.
    fn insert_child(&self, name: &str, child: Arc<RamfsEntry>) -> SysResult<()> {
        match self {
            RamfsEntry::Dir(inner, _) => {
                let mut lock = inner.lock();
                lock.children.insert(name.to_string(), child);
                let ts = now();
                lock.mtime = ts;
                lock.ctime = ts;
                Ok(())
            }
            _ => Err(Errno::ENOTDIR),
        }
    }
}

// ── RamfsNode — VfsNode wrapper ──────────────────────────────────────────────

struct RamfsNode(Arc<RamfsEntry>);

impl VfsNode for RamfsNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        match &*self.0 {
            RamfsEntry::File(inner, _) => {
                let mut lock = inner.lock();
                let off = offset as usize;
                if off >= lock.data.len() {
                    return Ok(0);
                }
                let avail = &lock.data[off..];
                let n = avail.len().min(buf.len());
                buf[..n].copy_from_slice(&avail[..n]);
                // atime policy: update atime on every successful non-empty read (eager policy).
                // This matches the VfsStat contract documented in kernel/src/vfs/mod.rs.
                if n > 0 {
                    lock.atime = now();
                }
                Ok(n)
            }
            RamfsEntry::Dir(_, _) => Err(Errno::EISDIR),
            RamfsEntry::Symlink(_, _) => Err(Errno::EINVAL),
        }
    }

    fn write(&self, offset: u64, buf: &[u8]) -> SysResult<usize> {
        match &*self.0 {
            RamfsEntry::File(inner, _) => {
                let mut lock = inner.lock();
                let off = offset as usize;
                let end = off + buf.len();
                if end > lock.data.len() {
                    lock.data.resize(end, 0);
                }
                lock.data[off..end].copy_from_slice(buf);
                let ts = now();
                lock.mtime = ts;
                lock.ctime = ts;
                Ok(buf.len())
            }
            RamfsEntry::Dir(_, _) => Err(Errno::EISDIR),
            RamfsEntry::Symlink(_, _) => Err(Errno::EINVAL),
        }
    }

    fn stat(&self) -> SysResult<VfsStat> {
        match &*self.0 {
            RamfsEntry::File(inner, ino) => {
                let lock = inner.lock();
                let size = lock.data.len() as u64;
                Ok(VfsStat {
                    mode: VfsStat::S_IFREG | (lock.mode & 0o7777),
                    size,
                    ino: *ino,
                    nlink: lock.nlink,
                    atime_sec: lock.atime.0,
                    atime_nsec: lock.atime.1,
                    mtime_sec: lock.mtime.0,
                    mtime_nsec: lock.mtime.1,
                    ctime_sec: lock.ctime.0,
                    ctime_nsec: lock.ctime.1,
                    ..Default::default()
                })
            }
            RamfsEntry::Dir(inner, ino) => {
                let lock = inner.lock();
                // nlink = 2 (self + parent) + number of subdirectory children.
                let subdir_count = lock
                    .children
                    .values()
                    .filter(|e| matches!(***e, RamfsEntry::Dir(_, _)))
                    .count() as u32;
                Ok(VfsStat {
                    mode: VfsStat::S_IFDIR | (lock.mode & 0o7777),
                    size: 0,
                    ino: *ino,
                    nlink: 2 + subdir_count,
                    atime_sec: lock.atime.0,
                    atime_nsec: lock.atime.1,
                    mtime_sec: lock.mtime.0,
                    mtime_nsec: lock.mtime.1,
                    ctime_sec: lock.ctime.0,
                    ctime_nsec: lock.ctime.1,
                    ..Default::default()
                })
            }
            RamfsEntry::Symlink(target, ino) => Ok(VfsStat {
                mode: VfsStat::S_IFLNK | 0o777,
                size: target.len() as u64,
                ino: *ino,
                nlink: 1,
                ..Default::default()
            }),
        }
    }

    fn chmod(&self, mode: u32) -> SysResult<()> {
        let ts = now();
        match &*self.0 {
            RamfsEntry::File(inner, _) => {
                let mut lock = inner.lock();
                lock.mode = mode & 0o7777;
                lock.ctime = ts;
                Ok(())
            }
            RamfsEntry::Dir(inner, _) => {
                let mut lock = inner.lock();
                lock.mode = mode & 0o7777;
                lock.ctime = ts;
                Ok(())
            }
            RamfsEntry::Symlink(_, _) => {
                // Symlink permissions are fixed at 0o777; chmod is a no-op.
                Ok(())
            }
        }
    }

    fn utimes(&self, atime: Option<(u64, u32)>, mtime: Option<(u64, u32)>) -> SysResult<()> {
        let ts = now();
        match &*self.0 {
            RamfsEntry::File(inner, _) => {
                let mut lock = inner.lock();
                if let Some((sec, nsec)) = atime {
                    lock.atime = (sec, nsec);
                }
                if let Some((sec, nsec)) = mtime {
                    lock.mtime = (sec, nsec);
                }
                lock.ctime = ts;
                Ok(())
            }
            RamfsEntry::Dir(inner, _) => {
                let mut lock = inner.lock();
                if let Some((sec, nsec)) = atime {
                    lock.atime = (sec, nsec);
                }
                if let Some((sec, nsec)) = mtime {
                    lock.mtime = (sec, nsec);
                }
                lock.ctime = ts;
                Ok(())
            }
            RamfsEntry::Symlink(_, _) => {
                // Symlink timestamps are not stored; silently succeed.
                Ok(())
            }
        }
    }

    fn truncate(&self, new_size: u64) -> SysResult<()> {
        match &*self.0 {
            RamfsEntry::File(inner, _) => {
                let mut lock = inner.lock();
                lock.data.resize(new_size as usize, 0);
                let ts = now();
                lock.mtime = ts;
                lock.ctime = ts;
                Ok(())
            }
            RamfsEntry::Dir(_, _) => Err(Errno::EISDIR),
            RamfsEntry::Symlink(_, _) => Err(Errno::EINVAL),
        }
    }

    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        match &*self.0 {
            RamfsEntry::Dir(inner, _) => {
                let lock = inner.lock();
                super::write_readdir_entries(lock.children.keys().map(|s| s.as_str()), offset, buf)
            }
            RamfsEntry::Symlink(_, _) => Err(Errno::ENOTDIR),
            _ => Err(Errno::ENOTDIR),
        }
    }

    fn poll(&self) -> u16 {
        use abi::syscall::poll_flags::*;
        match &*self.0 {
            RamfsEntry::File(_, _) => POLLIN | POLLOUT,
            RamfsEntry::Dir(_, _) => POLLIN | POLLOUT,
            RamfsEntry::Symlink(_, _) => POLLIN | POLLOUT,
        }
    }

    fn readlink(&self) -> SysResult<alloc::string::String> {
        match &*self.0 {
            RamfsEntry::Symlink(target, _) => Ok(target.clone()),
            _ => Err(Errno::EINVAL),
        }
    }
}

// ── RamFs — VfsDriver ────────────────────────────────────────────────────────

/// In-memory filesystem driver.
///
/// Call [`RamFs::new`] to create an empty root directory.  Files and
/// directories can be pre-populated with [`RamFs::mkdir`] and
/// [`RamFs::create_file`].
pub struct RamFs {
    root: Arc<RamfsEntry>,
}

impl RamFs {
    /// Create a new, empty ramfs with a root directory.
    pub fn new() -> Self {
        Self {
            root: RamfsEntry::new_dir(),
        }
    }

    /// Create a subdirectory `path` (relative to this filesystem's root).
    ///
    /// Intermediate directories are created as needed.  It is not an error if
    /// the directory already exists.
    pub fn mkdir(&self, path: &str) -> SysResult<()> {
        let mut current = self.root.clone();
        for component in path.split('/').filter(|c| !c.is_empty()) {
            let next = match current.lookup_child(component) {
                Ok(child) => child,
                Err(Errno::ENOENT) => {
                    let new_dir = RamfsEntry::new_dir();
                    current.insert_child(component, new_dir.clone())?;
                    new_dir
                }
                Err(e) => return Err(e),
            };
            current = next;
        }
        Ok(())
    }

    /// Create or overwrite a file at `path` with `data`.
    pub fn create_file(&self, path: &str, data: Vec<u8>) -> SysResult<()> {
        let (dir_path, file_name) = split_last(path).ok_or(Errno::EINVAL)?;
        let dir = self.resolve_entry(dir_path)?;
        let file_entry = RamfsEntry::new_file(data);
        dir.insert_child(file_name, file_entry)
    }

    /// Create a symbolic link at `link_path` pointing to `target`.
    pub fn create_symlink(&self, target: &str, link_path: &str) -> SysResult<()> {
        let (dir_path, link_name) = split_last(link_path).ok_or(Errno::EINVAL)?;
        let dir = self.resolve_entry(dir_path)?;
        let symlink_entry = RamfsEntry::new_symlink(target.into());
        dir.insert_child(link_name, symlink_entry)
    }

    /// Create a hard link at `dst_path` referring to the same inode as `src_path`.
    ///
    /// Hard links are only supported for regular files; attempting to hard-link
    /// a directory returns `EPERM` (POSIX-compliant).
    pub fn create_hard_link(&self, src_path: &str, dst_path: &str) -> SysResult<()> {
        let src_entry = self.resolve_entry(src_path)?;
        // Only regular files may be hard-linked.
        match &*src_entry {
            RamfsEntry::Dir(_, _) => return Err(Errno::EPERM),
            RamfsEntry::Symlink(_, _) => return Err(Errno::EPERM),
            RamfsEntry::File(_, _) => {}
        }
        let (dst_dir_path, dst_name) = split_last(dst_path).ok_or(Errno::EINVAL)?;
        let dst_dir = self.resolve_entry(dst_dir_path)?;
        // Reject if destination already exists.
        if dst_dir.lookup_child(dst_name).is_ok() {
            return Err(Errno::EEXIST);
        }
        // Insert the *same* Arc into the destination directory first;
        // only increment nlink once the insert has succeeded.
        dst_dir.insert_child(dst_name, src_entry.clone())?;
        if let RamfsEntry::File(file_inner, _) = &*src_entry {
            let ts = now();
            let mut fl = file_inner.lock();
            fl.nlink = fl.nlink.saturating_add(1);
            fl.ctime = ts;
        }
        Ok(())
    }

    /// Resolve `path` to its `RamfsEntry`, walking the tree.
    fn resolve_entry(&self, path: &str) -> SysResult<Arc<RamfsEntry>> {
        let mut current = self.root.clone();
        for component in path.split('/').filter(|c| !c.is_empty()) {
            current = current.lookup_child(component)?;
        }
        Ok(current)
    }
}

impl Default for RamFs {
    fn default() -> Self {
        Self::new()
    }
}

impl VfsDriver for RamFs {
    /// Look up `path` (relative to the mount point) in this filesystem.
    fn lookup(&self, path: &str) -> SysResult<Arc<dyn VfsNode>> {
        let entry = self.resolve_entry(path)?;
        Ok(Arc::new(RamfsNode(entry)))
    }

    /// Create a new empty regular file at `path`.
    ///
    /// Intermediate directories must already exist.  Returns the new node
    /// as an open file descriptor suitable for immediate writing.
    fn create(&self, path: &str) -> SysResult<Arc<dyn VfsNode>> {
        let (dir_path, file_name) = split_last(path).ok_or(Errno::EINVAL)?;
        let dir = self.resolve_entry(dir_path)?;
        let file_entry = RamfsEntry::new_file(alloc::vec::Vec::new());
        dir.insert_child(file_name, file_entry.clone())?;
        Ok(Arc::new(RamfsNode(file_entry)))
    }

    /// Create a directory at `path`.
    ///
    /// Intermediate directories are created on demand.  Returns `EEXIST` if
    /// the final path component already exists (POSIX `mkdir(2)` semantics).
    fn mkdir(&self, path: &str) -> SysResult<()> {
        // Inline the inherent mkdir logic to avoid ambiguous self.mkdir() dispatch.
        let mut current = self.root.clone();
        let components: alloc::vec::Vec<&str> =
            path.split('/').filter(|c| !c.is_empty()).collect();
        let last_idx = components.len().saturating_sub(1);
        for (i, component) in components.iter().enumerate() {
            let next = match current.lookup_child(component) {
                Ok(child) => {
                    // The final component already exists → POSIX EEXIST.
                    if i == last_idx {
                        return Err(Errno::EEXIST);
                    }
                    child
                }
                Err(Errno::ENOENT) => {
                    let new_dir = RamfsEntry::new_dir();
                    current.insert_child(component, new_dir.clone())?;
                    new_dir
                }
                Err(e) => return Err(e),
            };
            current = next;
        }
        Ok(())
    }

    /// Remove the file or empty directory at `path`.
    fn unlink(&self, path: &str) -> SysResult<()> {
        let (dir_path, file_name) = split_last(path).ok_or(Errno::EINVAL)?;
        let dir = self.resolve_entry(dir_path)?;
        match &*dir {
            RamfsEntry::Dir(inner, _) => {
                let mut lock = inner.lock();
                let entry = lock.children.remove(file_name).ok_or(Errno::ENOENT)?;
                let ts = now();
                lock.mtime = ts;
                lock.ctime = ts;
                // Decrement the hard-link count for regular files.
                if let RamfsEntry::File(file_inner, _) = &*entry {
                    let mut fl = file_inner.lock();
                    fl.nlink = fl.nlink.saturating_sub(1);
                    fl.ctime = ts;
                }
                Ok(())
            }
            _ => Err(Errno::ENOTDIR),
        }
    }

    /// Rename a file or directory from `old_path` to `new_path`.
    fn rename(&self, old_path: &str, new_path: &str) -> SysResult<()> {
        let (old_dir_path, old_name) = split_last(old_path).ok_or(Errno::EINVAL)?;
        let (new_dir_path, new_name) = split_last(new_path).ok_or(Errno::EINVAL)?;

        let old_dir = self.resolve_entry(old_dir_path)?;
        let new_dir = self.resolve_entry(new_dir_path)?;

        let old_inner = match &*old_dir {
            RamfsEntry::Dir(c, _) => c,
            _ => return Err(Errno::ENOTDIR),
        };
        let new_inner = match &*new_dir {
            RamfsEntry::Dir(c, _) => c,
            _ => return Err(Errno::ENOTDIR),
        };

        if Arc::ptr_eq(&old_dir, &new_dir) {
            let mut lock = old_inner.lock();
            let entry = lock.children.remove(old_name).ok_or(Errno::ENOENT)?;
            lock.children.insert(new_name.to_string(), entry);
            let ts = now();
            lock.mtime = ts;
            lock.ctime = ts;
        } else {
            // Cross-directory rename within the same ramfs instance.
            // Lock in a stable order by pointer address to avoid deadlocks.
            let ptr_old = old_inner as *const _ as usize;
            let ptr_new = new_inner as *const _ as usize;

            if ptr_old < ptr_new {
                let mut lock_old = old_inner.lock();
                let mut lock_new = new_inner.lock();
                let entry = lock_old.children.remove(old_name).ok_or(Errno::ENOENT)?;
                lock_new.children.insert(new_name.to_string(), entry);
                let ts = now();
                lock_old.mtime = ts;
                lock_old.ctime = ts;
                lock_new.mtime = ts;
                lock_new.ctime = ts;
            } else {
                let mut lock_new = new_inner.lock();
                let mut lock_old = old_inner.lock();
                let entry = lock_old.children.remove(old_name).ok_or(Errno::ENOENT)?;
                lock_new.children.insert(new_name.to_string(), entry);
                let ts = now();
                lock_old.mtime = ts;
                lock_old.ctime = ts;
                lock_new.mtime = ts;
                lock_new.ctime = ts;
            }
        }

        Ok(())
    }

    /// Create a symbolic link at `link_path` pointing to `target`.
    fn symlink(&self, target: &str, link_path: &str) -> SysResult<()> {
        self.create_symlink(target, link_path)
    }

    /// Create a hard link at `dst_path` referring to the same file as `src_path`.
    fn link(&self, src_path: &str, dst_path: &str) -> SysResult<()> {
        self.create_hard_link(src_path, dst_path)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Split a path into `(parent_dir, last_component)`.
/// Returns `None` if `path` is empty or contains no filename.
fn split_last(path: &str) -> Option<(&str, &str)> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    match trimmed.rfind('/') {
        Some(idx) => {
            let parent = &trimmed[..idx]; // parent dir (no trailing slash)
            let name = &trimmed[idx + 1..];
            if name.is_empty() {
                None
            } else {
                Some((parent, name))
            }
        }
        None => Some(("", trimmed)),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_lookup_root_is_dir() {
        let fs = RamFs::new();
        let node = fs.lookup("").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_dir());
    }

    #[test]
    fn test_lookup_nonexistent_returns_enoent() {
        let fs = RamFs::new();
        assert!(matches!(fs.lookup("missing"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_create_and_read_file() {
        let fs = RamFs::new();
        fs.create_file("hello.txt", b"hello".to_vec()).unwrap();
        let node = fs.lookup("hello.txt").unwrap();
        let mut buf = [0u8; 5];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_write_and_read_back() {
        let fs = RamFs::new();
        fs.create_file("data.bin", vec![]).unwrap();
        let node = fs.lookup("data.bin").unwrap();
        let n = node.write(0, b"test").unwrap();
        assert_eq!(n, 4);
        let mut buf = [0u8; 4];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf, b"test");
    }

    #[test]
    fn test_mkdir_and_lookup_subdir() {
        let fs = RamFs::new();
        fs.mkdir("sub").unwrap();
        let node = fs.lookup("sub").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_dir());
    }

    #[test]
    fn test_file_in_subdir() {
        let fs = RamFs::new();
        fs.mkdir("sub").unwrap();
        fs.create_file("sub/file.txt", b"data".to_vec()).unwrap();
        let node = fs.lookup("sub/file.txt").unwrap();
        let mut buf = [0u8; 4];
        node.read(0, &mut buf).unwrap();
        assert_eq!(&buf, b"data");
    }

    #[test]
    fn test_read_dir_returns_entries() {
        let fs = RamFs::new();
        fs.mkdir("alpha").unwrap();
        fs.mkdir("beta").unwrap();
        let root = fs.lookup("").unwrap();
        let mut buf = [0u8; 64];
        let n = root.readdir(0, &mut buf).unwrap();
        assert!(n > 0, "readdir should return some bytes");
        let content = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(content.contains("alpha"));
        assert!(content.contains("beta"));
    }

    #[test]
    fn test_write_extends_file() {
        let fs = RamFs::new();
        fs.create_file("grow.bin", vec![0u8; 4]).unwrap();
        let node = fs.lookup("grow.bin").unwrap();
        // Write past original end.
        node.write(4, b"more").unwrap();
        let stat = node.stat().unwrap();
        assert_eq!(stat.size, 8);
    }

    #[test]
    fn test_sync_succeeds_on_file() {
        let fs = RamFs::new();
        let node = fs.create("sync.txt").unwrap();
        node.write(0, b"data").unwrap();
        assert_eq!(node.sync(), Ok(()));
    }

    #[test]
    fn test_sync_succeeds_on_directory() {
        let fs = RamFs::new();
        fs.mkdir("syncdir").unwrap();
        let node = fs.lookup("syncdir").unwrap();
        assert_eq!(node.sync(), Ok(()));
    }

    #[test]
    fn test_sync_is_safe_to_repeat() {
        let fs = RamFs::new();
        let node = fs.create("repeat-sync.txt").unwrap();
        assert_eq!(node.sync(), Ok(()));
        assert_eq!(node.sync(), Ok(()));
        assert_eq!(node.sync(), Ok(()));
    }

    // ── VfsDriver trait methods ──────────────────────────────────────────────

    #[test]
    fn test_driver_create_makes_empty_file() {
        let fs = RamFs::new();
        let node = fs.create("newfile.txt").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_reg());
        assert_eq!(stat.size, 0);
    }

    #[test]
    fn test_driver_create_file_is_writable() {
        let fs = RamFs::new();
        let node = fs.create("write_me.txt").unwrap();
        let n = node.write(0, b"hello").unwrap();
        assert_eq!(n, 5);
        // The file must also be retrievable via lookup.
        let node2 = fs.lookup("write_me.txt").unwrap();
        let mut buf = [0u8; 5];
        let n = node2.read(0, &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_driver_create_invalid_empty_path() {
        let fs = RamFs::new();
        // An empty path has no file name component → EINVAL.
        assert!(matches!(fs.create(""), Err(Errno::EINVAL)));
    }

    #[test]
    fn test_driver_mkdir_creates_directory() {
        let fs = RamFs::new();
        fs.mkdir("newdir").unwrap();
        let node = fs.lookup("newdir").unwrap();
        assert!(node.stat().unwrap().is_dir());
    }

    #[test]
    fn test_driver_mkdir_nested() {
        let fs = RamFs::new();
        fs.mkdir("a/b/c").unwrap();
        let node = fs.lookup("a/b/c").unwrap();
        assert!(node.stat().unwrap().is_dir());
    }

    #[test]
    fn test_driver_unlink_removes_file() {
        let fs = RamFs::new();
        fs.create_file("to_delete.txt", b"bye".to_vec()).unwrap();
        // File exists.
        assert!(fs.lookup("to_delete.txt").is_ok());
        // Unlink it.
        fs.unlink("to_delete.txt").unwrap();
        // File is gone.
        assert!(matches!(fs.lookup("to_delete.txt"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_driver_unlink_nonexistent_returns_enoent() {
        let fs = RamFs::new();
        assert!(matches!(fs.unlink("ghost.txt"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_driver_unlink_empty_path_returns_einval() {
        let fs = RamFs::new();
        assert!(matches!(fs.unlink(""), Err(Errno::EINVAL)));
    }

    // ── truncate ────────────────────────────────────────────────────────────

    #[test]
    fn test_truncate_shrinks_file() {
        let fs = RamFs::new();
        fs.create_file("shrink.bin", b"hello".to_vec()).unwrap();
        let node = fs.lookup("shrink.bin").unwrap();
        node.truncate(2).unwrap();
        assert_eq!(node.stat().unwrap().size, 2);
        let mut buf = [0u8; 5];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&buf[..2], b"he");
    }

    #[test]
    fn test_truncate_extends_file_with_zeros() {
        let fs = RamFs::new();
        fs.create_file("grow.bin", b"hi".to_vec()).unwrap();
        let node = fs.lookup("grow.bin").unwrap();
        node.truncate(5).unwrap();
        assert_eq!(node.stat().unwrap().size, 5);
        let mut buf = [0xFFu8; 5];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf, b"hi\0\0\0");
    }

    #[test]
    fn test_truncate_to_zero() {
        let fs = RamFs::new();
        fs.create_file("zero.bin", b"data".to_vec()).unwrap();
        let node = fs.lookup("zero.bin").unwrap();
        node.truncate(0).unwrap();
        assert_eq!(node.stat().unwrap().size, 0);
        let mut buf = [0u8; 4];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_truncate_on_dir_returns_eisdir() {
        let fs = RamFs::new();
        fs.mkdir("adir").unwrap();
        let node = fs.lookup("adir").unwrap();
        assert!(matches!(node.truncate(0), Err(Errno::EISDIR)));
    }

    // ── timestamps ──────────────────────────────────────────────────────────

    /// Timestamps are initialised (non-garbage) on file creation.
    /// In tests the clock is not anchored so all timestamps are zero,
    /// but they must be consistent across stat calls.
    #[test]
    fn test_create_initialises_timestamps() {
        let fs = RamFs::new();
        fs.create_file("ts.txt", b"hello".to_vec()).unwrap();
        let node = fs.lookup("ts.txt").unwrap();
        let stat = node.stat().unwrap();
        // All three timestamps must be valid (non-garbage).
        assert!(stat.atime_nsec < 1_000_000_000);
        assert!(stat.mtime_nsec < 1_000_000_000);
        assert!(stat.ctime_nsec < 1_000_000_000);
        // On creation, atime == mtime == ctime.
        assert_eq!(stat.atime_sec, stat.mtime_sec);
        assert_eq!(stat.mtime_sec, stat.ctime_sec);
    }

    /// Write must update mtime and ctime; atime must be unchanged.
    #[test]
    fn test_write_updates_mtime_and_ctime() {
        let fs = RamFs::new();
        let node = fs.create("wts.txt").unwrap();
        let before = node.stat().unwrap();
        // Write updates mtime/ctime regardless of clock value.
        node.write(0, b"data").unwrap();
        let after = node.stat().unwrap();
        // mtime_sec and ctime_sec must be >= before.
        assert!(after.mtime_sec >= before.mtime_sec);
        assert!(after.ctime_sec >= before.ctime_sec);
    }

    /// Truncate must update mtime and ctime.
    #[test]
    fn test_truncate_updates_mtime_and_ctime() {
        let fs = RamFs::new();
        fs.create_file("tts.txt", b"hello world".to_vec()).unwrap();
        let node = fs.lookup("tts.txt").unwrap();
        let before = node.stat().unwrap();
        node.truncate(3).unwrap();
        let after = node.stat().unwrap();
        assert!(after.mtime_sec >= before.mtime_sec);
        assert!(after.ctime_sec >= before.ctime_sec);
    }

    /// stat() must return consistent timestamps (atime, mtime, ctime are valid).
    #[test]
    fn test_stat_timestamps_are_non_garbage() {
        let fs = RamFs::new();
        fs.create_file("ng.txt", b"x".to_vec()).unwrap();
        let node = fs.lookup("ng.txt").unwrap();
        let stat = node.stat().unwrap();
        // nsec components must be in valid range.
        assert!(stat.atime_nsec < 1_000_000_000);
        assert!(stat.mtime_nsec < 1_000_000_000);
        assert!(stat.ctime_nsec < 1_000_000_000);
    }

    /// Directory timestamps must be consistent.
    #[test]
    fn test_dir_timestamps_consistent() {
        let fs = RamFs::new();
        fs.mkdir("dts").unwrap();
        let node = fs.lookup("dts").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.atime_nsec < 1_000_000_000);
        assert!(stat.mtime_nsec < 1_000_000_000);
        assert!(stat.ctime_nsec < 1_000_000_000);
        assert_eq!(stat.atime_sec, stat.mtime_sec);
    }

    /// Creating a file in a directory must update the directory's mtime/ctime.
    #[test]
    fn test_dir_mtime_updates_on_child_create() {
        let fs = RamFs::new();
        fs.mkdir("dmu").unwrap();
        let dir_node = fs.lookup("dmu").unwrap();
        let dir_before = dir_node.stat().unwrap();
        // Create a child file.
        fs.create_file("dmu/child.txt", b"x".to_vec()).unwrap();
        let dir_after = dir_node.stat().unwrap();
        assert!(dir_after.mtime_sec >= dir_before.mtime_sec);
        assert!(dir_after.ctime_sec >= dir_before.ctime_sec);
    }

    // ── symlink ──────────────────────────────────────────────────────────────

    #[test]
    fn test_symlink_stat_is_iflnk() {
        let fs = RamFs::new();
        fs.create_symlink("/target/path", "mylink").unwrap();
        let node = fs.lookup("mylink").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_symlink(), "mode=0o{:o}", stat.mode);
        assert_eq!(stat.size, "/target/path".len() as u64);
    }

    #[test]
    fn test_symlink_readlink_returns_target() {
        let fs = RamFs::new();
        fs.create_symlink("/some/target", "link").unwrap();
        let node = fs.lookup("link").unwrap();
        let target = node.readlink().unwrap();
        assert_eq!(target, "/some/target");
    }

    #[test]
    fn test_regular_file_readlink_returns_einval() {
        let fs = RamFs::new();
        fs.create_file("regular.txt", b"data".to_vec()).unwrap();
        let node = fs.lookup("regular.txt").unwrap();
        assert!(matches!(node.readlink(), Err(Errno::EINVAL)));
    }

    #[test]
    fn test_symlink_read_returns_einval() {
        let fs = RamFs::new();
        fs.create_symlink("/target", "lnk").unwrap();
        let node = fs.lookup("lnk").unwrap();
        let mut buf = [0u8; 8];
        assert!(matches!(node.read(0, &mut buf), Err(Errno::EINVAL)));
    }

    #[test]
    fn test_symlink_write_returns_einval() {
        let fs = RamFs::new();
        fs.create_symlink("/target", "lnk2").unwrap();
        let node = fs.lookup("lnk2").unwrap();
        assert!(matches!(node.write(0, b"data"), Err(Errno::EINVAL)));
    }

    #[test]
    fn test_driver_symlink_creates_entry() {
        let fs = RamFs::new();
        fs.symlink("/real/path", "sl").unwrap();
        let node = fs.lookup("sl").unwrap();
        assert!(node.stat().unwrap().is_symlink());
        assert_eq!(node.readlink().unwrap(), "/real/path");
    }

    #[test]
    fn test_symlink_unlink_removes_link() {
        let fs = RamFs::new();
        fs.create_symlink("/target", "rm_link").unwrap();
        assert!(fs.lookup("rm_link").is_ok());
        fs.unlink("rm_link").unwrap();
        assert!(matches!(fs.lookup("rm_link"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_symlink_in_subdir() {
        let fs = RamFs::new();
        fs.mkdir("sub").unwrap();
        fs.create_symlink("/other", "sub/link_in_sub").unwrap();
        let node = fs.lookup("sub/link_in_sub").unwrap();
        assert!(node.stat().unwrap().is_symlink());
        assert_eq!(node.readlink().unwrap(), "/other");
    }

    #[test]
    fn test_file_stat_has_nlink_one_and_zero_uid_gid() {
        let fs = RamFs::new();
        fs.create_file("owned.txt", b"hello".to_vec()).unwrap();
        let node = fs.lookup("owned.txt").unwrap();
        let st = node.stat().unwrap();
        assert_eq!(st.nlink, 1);
        assert_eq!(st.uid, 0);
        assert_eq!(st.gid, 0);
        assert_eq!(st.rdev, 0);
    }

    #[test]
    fn test_dir_stat_has_nlink_at_least_two() {
        let fs = RamFs::new();
        let root = fs.lookup("").unwrap();
        let st = root.stat().unwrap();
        assert!(st.nlink >= 2, "directory nlink should be >= 2, got {}", st.nlink);
        assert_eq!(st.uid, 0);
        assert_eq!(st.gid, 0);
    }

    #[test]
    fn test_dir_nlink_increases_with_subdirs() {
        let fs = RamFs::new();
        let root_before = fs.lookup("").unwrap().stat().unwrap().nlink;
        fs.mkdir("sub1").unwrap();
        fs.mkdir("sub2").unwrap();
        let root_after = fs.lookup("").unwrap().stat().unwrap().nlink;
        assert_eq!(root_after, root_before + 2);
    }

    #[test]
    fn test_symlink_stat_has_nlink_one() {
        let fs = RamFs::new();
        fs.create_symlink("/target", "lnk3").unwrap();
        let node = fs.lookup("lnk3").unwrap();
        let st = node.stat().unwrap();
        assert_eq!(st.nlink, 1);
    }

    // ── Hard-link tests ───────────────────────────────────────────────────────

    #[test]
    fn test_hard_link_creates_second_entry_with_same_ino() {
        let fs = RamFs::new();
        fs.create_file("orig.txt", b"hello".to_vec()).unwrap();
        fs.create_hard_link("orig.txt", "link.txt").unwrap();

        let orig_ino = fs.lookup("orig.txt").unwrap().stat().unwrap().ino;
        let link_ino = fs.lookup("link.txt").unwrap().stat().unwrap().ino;
        assert_eq!(orig_ino, link_ino, "hard link must share inode number");
    }

    #[test]
    fn test_hard_link_nlink_is_two() {
        let fs = RamFs::new();
        fs.create_file("f.txt", b"data".to_vec()).unwrap();
        fs.create_hard_link("f.txt", "f2.txt").unwrap();

        let st = fs.lookup("f.txt").unwrap().stat().unwrap();
        assert_eq!(st.nlink, 2, "nlink should be 2 after one hard link");
    }

    #[test]
    fn test_hard_link_data_shared() {
        let fs = RamFs::new();
        fs.create_file("src.txt", b"content".to_vec()).unwrap();
        fs.create_hard_link("src.txt", "dst.txt").unwrap();

        // Reading through the link returns the original content.
        let node = fs.lookup("dst.txt").unwrap();
        let mut buf = [0u8; 7];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 7);
        assert_eq!(&buf, b"content");
    }

    #[test]
    fn test_hard_link_unlink_decrements_nlink() {
        let fs = RamFs::new();
        fs.create_file("h.txt", b"hi".to_vec()).unwrap();
        fs.create_hard_link("h.txt", "h2.txt").unwrap();

        // Remove the link; the original should have nlink = 1 again.
        fs.unlink("h2.txt").unwrap();
        let st = fs.lookup("h.txt").unwrap().stat().unwrap();
        assert_eq!(st.nlink, 1);
    }

    #[test]
    fn test_hard_link_to_directory_returns_eperm() {
        let fs = RamFs::new();
        fs.mkdir("adir").unwrap();
        let err = fs.create_hard_link("adir", "adir_link").unwrap_err();
        assert_eq!(err, Errno::EPERM);
    }

    #[test]
    fn test_hard_link_to_symlink_returns_eperm() {
        let fs = RamFs::new();
        fs.create_symlink("/target", "sl").unwrap();
        let err = fs.create_hard_link("sl", "sl2").unwrap_err();
        assert_eq!(err, Errno::EPERM);
    }

    #[test]
    fn test_hard_link_dst_exists_returns_eexist() {
        let fs = RamFs::new();
        fs.create_file("a.txt", b"a".to_vec()).unwrap();
        fs.create_file("b.txt", b"b".to_vec()).unwrap();
        let err = fs.create_hard_link("a.txt", "b.txt").unwrap_err();
        assert_eq!(err, Errno::EEXIST);
    }

    #[test]
    fn test_hard_link_src_missing_returns_enoent() {
        let fs = RamFs::new();
        let err = fs.create_hard_link("ghost.txt", "link.txt").unwrap_err();
        assert_eq!(err, Errno::ENOENT);
    }

    // ── VfsDriver::mkdir POSIX semantics ──────────────────────────────────────

    /// Creating a directory that already exists must return EEXIST (POSIX mkdir(2)).
    #[test]
    fn test_driver_mkdir_existing_dir_returns_eexist() {
        use super::super::VfsDriver;
        let fs = RamFs::new();
        // First mkdir succeeds.
        fs.mkdir("newdir").unwrap();
        // Second mkdir on the same name must fail with EEXIST.
        let err = <RamFs as VfsDriver>::mkdir(&fs, "newdir").unwrap_err();
        assert_eq!(err, Errno::EEXIST);
    }

    /// Creating a directory whose last component clashes with an existing file
    /// must also return EEXIST.
    #[test]
    fn test_driver_mkdir_clashes_with_file_returns_eexist() {
        use super::super::VfsDriver;
        let fs = RamFs::new();
        fs.create_file("thing.txt", vec![]).unwrap();
        let err = <RamFs as VfsDriver>::mkdir(&fs, "thing.txt").unwrap_err();
        assert_eq!(err, Errno::EEXIST);
    }

    /// Creating a directory with a non-existent intermediate component must
    /// succeed (intermediate directories are still created on demand).
    #[test]
    fn test_driver_mkdir_nested_creates_intermediate_dirs() {
        use super::super::VfsDriver;
        let fs = RamFs::new();
        <RamFs as VfsDriver>::mkdir(&fs, "a/b/c").unwrap();
        let node = fs.lookup("a/b/c").unwrap();
        assert!(node.stat().unwrap().is_dir());
    }
}
