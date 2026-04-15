//! VFS mount table.
//!
//! Maintains a list of (`mount_point`, `Arc<dyn VfsDriver>`) pairs sorted
//! longest-prefix-first so that the most specific mount is tried first.
//!
//! # Thread safety
//! The table is protected by a spin-lock.  Mounts happen once at boot; reads
//! happen on every `open(2)` syscall.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use super::VfsDriver;
use abi::errors::{Errno, SysResult};

struct MountEntry {
    /// The canonical mount point, e.g. `"/dev"` (no trailing slash).
    prefix: String,
    driver: Arc<dyn VfsDriver>,
    /// Stable per-mount identifier, assigned once at mount time.
    id: u64,
}

static MOUNT_TABLE: Mutex<Vec<MountEntry>> = Mutex::new(Vec::new());
static INIT_DONE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static NEXT_MOUNT_ID: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);

/// Initialise the mount table storage.  Must be called once before any
/// [`mount`] or [`lookup`] call.
pub fn init() {
    // The static Mutex<Vec<_>> is already valid; mark init complete.
    INIT_DONE.store(true, core::sync::atomic::Ordering::SeqCst);
}

/// Mount a filesystem driver at `mount_point` (e.g. `"/dev"`).
///
/// Replaces any existing mount at the same point.  Thread-safe.
pub fn mount(mount_point: &str, driver: Arc<dyn VfsDriver>) {
    let prefix = normalise(mount_point);
    let id = NEXT_MOUNT_ID.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    let mut table = MOUNT_TABLE.lock();
    // Remove duplicate.
    table.retain(|e| e.prefix != prefix);
    table.push(MountEntry { prefix, driver, id });
    // Keep longest-prefix first so that `/dev/pts` beats `/dev`.
    table.sort_by(|a, b| b.prefix.len().cmp(&a.prefix.len()));
}

/// Unmount the filesystem at `mount_point`.
///
/// Returns `Err(ENOENT)` if nothing is mounted there.
pub fn umount(mount_point: &str) -> SysResult<()> {
    let prefix = normalise(mount_point);
    let mut table = MOUNT_TABLE.lock();
    let before = table.len();
    table.retain(|e| e.prefix != prefix);
    if table.len() == before {
        Err(Errno::ENOENT)
    } else {
        Ok(())
    }
}

/// Resolve `path` to a VFS node by finding the best-matching mount and
/// calling its `lookup` with the remaining path component(s).
///
/// `path` must be absolute (start with `/`).
pub fn lookup(path: &str) -> SysResult<alloc::sync::Arc<dyn super::VfsNode>> {
    if !path.starts_with('/') {
        return Err(Errno::ENOENT);
    }

    // Capture matching drivers into a local list to avoid holding the spinlock
    // during potentially blocking driver lookups.
    let matches: Vec<(String, Arc<dyn VfsDriver>)> = {
        let table = MOUNT_TABLE.lock();
        table
            .iter()
            .filter_map(|entry| {
                strip_prefix(path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .collect()
    };

    for (rel, driver) in matches {
        match driver.lookup(&rel) {
            Ok(node) => return Ok(node),
            Err(Errno::ENOENT) => {
                // Specialized fallback for fb0 if the driver doesn't have it.
                // This is a legacy hack for early-boot framebuffer access.
                if rel == "fb0" && path.starts_with("/dev") {
                    if let Ok(node) = crate::vfs::devfs::DevFs::new().lookup("fb0") {
                        return Ok(node);
                    }
                }
                // Continue to next matching mount point (shorter prefix).
                continue;
            }
            Err(err) => {
                return Err(err);
            }
        }
    }

    Err(Errno::ENOENT)
}

/// Return all mount points that are immediate children of `parent_path`.
///
/// For example, if we have mounts at `/dev`, `/dev/display/card0`, and `/sys`,
/// `get_mounts_under("/dev/display")` would return `["card0"]`.
pub fn get_mounts_under(parent_path: &str) -> Vec<String> {
    let prefix = normalise(parent_path);
    let table = MOUNT_TABLE.lock();
    let mut results = Vec::new();

    for entry in table.iter() {
        if entry.prefix == prefix {
            continue;
        }

        if let Some(rel) = strip_prefix(&entry.prefix, &prefix) {
            // Check if it's an immediate child (no further slashes).
            let rel = rel.trim_start_matches('/');
            if !rel.is_empty() && !rel.contains('/') {
                results.push(rel.to_string());
            }
        }
    }

    results
}

/// Return the stable mount ID for the best-matching mount covering `path`.
///
/// Returns `0` if no mount covers `path` (which should not happen for valid
/// absolute paths after the VFS is initialised).
pub fn mount_id_for_path(path: &str) -> u64 {
    if !path.starts_with('/') {
        return 0;
    }
    let table = MOUNT_TABLE.lock();
    // The table is sorted longest-prefix first, so the first match is the
    // most specific mount.
    for entry in table.iter() {
        if strip_prefix(path, &entry.prefix).is_some() {
            return entry.id;
        }
    }
    0
}

/// Create a new regular file at `path` by finding the best-matching mount.
///
/// `path` must be absolute.  Returns the new open node on success.
pub fn create(path: &str) -> SysResult<alloc::sync::Arc<dyn super::VfsNode>> {
    if !path.starts_with('/') {
        return Err(Errno::ENOENT);
    }
    let (rel, driver): (String, Arc<dyn VfsDriver>) = {
        let table = MOUNT_TABLE.lock();
        table
            .iter()
            .find_map(|entry| {
                strip_prefix(path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .ok_or(Errno::ENOENT)?
    };
    driver.create(&rel)
}

/// Create a directory at `path` by finding the best-matching mount.
///
/// `path` must be absolute.
pub fn mkdir(path: &str) -> SysResult<()> {
    if !path.starts_with('/') {
        return Err(Errno::ENOENT);
    }
    let (rel, driver): (String, Arc<dyn VfsDriver>) = {
        let table = MOUNT_TABLE.lock();
        table
            .iter()
            .find_map(|entry| {
                strip_prefix(path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .ok_or(Errno::ENOENT)?
    };
    driver.mkdir(&rel)
}

/// Remove the file or empty directory at `path`.
///
/// `path` must be absolute.
pub fn unlink(path: &str) -> SysResult<()> {
    if !path.starts_with('/') {
        return Err(Errno::ENOENT);
    }
    let (rel, driver): (String, Arc<dyn VfsDriver>) = {
        let table = MOUNT_TABLE.lock();
        table
            .iter()
            .find_map(|entry| {
                strip_prefix(path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .ok_or(Errno::ENOENT)?
    };
    driver.unlink(&rel)
}

/// Create a symbolic link at `link_path` pointing to `target`.
///
/// `link_path` must be absolute.
pub fn symlink(target: &str, link_path: &str) -> SysResult<()> {
    if !link_path.starts_with('/') {
        return Err(Errno::ENOENT);
    }
    let (rel, driver): (String, Arc<dyn VfsDriver>) = {
        let table = MOUNT_TABLE.lock();
        table
            .iter()
            .find_map(|entry| {
                strip_prefix(link_path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .ok_or(Errno::ENOENT)?
    };
    driver.symlink(target, &rel)
}

/// Create a hard link at `dst_path` referring to the same file as `src_path`.
///
/// Both paths must be absolute and within the same mount point.
/// Returns `EXDEV` if the paths are on different mount points, and `ENOENT`
/// if either path has no matching mount.
pub fn link(src_path: &str, dst_path: &str) -> SysResult<()> {
    if !src_path.starts_with('/') || !dst_path.starts_with('/') {
        return Err(Errno::ENOENT);
    }
    let (src_rel, dst_rel, driver): (String, String, Arc<dyn VfsDriver>) = {
        let table = MOUNT_TABLE.lock();
        // Find the best (longest-prefix) mount for each path individually so
        // we can distinguish "no mount" (ENOENT) from "different mounts" (EXDEV).
        let src_entry = table
            .iter()
            .find_map(|entry| {
                strip_prefix(src_path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .ok_or(Errno::ENOENT)?;
        let dst_entry = table
            .iter()
            .find_map(|entry| {
                strip_prefix(dst_path, &entry.prefix)
                    .map(|rel| (rel.to_string(), Arc::clone(&entry.driver)))
            })
            .ok_or(Errno::ENOENT)?;
        // Both paths must resolve to the same driver (same mount point).
        if !Arc::ptr_eq(&src_entry.1, &dst_entry.1) {
            return Err(Errno::EXDEV);
        }
        (src_entry.0, dst_entry.0, src_entry.1)
    };
    driver.link(&src_rel, &dst_rel)
}

/// Rename a file or directory from `old_path` to `new_path`.
///
/// Both paths must be absolute and within the same mount point.
pub fn rename(old_path: &str, new_path: &str) -> SysResult<()> {
    if !old_path.starts_with('/') || !new_path.starts_with('/') {
        return Err(Errno::ENOENT);
    }
    let (old_rel, new_rel, driver): (String, String, Arc<dyn VfsDriver>) = {
        let table = MOUNT_TABLE.lock();
        table
            .iter()
            .find_map(|entry| {
                let old_rel = strip_prefix(old_path, &entry.prefix)?;
                let new_rel = strip_prefix(new_path, &entry.prefix)?;
                Some((
                    old_rel.to_string(),
                    new_rel.to_string(),
                    Arc::clone(&entry.driver),
                ))
            })
            .ok_or(Errno::EXDEV)?
    };
    driver.rename(&old_rel, &new_rel)
}

/// Return a human-readable text listing of all active mount points.
///
/// Format:
/// ```text
/// <mount_point> <type> rw 0 0
/// ```
/// Used by `/proc/mounts`.
pub fn mounts_text() -> alloc::string::String {
    let table = MOUNT_TABLE.lock();
    let mut out = alloc::string::String::new();
    for entry in table.iter() {
        out.push_str(&entry.prefix);
        out.push_str(" vfs rw 0 0\n");
    }
    out
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn normalise(p: &str) -> String {
    // Strip trailing slash unless it is the root itself.
    let s = p.trim_end_matches('/');
    if s.is_empty() {
        String::from("/")
    } else {
        String::from(s)
    }
}

/// Returns the relative portion of `path` after `prefix`, if `path` starts
/// with that prefix followed by `/` or is exactly equal.
fn strip_prefix<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix == "/" {
        // Root mount: everything after the leading slash.
        return Some(&path[1..]);
    }
    if path == prefix {
        return Some("");
    }
    // Avoid allocation: check if path starts with prefix followed by '/'.
    let prefix_with_slash = prefix.trim_end_matches('/');
    let rest = path.strip_prefix(prefix_with_slash)?;
    rest.strip_prefix('/')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::{VfsDriver, VfsNode, VfsStat};
    use abi::errors::Errno;
    use alloc::sync::Arc;
    use alloc::vec;

    struct DummyNode;
    impl VfsNode for DummyNode {
        fn read(&self, _: u64, _: &mut [u8]) -> abi::errors::SysResult<usize> {
            Ok(0)
        }
        fn write(&self, _: u64, _: &[u8]) -> abi::errors::SysResult<usize> {
            Ok(0)
        }
        fn stat(&self) -> abi::errors::SysResult<VfsStat> {
            Ok(VfsStat {
                mode: VfsStat::S_IFCHR | 0o666,
                size: 0,
                ino: 99,
                ..Default::default()
            })
        }
    }

    struct DummyFs;
    impl VfsDriver for DummyFs {
        fn lookup(&self, path: &str) -> abi::errors::SysResult<Arc<dyn VfsNode>> {
            if path == "thing" {
                Ok(Arc::new(DummyNode))
            } else {
                Err(Errno::ENOENT)
            }
        }
    }

    fn fresh_table() {
        MOUNT_TABLE.lock().clear();
        INIT_DONE.store(true, core::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn test_mount_and_lookup() {
        fresh_table();
        mount("/test", Arc::new(DummyFs));
        assert!(lookup("/test/thing").is_ok());
    }

    #[test]
    fn test_lookup_unknown_path_returns_enoent() {
        fresh_table();
        mount("/test", Arc::new(DummyFs));
        assert!(matches!(lookup("/other/thing"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_umount_removes_mount() {
        fresh_table();
        mount("/rm", Arc::new(DummyFs));
        assert!(lookup("/rm/thing").is_ok());
        umount("/rm").unwrap();
        assert!(matches!(lookup("/rm/thing"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_longer_prefix_wins() {
        fresh_table();

        struct Short;
        impl VfsDriver for Short {
            fn lookup(&self, _: &str) -> abi::errors::SysResult<Arc<dyn VfsNode>> {
                Err(Errno::EIO)
            }
        }
        mount("/a", Arc::new(Short));
        mount("/a/b", Arc::new(DummyFs));

        // "/a/b/thing" should match the longer prefix "/a/b".
        assert!(lookup("/a/b/thing").is_ok());
        // "/a/thing" hits the short driver which returns EIO (not ENOENT).
        assert!(matches!(lookup("/a/thing"), Err(Errno::EIO)));
    }

    #[test]
    fn test_strip_prefix_helper() {
        assert_eq!(strip_prefix("/dev/console", "/dev"), Some("console"));
        assert_eq!(strip_prefix("/dev", "/dev"), Some(""));
        assert_eq!(strip_prefix("/other/path", "/dev"), None);
        assert_eq!(strip_prefix("/hello", "/"), Some("hello"));
    }
}
