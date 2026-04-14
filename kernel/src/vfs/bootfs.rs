//! bootfs — minimal static boot filesystem.
//!
//! Provides a read-only filesystem whose contents are compiled into the kernel
//! binary at link time, or passed as boot modules by the loader.
//!
//! The boot filesystem is mounted at `/boot` by [`crate::vfs::init`].

use crate::BootModuleDesc;
use abi::errors::{Errno, SysResult};
use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::sync::Arc;

use super::{VfsDriver, VfsNode, VfsStat};

// ── Embedded file contents ────────────────────────────────────────────────────

const VERSION_DATA: &[u8] = b"Thing-OS v0.1 (janix ACT IV)\n";
const MOTD_DATA: &[u8] = b"Welcome to Thing-OS.\nBooting into path-based namespace...\n";

// ── BootFs driver ─────────────────────────────────────────────────────────────

/// The boot filesystem driver.  Mounted at `/boot` by `vfs::init`.
pub struct BootFs {
    modules: &'static [BootModuleDesc],
}

impl BootFs {
    pub fn new(modules: &'static [BootModuleDesc]) -> Self {
        Self { modules }
    }
}

impl VfsDriver for BootFs {
    fn lookup(&self, path: &str) -> SysResult<Arc<dyn VfsNode>> {
        let path = path.strip_prefix('/').unwrap_or(path);
        if path.is_empty() {
            return Ok(Arc::new(BootDirNode {
                prefix: String::new(),
                modules: self.modules,
            }));
        }

        if path == "version" {
            return Ok(Arc::new(StaticFileNode::new(VERSION_DATA, 10)));
        }
        if path == "motd" {
            return Ok(Arc::new(StaticFileNode::new(MOTD_DATA, 11)));
        }

        // Search for exact match in modules
        for (i, m) in self.modules.iter().enumerate() {
            let name = m.name.trim_matches('\0').trim();
            let clean_name = name.strip_prefix('/').unwrap_or(name);

            if clean_name == path {
                crate::kdebug!("BootFs: EXACT match for '{}' at index {}", path, i);
                return Ok(Arc::new(StaticFileNode::new(m.bytes, 100 + i as u64)));
            }
        }

        // Check if `path` is a directory prefix
        let dir_prefix = if path.ends_with('/') {
            path.to_string()
        } else {
            alloc::format!("{}/", path)
        };
        let mut found_subdir = false;
        for m in self.modules {
            let name = m.name.trim_matches('\0').trim();
            let clean_name = name.strip_prefix('/').unwrap_or(name);
            if clean_name.starts_with(&dir_prefix) {
                found_subdir = true;
                break;
            }
        }

        if found_subdir {
            return Ok(Arc::new(BootDirNode {
                prefix: path.to_string(),
                modules: self.modules,
            }));
        }

        Err(Errno::ENOENT)
    }
}

// ── /boot directory node ──────────────────────────────────────────────────────

struct BootDirNode {
    prefix: String,
    modules: &'static [BootModuleDesc],
}

impl VfsNode for BootDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o555,
            size: 0,
            ino: 9,
            ..Default::default()
        })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let mut components = BTreeSet::new();

        // Standard files at root
        if self.prefix.is_empty() {
            components.insert("version".to_string());
            components.insert("motd".to_string());
        }

        let prefix_with_slash = if self.prefix.is_empty() {
            String::new()
        } else if self.prefix.ends_with('/') {
            self.prefix.clone()
        } else {
            alloc::format!("{}/", self.prefix)
        };

        for m in self.modules {
            let name = m.name.trim_matches('\0').trim();
            let clean_name = name.strip_prefix('/').unwrap_or(name);

            if clean_name.starts_with(&prefix_with_slash) {
                let rest = &clean_name[prefix_with_slash.len()..];
                if let Some(slash_idx) = rest.find('/') {
                    // It's a directory component
                    components.insert(rest[..slash_idx].to_string());
                } else {
                    // It's a file component
                    if !rest.is_empty() {
                        components.insert(rest.to_string());
                    }
                }
            }
        }

        super::write_readdir_entries(components.iter().map(|s| s.as_str()), offset, buf)
    }
}

// ── Static file node ──────────────────────────────────────────────────────────

struct StaticFileNode {
    data: &'static [u8],
    ino: u64,
}

impl StaticFileNode {
    const fn new(data: &'static [u8], ino: u64) -> Self {
        Self { data, ino }
    }
}

impl VfsNode for StaticFileNode {
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
            mode: VfsStat::S_IFREG | 0o444,
            size: self.data.len() as u64,
            ino: self.ino,
            ..Default::default()
        })
    }
}
