//! High-level VFS utilities.

use crate::syscall::{vfs_close, vfs_read, vfs_watch_path};
use abi::errors::{Errno, SysResult};
use abi::vfs_watch::{flags, mask, WatchEvent};
use alloc::vec;

/// A file system event watcher.
pub struct Watcher {
    fd: u32,
}

impl Watcher {
    /// Create a new watcher for the given path.
    pub fn from_path(path: &str, mask: u32, flags: u32) -> SysResult<Self> {
        let fd = vfs_watch_path(path, mask, flags)?;
        Ok(Self { fd })
    }

    /// Read the next event from the watch stream.
    ///
    /// If no events are available and the watch was not opened with `NONBLOCK`,
    /// this will block until an event occurs.
    pub fn read_event(&self) -> SysResult<Option<(WatchEvent, alloc::string::String)>> {
        let mut buf = [0u8; 512];
        let n = match vfs_read(self.fd, &mut buf) {
            Ok(n) if n >= core::mem::size_of::<WatchEvent>() => n,
            Ok(_) => return Ok(None),
            Err(Errno::EAGAIN) => return Ok(None),
            Err(e) => return Err(e),
        };

        let event = unsafe { *(buf.as_ptr() as *const WatchEvent) };
        let name_start = core::mem::size_of::<WatchEvent>();
        let name_end = name_start + event.name_len as usize;

        let name = if event.name_len > 0 && name_end <= n {
            alloc::string::String::from_utf8_lossy(&buf[name_start..name_end]).into_owned()
        } else {
            alloc::string::String::new()
        };

        Ok(Some((event, name)))
    }

    /// Returns the raw file descriptor.
    pub fn fd(&self) -> u32 {
        self.fd
    }
}

impl Drop for Watcher {
    fn drop(&mut self) {
        let _ = vfs_close(self.fd);
    }
}

/// Block until a file at `path` exists.
pub fn wait_until_exists(path: &str) -> SysResult<()> {
    // Check if it already exists
    if crate::syscall::vfs_open(path, abi::syscall::vfs_flags::O_RDONLY).is_ok() {
        return Ok(());
    }

    // Identify parent directory
    let (parent, name) = if let Some(idx) = path.rfind('/') {
        if idx == 0 {
            ("/", &path[1..])
        } else {
            (&path[..idx], &path[idx + 1..])
        }
    } else {
        (".", path)
    };

    // Watch parent for creation
    let watcher = Watcher::from_path(parent, mask::CREATE | mask::MOVE_TO, 0)?;

    // Check again to avoid race
    if crate::syscall::vfs_open(path, abi::syscall::vfs_flags::O_RDONLY).is_ok() {
        return Ok(());
    }

    loop {
        if let Some((event, event_name)) = watcher.read_event()? {
            if event_name == name {
                return Ok(());
            }
        }
    }
}
