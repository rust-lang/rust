//! Kernel-native VFS watch system.
//!
//! Implements Act I of the watch migration: replacing Root-service watches
//! with a first-class kernel mechanism integrated into the VFS.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use super::{VfsNode, VfsStat};
use abi::errors::{Errno, SysResult};
use abi::vfs_watch::{self, WatchEvent};

/// A ring buffer of VFS events.
pub struct EventQueue {
    /// Queue of (header, optional_name) pairs.
    events: Mutex<VecDeque<(WatchEvent, Option<String>)>>,
    max_size: usize,
    /// Tasks waiting for events.
    waiters: Mutex<Vec<u64>>,
}

impl EventQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            events: Mutex::new(VecDeque::new()),
            max_size,
            waiters: Mutex::new(Vec::new()),
        }
    }

    pub fn push(&self, mut event: WatchEvent, name: Option<&str>) {
        let mut lock = self.events.lock();
        if lock.len() >= self.max_size {
            // Check if last event was an overflow to avoid flooding
            if let Some((last, _)) = lock.back() {
                if last.mask & vfs_watch::mask::OVERFLOW != 0 {
                    return;
                }
            }
            // Push overflow event
            event.mask = vfs_watch::mask::OVERFLOW;
            lock.push_back((event, None));
            return;
        }
        lock.push_back((event, name.map(|s| s.into())));

        // Wake up waiters
        let waiters = self.waiters.lock();
        for &tid in waiters.iter() {
            unsafe {
                crate::sched::wake_task_erased(tid);
            }
        }
    }

    pub fn pop(&self) -> Option<(WatchEvent, Option<String>)> {
        self.events.lock().pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }

    pub fn add_waiter(&self, tid: u64) {
        let mut lock = self.waiters.lock();
        if !lock.contains(&tid) {
            lock.push(tid);
        }
    }

    pub fn remove_waiter(&self, tid: u64) {
        let mut lock = self.waiters.lock();
        lock.retain(|&t| t != tid);
    }
}

/// A kernel watch object.
///
/// Implements [`VfsNode`] so it can be used as a file descriptor.
pub struct Watch {
    pub queue: Arc<EventQueue>,
    pub mask: u32,
    pub flags: u32,
}

impl Watch {
    pub fn new(mask: u32, flags: u32) -> Self {
        Self {
            queue: Arc::new(EventQueue::new(1024)),
            mask,
            flags,
        }
    }
}

impl VfsNode for Watch {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let (event, name) = match self.queue.pop() {
            Some(e) => e,
            None => {
                if self.flags & vfs_watch::flags::NONBLOCK != 0 {
                    return Err(Errno::EAGAIN);
                }
                return Err(Errno::EAGAIN); // Should block in a real impl, but wait_many handles it
            }
        };

        let mut event = event;
        let name_bytes = name.as_ref().map(|s| s.as_bytes()).unwrap_or(&[]);
        event.name_len = name_bytes.len() as u16;

        let header_size = core::mem::size_of::<WatchEvent>();
        let total_size = header_size + name_bytes.len();

        if buf.len() < total_size {
            // TODO: Re-push or handle partial reads? POSIX inotify returns EINVAL if buffer is too small for one event.
            return Err(Errno::EINVAL);
        }

        // Copy header
        unsafe {
            let ptr = &event as *const _ as *const u8;
            buf[..header_size].copy_from_slice(core::slice::from_raw_parts(ptr, header_size));
        }

        // Copy name
        if !name_bytes.is_empty() {
            buf[header_size..total_size].copy_from_slice(name_bytes);
        }

        Ok(total_size)
    }

    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EBADF)
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 0,
            ..Default::default()
        })
    }

    fn poll(&self) -> u16 {
        use abi::syscall::poll_flags::*;
        if self.queue.is_empty() { 0 } else { POLLIN }
    }

    fn add_waiter(&self, tid: u64) {
        self.queue.add_waiter(tid);
    }

    fn remove_waiter(&self, tid: u64) {
        self.queue.remove_waiter(tid);
    }
}

// ── Registry ────────────────────────────────────────────────────────────────

struct WatchRecord {
    watch: Arc<Watch>,
    target_ino: u64,
    // TODO: add mount_id or similar to distinguish same inos on different mounts
}

static REGISTRY: Mutex<Vec<WatchRecord>> = Mutex::new(Vec::new());

pub fn register_watch(node: &Arc<dyn VfsNode>, watch: Arc<Watch>) -> SysResult<()> {
    let stat = node.stat()?;
    let mut lock = REGISTRY.lock();
    lock.push(WatchRecord {
        watch,
        target_ino: stat.ino,
    });
    Ok(())
}

pub fn emit_event(node: &dyn VfsNode, mask: u32, name: Option<&str>, cookie: u32) {
    let stat = match node.stat() {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut lock = REGISTRY.lock();
    // Use a temporary list to avoid holding the lock while calling watch.queue.push (which might wake tasks)
    // Actually, EventQueue::push uses its own lock, so it's fine.
    for record in lock.iter() {
        if record.target_ino == stat.ino {
            if record.watch.mask & mask != 0 {
                let mut event = WatchEvent::default();
                event.mask = mask;
                event.cookie = cookie;
                event.subject_id = stat.ino; // TODO: use a more stable ID?
                if stat.is_dir() {
                    event.flags |= vfs_watch::event_flags::IS_DIR;
                }
                record.watch.queue.push(event, name);
            }
        }
    }
}
