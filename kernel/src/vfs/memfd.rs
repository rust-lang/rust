use super::{SysResult, VfsNode, VfsStat};
use abi::errors::Errno;
use alloc::sync::Arc;
use spin::Mutex;

/// A memory-backed file descriptor (MemFD).
/// Allocates a contiguous physical memory region at creation.
pub struct MemFdNode {
    inner: Arc<Mutex<MemFdInner>>,
}

struct MemFdInner {
    phys_base: u64,
    size: usize,
    name: alloc::string::String,
    /// Creation / initial access time.
    atime: (u64, u32),
    /// Last modification time (write).
    mtime: (u64, u32),
    /// Last status-change time.
    ctime: (u64, u32),
}

impl MemFdNode {
    pub fn new(name: &str, size: usize) -> SysResult<Self> {
        let page_size = 4096usize;
        let aligned_size = (size + page_size - 1) & !(page_size - 1);
        let page_count = aligned_size / page_size;

        // Allocate contiguous physical memory
        // ACT III: Using the system's contiguous allocator
        let phys_base = crate::memory::alloc_contiguous_frames(page_count).ok_or(Errno::ENOMEM)?;

        // Zero the memory
        let hhdm = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
        unsafe {
            core::ptr::write_bytes((phys_base + hhdm) as *mut u8, 0, aligned_size);
        }

        let ts = crate::time::now_timespec();
        Ok(Self {
            inner: Arc::new(Mutex::new(MemFdInner {
                phys_base,
                size: aligned_size,
                name: alloc::string::String::from(name),
                atime: ts,
                mtime: ts,
                ctime: ts,
            })),
        })
    }
}

impl VfsNode for MemFdNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let mut inner = self.inner.lock();
        let off = offset as usize;
        if off >= inner.size {
            return Ok(0);
        }
        let avail = inner.size - off;
        let n = buf.len().min(avail);

        let hhdm = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
        let src = unsafe {
            core::slice::from_raw_parts((inner.phys_base + hhdm + offset) as *const u8, n)
        };
        buf[..n].copy_from_slice(src);
        if n > 0 {
            inner.atime = crate::time::now_timespec();
        }
        Ok(n)
    }

    fn write(&self, offset: u64, buf: &[u8]) -> SysResult<usize> {
        let mut inner = self.inner.lock();
        let off = offset as usize;
        if off >= inner.size {
            return Err(Errno::ENOSPC);
        }
        let avail = inner.size - off;
        let n = buf.len().min(avail);

        let hhdm = crate::boot_info::get().map(|i| i.hhdm_offset).unwrap_or(0);
        let dst = unsafe {
            core::slice::from_raw_parts_mut((inner.phys_base + hhdm + offset) as *mut u8, n)
        };
        dst[..n].copy_from_slice(&buf[..n]);
        let ts = crate::time::now_timespec();
        inner.mtime = ts;
        inner.ctime = ts;
        Ok(n)
    }

    fn stat(&self) -> SysResult<VfsStat> {
        let inner = self.inner.lock();
        Ok(VfsStat {
            mode: VfsStat::S_IFREG | 0o666,
            size: inner.size as u64,
            ino: inner.phys_base, // Use phys_base as unique ino for now
            nlink: 1,
            atime_sec: inner.atime.0,
            atime_nsec: inner.atime.1,
            mtime_sec: inner.mtime.0,
            mtime_nsec: inner.mtime.1,
            ctime_sec: inner.ctime.0,
            ctime_nsec: inner.ctime.1,
            ..Default::default()
        })
    }

    fn phys_region(&self) -> SysResult<(u64, usize)> {
        let inner = self.inner.lock();
        Ok((inner.phys_base, inner.size))
    }
}

impl Drop for MemFdInner {
    fn drop(&mut self) {
        let page_count = self.size / 4096;
        unsafe {
            crate::memory::free_contiguous_frames(self.phys_base, page_count);
        }
    }
}
