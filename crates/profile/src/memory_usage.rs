//! Like [`std::time::Instant`], but for memory.
//!
//! Measures the total size of all currently allocated objects.
use std::fmt;

use cfg_if::cfg_if;

#[derive(Copy, Clone)]
pub struct MemoryUsage {
    pub allocated: Bytes,
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.allocated.fmt(f)
    }
}

impl std::ops::Sub for MemoryUsage {
    type Output = MemoryUsage;
    fn sub(self, rhs: MemoryUsage) -> MemoryUsage {
        MemoryUsage { allocated: self.allocated - rhs.allocated }
    }
}

impl MemoryUsage {
    pub fn now() -> MemoryUsage {
        cfg_if! {
            if #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))] {
                jemalloc_ctl::epoch::advance().unwrap();
                MemoryUsage {
                    allocated: Bytes(jemalloc_ctl::stats::allocated::read().unwrap() as isize),
                }
            } else if #[cfg(all(target_os = "linux", target_env = "gnu"))] {
                memusage_linux()
            } else if #[cfg(windows)] {
                // There doesn't seem to be an API for determining heap usage, so we try to
                // approximate that by using the Commit Charge value.

                use winapi::um::processthreadsapi::*;
                use winapi::um::psapi::*;
                use std::mem::{MaybeUninit, size_of};

                let proc = unsafe { GetCurrentProcess() };
                let mut mem_counters = MaybeUninit::uninit();
                let cb = size_of::<PROCESS_MEMORY_COUNTERS>();
                let ret = unsafe { GetProcessMemoryInfo(proc, mem_counters.as_mut_ptr(), cb as u32) };
                assert!(ret != 0);

                let usage = unsafe { mem_counters.assume_init().PagefileUsage };
                MemoryUsage { allocated: Bytes(usage as isize) }
            } else {
                MemoryUsage { allocated: Bytes(0) }
            }
        }
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu", not(feature = "jemalloc")))]
fn memusage_linux() -> MemoryUsage {
    // Linux/glibc has 2 APIs for allocator introspection that we can use: mallinfo and mallinfo2.
    // mallinfo uses `int` fields and cannot handle memory usage exceeding 2 GB.
    // mallinfo2 is very recent, so its presence needs to be detected at runtime.
    // Both are abysmally slow.

    use std::ffi::CStr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static MALLINFO2: AtomicUsize = AtomicUsize::new(1);

    let mut mallinfo2 = MALLINFO2.load(Ordering::Relaxed);
    if mallinfo2 == 1 {
        let cstr = CStr::from_bytes_with_nul(b"mallinfo2\0").unwrap();
        mallinfo2 = unsafe { libc::dlsym(libc::RTLD_DEFAULT, cstr.as_ptr()) } as usize;
        // NB: races don't matter here, since they'll always store the same value
        MALLINFO2.store(mallinfo2, Ordering::Relaxed);
    }

    if mallinfo2 == 0 {
        // mallinfo2 does not exist, use mallinfo.
        let alloc = unsafe { libc::mallinfo() }.uordblks as isize;
        MemoryUsage { allocated: Bytes(alloc) }
    } else {
        let mallinfo2: fn() -> libc::mallinfo2 = unsafe { std::mem::transmute(mallinfo2) };
        let alloc = mallinfo2().uordblks as isize;
        MemoryUsage { allocated: Bytes(alloc) }
    }
}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Bytes(isize);

impl Bytes {
    pub fn new(bytes: isize) -> Bytes {
        Bytes(bytes)
    }
}

impl Bytes {
    pub fn megabytes(self) -> isize {
        self.0 / 1024 / 1024
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.0;
        let mut value = bytes;
        let mut suffix = "b";
        if value.abs() > 4096 {
            value /= 1024;
            suffix = "kb";
            if value.abs() > 4096 {
                value /= 1024;
                suffix = "mb";
            }
        }
        f.pad(&format!("{value}{suffix}"))
    }
}

impl std::ops::AddAssign<usize> for Bytes {
    fn add_assign(&mut self, x: usize) {
        self.0 += x as isize;
    }
}

impl std::ops::Sub for Bytes {
    type Output = Bytes;
    fn sub(self, rhs: Bytes) -> Bytes {
        Bytes(self.0 - rhs.0)
    }
}
