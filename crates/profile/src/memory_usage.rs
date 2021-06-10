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
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.allocated)
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
                // Note: This is incredibly slow.
                let alloc = unsafe { libc::mallinfo() }.uordblks as isize;
                MemoryUsage { allocated: Bytes(alloc) }
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

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Bytes(isize);

impl Bytes {
    pub fn megabytes(self) -> isize {
        self.0 / 1024 / 1024
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
        f.pad(&format!("{}{}", value, suffix))
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
