//! FIXME: write short doc here
use std::fmt;

use cfg_if::cfg_if;

pub struct MemoryUsage {
    pub allocated: Bytes,
    pub resident: Bytes,
}

impl MemoryUsage {
    pub fn current() -> MemoryUsage {
        cfg_if! {
            if #[cfg(target_os = "linux")] {
                // Note: This is incredibly slow.
                let alloc = unsafe { libc::mallinfo() }.uordblks as u32 as usize;
                MemoryUsage { allocated: Bytes(alloc), resident: Bytes(0) }
            } else {
                MemoryUsage { allocated: Bytes(0), resident: Bytes(0) }
            }
        }
    }
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{} allocated {} resident", self.allocated, self.resident,)
    }
}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Bytes(usize);

impl Bytes {
    pub fn megabytes(self) -> usize {
        self.0 / 1024 / 1024
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes = self.0;
        let mut value = bytes;
        let mut suffix = "b";
        if value > 4096 {
            value /= 1024;
            suffix = "kb";
            if value > 4096 {
                value /= 1024;
                suffix = "mb";
            }
        }
        f.pad(&format!("{}{}", value, suffix))
    }
}

impl std::ops::AddAssign<usize> for Bytes {
    fn add_assign(&mut self, x: usize) {
        self.0 += x;
    }
}

impl std::ops::Sub for Bytes {
    type Output = Bytes;
    fn sub(self, rhs: Bytes) -> Bytes {
        Bytes(self.0 - rhs.0)
    }
}
