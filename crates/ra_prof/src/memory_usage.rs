//! FIXME: write short doc here

use std::fmt;

pub struct MemoryUsage {
    pub allocated: Bytes,
    pub resident: Bytes,
}

impl MemoryUsage {
    #[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
    pub fn current() -> MemoryUsage {
        jemalloc_ctl::epoch::advance().unwrap();
        MemoryUsage {
            allocated: Bytes(jemalloc_ctl::stats::allocated::read().unwrap()),
            resident: Bytes(jemalloc_ctl::stats::resident::read().unwrap()),
        }
    }

    #[cfg(any(not(feature = "jemalloc"), target_env = "msvc"))]
    pub fn current() -> MemoryUsage {
        MemoryUsage { allocated: Bytes(0), resident: Bytes(0) }
    }
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{} allocated {} resident", self.allocated, self.resident,)
    }
}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Bytes(usize);

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
