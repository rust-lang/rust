use std::fmt;

pub struct MemoryUsage {
    pub allocated: Bytes,
    pub resident: Bytes,
}

impl MemoryUsage {
    #[cfg(feature = "jemalloc")]
    pub fn current() -> MemoryUsage {
        jemalloc_ctl::epoch().unwrap();
        MemoryUsage {
            allocated: Bytes(jemalloc_ctl::stats::allocated().unwrap()),
            resident: Bytes(jemalloc_ctl::stats::resident().unwrap()),
        }
    }

    #[cfg(not(feature = "jemalloc"))]
    pub fn current() -> MemoryUsage {
        MemoryUsage { allocated: Bytes(0), resident: Bytes(0) }
    }
}

impl fmt::Display for MemoryUsage {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{} allocated {} resident", self.allocated, self.resident,)
    }
}

#[derive(Default)]
pub struct Bytes(usize);

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes = self.0;
        if bytes < 4096 {
            return write!(f, "{} bytes", bytes);
        }
        let kb = bytes / 1024;
        if kb < 4096 {
            return write!(f, "{}kb", kb);
        }
        let mb = kb / 1024;
        write!(f, "{}mb", mb)
    }
}

impl std::ops::AddAssign<usize> for Bytes {
    fn add_assign(&mut self, x: usize) {
        self.0 += x;
    }
}
