#![unstable(feature = "thread_scheduling", issue = "none")]

use crate::fmt;

pub struct Priority(pub(crate) libc::c_int);

impl Priority {
    fn new(priority: i32) -> Self {
        Self(priority)
    }
}

impl crate::fmt::Debug for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Priority").finish_non_exhaustive()
    }
}

pub struct Affinity(pub(crate) libc::cpu_set_t);

impl Affinity {
    pub fn new() -> Self {
        unsafe {
            let mut set = crate::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            Self(set)
        }
    }

    pub fn set(&mut self, cpu: usize) {
        unsafe {
            libc::CPU_SET(cpu, &mut self.0);
        }
    }
}

impl crate::fmt::Debug for Affinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affinity").finish_non_exhaustive()
    }
}
