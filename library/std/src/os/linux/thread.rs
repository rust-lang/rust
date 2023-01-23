//! Linux-specific extensions for working with the [`std::thread`] module.
//!
//! [`std::thread`]: crate::thread

#![unstable(feature = "thread_scheduling", issue = "none")]

use crate::fmt;

/// The relative scheduling priority of a thread, corresponding to the
/// `sched_priority` scheduling parameter.
///
/// Refer to the man page for [`pthread_attr_setschedparam(3)`] for more details.
///
/// [`pthread_attr_setschedparam(3)`]: https://man7.org/linux/man-pages/man3/pthread_attr_setschedparam.3.html
pub struct Priority(pub(crate) libc::c_int);

impl Priority {
    /// Create an integer priority.
    pub fn new(priority: i32) -> Self {
        Self(priority)
    }
}

impl crate::fmt::Debug for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Priority").finish_non_exhaustive()
    }
}

/// The CPU affinity mask of a thread, which determines what CPUs a thread is
/// eligible to run on.
///
/// Refer to the man page for [`pthread_attr_setaffinity_np(3)`] for more details.
///
/// [`pthread_attr_setaffinity_np(3)`]: https://man7.org/linux/man-pages/man3/pthread_attr_setaffinity_np.3.html
pub struct Affinity(pub(crate) libc::cpu_set_t);

impl Affinity {
    /// Create an affinity mask with no CPUs in it.
    /// See the man page entry for [`CPU_ZERO`] for more details.
    ///
    /// [`CPU_ZERO`]: https://man7.org/linux/man-pages/man3/CPU_SET.3.html
    pub fn new() -> Self {
        unsafe {
            let mut set = crate::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            Self(set)
        }
    }

    /// Add a CPU to the affinity mask.
    /// See the man page entry for [`CPU_SET`] for more details.
    ///
    /// [`CPU_SET`]: https://man7.org/linux/man-pages/man3/CPU_SET.3.html
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
