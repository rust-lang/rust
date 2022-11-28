use super::imp;

/// The relative priority of an thread. The representation of this priority is
/// platform-dependent and setting priority may not be supported on all platforms.
///
/// To set a thread's priority on supported platforms, use helpers in [`std::os`]
/// (e.g. [`os::linux::thread::Priority`]) to create a `thread::Priority` and
/// [`thread::Builder::priority`] to set the thread's priority.
///
/// [`std::os`]: crate::os
/// [`os::linux::thread::Priority`]: crate::os::linux::thread::Priority
/// [`thread::Builder::priority`]: crate::thread::Builder::priority
#[derive(Debug)]
pub struct Priority(imp::Priority);

impl From<imp::Priority> for Priority {
    fn from(priority: imp::Priority) -> Self {
        Self(priority)
    }
}

impl From<Priority> for imp::Priority {
    fn from(priority: Priority) -> Self {
        priority.0
    }
}

/// The affinity of an thread, i.e. which CPU(s) a thread may run on. The
/// meaning and representation of a thread's affinity is platform-dependent
/// and setting affinity may not be supported on all platforms.
///
/// To set a thread's affinity on supported platforms, use helpers from [`std::os`]
/// (e.g. [`os::linux::thread::Affinity`]) to create a `thread::Affinity` and
/// [`thread::Builder::affinity`] to set the thread's affinity.
///
/// [`std::os`]: crate::os
/// [`os::linux::thread::Affinity`]: crate::os::linux::thread::Affinity
/// [`thread::Builder::affinity`]: crate::thread::Builder::affinity
#[derive(Debug)]
pub struct Affinity(pub(crate) imp::Affinity);

impl From<imp::Affinity> for Affinity {
    fn from(affinity: imp::Affinity) -> Self {
        Self(affinity)
    }
}

impl From<Affinity> for imp::Affinity {
    fn from(affinity: Affinity) -> Self {
        affinity.0
    }
}
