//! Unified wait-many ABI.

/// Maximum number of wait specs/results accepted by the kernel in one call.
pub const WAIT_MANY_MAX_ITEMS: usize = 32;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitKind {
    /// Wait for a message-passing port to become readable or writable.
    Port = 1,
    /// Legacy graph-watch kind. Returns `ENOSYS`; use `WaitKind::Fd` instead.
    #[deprecated(note = "Graph watches are removed; open an FD and use WaitKind::Fd")]
    RootWatch = 2,
    /// Wait for a task (thread) to exit.
    TaskExit = 3,
    /// Wait for an interrupt to fire.
    Irq = 4,
    /// Internal: the `wait_many` global timeout expired.
    Timeout = 5,
    /// Legacy async graph-op kind. Returns `ENOSYS`; use `WaitKind::Fd` instead.
    #[deprecated(note = "Graph ops are removed; use file-descriptor–based I/O instead")]
    GraphOp = 6,
    /// Wait for a VFS file descriptor to become readable or writable.
    ///
    /// This is the primary readiness kind for all VFS-backed resources:
    /// pipes, sockets, channels bridged via `SYS_FS_FD_FROM_HANDLE`, and
    /// device nodes.
    Fd = 7,
}

impl WaitKind {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Port),
            #[allow(deprecated)]
            2 => Some(Self::RootWatch),
            3 => Some(Self::TaskExit),
            4 => Some(Self::Irq),
            5 => Some(Self::Timeout),
            #[allow(deprecated)]
            6 => Some(Self::GraphOp),
            7 => Some(Self::Fd),
            _ => None,
        }
    }
}

pub mod interest {
    pub const READABLE: u32 = 1 << 0;
    pub const WRITABLE: u32 = 1 << 1;
}

pub mod ready {
    pub const READABLE: u32 = 1 << 0;
    pub const WRITABLE: u32 = 1 << 1;
    pub const HANGUP: u32 = 1 << 2;
    pub const EXITED: u32 = 1 << 3;
    pub const TIMEOUT: u32 = 1 << 4;
    pub const OVERFLOW: u32 = 1 << 5;
    pub const ERROR: u32 = 1 << 6;
    pub const IRQ: u32 = 1 << 7;
    pub const DONE: u32 = 1 << 8;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct WaitSpec {
    pub kind: u32,
    pub flags: u32,
    pub object: u64,
    pub token: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct WaitResult {
    pub kind: u32,
    pub flags: u32,
    pub object: u64,
    pub token: u64,
    /// Optional payload:
    /// - port/watch/irq: bytes/pending count when known
    /// - task exit: exit code
    /// - error: Errno value
    pub value: i64,
    pub reserved: u64,
}
