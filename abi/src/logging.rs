//! Unified Logging ABI
//!
//! Defines the canonical log record format used throughout the system.

/// Log severity levels
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Level::Error),
            2 => Some(Level::Warn),
            3 => Some(Level::Info),
            4 => Some(Level::Debug),
            5 => Some(Level::Trace),
            _ => None,
        }
    }
}

/// Special value indicating kernel context (no userspace pid)
pub const PID_KERNEL: u64 = u64::MAX;

/// Canonical log record with all required fields.
/// This struct defines the wire format for structured logging.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LogRecord {
    /// Monotonic timestamp (ticks from boot)
    pub ts: u64,
    /// Global sequence number for ordering
    pub seq: u64,
    /// Log level
    pub level: u8,
    /// CPU id (0 for single-core)
    pub cpu: u8,
    /// Padding
    pub _pad: [u8; 6],
    /// Kernel thread id
    pub tid: u64,
    /// Userspace process id (PID_KERNEL if kernel-only)
    pub pid: u64,
    /// Span correlation id (0 = no span)
    pub span: u64,
}

impl Default for LogRecord {
    fn default() -> Self {
        Self {
            ts: 0,
            seq: 0,
            level: Level::Info as u8,
            cpu: 0,
            _pad: [0; 6],
            tid: 0,
            pid: PID_KERNEL,
            span: 0,
        }
    }
}

/// Span identifier for correlating related log entries
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanId(pub u64);

impl SpanId {
    pub const NONE: SpanId = SpanId(0);

    pub fn is_none(&self) -> bool {
        self.0 == 0
    }
}

/// User log request (sent via syscall)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UserLogRequest {
    pub level: u32,
    pub line: u32,
    pub msg_ptr: u64,
    pub msg_len: u64,
    pub file_ptr: u64,
    pub file_len: u64,
    pub mod_ptr: u64,
    pub mod_len: u64,
}
