//! Shared types used in syscall payloads.
//! Must be #[repr(C)] to ensure stable layout.

use crate::{BlobId, SymbolId};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TimeSpec {
    pub seconds: u64,
    pub nanoseconds: u32,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ThingId(pub u64);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamStatus {
    pub readable: bool,
    pub writable: bool,
    pub closed: bool,
    pub error: bool,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct StreamId(pub u64);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct WatchId(pub u64);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EventHeader {
    pub type_: u32,
    pub size: u32,
    pub stream_id: StreamId,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RootWatchEvent {
    pub target: u64,
    pub key: u64,
    pub value: u64,
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Unknown = 0,
    Runnable = 1,
    Running = 2,
    Blocked = 3,
    Dead = 4,
}

/// Flags for the `SYS_WAITPID` syscall.
pub mod waitpid_flags {
    /// Do not block; return immediately if no child has exited yet.
    pub const WNOHANG: u32 = 1;
    /// Report stopped children in addition to exited/signalled children.
    pub const WUNTRACED: u32 = 1 << 1;
    /// Report children resumed by `SIGCONT`.
    pub const WCONTINUED: u32 = 1 << 2;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StackInfo {
    pub guard_start: usize,
    pub guard_end: usize,
    pub reserve_start: usize,
    pub reserve_end: usize,
    pub committed_start: usize,
    pub grow_chunk_bytes: usize,
}

/// Flags for [`SpawnThreadReq`].
pub mod spawn_thread_flags {
    /// The new thread is detached: it cannot be joined and its resources are
    /// reclaimed automatically when it exits.  Passing this flag and then
    /// calling `SYS_TASK_WAIT` on the returned TID returns `EINVAL`.
    pub const DETACHED: u32 = 1;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnThreadReq {
    pub entry: usize,
    pub sp: usize,
    pub arg: usize,
    pub stack: StackInfo,
    /// Initial user-mode TLS base for the new thread (e.g. FS_BASE on x86_64).
    ///
    /// When non-zero the kernel writes this value into the new thread's
    /// hardware TLS register before the thread is first scheduled.  Passing
    /// zero leaves the TLS register in its architecture-defined initial state
    /// (typically zero as well).
    pub tls_base: usize,
    /// Spawn flags — see [`spawn_thread_flags`].
    pub flags: u32,
    pub _pad: u32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchMode {
    QueryThenStream = 0,
    StreamOnly = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct WatchSpec {
    pub query_ptr: u64,
    pub query_len: u64,
    pub mode: u32,
    pub _padding: u32, // Alignment padding
    pub start_seq: u64,
    /// Pointer to RootWatchFilter (0 = no filter, match all)
    pub filter_ptr: u64,
    /// Size of filter struct (for versioning, should be RootWatchFilter::SIZE)
    pub filter_len: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct WatchEvent {
    pub kind: u32, // 1=Found, 2=Lost
    pub node_id: u64,
    pub handle: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphEdge {
    pub rel: u64,
    pub target: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphProp {
    pub key: u32,
    pub _pad: u32,
    pub value: u64,
}

// ============================================================================
// Watch Semantics
// ============================================================================

/// Start from next commit only (no history replay)
/// Use this to ignore historical events and only see future mutations.
pub const WATCH_START_LATEST: u64 = u64::MAX;

impl WatchMode {
    /// Convert raw u32 to WatchMode, returning None for unknown values.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::QueryThenStream),
            1 => Some(Self::StreamOnly),
            _ => None,
        }
    }
}

// ============================================================================
// Bulk Property Fetch (Performance Optimization)
// ============================================================================

/// Maximum number of properties that can be fetched in one bulk call
pub const BULK_PROPS_MAX_KEYS: usize = 32;

/// Request for bulk property fetch
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BulkPropsRequest {
    /// Node ID to query
    pub node_id: u64,
    /// Array of property key IDs (pre-interned SymbolIds)
    pub keys: [u32; BULK_PROPS_MAX_KEYS],
    /// Number of valid keys in the array
    pub key_count: u8,
    pub _pad: [u8; 3],
}

impl Default for BulkPropsRequest {
    fn default() -> Self {
        Self {
            node_id: 0,
            keys: [0; BULK_PROPS_MAX_KEYS],
            key_count: 0,
            _pad: [0; 3],
        }
    }
}

/// Response from bulk property fetch
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BulkPropsResponse {
    /// Node ID that was queried
    pub node_id: u64,
    /// Property values (parallel to request keys array)
    pub values: [u64; BULK_PROPS_MAX_KEYS],
    /// Bitmask indicating which keys had values (bit N = key N present)
    pub present_mask: u32,
    pub _pad: u32,
}

impl Default for BulkPropsResponse {
    fn default() -> Self {
        Self {
            node_id: 0,
            values: [0; BULK_PROPS_MAX_KEYS],
            present_mask: 0,
            _pad: 0,
        }
    }
}

// ============================================================================
// Enhanced Process Spawn (SYS_SPAWN_PROCESS_EX)
// ============================================================================

/// Mapping from a parent file descriptor to a destination file descriptor in the child.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FdRemap {
    pub src_fd: u32,
    pub dst_fd: u32,
}

/// Stdio mode for child streams in SpawnProcessExReq.
pub mod stdio_mode {
    /// Inherit the parent's handle for this stream.
    pub const INHERIT: u32 = 0;
    /// Attach to a null sink/source (discard output, empty input).
    pub const NULL: u32 = 1;
    /// Create a kernel pipe; parent gets the opposite end.
    pub const PIPE: u32 = 2;

    const FD_BIT: u32 = 1 << 31;

    /// Use an explicitly inherited parent file descriptor for this stream.
    #[inline]
    pub const fn fd(fd: u32) -> u32 {
        FD_BIT | fd
    }

    /// Decode an explicit parent file descriptor, if `mode` encodes one.
    #[inline]
    pub const fn explicit_fd(mode: u32) -> Option<u32> {
        if (mode & FD_BIT) != 0 {
            Some(mode & !FD_BIT)
        } else {
            None
        }
    }
}

/// Request payload for SYS_SPAWN_PROCESS_EX.
///
/// Passed by pointer from userspace.  All pointer/length fields reference
/// userspace memory and are copied-in by the kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnProcessExReq {
    /// Pointer to program name bytes (NOT null-terminated).
    pub name_ptr: u64,
    /// Length of program name in bytes.
    pub name_len: u32,
    pub _pad0: u32,
    /// Pointer to serialized argv blob.
    /// Format: count:u32, then for each arg: len:u32, bytes...
    pub argv_ptr: u64,
    /// Length of argv blob in bytes (0 = no argv override).
    pub argv_len: u32,
    pub _pad1: u32,
    /// Pointer to serialized env blob.
    /// Format: count:u32, then for each: klen:u32, key, vlen:u32, val
    pub env_ptr: u64,
    /// Length of env blob in bytes (0 = no env override).
    pub env_len: u32,
    pub _pad2: u32,
    /// Stdio modes (see `stdio_mode`).
    pub stdin_mode: u32,
    pub stdout_mode: u32,
    pub stderr_mode: u32,
    pub _reserved: u32,
    pub boot_arg: u64,
    /// Explicit handles to inherit from parent.
    /// The kernel will clone these into the child's handle table.
    pub handles_to_inherit: [u64; 8],
    /// Number of handles in the array to inherit.
    pub num_inherited_handles: u32,
    pub _pad3: u32,
    /// Pointer to the desired working directory bytes (NOT null-terminated).
    /// Set to 0 to inherit the parent's cwd.
    pub cwd_ptr: u64,
    pub cwd_len: u32,
    pub _pad4: u32,
    /// Pointer to an array of [`FdRemap`] entries.
    pub fd_remap_ptr: u64,
    /// Number of entries in the `fd_remap_ptr` array.
    pub fd_remap_len: u32,
    pub _pad5: u32,
}

/// Response payload for SYS_SPAWN_PROCESS_EX.
///
/// Written by the kernel into userspace via `resp_ptr`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnProcessExResp {
    /// Child task/process ID.
    pub child_tid: u64,
    /// Child process ID (for waitpid).
    pub child_pid: u32,
    pub _pad: u32,
    /// Parent's fd for piped stdin (parent writes to this fd); 0 = not piped.
    pub stdin_pipe: u64,
    /// Parent's fd for piped stdout (parent reads from this fd); 0 = not piped.
    pub stdout_pipe: u64,
    /// Parent's fd for piped stderr (parent reads from this fd); 0 = not piped.
    pub stderr_pipe: u64,
}
