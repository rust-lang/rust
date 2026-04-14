// ThingOS Syscall Numbers
//
// One single source of truth for syscall numbers.
// This file is explicitly free of any dependencies (like serde) so it can
// be shared between the kernel, the ABI, and the Platform Abstraction Layer (PAL).

// ============================================================================
// Process & Thread Management (0x1000)
// ============================================================================
pub const SYS_EXIT: u32 = 0x1000;
pub const SYS_GET_TID: u32 = 0x1001;
pub const SYS_GETPID: u32 = 0x1002;
pub const SYS_GETPPID: u32 = 0x1003;
pub const SYS_SPAWN_THREAD: u32 = 0x1004;
pub const SYS_SPAWN_PROCESS: u32 = 0x1005;
pub const SYS_SPAWN_PROCESS_EX: u32 = 0x1006;
pub const SYS_TASK_WAIT: u32 = 0x1007;
pub const SYS_TASK_KILL: u32 = 0x1008;
pub const SYS_TASK_DUMP: u32 = 0x1009;
pub const SYS_TASK_POLL: u32 = 0x100A;
pub const SYS_YIELD: u32 = 0x100B;
pub const SYS_SET_PRIORITY: u32 = 0x100C;
pub const SYS_TASK_EXEC: u32 = 0x100D;
pub const SYS_TASK_SET_TLS_BASE: u32 = 0x100E;
pub const SYS_TASK_GET_TLS_BASE: u32 = 0x100F;
pub const SYS_TASK_INTERRUPT: u32 = 0x1010;
/// Wait for a child process to exit and retrieve its exit status.
/// Supports `WNOHANG` for non-blocking polling.  Analogous to POSIX `waitpid`.
pub const SYS_WAITPID: u32 = 0x1011;
/// Return the current thread's effective CPU parallelism.
///
/// This is affinity-aware: pinned threads report 1, while unpinned threads
/// report the current online CPU count.
pub const SYS_AVAILABLE_PARALLELISM: u32 = 0x1012;

// ── Signal Management (0x1013–0x101F) ────────────────────────────────────────
/// Send a signal to a process. Analogous to POSIX `kill(2)`.
/// Args: pid (i64 as usize), signum (u8 as usize)
pub const SYS_KILL: u32 = 0x1013;
/// Send a signal to the calling process. Analogous to POSIX `raise(3)`.
/// Args: signum (u8 as usize)
pub const SYS_RAISE: u32 = 0x1014;
/// Examine and change a per-signal action. Analogous to POSIX `sigaction(2)`.
/// Args: signum, act_ptr, oldact_ptr
pub const SYS_SIGACTION: u32 = 0x1015;
/// Examine and change the per-thread signal mask. Analogous to POSIX `sigprocmask(2)`.
/// Args: how, set_ptr, oldset_ptr
pub const SYS_SIGPROCMASK: u32 = 0x1016;
/// Return the set of pending blocked signals. Analogous to POSIX `sigpending(2)`.
/// Args: set_ptr
pub const SYS_SIGPENDING: u32 = 0x1017;
/// Suspend until a signal arrives, temporarily replacing the mask.
/// Analogous to POSIX `sigsuspend(2)`.
/// Args: mask_ptr
pub const SYS_SIGSUSPEND: u32 = 0x1018;
/// Return from a signal handler and restore the saved context.
/// Analogous to Linux `rt_sigreturn`.  No args (uses saved stack frame).
pub const SYS_SIGRETURN: u32 = 0x1019;
/// Set an alarm clock for delivery of SIGALRM. Analogous to POSIX `alarm(2)`.
/// Args: seconds (u64 as usize)
pub const SYS_ALARM: u32 = 0x101A;
/// Suspend until a signal whose disposition is not SIG_IGN arrives.
/// Analogous to POSIX `pause(2)`.  No args.
pub const SYS_PAUSE: u32 = 0x101B;
/// Set process group ID. Analogous to POSIX `setpgid(2)`.
/// Args: pid (i64 as usize), pgid (i64 as usize)
pub const SYS_SETPGID: u32 = 0x101C;
/// Get caller process group ID. Analogous to POSIX `getpgrp(2)`.
pub const SYS_GETPGRP: u32 = 0x101D;
/// Create a new session. Analogous to POSIX `setsid(2)`.
/// Args: none
pub const SYS_SETSID: u32 = 0x101E;
/// Set the calling thread's human-readable name.
/// Args: (name_ptr, name_len) — UTF-8 string, clamped to 31 bytes.
pub const SYS_TASK_SET_NAME: u32 = 0x101F;

// ============================================================================
// Process Environment (0x1100)
// ============================================================================
pub const SYS_ARGV_GET: u32 = 0x1100;
pub const SYS_ENV_GET: u32 = 0x1101;
pub const SYS_ENV_SET: u32 = 0x1102;
pub const SYS_ENV_UNSET: u32 = 0x1103;
pub const SYS_ENV_LIST: u32 = 0x1104;
pub const SYS_AUXV_GET: u32 = 0x1105;

// ============================================================================
// Time & Waiting (0x1200)
// ============================================================================
pub const SYS_SLEEP: u32 = 0x1200;
pub const SYS_SLEEP_NS: u32 = SYS_SLEEP;
pub const SYS_SLEEP_MS: u32 = 0x1201;
pub const SYS_TIME_MONOTONIC: u32 = 0x1202;
pub const SYS_TIME_NOW: u32 = 0x1203;
pub const SYS_TIME_ANCHOR: u32 = 0x1204;
pub const SYS_WAIT_MANY: u32 = 0x1205;

// ============================================================================
// Synchronization (0x1300)
// ============================================================================
pub const SYS_FUTEX_WAIT: u32 = 0x1300;
pub const SYS_FUTEX_WAKE: u32 = 0x1301;

// ============================================================================
// Basic I/O & Console (0x1400)
// ============================================================================
pub const SYS_READ: u32 = 0x1400;
pub const SYS_WRITE: u32 = 0x1401;
pub const SYS_DEBUG_WRITE: u32 = 0x1402;
pub const SYS_LOG_WRITE: u32 = 0x1403;
pub const SYS_TRACE_READ: u32 = 0x1404;
pub const SYS_CONSOLE_DISABLE: u32 = 0x1405;

// ============================================================================
// Memory & Virtual Mapping (0x2000)
// ============================================================================
pub const SYS_ALLOC_STACK: u32 = 0x2000;
pub const SYS_VM_MAP: u32 = 0x2001;
pub const SYS_VM_UNMAP: u32 = 0x2002;
pub const SYS_VM_PROTECT: u32 = 0x2003;
pub const SYS_VM_ADVISE: u32 = 0x2004;
pub const SYS_VM_QUERY: u32 = 0x2005;
pub const SYS_MEMFD_CREATE: u32 = 0x2006;
pub const SYS_MEMFD_PHYS: u32 = 0x2007;

// ============================================================================
// IPC (0x3000)
// ============================================================================
pub const SYS_PORT_CREATE: u32 = 0x3000;
pub const SYS_CHANNEL_CREATE: u32 = 0x3000;
pub const SYS_CHANNEL_SEND: u32 = 0x3001;
pub const SYS_CHANNEL_RECV: u32 = 0x3002;
pub const SYS_CHANNEL_TRY_RECV: u32 = 0x3003;
pub const SYS_CHANNEL_SEND_ALL: u32 = 0x3004;
/// **Deprecated** — prefer `SYS_CHANNEL_SEND_MSG` which bundles data and FDs atomically.
pub const SYS_CHANNEL_SEND_HANDLE: u32 = 0x3005;
/// **Deprecated** — prefer `SYS_CHANNEL_RECV_MSG` which receives data and FDs atomically.
pub const SYS_CHANNEL_RECV_HANDLE: u32 = 0x3006;
pub const SYS_CHANNEL_INFO: u32 = 0x3007;
pub const SYS_CHANNEL_CLOSE: u32 = 0x3008;
/// **Deprecated** — convert handles to FDs with `SYS_FD_FROM_HANDLE` and use `SYS_FS_POLL`.
pub const SYS_CHANNEL_WAIT: u32 = 0x3009;
/// Send a message with zero or more attached handles over a channel.
/// Args: (channel, data_ptr, data_len, handles_ptr, handles_count, 0)
pub const SYS_CHANNEL_SEND_MSG: u32 = 0x300A;
/// Receive a message with zero or more attached handles from a channel.
/// Args: (channel, data_ptr, data_cap, handles_ptr, handles_cap, out_lens_ptr)
/// out_lens_ptr → [usize; 2] = [actual_data_len, actual_handles_count]
pub const SYS_CHANNEL_RECV_MSG: u32 = 0x300B;
pub const SYS_PIPE: u32 = 0x3015;

// ── Unix Domain Sockets (0x3020) ─────────────────────────────────────────────
/// Create a Unix domain socket. Args: (domain, type, protocol) → fd
pub const SYS_SOCKET: u32 = 0x3020;
/// Bind a socket to a filesystem path. Args: (fd, path_ptr, path_len)
pub const SYS_BIND: u32 = 0x3021;
/// Mark a socket as listening for connections. Args: (fd, backlog)
pub const SYS_LISTEN: u32 = 0x3022;
/// Accept an incoming connection. Args: (fd) → new_fd
pub const SYS_ACCEPT: u32 = 0x3023;
/// Connect to a listening socket. Args: (fd, path_ptr, path_len)
pub const SYS_CONNECT: u32 = 0x3024;
/// Shut down part or all of a socket connection. Args: (fd, how)
pub const SYS_SHUTDOWN: u32 = 0x3025;
/// Create a connected socket pair. Args: (domain, type, protocol, fds_ptr) → 0
pub const SYS_SOCKETPAIR: u32 = 0x3026;

// ── Typed Message Delivery (0x3030) ──────────────────────────────────────────
//
// Prototype group-broadcast using the typed message delivery model.
// Broadcast is implemented as repeated per-recipient typed delivery;
// there is no separate metaphysical channel.
//
// Membership is snapshotted once at send time (snapshot semantics).
//
// Args layout shared by both syscalls:
//   arg0 – target: pid (direct) or pgid (broadcast)
//   arg1 – kind_id_ptr: *const u8  (16-byte KindId, user pointer)
//   arg2 – payload_ptr: *const u8  (arbitrary bytes, user pointer)
//   arg3 – payload_len: usize

/// Send one typed message directly to a process by PID.
///
/// Returns 0 on success, or an `Errno` error on failure:
/// - `ESRCH`  — recipient process not found
/// - `EAGAIN` — recipient inbox full
/// - `EINVAL` — invalid arguments / null pointer
pub const SYS_MSG_SEND: u32 = 0x3030;

/// Broadcast one typed message to all members of a process group (pgid).
///
/// Membership is snapshotted at send time. Fanout continues past individual
/// recipient failures. Returns a compact status word in the low bits:
///   bits 31:16 – number of failures (saturated to 0xFFFF)
///   bits 15:0  – number of successes (saturated to 0xFFFF)
///
/// Returns a negative `Errno` only for top-level errors:
/// - `EINVAL` — pgid == 0 or invalid pointer arguments
pub const SYS_MSG_BROADCAST: u32 = 0x3031;

// ============================================================================
// Virtual File System (VFS) (0x4000)
// ============================================================================
pub const SYS_FS_OPEN: u32 = 0x4000;
pub const SYS_FS_CLOSE: u32 = 0x4001;
pub const SYS_FS_READ: u32 = 0x4002;
pub const SYS_FS_WRITE: u32 = 0x4003;
pub const SYS_FS_SEEK: u32 = 0x4004;
pub const SYS_FS_STAT: u32 = 0x4005;
pub const SYS_FS_READDIR: u32 = 0x4006;
pub const SYS_FS_MKDIR: u32 = 0x4007;
pub const SYS_FS_UNLINK: u32 = 0x4008;
pub const SYS_FS_MOUNT: u32 = 0x4009;
pub const SYS_FS_UMOUNT: u32 = 0x400A;
pub const SYS_FS_POLL: u32 = 0x400B;
pub const SYS_FS_DUP: u32 = 0x400C;
pub const SYS_FS_DUP2: u32 = 0x400D;
pub const SYS_FS_WATCH_FD: u32 = 0x400E;
pub const SYS_FS_WATCH_PATH: u32 = 0x400F;
pub const SYS_FS_RENAME: u32 = 0x4010;
pub const SYS_FS_DEVICE_CALL: u32 = 0x4011;
pub const SYS_FS_CHDIR: u32 = 0x4012;
pub const SYS_FS_GETCWD: u32 = 0x4013;
pub const SYS_FD_FROM_HANDLE: u32 = 0x4014;
pub const SYS_FS_NOTIFY: u32 = 0x4015;
pub const SYS_FS_ISATTY: u32 = 0x4020;
pub const SYS_FS_REALPATH: u32 = 0x4016;
pub const SYS_FS_SYNC: u32 = 0x4017;
pub const SYS_FS_FCNTL: u32 = 0x4018;
pub const SYS_FS_SYMLINK: u32 = 0x4019;
pub const SYS_FS_READLINK: u32 = 0x401A;
/// Truncate an open file to `size` bytes (ftruncate semantics).
pub const SYS_FS_FTRUNCATE: u32 = 0x401B;
/// Change file permission bits by path (chmod).
pub const SYS_FS_CHMOD: u32 = 0x401C;
/// Change file permission bits by open file descriptor (fchmod).
pub const SYS_FS_FCHMOD: u32 = 0x401D;
/// Set access and modification timestamps by path (utimes).
pub const SYS_FS_UTIMES: u32 = 0x401E;
/// Set access and modification timestamps by open file descriptor (futimes).
pub const SYS_FS_FUTIMES: u32 = 0x401F;
/// Stat a path without following the final symlink (lstat semantics).
/// Args: (path_ptr, path_len, stat_ptr) → 0
pub const SYS_FS_LSTAT: u32 = 0x4020;
/// Scatter-gather read from an open file descriptor.
/// Args: (fd, iovec_ptr, iovec_count) → total_bytes_read
pub const SYS_FS_READV: u32 = 0x4021;
/// Scatter-gather write to an open file descriptor.
/// Args: (fd, iovec_ptr, iovec_count) → total_bytes_written
pub const SYS_FS_WRITEV: u32 = 0x4022;
/// Create a hard link at `dst` pointing to the same inode as `src`.
/// Args: (src_ptr, src_len, dst_ptr, dst_len) → 0
pub const SYS_FS_LINK: u32 = 0x4023;
/// Advisory file lock / unlock (flock semantics).
/// Args: (fd, how) where `how` is a combination of [`flock_flags`] constants.
/// Returns 0 on success; EWOULDBLOCK if the lock is held and LOCK_NB was set.
pub const SYS_FS_FLOCK: u32 = 0x4024;
/// Set access and modification timestamps for a path without following symlinks (lutimes).
pub const SYS_FS_LUTIMES: u32 = 0x4025;

/// Flags for [`SYS_FS_FLOCK`].
///
/// Mirrors the POSIX / Linux `flock(2)` flag values.
pub mod flock_flags {
    /// Acquire a shared (read) lock.
    pub const LOCK_SH: u32 = 1;
    /// Acquire an exclusive (write) lock.
    pub const LOCK_EX: u32 = 2;
    /// Non-blocking: return `EWOULDBLOCK` instead of blocking.
    pub const LOCK_NB: u32 = 4;
    /// Release the lock held on the file.
    pub const LOCK_UN: u32 = 8;
}

// ============================================================================
// Hardware & Device Interfaces (0x5000)
// ============================================================================
pub const SYS_DEVICE_CLAIM: u32 = 0x5000;
pub const SYS_DEVICE_CALL: u32 = 0x5001;
pub const SYS_DEVICE_MAP_MMIO: u32 = 0x5002;
pub const SYS_DEVICE_ALLOC_DMA: u32 = 0x5003;
pub const SYS_DEVICE_DMA_PHYS: u32 = 0x5004;
pub const SYS_DEVICE_IOPORT_READ: u32 = 0x5005;
pub const SYS_DEVICE_IOPORT_WRITE: u32 = 0x5006;
pub const SYS_DEVICE_IRQ_SUBSCRIBE: u32 = 0x5007;
pub const SYS_DEVICE_IRQ_WAIT: u32 = 0x5008;

// ============================================================================
// System Control & Misc (0x7000)
// ============================================================================
pub const SYS_REBOOT: u32 = 0x7000;
pub const SYS_GETRANDOM: u32 = 0x7001;
pub const SYS_CONSOLE_ENABLE: u32 = 0x7002;
pub const SYS_LOG_SET_LEVEL: u32 = 0x7003;
/// Mix caller-supplied bytes into the kernel entropy pool and mark it seeded.
/// Intended for privileged entropy-source drivers (analogous to `SYS_TIME_ANCHOR`).
pub const SYS_ENTROPY_SEED: u32 = 0x7004;

pub mod reboot_cmd {
    pub const RESTART: u32 = 0;
    pub const HALT: u32 = 1;
    pub const POWER_OFF: u32 = 2;
}

// ============================================================================
// ABI Flags & Constants
// ============================================================================

pub mod channel_wait {
    pub const READABLE: u32 = 1 << 0;
    pub const WRITABLE: u32 = 1 << 1;
}

/// Socket address family constants (analogous to POSIX AF_* values).
pub mod socket_domain {
    /// Unix domain sockets (local IPC via filesystem paths).
    pub const AF_UNIX: u32 = 1;
}

/// Socket type constants (analogous to POSIX SOCK_* values).
pub mod socket_type {
    /// Reliable, sequenced, bidirectional byte stream.
    pub const SOCK_STREAM: u32 = 1;
    /// Connectionless, unreliable datagrams with message boundaries.
    pub const SOCK_DGRAM: u32 = 2;
}

/// Shutdown direction constants for `sys_shutdown`.
pub mod shutdown_how {
    /// Shut down the receive direction.
    pub const SHUT_RD: u32 = 0;
    /// Shut down the transmit direction.
    pub const SHUT_WR: u32 = 1;
    /// Shut down both directions.
    pub const SHUT_RDWR: u32 = 2;
}

pub mod pipe_flags {
    pub const NONBLOCK: u32 = 1 << 0;
    pub const CLOEXEC: u32 = 1 << 1;
}

pub mod vfs_flags {
    pub const O_RDONLY: u32 = 0x0000;
    pub const O_WRONLY: u32 = 0x0001;
    pub const O_RDWR: u32 = 0x0002;
    pub const O_CREAT: u32 = 0x0040;
    /// Fail if the file already exists (used with O_CREAT for create_new semantics).
    pub const O_EXCL: u32 = 0x0080;
    pub const O_TRUNC: u32 = 0x0200;
    pub const O_APPEND: u32 = 0x0400;
    pub const O_NONBLOCK: u32 = 0x0800;
}

pub mod fcntl_cmd {
    pub const F_GETFD: u32 = 1;
    pub const F_SETFD: u32 = 2;
    pub const F_GETFL: u32 = 3;
    pub const F_SETFL: u32 = 4;
}

pub mod fd_flags {
    pub const FD_CLOEXEC: u32 = 0x1;
}

pub mod poll_flags {
    pub const POLLIN: u16 = 0x0001;
    pub const POLLOUT: u16 = 0x0004;
    pub const POLLERR: u16 = 0x0008;
    pub const POLLHUP: u16 = 0x0010;
    pub const POLLNVAL: u16 = 0x0020;
}
