//! ThingOS system call interface.
//!
//! This module provides `raw_syscall6`, the low-level syscall entry point used
//! by all higher-level ThingOS APIs, plus the canonical syscall-number
//! constants for every architecture that ThingOS supports.

#![allow(dead_code)]

// ── Syscall constants ────────────────────────────────────────────────────────
// These numbers are ThingOS-specific and do **not** match Linux.
pub const SYS_READ: u64 = 0;
pub const SYS_WRITE: u64 = 1;
pub const SYS_OPEN: u64 = 2;
pub const SYS_CLOSE: u64 = 3;
pub const SYS_STAT: u64 = 4;
pub const SYS_FSTAT: u64 = 5;
pub const SYS_LSTAT: u64 = 6;
pub const SYS_LSEEK: u64 = 7;
/// Anonymous memory mapping (ThingOS equivalent of mmap(MAP_ANON)).
pub const SYS_VM_MAP: u64 = 8;
/// Unmap a previously mapped region.
pub const SYS_VM_UNMAP: u64 = 9;
pub const SYS_GETPID: u64 = 10;
pub const SYS_EXIT: u64 = 11;
pub const SYS_EXIT_GROUP: u64 = 12;
/// Spawn a new thread (clone-like).
pub const SYS_THREAD_SPAWN: u64 = 13;
/// Join a thread by its handle.
pub const SYS_THREAD_JOIN: u64 = 14;
/// Voluntarily yield the CPU.
pub const SYS_THREAD_YIELD: u64 = 15;
/// Futex wait/wake.
pub const SYS_FUTEX: u64 = 16;
/// Sleep for `ns` nanoseconds.
pub const SYS_THREAD_SLEEP_NS: u64 = 17;
pub const SYS_GETCWD: u64 = 18;
pub const SYS_CHDIR: u64 = 19;
pub const SYS_MKDIR: u64 = 20;
pub const SYS_RMDIR: u64 = 21;
pub const SYS_UNLINK: u64 = 22;
pub const SYS_RENAME: u64 = 23;
pub const SYS_GETDENTS: u64 = 24;
pub const SYS_READLINK: u64 = 25;
pub const SYS_SYMLINK: u64 = 26;
pub const SYS_CLOCK_GETTIME: u64 = 27;
pub const SYS_GETRANDOM: u64 = 28;
pub const SYS_SOCKET: u64 = 29;
pub const SYS_CONNECT: u64 = 30;
pub const SYS_ACCEPT: u64 = 31;
pub const SYS_SEND: u64 = 32;
pub const SYS_RECV: u64 = 33;
pub const SYS_BIND: u64 = 34;
pub const SYS_LISTEN: u64 = 35;
pub const SYS_SHUTDOWN: u64 = 36;
pub const SYS_SETSOCKOPT: u64 = 37;
pub const SYS_GETSOCKOPT: u64 = 38;
pub const SYS_GETPEERNAME: u64 = 39;
pub const SYS_GETSOCKNAME: u64 = 40;
pub const SYS_PIPE: u64 = 41;
pub const SYS_FCNTL: u64 = 42;
pub const SYS_DUP: u64 = 43;
/// Query the number of logical CPUs available (for `available_parallelism`).
pub const SYS_CPUS: u64 = 44;
/// Spawn a child process.
pub const SYS_SPAWN: u64 = 45;
/// Wait for a child process.
pub const SYS_WAIT: u64 = 46;
/// Send a signal to a process.
pub const SYS_KILL: u64 = 47;
/// Read the path of the running executable.
pub const SYS_CURRENT_EXE: u64 = 48;
/// Test whether an fd refers to a terminal.
pub const SYS_ISATTY: u64 = 49;
/// Set the name of the current thread.
pub const SYS_SET_THREAD_NAME: u64 = 50;
/// Architecture-specific thread-pointer setup (x86_64: set FS.base).
pub const SYS_ARCH_PRCTL: u64 = 51;
pub const SYS_FTRUNCATE: u64 = 52;
pub const SYS_CHMOD: u64 = 53;
pub const SYS_FCHMOD: u64 = 54;
pub const SYS_LINK: u64 = 55;
pub const SYS_TRUNCATE: u64 = 56;
pub const SYS_FSYNC: u64 = 57;
pub const SYS_IOCTL: u64 = 58;
/// Get the current thread's numeric ID.
pub const SYS_GETTID: u64 = 59;
pub const SYS_SET_SOCKET_TIMEOUT: u64 = 60;

// ── Futex operation codes ────────────────────────────────────────────────────
pub const FUTEX_WAIT: u64 = 0;
pub const FUTEX_WAKE: u64 = 1;

// ── clock IDs for SYS_CLOCK_GETTIME ─────────────────────────────────────────
pub const CLOCK_REALTIME: u64 = 0;
pub const CLOCK_MONOTONIC: u64 = 1;

// ── SYS_VM_MAP flags ─────────────────────────────────────────────────────────
/// Map anonymous, read-write, private memory.
pub const VM_MAP_ANON_RW: u64 = 0;

// ── Socket address families ──────────────────────────────────────────────────
pub const AF_INET: u32 = 2;
pub const AF_INET6: u32 = 10;

// ── Socket types ─────────────────────────────────────────────────────────────
pub const SOCK_STREAM: u32 = 1;
pub const SOCK_DGRAM: u32 = 2;

// ── Shutdown flags ───────────────────────────────────────────────────────────
pub const SHUT_RD: u32 = 0;
pub const SHUT_WR: u32 = 1;
pub const SHUT_RDWR: u32 = 2;

// ── Socket option levels / names ─────────────────────────────────────────────
pub const SOL_SOCKET: u32 = 1;
pub const SO_KEEPALIVE: u32 = 9;
pub const SO_RCVTIMEO: u32 = 20;
pub const SO_SNDTIMEO: u32 = 21;
pub const IPPROTO_TCP: u32 = 6;
pub const TCP_NODELAY: u32 = 1;
pub const IPPROTO_IP: u32 = 0;
pub const IP_TTL: u32 = 2;
pub const IPPROTO_IPV6: u32 = 41;
pub const IPV6_UNICAST_HOPS: u32 = 16;
pub const SO_ERROR: u32 = 4;

// ── fcntl commands ───────────────────────────────────────────────────────────
pub const F_GETFL: u64 = 3;
pub const F_SETFL: u64 = 4;
pub const O_NONBLOCK: u64 = 2048;

// ── Open flags ───────────────────────────────────────────────────────────────
pub const O_RDONLY: u64 = 0;
pub const O_WRONLY: u64 = 1;
pub const O_RDWR: u64 = 2;
pub const O_CREAT: u64 = 64;
pub const O_TRUNC: u64 = 512;
pub const O_APPEND: u64 = 1024;
pub const O_EXCL: u64 = 128;
pub const O_CLOEXEC: u64 = 524288;

// ── Seek whence values ───────────────────────────────────────────────────────
pub const SEEK_SET: u64 = 0;
pub const SEEK_CUR: u64 = 1;
pub const SEEK_END: u64 = 2;

// ── Directory entry file types ───────────────────────────────────────────────
pub const DT_UNKNOWN: u8 = 0;
pub const DT_REG: u8 = 8;
pub const DT_DIR: u8 = 4;
pub const DT_LNK: u8 = 10;

// ── Error codes ──────────────────────────────────────────────────────────────
pub const ENOTSUP: i32 = 95;
pub const EAFNOSUPPORT: i32 = 97;
pub const ETIMEDOUT: i32 = 110;

// ── SYS_SPAWN flags ──────────────────────────────────────────────────────────
pub const SPAWN_FLAG_CLOEXEC_STDIN: u64 = 1;
pub const SPAWN_FLAG_CLOEXEC_STDOUT: u64 = 2;
pub const SPAWN_FLAG_CLOEXEC_STDERR: u64 = 4;
pub const SPAWN_STDIO_INHERIT: i32 = -1;
pub const SPAWN_STDIO_NULL: i32 = -2;
pub const SPAWN_STDIO_PIPE: i32 = -3;

// ── Timespec (used by clock_gettime and futex_wait) ──────────────────────────
#[repr(C)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

// ── Stat structure ───────────────────────────────────────────────────────────
#[repr(C)]
pub struct Stat {
    pub size: u64,
    pub file_type: u8,
    pub perm: u8,
    pub _pad: [u8; 6],
    pub modified: u64,  // nanoseconds since epoch
    pub accessed: u64,
    pub created: u64,
}

impl Default for Stat {
    fn default() -> Self {
        Self { size: 0, file_type: 0, perm: 0, _pad: [0; 6], modified: 0, accessed: 0, created: 0 }
    }
}

// ── Directory entry ───────────────────────────────────────────────────────────
#[repr(C)]
pub struct Dirent {
    pub ino: u64,
    pub off: i64,
    pub reclen: u16,
    pub file_type: u8,
    // name follows: variable-length, NUL-terminated
}

// ── Sockaddr structures ───────────────────────────────────────────────────────
#[repr(C)]
pub struct SockaddrIn {
    pub family: u16,
    pub port: u16,   // network byte order
    pub addr: u32,   // network byte order
    pub _pad: [u8; 8],
}

#[repr(C)]
pub struct SockaddrIn6 {
    pub family: u16,
    pub port: u16,
    pub flowinfo: u32,
    pub addr: [u8; 16],
    pub scope_id: u32,
}

// ── raw_syscall6 ─────────────────────────────────────────────────────────────

/// Issue a ThingOS system call with up to six arguments.
///
/// Returns the kernel's result: negative values indicate `-errno`.
///
/// # Safety
/// The caller must ensure that all arguments are valid for the given syscall.
#[inline]
pub unsafe fn raw_syscall6(nr: u64, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        let ret: i64;
        unsafe {
            core::arch::asm!(
                "syscall",
                in("rax") nr,
                in("rdi") a0,
                in("rsi") a1,
                in("rdx") a2,
                in("r10") a3,
                in("r8")  a4,
                in("r9")  a5,
                out("rcx") _,
                out("r11") _,
                lateout("rax") ret,
                options(nostack, preserves_flags),
            );
        }
        ret
    }
    #[cfg(target_arch = "aarch64")]
    {
        let ret: i64;
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8")  nr,
                in("x0")  a0,
                in("x1")  a1,
                in("x2")  a2,
                in("x3")  a3,
                in("x4")  a4,
                in("x5")  a5,
                lateout("x0") ret,
                options(nostack, preserves_flags),
            );
        }
        ret
    }
    #[cfg(target_arch = "riscv64")]
    {
        let ret: i64;
        unsafe {
            core::arch::asm!(
                "ecall",
                in("a7") nr,
                in("a0") a0,
                in("a1") a1,
                in("a2") a2,
                in("a3") a3,
                in("a4") a4,
                in("a5") a5,
                lateout("a0") ret,
                options(nostack, preserves_flags),
            );
        }
        ret
    }
    #[cfg(target_arch = "loongarch64")]
    {
        let ret: i64;
        unsafe {
            core::arch::asm!(
                "syscall 0",
                in("$a7") nr,
                in("$a0") a0,
                in("$a1") a1,
                in("$a2") a2,
                in("$a3") a3,
                in("$a4") a4,
                in("$a5") a5,
                lateout("$a0") ret,
                options(nostack, preserves_flags),
            );
        }
        ret
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "loongarch64",
    )))]
    {
        // Unsupported architecture: make a link-time error obvious.
        let _ = (nr, a0, a1, a2, a3, a4, a5);
        compile_error!("ThingOS does not support this architecture")
    }
}

/// Convenience wrapper for syscalls with fewer than six arguments.
#[inline]
pub unsafe fn syscall0(nr: u64) -> i64 {
    unsafe { raw_syscall6(nr, 0, 0, 0, 0, 0, 0) }
}

#[inline]
pub unsafe fn syscall1(nr: u64, a0: u64) -> i64 {
    unsafe { raw_syscall6(nr, a0, 0, 0, 0, 0, 0) }
}

#[inline]
pub unsafe fn syscall2(nr: u64, a0: u64, a1: u64) -> i64 {
    unsafe { raw_syscall6(nr, a0, a1, 0, 0, 0, 0) }
}

#[inline]
pub unsafe fn syscall3(nr: u64, a0: u64, a1: u64, a2: u64) -> i64 {
    unsafe { raw_syscall6(nr, a0, a1, a2, 0, 0, 0) }
}

#[inline]
pub unsafe fn syscall4(nr: u64, a0: u64, a1: u64, a2: u64, a3: u64) -> i64 {
    unsafe { raw_syscall6(nr, a0, a1, a2, a3, 0, 0) }
}

#[inline]
pub unsafe fn syscall5(nr: u64, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> i64 {
    unsafe { raw_syscall6(nr, a0, a1, a2, a3, a4, 0) }
}

/// Convert a (possibly negative) syscall return value into an `io::Result`.
///
/// A return value `< 0` is interpreted as `-errno`.
#[inline]
pub fn cvt(ret: i64) -> crate::io::Result<i64> {
    if ret < 0 {
        Err(crate::io::Error::from_raw_os_error((-ret) as i32))
    } else {
        Ok(ret)
    }
}

/// Like `cvt`, but retries automatically on `EINTR`.
#[inline]
pub fn cvt_r<F: FnMut() -> i64>(mut f: F) -> crate::io::Result<i64> {
    loop {
        let ret = f();
        if ret == -4 {
            // EINTR: retry
            continue;
        }
        return cvt(ret);
    }
}
