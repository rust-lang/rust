pub mod arch;
pub mod channel;
pub mod signal;
pub mod socket;
pub mod vfs;
pub mod wait;

use abi::device::{DEVICE_IRQ_SUBSCRIBE_DEVICE, DEVICE_IRQ_SUBSCRIBE_VECTOR};
use abi::errors::Errno;
pub use abi::syscall::*;
use abi::time::{ClockId, TimeSpec};
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use arch::raw_syscall6;

/// Helper to expose raw syscalls safely to other modules if needed.
#[inline(always)]
pub unsafe fn syscall6(
    n: u32,
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> isize {
    unsafe { raw_syscall6(n, a0, a1, a2, a3, a4, a5) }
}

// Low-level wrappers

pub fn exit(code: i32) -> ! {
    unsafe {
        raw_syscall6(SYS_EXIT, code as usize, 0, 0, 0, 0, 0);
        core::hint::unreachable_unchecked();
    }
}

/// Reboot or shutdown the system. This call does not return.
pub fn reboot_raw(cmd: u32) -> ! {
    unsafe {
        raw_syscall6(SYS_REBOOT, cmd as usize, 0, 0, 0, 0, 0);
        core::hint::unreachable_unchecked();
    }
}

/// Reboot the system. This call does not return.
pub fn reboot() -> ! {
    reboot_raw(abi::syscall::reboot_cmd::RESTART)
}

/// Shutdown the system. This call does not return.
pub fn shutdown() -> ! {
    reboot_raw(abi::syscall::reboot_cmd::POWER_OFF)
}

pub fn log_write(msg: &str, level: usize) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_LOG_WRITE,
            msg.as_ptr() as usize,
            msg.len(),
            level,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

pub use log_write as debug_write;

pub fn log_set_level(level: u8) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_LOG_SET_LEVEL, level as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

pub fn read(fd: usize, buf: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe { raw_syscall6(SYS_READ, fd, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0) };
    abi::errors::errno(ret)
}

pub fn write(fd: usize, buf: &[u8]) -> Result<usize, Errno> {
    let ret = unsafe { raw_syscall6(SYS_WRITE, fd, buf.as_ptr() as usize, buf.len(), 0, 0, 0) };
    abi::errors::errno(ret)
}

#[allow(deprecated)]
pub use channel::{
    channel_capacity, channel_close, channel_create, channel_create_fds, channel_len, channel_recv,
    channel_recv_handle, channel_recv_msg, channel_send, channel_send_all, channel_send_handle,
    channel_send_msg, channel_try_recv, channel_wait, ChannelHandle,
};
pub use vfs::{
    dup, dup2, pipe, tcgetattr, tcsetattr, vfs_chdir, vfs_chmod, vfs_close, vfs_fchmod, vfs_fcntl,
    vfs_fd_from_handle, vfs_fsync, vfs_futimes, vfs_getcwd, vfs_isatty, vfs_mkdir, vfs_mount,
    vfs_open, vfs_poll, vfs_read, vfs_readdir, vfs_readv, vfs_realpath, vfs_rename, vfs_seek,
    vfs_stat, vfs_umount, vfs_unlink, vfs_utimes, vfs_watch_fd, vfs_watch_path, vfs_write,
    vfs_writev,
};
pub use wait::wait_many;
pub use signal::{
    alarm, kill, pause, raise, sig_block, sig_setmask, sig_unblock, sigaction, sigpending,
    sigprocmask, sigsuspend,
};

// ============================================================================
// Futex (fast userspace mutex) syscall wrappers
// ============================================================================

/// Block the calling thread until `*addr != expected` or until `timeout_ns`
/// nanoseconds elapse (pass `0` for an indefinite wait).
///
/// Returns `Ok(())` on wake, `Err(EAGAIN)` if the value had already changed,
/// or `Err(ETIMEDOUT)` on timeout.
///
/// Note: Thing-OS only supports 64-bit targets, so `u64` timeout values
/// fit safely in a `usize` register argument.
pub fn futex_wait(
    addr: &core::sync::atomic::AtomicU32,
    expected: u32,
    timeout_ns: u64,
) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FUTEX_WAIT,
            addr as *const core::sync::atomic::AtomicU32 as usize,
            expected as usize,
            timeout_ns as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Wake up to `count` threads waiting on `addr`.
///
/// Returns the number of threads actually woken.
pub fn futex_wake(addr: &core::sync::atomic::AtomicU32, count: u32) -> Result<u32, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_FUTEX_WAKE,
            addr as *const core::sync::atomic::AtomicU32 as usize,
            count as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

pub fn yield_now() {
    unsafe {
        raw_syscall6(SYS_YIELD, 0, 0, 0, 0, 0, 0);
    }
}

pub fn sleep_ns(ns: u64) {
    unsafe {
        raw_syscall6(SYS_SLEEP, ns as usize, 0, 0, 0, 0, 0);
    }
}

pub fn sleep_ms(ms: u64) {
    // Legacy support, or use ns
    sleep_ns(ms * 1_000_000);
}

pub fn get_tid() -> Result<u64, Errno> {
    let ret = unsafe { raw_syscall6(SYS_GET_TID, 0, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u64)
}

/// Get the current process ID.
pub fn getpid() -> u32 {
    let ret = unsafe { raw_syscall6(SYS_GETPID, 0, 0, 0, 0, 0, 0) };
    ret as u32
}

/// Get the parent process ID.
pub fn getppid() -> u32 {
    let ret = unsafe { raw_syscall6(SYS_GETPPID, 0, 0, 0, 0, 0, 0) };
    ret as u32
}

/// Retrieve the process argv into `buf`. Returns total bytes needed.
/// First call with an empty/small buffer to learn the size, then retry.
pub fn argv_get(buf: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_ARGV_GET,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Get a single environment variable by key. Returns value length needed.
pub fn env_get(key: &[u8], val: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_ENV_GET,
            key.as_ptr() as usize,
            key.len(),
            val.as_mut_ptr() as usize,
            val.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Set an environment variable.
pub fn env_set(key: &[u8], val: &[u8]) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_ENV_SET,
            key.as_ptr() as usize,
            key.len(),
            val.as_ptr() as usize,
            val.len(),
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Remove an environment variable.
pub fn env_unset(key: &[u8]) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_ENV_UNSET, key.as_ptr() as usize, key.len(), 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// List all environment variables. Returns total bytes needed.
pub fn env_list(buf: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_ENV_LIST,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Retrieve the process auxiliary vector (AT_* entries) into `buf`.
///
/// Returns the total bytes needed.  On the first call pass an empty slice to
/// learn the required buffer size, then retry with a buffer of that size.
///
/// The serialized format is:
/// - `count: u32 LE` — number of entries including the terminating `AT_NULL` sentinel
/// - For each entry: `type: u64 LE`, `value: u64 LE`
///
/// The last entry is always `(AT_NULL=0, 0)`.
pub fn auxv_get(buf: &mut [u8]) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_AUXV_GET,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

pub fn monotonic_ns() -> u64 {
    time_now(ClockId::Monotonic)
        .ok()
        .and_then(|spec| spec.as_nanos())
        .unwrap_or(0)
}

pub fn time_now(clock_id: ClockId) -> Result<TimeSpec, Errno> {
    time_now_raw(clock_id as u32)
}

pub fn time_now_raw(clock_id: u32) -> Result<TimeSpec, Errno> {
    let mut spec = TimeSpec::ZERO;
    let ret = unsafe {
        raw_syscall6(
            SYS_TIME_NOW,
            clock_id as usize,
            (&mut spec as *mut TimeSpec).cast::<u8>() as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)?;
    Ok(spec)
}

pub fn spawn_process(name: &str, arg: usize) -> Result<u64, abi::errors::Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_SPAWN_PROCESS,
            name.as_ptr() as usize,
            name.len(),
            arg,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u64)
}

/// Replace the current process image with a new executable from the given FD.
/// PID and file descriptors are preserved.
pub fn task_exec(fd: u32, argv: &[&[u8]], env: &BTreeMap<Vec<u8>, Vec<u8>>) -> Result<(), Errno> {
    let argv_blob = serialize_argv(argv);
    let env_blob = serialize_env(env);

    let ret = unsafe {
        raw_syscall6(
            SYS_TASK_EXEC,
            fd as usize,
            argv_blob.as_ptr() as usize,
            argv_blob.len(),
            env_blob.as_ptr() as usize,
            env_blob.len(),
            0,
        )
    };

    abi::errors::errno(ret).map(|_| ())
}

/// Returns `true` if the first 4 bytes of `header` match the ELF magic
/// `\x7fELF`.
///
/// Callers may use this to quickly validate a file before calling
/// [`execve`] or [`task_exec`], though validation is also performed by the
/// kernel loader.
#[inline]
pub fn is_elf_magic(header: &[u8]) -> bool {
    header.len() >= 4 && header[..4] == *b"\x7fELF"
}

/// Parse a shebang (`#!`) line from a file header.
///
/// Returns `Some((interpreter_path, optional_arg))` when `header` begins with
/// `#!` and contains at least one non-empty interpreter path.  Both returned
/// string slices are sub-slices of `header`, so their lifetime is tied to it.
///
/// The interpreter path is the first whitespace-delimited token after `#!`.
/// If a second token exists on the same line it is returned as the optional
/// argument (the entire remainder of the line after the first separator,
/// trimmed).  This matches the POSIX single-optional-arg convention used by
/// Linux and BSD kernels.
///
/// Returns `None` if the header does not begin with `#!`, the line is not
/// valid UTF-8, or no non-empty interpreter path can be found.
pub fn parse_shebang(header: &[u8]) -> Option<(&str, Option<&str>)> {
    if header.len() < 2 || header[0] != b'#' || header[1] != b'!' {
        return None;
    }

    // Find the end of the shebang line (CR or LF).
    let rest = &header[2..];
    let line_len = rest
        .iter()
        .position(|&b| b == b'\n' || b == b'\r')
        .unwrap_or(rest.len());
    let line = core::str::from_utf8(&rest[..line_len]).ok()?.trim();

    if line.is_empty() {
        return None;
    }

    // Split interpreter path from optional argument on the first run of
    // whitespace.  Only two tokens are recognised (POSIX allows exactly one
    // optional argument).
    if let Some(sep) = line.find(|c: char| c == ' ' || c == '\t') {
        let interp = line[..sep].trim();
        let arg = line[sep..].trim();
        if interp.is_empty() {
            return None;
        }
        Some((interp, if arg.is_empty() { None } else { Some(arg) }))
    } else {
        Some((line, None))
    }
}

/// Higher-level POSIX-friendly execve.  Opens the path, optionally verifies
/// the executable type, and calls [`task_exec`].
///
/// The function:
/// 1. Opens `path` with `O_RDONLY`.
/// 2. Reads up to 256 bytes to check for a recognised magic number.
///    - ELF binaries are executed directly via [`task_exec`].
///    - Files starting with `#!` are handled as interpreter scripts: the
///      shebang line is parsed, the interpreter binary is opened, `argv` is
///      rewritten to `[interpreter, opt_arg?, script_path, original_argv[1:]…]`,
///      and [`task_exec`] is called with the interpreter's FD.
///    - Any other file causes the FD to be closed and [`Errno::ENOEXEC`] to be
///      returned.
/// 3. On failure the file descriptor is closed before returning the error.
pub fn execve(path: &str, argv: &[&[u8]], env: &BTreeMap<Vec<u8>, Vec<u8>>) -> Result<(), Errno> {
    let fd = vfs_open(path, abi::syscall::vfs_flags::O_RDONLY)?;

    // Read enough bytes to detect the magic number and, for shebang scripts,
    // parse the interpreter line.  POSIX shebang lines are at most 255 bytes
    // after `#!`; reading 256 bytes total covers the common maximum.
    let mut header_buf = [0u8; 256];
    let n = match vfs_read(fd, &mut header_buf) {
        Ok(n) => n,
        Err(e) => {
            let _ = vfs_close(fd);
            return Err(e);
        }
    };

    if n < 2 {
        let _ = vfs_close(fd);
        return Err(Errno::ENOEXEC);
    }

    let header = &header_buf[..n];

    if is_elf_magic(header) {
        // ELF binary: the kernel reads the full file from the VFS node, so
        // the file-offset position does not matter here.
        let res = task_exec(fd, argv, env);
        if res.is_err() {
            let _ = vfs_close(fd);
        }
        return res;
    }

    if header[0] == b'#' && header[1] == b'!' {
        // Shebang script: the script fd is no longer needed once we have the
        // interpreter path.
        let _ = vfs_close(fd);

        let (interp_path, interp_arg) = match parse_shebang(header) {
            Some(v) => v,
            None => return Err(Errno::ENOEXEC),
        };

        // Open the interpreter binary.
        let interp_fd = vfs_open(interp_path, abi::syscall::vfs_flags::O_RDONLY)?;

        // Verify the interpreter is an ELF so we give a clear error for
        // misconfigured scripts rather than a confusing kernel-loader error.
        let mut interp_magic = [0u8; 4];
        let nm = match vfs_read(interp_fd, &mut interp_magic) {
            Ok(n) => n,
            Err(e) => {
                let _ = vfs_close(interp_fd);
                return Err(e);
            }
        };
        if nm < 4 || !is_elf_magic(&interp_magic) {
            let _ = vfs_close(interp_fd);
            return Err(Errno::ENOEXEC);
        }

        // Rewrite argv:
        //   argv[0] = interpreter path
        //   argv[1] = optional interpreter argument (if present)
        //   argv[N] = script path  (original argv[0] is replaced)
        //   argv[N+1..] = original argv[1..]
        let mut new_argv: Vec<&[u8]> = Vec::new();
        new_argv.push(interp_path.as_bytes());
        if let Some(arg) = interp_arg {
            new_argv.push(arg.as_bytes());
        }
        new_argv.push(path.as_bytes());
        if argv.len() > 1 {
            new_argv.extend_from_slice(&argv[1..]);
        }

        let res = task_exec(interp_fd, &new_argv, env);
        if res.is_err() {
            let _ = vfs_close(interp_fd);
        }
        return res;
    }

    // Not an ELF and not a shebang script.
    let _ = vfs_close(fd);
    Err(Errno::ENOEXEC)
}

/// POSIX-style execv: replace the current process image with the executable
/// at `path`, passing `argv` and an **empty** environment.
///
/// This is a convenience wrapper around [`execve`].  If you need to pass an
/// explicit environment, use [`execve`] directly.
pub fn execv(path: &str, argv: &[&[u8]]) -> Result<(), Errno> {
    execve(path, argv, &BTreeMap::new())
}

#[cfg(test)]
mod exec_tests {
    use super::*;

    // ── is_elf_magic ──────────────────────────────────────────────────────────

    #[test]
    fn elf_magic_valid() {
        assert!(is_elf_magic(b"\x7fELF\x02\x01\x01\x00"));
    }

    #[test]
    fn elf_magic_too_short() {
        assert!(!is_elf_magic(b"\x7fEL"));
        assert!(!is_elf_magic(b""));
    }

    #[test]
    fn elf_magic_wrong_bytes() {
        assert!(!is_elf_magic(b"#!/bin/sh\n"));
        assert!(!is_elf_magic(b"\x00\x00\x00\x00"));
    }

    // ── serialize_argv / serialize_env round-trip ─────────────────────────────

    #[test]
    fn serialize_argv_empty() {
        let blob = serialize_argv(&[]);
        // count field only
        assert_eq!(blob, &[0, 0, 0, 0]);
    }

    #[test]
    fn serialize_argv_single_arg() {
        let blob = serialize_argv(&[b"hello"]);
        // count=1, len=5, "hello"
        let mut expected: Vec<u8> = Vec::new();
        expected.extend_from_slice(&1u32.to_le_bytes());
        expected.extend_from_slice(&5u32.to_le_bytes());
        expected.extend_from_slice(b"hello");
        assert_eq!(blob, expected);
    }

    #[test]
    fn serialize_argv_multiple_args() {
        let args: &[&[u8]] = &[b"foo", b"bar", b"baz"];
        let blob = serialize_argv(args);
        // count=3
        assert_eq!(&blob[..4], &3u32.to_le_bytes());
        // total length: 4 (count) + 3*(4+3) = 4 + 21 = 25
        assert_eq!(blob.len(), 4 + 3 * (4 + 3));
    }

    #[test]
    fn serialize_env_empty() {
        let blob = serialize_env(&BTreeMap::new());
        assert_eq!(blob, &[0, 0, 0, 0]);
    }

    #[test]
    fn serialize_env_single_entry() {
        let mut env = BTreeMap::new();
        env.insert(b"KEY".to_vec(), b"val".to_vec());
        let blob = serialize_env(&env);
        // count=1
        assert_eq!(&blob[..4], &1u32.to_le_bytes());
        // key len=3, key="KEY", val len=3, val="val"
        let expected_len = 4 + (4 + 3) + (4 + 3);
        assert_eq!(blob.len(), expected_len);
    }

    // ── parse_shebang ─────────────────────────────────────────────────────────

    #[test]
    fn shebang_no_arg() {
        let (interp, arg) = parse_shebang(b"#!/bin/sh\n").unwrap();
        assert_eq!(interp, "/bin/sh");
        assert_eq!(arg, None);
    }

    #[test]
    fn shebang_with_arg() {
        let (interp, arg) = parse_shebang(b"#!/usr/bin/env python3\n").unwrap();
        assert_eq!(interp, "/usr/bin/env");
        assert_eq!(arg, Some("python3"));
    }

    #[test]
    fn shebang_arg_with_leading_spaces() {
        let (interp, arg) = parse_shebang(b"#!/usr/bin/awk -f\n").unwrap();
        assert_eq!(interp, "/usr/bin/awk");
        assert_eq!(arg, Some("-f"));
    }

    #[test]
    fn shebang_no_newline() {
        // No trailing newline: interpret to end of buffer.
        let (interp, arg) = parse_shebang(b"#!/bin/bash").unwrap();
        assert_eq!(interp, "/bin/bash");
        assert_eq!(arg, None);
    }

    #[test]
    fn shebang_crlf_line_ending() {
        let (interp, arg) = parse_shebang(b"#!/bin/sh\r\n").unwrap();
        assert_eq!(interp, "/bin/sh");
        assert_eq!(arg, None);
    }

    #[test]
    fn shebang_with_spaces_before_interp() {
        // Leading space after `#!` is unusual but we trim it.
        let (interp, arg) = parse_shebang(b"#! /bin/sh\n").unwrap();
        assert_eq!(interp, "/bin/sh");
        assert_eq!(arg, None);
    }

    #[test]
    fn shebang_arg_spaces_trimmed() {
        // Multiple spaces between tokens: only first whitespace-run is the
        // separator; everything after it is the single optional arg, trimmed.
        let (interp, arg) = parse_shebang(b"#!/usr/bin/env  python3\n").unwrap();
        assert_eq!(interp, "/usr/bin/env");
        // The remainder " python3" trimmed is "python3".
        assert_eq!(arg, Some("python3"));
    }

    #[test]
    fn shebang_not_present() {
        assert!(parse_shebang(b"\x7fELF").is_none());
        assert!(parse_shebang(b"hello world").is_none());
    }

    #[test]
    fn shebang_too_short() {
        assert!(parse_shebang(b"").is_none());
        assert!(parse_shebang(b"#").is_none());
    }

    #[test]
    fn shebang_empty_interpreter() {
        // `#!` with only whitespace on the line → no valid interpreter.
        assert!(parse_shebang(b"#!   \n").is_none());
        assert!(parse_shebang(b"#!\n").is_none());
    }
}

/// Set the calling thread's user TLS base (FS_BASE on x86_64) to `base`.
///
/// The change is applied to hardware immediately and is preserved across
/// context switches.  Returns an error if `base` is a non-canonical address.
pub fn task_set_tls_base(base: usize) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_TASK_SET_TLS_BASE, base, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Return the calling thread's current user TLS base (FS_BASE on x86_64).
pub fn task_get_tls_base() -> Result<usize, Errno> {
    let ret = unsafe { raw_syscall6(SYS_TASK_GET_TLS_BASE, 0, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as usize)
}

/// Set the calling thread's human-readable name.
///
/// The name is stored in the kernel's thread record and is visible in
/// `/proc/<pid>/task/<tid>/name`.  Names longer than 31 bytes are
/// silently truncated by the kernel.
pub fn task_set_name(name: &[u8]) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_TASK_SET_NAME,
            name.as_ptr() as usize,
            name.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

pub fn task_interrupt(tid: u64) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_TASK_INTERRUPT, tid as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

pub fn spawn_process_ex(
    name: &str,
    argv: &[&[u8]],
    env: &BTreeMap<Vec<u8>, Vec<u8>>,
    stdin_mode: u32,
    stdout_mode: u32,
    stderr_mode: u32,
    boot_arg: u64,
    handles: &[u64],
) -> Result<abi::types::SpawnProcessExResp, Errno> {
    spawn_process_ex_cwd(
        name,
        argv,
        env,
        stdin_mode,
        stdout_mode,
        stderr_mode,
        boot_arg,
        handles,
        None,
    )
}

pub fn spawn_process_ex_cwd(
    name: &str,
    argv: &[&[u8]],
    env: &BTreeMap<Vec<u8>, Vec<u8>>,
    stdin_mode: u32,
    stdout_mode: u32,
    stderr_mode: u32,
    boot_arg: u64,
    handles: &[u64],
    cwd: Option<&str>,
) -> Result<abi::types::SpawnProcessExResp, Errno> {
    let argv_blob = serialize_argv(argv);
    let env_blob = serialize_env(env);

    let mut h_to_inherit = [0u64; 8];
    let num_inherited = handles.len().min(8);
    for i in 0..num_inherited {
        h_to_inherit[i] = handles[i];
    }

    let req = abi::types::SpawnProcessExReq {
        name_ptr: name.as_ptr() as u64,
        name_len: name.len() as u32,
        _pad0: 0,
        argv_ptr: argv_blob.as_ptr() as u64,
        argv_len: argv_blob.len() as u32,
        _pad1: 0,
        env_ptr: env_blob.as_ptr() as u64,
        env_len: env_blob.len() as u32,
        _pad2: 0,
        stdin_mode,
        stdout_mode,
        stderr_mode,
        _reserved: 0,
        boot_arg,
        handles_to_inherit: h_to_inherit,
        num_inherited_handles: num_inherited as u32,
        _pad3: 0,
        cwd_ptr: cwd.map_or(0, |s| s.as_ptr() as u64),
        cwd_len: cwd.map_or(0, |s| s.len() as u32),
        _pad4: 0,
        fd_remap_ptr: 0,
        fd_remap_len: 0,
        _pad5: 0,
    };
    // SAFETY: `req`, `argv_blob`, `env_blob`, and `cwd` all live on the stack
    // until after `raw_syscall6` returns.  The kernel copies all pointer fields
    // synchronously during the syscall, so there is no use-after-free risk.

    let mut resp = abi::types::SpawnProcessExResp::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_SPAWN_PROCESS_EX,
            &req as *const _ as usize,
            &mut resp as *mut _ as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| resp)
}

fn serialize_argv(argv: &[&[u8]]) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    for arg in argv {
        blob.extend_from_slice(&(arg.len() as u32).to_le_bytes());
        blob.extend_from_slice(arg);
    }
    blob
}

fn serialize_env(env: &BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(env.len() as u32).to_le_bytes());
    for (key, value) in env {
        blob.extend_from_slice(&(key.len() as u32).to_le_bytes());
        blob.extend_from_slice(key);
        blob.extend_from_slice(&(value.len() as u32).to_le_bytes());
        blob.extend_from_slice(value);
    }
    blob
}

pub fn set_priority(tid: u64, priority: usize) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_SET_PRIORITY, tid as usize, priority, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

pub fn alloc_stack(pages: usize) -> Result<usize, Errno> {
    let ret = unsafe { raw_syscall6(SYS_ALLOC_STACK, pages, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret)
}

pub fn spawn_thread(entry: usize, arg: usize, stack: &crate::stack::Stack) -> Result<u64, Errno> {
    spawn_thread_ex(entry, arg, stack, 0, 0)
}

/// Extended thread spawn with explicit TLS base and flags.
///
/// - `tls_base`: initial value for the thread-local storage base register
///   (FS_BASE on x86_64).  Pass `0` to leave the register in its default
///   initial state.
/// - `flags`: bitmask of [`abi::types::spawn_thread_flags`] constants.
///   Use `DETACHED` to create a thread that cannot be joined.
pub fn spawn_thread_ex(
    entry: usize,
    arg: usize,
    stack: &crate::stack::Stack,
    tls_base: usize,
    flags: u32,
) -> Result<u64, Errno> {
    let req = abi::types::SpawnThreadReq {
        entry,
        sp: stack.sp as usize,
        arg,
        stack: stack.info,
        tls_base,
        flags,
        _pad: 0,
    };
    let ret = unsafe {
        raw_syscall6(
            SYS_SPAWN_THREAD,
            &req as *const abi::types::SpawnThreadReq as usize,
            0,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u64)
}

pub fn task_poll(pid: u64) -> Result<(abi::types::TaskStatus, i32), Errno> {
    let ret = unsafe { raw_syscall6(abi::syscall::SYS_TASK_POLL, pid as usize, 0, 0, 0, 0, 0) };
    if ret < 0 {
        abi::errors::errno(ret).map(|_| (abi::types::TaskStatus::Unknown, 0))
    } else {
        let val = ret as u64;
        let status_val = val & 0xFFFFFFFF;
        let code_val = (val >> 32) as i32;
        let status = match status_val {
            0 => abi::types::TaskStatus::Unknown,
            1 => abi::types::TaskStatus::Runnable,
            2 => abi::types::TaskStatus::Running,
            3 => abi::types::TaskStatus::Blocked,
            4 => abi::types::TaskStatus::Dead,
            _ => abi::types::TaskStatus::Unknown,
        };
        Ok((status, code_val))
    }
}

pub fn time_anchor(unix_secs: u64) {
    unsafe {
        raw_syscall6(SYS_TIME_ANCHOR, unix_secs as usize, 0, 0, 0, 0, 0);
    }
}

pub fn ioport_read(port: usize, width: usize) -> usize {
    let ret = unsafe { raw_syscall6(SYS_DEVICE_IOPORT_READ, port, width, 0, 0, 0, 0) };
    if ret < 0 {
        0
    } else {
        ret as usize
    }
}

pub fn ioport_write(port: usize, value: usize, width: usize) {
    unsafe {
        raw_syscall6(SYS_DEVICE_IOPORT_WRITE, port, value, width, 0, 0, 0);
    }
}

pub fn task_wait(tid: u64) -> Result<i32, Errno> {
    let ret = unsafe { raw_syscall6(SYS_TASK_WAIT, tid as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as i32)
}

/// Wait for a child process to exit, analogous to POSIX `waitpid`.
///
/// - `pid > 0`: wait for the specific child with that PID.
/// - `pid == -1` or `pid == 0`: wait for any child.
/// - `flags`: combine `WNOHANG`, `WUNTRACED`, and `WCONTINUED` as needed.
///
/// On success returns `(child_pid, wait_status)`. With `WNOHANG` and no matching
/// child state change ready, returns `Ok((0, 0))`. Returns `Err(ECHILD)` when no
/// matching children exist.
pub fn waitpid(pid: i64, flags: u32) -> Result<(i64, i32), Errno> {
    let mut status: i32 = 0;
    let ret = unsafe {
        raw_syscall6(
            SYS_WAITPID,
            pid as usize,
            &mut status as *mut i32 as usize,
            flags as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| (v as i64, status))
}

/// Kill a task by TID. Returns Ok(()) if the task was killed, Err(ESRCH) if not found.
pub fn task_kill(tid: u64) -> Result<(), Errno> {
    let ret = unsafe { raw_syscall6(SYS_TASK_KILL, tid as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|_| ())
}

/// Dump all task information to the kernel serial console (like `top`).
pub fn task_dump() {
    unsafe {
        raw_syscall6(SYS_TASK_DUMP, 0, 0, 0, 0, 0, 0);
    }
}

// ...
pub fn trace_read(buf: &mut [abi::trace::TraceEvent]) -> Result<usize, Errno> {
    let ret = unsafe {
        match raw_syscall6(
            abi::syscall::SYS_TRACE_READ,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
            0,
        ) {
            r if r < 0 => return Err(core::mem::transmute(-(r as i32))),
            r => r as usize,
        }
    };
    Ok(ret)
}

/// Disable the boot console (call when compositor takes over framebuffer)
pub fn console_disable() {
    unsafe {
        raw_syscall6(SYS_CONSOLE_DISABLE, 0, 0, 0, 0, 0, 0);
    }
}

// Device MMIO and DMA syscalls

/// Claim a device identified by its sysfs path (e.g. `/sys/devices/pci-0000:00:1f.2`).
///
/// The kernel resolves the path to a device in the registry and returns a
/// capability handle. Use the handle for `device_map_mmio`, `device_alloc_dma`,
/// and related calls.
pub fn device_claim(path: &str) -> Result<usize, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_DEVICE_CLAIM,
            path.as_ptr() as usize,
            path.len(),
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret)
}

/// Map a device MMIO BAR into memory
/// Returns the virtual address where the BAR is mapped
pub fn device_map_mmio(claim_handle: usize, bar_index: usize) -> Result<u64, Errno> {
    let ret = unsafe { raw_syscall6(SYS_DEVICE_MAP_MMIO, claim_handle, bar_index, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u64)
}

/// Allocate DMA-safe memory for a device
/// Returns the virtual address of the allocated buffer
pub fn device_alloc_dma(claim_handle: usize, page_count: usize) -> Result<u64, Errno> {
    let ret = unsafe { raw_syscall6(SYS_DEVICE_ALLOC_DMA, claim_handle, page_count, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u64)
}

/// Get the physical address for a DMA virtual address
pub fn device_dma_phys(virt_addr: u64) -> Result<u64, Errno> {
    let ret = unsafe { raw_syscall6(SYS_DEVICE_DMA_PHYS, virt_addr as usize, 0, 0, 0, 0, 0) };
    abi::errors::errno(ret).map(|v| v as u64)
}

// ============================================================================
// IRQ Subscription and Waiting
// ============================================================================

/// Subscribe the current task to receive interrupts for a given vector.
/// For PS/2 keyboard: vector 0x21 (IRQ1), PS/2 mouse: vector 0x2C (IRQ12)
pub fn irq_subscribe(vector: u8) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_DEVICE_IRQ_SUBSCRIBE,
            vector as usize,
            0,
            DEVICE_IRQ_SUBSCRIBE_VECTOR as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

pub fn device_irq_subscribe(claim_handle: usize, irq_index: u8) -> Result<(), Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_DEVICE_IRQ_SUBSCRIBE,
            claim_handle,
            irq_index as usize,
            DEVICE_IRQ_SUBSCRIBE_DEVICE as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

/// Wait for an interrupt on the given vector. Blocks until interrupt fires.
/// Returns the number of pending interrupts since last wait.
pub fn irq_wait(vector: u8) -> Result<u32, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_DEVICE_IRQ_WAIT,
            vector as usize,
            0,
            DEVICE_IRQ_SUBSCRIBE_VECTOR as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

pub fn device_irq_wait(claim_handle: usize, irq_index: u8) -> Result<u32, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_DEVICE_IRQ_WAIT,
            claim_handle,
            irq_index as usize,
            DEVICE_IRQ_SUBSCRIBE_DEVICE as usize,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

pub fn memfd_create(name: &str, size: usize) -> Result<u32, Errno> {
    let ret = unsafe {
        raw_syscall6(
            SYS_MEMFD_CREATE,
            name.as_ptr() as usize,
            name.len(),
            size,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|v| v as u32)
}

pub fn memfd_phys(fd: u32) -> Result<u64, Errno> {
    let ret = unsafe {
        match raw_syscall6(SYS_MEMFD_PHYS, fd as usize, 0, 0, 0, 0, 0) {
            r if r < 0 => return Err(core::mem::transmute(-(r as i32))),
            r => r as u64,
        }
    };
    Ok(ret)
}

// ============================================================================
// Entropy / Random
// ============================================================================

/// Fill `buf` with random bytes from the kernel entropy pool.
pub fn getrandom(buf: &mut [u8]) -> Result<(), Errno> {
    let mut offset = 0;
    while offset < buf.len() {
        let chunk = &mut buf[offset..];
        let ret = unsafe {
            raw_syscall6(
                SYS_GETRANDOM,
                chunk.as_mut_ptr() as usize,
                chunk.len(),
                0,
                0,
                0,
                0,
            )
        };
        if ret < 0 {
            return Err(unsafe { core::mem::transmute(-(ret as i32)) });
        }
        // Kernel caps at 256 bytes per call, advance by the amount requested
        // (we know the kernel filled min(len, 256))
        let filled = chunk.len().min(256);
        offset += filled;
    }
    Ok(())
}

/// Mix `buf` bytes into the kernel entropy pool and mark it seeded.
///
/// This is the entropy analogue of [`time_anchor`]: a privileged entropy-source
/// driver collects hardware randomness and calls this to seed the kernel CSPRNG.
/// At most 256 bytes are consumed per call; call in a loop for larger inputs.
pub fn entropy_seed(buf: &[u8]) {
    let mut offset = 0;
    while offset < buf.len() {
        let chunk = &buf[offset..];
        let len = chunk.len().min(256);
        unsafe {
            raw_syscall6(SYS_ENTROPY_SEED, chunk.as_ptr() as usize, len, 0, 0, 0, 0);
        }
        offset += len;
    }
}

pub fn vm_map(req: &abi::vm::VmMapReq) -> Result<abi::vm::VmMapResp, Errno> {
    let mut resp = abi::vm::VmMapResp { addr: 0, len: 0 };
    let req_ptr = req as *const _ as usize;
    let resp_ptr = &mut resp as *mut _ as usize;
    let ret = unsafe { raw_syscall6(SYS_VM_MAP, req_ptr, resp_ptr, 0, 0, 0, 0) };
    if ret < 0 {
        Err(unsafe { core::mem::transmute(-(ret as i32)) })
    } else {
        Ok(resp)
    }
}

pub fn vm_unmap(addr: usize, len: usize) -> Result<(), Errno> {
    let req = abi::vm::VmUnmapReq { addr, len };
    let mut resp = abi::vm::VmUnmapResp::default();
    let ret = unsafe {
        raw_syscall6(
            SYS_VM_UNMAP,
            &req as *const _ as usize,
            &mut resp as *mut _ as usize,
            0,
            0,
            0,
            0,
        )
    };
    abi::errors::errno(ret).map(|_| ())
}

pub fn root_watch_open(_spec: &abi::types::WatchSpec) -> Result<u32, Errno> {
    Err(Errno::ENOSYS)
}

pub fn root_watch_try_next(_handle: u32, _seq: &mut u64, _buf: &mut [u8]) -> Result<usize, Errno> {
    Err(Errno::ENOSYS)
}

pub fn root_watch_close(_handle: u32) -> Result<(), Errno> {
    Err(Errno::ENOSYS)
}
