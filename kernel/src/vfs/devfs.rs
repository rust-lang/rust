//! devfs — kernel-native device filesystem mounted at `/dev`.
//!
//! Provides a minimal set of built-in device nodes plus a **runtime
//! registration** mechanism so that kernel subsystems and drivers can add
//! new device entries without modifying this file.
//!
//! # Built-in nodes
//!
//! | Path            | Kind     | Description                              |
//! |-----------------|----------|------------------------------------------|
//! | `/dev/console`  | char     | Writes go to the boot console; reads from the per-process console input queue |
//! | `/dev/null`     | char     | Discards writes; returns EOF on reads    |
//! | `/dev/zero`     | char     | Returns zero bytes; discards writes      |
//!
//! # Extensibility
//! Additional device nodes can be registered at runtime via the global
//! [`register`] function:
//!
//! ```ignore
//! devfs::register("ttyS0", Arc::new(my_uart_node));
//! ```
//!
//! Registered nodes are consulted **before** the built-in match, so they can
//! shadow built-in names when needed (last registration wins).  The global
//! registry is protected by a spin-lock.

use abi::errors::{Errno, SysResult};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use spin::Mutex;

use super::{VfsDriver, VfsNode, VfsStat};

// ── Global device registry ────────────────────────────────────────────────────

/// Global table of dynamically registered `/dev` entries.
///
/// Keys are bare device names (no leading `/dev/`).  The table is consulted
/// *after* the built-in match, so built-in names (`null`, `zero`, `console`)
/// can still be overridden if needed.
static DEVICE_REGISTRY: Mutex<BTreeMap<String, Arc<dyn VfsNode>>> = Mutex::new(BTreeMap::new());
static BOOT_FB_INFO: Mutex<Option<(crate::FramebufferInfo, u64)>> = Mutex::new(None);

/// Register a device node under the name `name` in `/dev`.
///
/// The `name` must be the bare device name, e.g. `"ttyS0"` (not `/dev/ttyS0`).
/// If a node with the same name was previously registered, it is replaced.
///
/// # Example
/// ```ignore
/// use kernel::vfs::devfs;
/// devfs::register("ttyS0", Arc::new(UartNode::new()));
/// ```
pub fn register(name: &str, node: Arc<dyn VfsNode>) {
    DEVICE_REGISTRY.lock().insert(name.to_string(), node);
}

/// Remove a previously registered device node.
///
/// Returns `true` if a node was found and removed, `false` if the name was
/// not registered.
pub fn unregister(name: &str) -> bool {
    DEVICE_REGISTRY.lock().remove(name).is_some()
}

pub fn set_boot_fb(fb: crate::FramebufferInfo, resource_id: u64) {
    crate::kinfo!(
        "devfs: set_boot_fb width={} height={} pitch={} resource_id=0x{:x}",
        fb.width,
        fb.height,
        fb.pitch,
        resource_id
    );
    *BOOT_FB_INFO.lock() = Some((fb, resource_id));
}

// ── DevFs driver ─────────────────────────────────────────────────────────────

/// The device filesystem driver.  Mounted at `/dev` by `vfs::init`.
pub struct DevFs;

impl DevFs {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DevFs {
    fn default() -> Self {
        Self::new()
    }
}

impl VfsDriver for DevFs {
    fn lookup(&self, path: &str) -> SysResult<Arc<dyn VfsNode>> {
        if path == "fb0" || path.starts_with("fb") {
            crate::kdebug!("devfs: lookup entry path='{}' len={}", path, path.len());
        }
        // Empty path → the /dev directory node itself.
        if path.is_empty() {
            return Ok(Arc::new(DevDirNode));
        }

        // Check the dynamic registry first; registered nodes take precedence
        // over built-in names, allowing callers to override defaults.
        {
            let reg = DEVICE_REGISTRY.lock();
            if let Some(node) = reg.get(path) {
                if path == "fb0" || path.starts_with("fb") {
                    crate::kdebug!("devfs: dynamic registry hit path='{}'", path);
                }
                return Ok(node.clone());
            }
        }

        // Handle synthetic subdirectories
        match path {
            "display" => return Ok(Arc::new(DevSubDirNode::new("display/"))),
            "input" => return Ok(Arc::new(DevSubDirNode::new("input/"))),
            "audio" => return Ok(Arc::new(DevSubDirNode::new("audio/"))),
            _ => {}
        }

        // Fall back to built-in nodes.
        match path {
            "console" => Ok(Arc::new(ConsoleNode)),
            "null" => Ok(Arc::new(NullNode)),
            "zero" => Ok(Arc::new(ZeroNode)),
            "fb0" => {
                if let Some((fb, resource_id)) = *BOOT_FB_INFO.lock() {
                    crate::kdebug!(
                        "devfs: lookup fb0 -> hit ({}x{} stride={})",
                        fb.width,
                        fb.height,
                        fb.pitch
                    );
                    Ok(Arc::new(FbNode::new(fb, resource_id)))
                } else {
                    crate::kwarn!("devfs: lookup fb0 -> missing boot fb state");
                    Err(Errno::ENOENT)
                }
            }
            "rtc" => Ok(Arc::new(RtcNode)),
            "random" => Ok(Arc::new(RandomNode)),
            "urandom" => Ok(Arc::new(UrandomNode)),
            "kmsg" => Ok(Arc::new(KmsgNode)),
            _ => Err(Errno::ENOENT),
        }
    }
}

// ── Synthetic subdirectory node ──────────────────────────────────────────────

struct DevSubDirNode {
    prefix: String,
}

impl DevSubDirNode {
    fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }
}

impl VfsNode for DevSubDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o755,
            size: 0,
            ino: 101, // arbitrary
            nlink: 2,
            ..Default::default()
        })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let mut names = Vec::new();
        {
            let reg = DEVICE_REGISTRY.lock();
            for name in reg.keys() {
                if name.starts_with(&self.prefix) {
                    let subname = &name[self.prefix.len()..];
                    if !subname.is_empty() && !subname.contains('/') {
                        names.push(subname.to_string());
                    }
                }
            }
        }
        super::write_readdir_entries(names.iter().map(|s| s.as_str()), offset, buf)
    }
}

// ── /dev directory node ───────────────────────────────────────────────────────

/// Directory node for `/dev` itself.
struct DevDirNode;

impl VfsNode for DevDirNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EISDIR)
    }
    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFDIR | 0o755,
            size: 0,
            ino: 100,
            nlink: 2,
            ..Default::default()
        })
    }
    fn readdir(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        let mut names = alloc::vec![
            "console".to_string(),
            "null".to_string(),
            "zero".to_string()
        ];
        if BOOT_FB_INFO.lock().is_some() {
            names.push("fb0".to_string());
        }
        names.push("display".to_string());
        names.push("input".to_string());
        names.push("audio".to_string());
        names.push("rtc".to_string());
        names.push("random".to_string());
        names.push("urandom".to_string());
        names.push("kmsg".to_string());
        {
            let reg = DEVICE_REGISTRY.lock();
            for name in reg.keys() {
                if !matches!(
                    name.as_str(),
                    "console" | "null" | "zero" | "fb0" | "rtc" | "random" | "urandom"
                ) {
                    names.push(name.clone());
                }
            }
        }
        super::write_readdir_entries(names.iter().map(|s: &String| s.as_str()), offset, buf)
    }
}

static CONSOLE_BUF: Mutex<alloc::collections::VecDeque<u8>> =
    Mutex::new(alloc::collections::VecDeque::new());

/// Runtime tty state for `/dev/console`.
#[derive(Clone, Copy)]
struct ConsoleTtyState {
    termios: abi::termios::Termios,
    controlling_sid: Option<u32>,
    foreground_pgid: Option<u32>,
}

impl Default for ConsoleTtyState {
    fn default() -> Self {
        Self {
            termios: abi::termios::DEFAULT_TERMIOS,
            controlling_sid: None,
            foreground_pgid: None,
        }
    }
}

#[derive(Clone, Copy)]
struct ConsoleCaller {
    sid: u32,
    pgid: u32,
    session_leader: bool,
}

/// Global tty state for `/dev/console`.
static CONSOLE_TTY_STATE: Mutex<ConsoleTtyState> = Mutex::new(ConsoleTtyState {
    termios: abi::termios::DEFAULT_TERMIOS,
    controlling_sid: None,
    foreground_pgid: None,
});

/// Character device node for `/dev/console`.
///
/// - **write**: each byte is forwarded to the kernel's boot console via
///   [`crate::runtime_base()`].
/// - **read**: reads from the boot console, honouring the current termios
///   settings (canonical vs. raw mode, echo, ISIG, etc.).  Blocks (yields)
///   until data is available, and returns `EINTR` if a pending interrupt is
///   detected or if Ctrl-C is received while `ISIG` is set.
/// - **device_call**: supports `TERMINAL_OP_TCGETS` and `TERMINAL_OP_TCSETS`
///   to query/update the termios settings from userspace.
pub struct ConsoleNode;

impl ConsoleNode {
    fn current_caller() -> Option<ConsoleCaller> {
        let pinfo = crate::sched::process_info_current()?;
        let p = pinfo.lock();
        Some(ConsoleCaller {
            sid: p.sid,
            pgid: p.pgid,
            session_leader: p.session_leader,
        })
    }

    fn maybe_acquire_controlling_tty(state: &mut ConsoleTtyState, caller: Option<ConsoleCaller>) {
        if state.controlling_sid.is_some() {
            return;
        }
        if let Some(c) = caller
            && c.session_leader
        {
            state.controlling_sid = Some(c.sid);
            state.foreground_pgid = Some(c.pgid);
        }
    }

    fn is_background_caller(state: &ConsoleTtyState, caller: ConsoleCaller) -> bool {
        match (state.controlling_sid, state.foreground_pgid) {
            (Some(sid), Some(fg_pgid)) => caller.sid == sid && caller.pgid != fg_pgid,
            _ => false,
        }
    }

    fn enforce_job_control_before_read() -> SysResult<()> {
        let caller = match Self::current_caller() {
            Some(c) => c,
            None => return Ok(()),
        };
        let (is_background, controlling_sid, foreground_pgid) = {
            let mut state = CONSOLE_TTY_STATE.lock();
            Self::maybe_acquire_controlling_tty(&mut state, Some(caller));
            (
                Self::is_background_caller(&state, caller),
                state.controlling_sid,
                state.foreground_pgid,
            )
        };
        if is_background {
            crate::kwarn!(
                "console read rejected by job control: pgid={} sid={} leader={} tty_sid={:?} tty_fg={:?}",
                caller.pgid,
                caller.sid,
                caller.session_leader,
                controlling_sid,
                foreground_pgid
            );
            crate::signal::send_signal_to_group(caller.pgid, abi::signal::SIGTTIN);
            return Err(Errno::EINTR);
        }
        Ok(())
    }

    fn enforce_job_control_before_write() -> SysResult<()> {
        let caller = match Self::current_caller() {
            Some(c) => c,
            None => return Ok(()),
        };
        let (is_background, controlling_sid, foreground_pgid) = {
            let mut state = CONSOLE_TTY_STATE.lock();
            Self::maybe_acquire_controlling_tty(&mut state, Some(caller));
            (
                Self::is_background_caller(&state, caller),
                state.controlling_sid,
                state.foreground_pgid,
            )
        };
        if is_background {
            crate::kwarn!(
                "console write rejected by job control: pgid={} sid={} leader={} tty_sid={:?} tty_fg={:?}",
                caller.pgid,
                caller.sid,
                caller.session_leader,
                controlling_sid,
                foreground_pgid
            );
            crate::signal::send_signal_to_group(caller.pgid, abi::signal::SIGTTOU);
            return Err(Errno::EINTR);
        }
        Ok(())
    }

    /// Return a copy of the current termios settings.
    pub fn get_termios() -> abi::termios::Termios {
        CONSOLE_TTY_STATE.lock().termios
    }

    /// Replace the current termios settings.
    pub fn set_termios(t: abi::termios::Termios) {
        CONSOLE_TTY_STATE.lock().termios = t;
    }

    #[cfg(test)]
    fn set_tty_owner_for_test(sid: Option<u32>, fg_pgid: Option<u32>) {
        let mut st = CONSOLE_TTY_STATE.lock();
        st.controlling_sid = sid;
        st.foreground_pgid = fg_pgid;
    }
}

impl VfsNode for ConsoleNode {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        use abi::termios::{ECHO, ECHOE, ICANON, ICRNL, ISIG, VINTR, VMIN, VQUIT, VSUSP};

        if buf.is_empty() {
            return Ok(0);
        }
        Self::enforce_job_control_before_read()?;
        let mut read_bytes = 0;

        loop {
            // ── Check for a pending interrupt (e.g. from SYS_TASK_INTERRUPT) ──
            if crate::sched::take_pending_interrupt_current() {
                return Err(abi::errors::Errno::EINTR);
            }

            let rt = crate::runtime_base();

            // Snapshot current terminal flags and special characters so we are
            // consistent across one drain + one dequeue pass.
            let tty_state = CONSOLE_TTY_STATE.lock();
            let termios = tty_state.termios;
            let foreground_pgid = tty_state.foreground_pgid;
            drop(tty_state);

            let canonical = termios.c_lflag & ICANON != 0;
            let do_echo = termios.c_lflag & ECHO != 0;
            let do_echo_erase = termios.c_lflag & ECHOE != 0;
            let isig = termios.c_lflag & ISIG != 0;
            let icrnl = termios.c_iflag & ICRNL != 0;

            // Get configured signal characters
            let vintr = termios.c_cc[VINTR];
            let vquit = termios.c_cc[VQUIT];
            let vsusp = termios.c_cc[VSUSP];

            // ── Drain hardware FIFO into the software buffer ──────────────────
            while let Some(c) = rt.getchar() {
                // ── Check for signal-generating characters ────────────────────
                if isig {
                    if c == vintr {
                        // SIGINT (interrupt) character
                        if do_echo {
                            rt.putchar(b'^');
                            rt.putchar(b'C');
                            rt.putchar(b'\r');
                            rt.putchar(b'\n');
                        }
                        CONSOLE_BUF.lock().clear();
                        if let Some(pgid) = foreground_pgid {
                            crate::signal::send_signal_to_group(pgid, abi::signal::SIGINT);
                        }
                        return Err(abi::errors::Errno::EINTR);
                    } else if c == vquit {
                        // SIGQUIT (quit) character
                        if do_echo {
                            rt.putchar(b'^');
                            rt.putchar(b'\\');
                            rt.putchar(b'\r');
                            rt.putchar(b'\n');
                        }
                        CONSOLE_BUF.lock().clear();
                        if let Some(pgid) = foreground_pgid {
                            crate::signal::send_signal_to_group(pgid, abi::signal::SIGQUIT);
                        }
                        return Err(abi::errors::Errno::EINTR);
                    } else if c == vsusp {
                        // SIGTSTP (suspend) character
                        if do_echo {
                            rt.putchar(b'^');
                            rt.putchar(b'Z');
                            rt.putchar(b'\r');
                            rt.putchar(b'\n');
                        }
                        CONSOLE_BUF.lock().clear();
                        if let Some(pgid) = foreground_pgid {
                            crate::signal::send_signal_to_group(pgid, abi::signal::SIGTSTP);
                        }
                        return Err(abi::errors::Errno::EINTR);
                    }
                }

                // ── Regular character processing ──────────────────────────────
                match c {
                    b'\r' | b'\n' => {
                        let mapped = if icrnl { b'\n' } else { c };
                        if do_echo {
                            rt.putchar(b'\r');
                            rt.putchar(b'\n');
                        }
                        CONSOLE_BUF.lock().push_back(mapped);
                    }
                    0x08 | 0x7f => {
                        // Backspace / DEL
                        if canonical {
                            let mut cb = CONSOLE_BUF.lock();
                            let last = cb.back().copied();
                            if last.is_some() && last != Some(b'\n') {
                                cb.pop_back();
                                if do_echo && do_echo_erase {
                                    rt.putchar(0x08);
                                    rt.putchar(b' ');
                                    rt.putchar(0x08);
                                }
                            }
                        } else {
                            CONSOLE_BUF.lock().push_back(c);
                        }
                    }
                    0x04 => {
                        // Ctrl-D (EOF in canonical mode)
                        if canonical {
                            CONSOLE_BUF.lock().push_back(0x04);
                        } else {
                            CONSOLE_BUF.lock().push_back(c);
                        }
                    }
                    0x20..=0x7e => {
                        if do_echo {
                            rt.putchar(c);
                        }
                        CONSOLE_BUF.lock().push_back(c);
                    }
                    _ => {
                        // In raw mode pass all bytes through; in canonical
                        // mode silently discard control chars we don't handle.
                        if !canonical {
                            CONSOLE_BUF.lock().push_back(c);
                        }
                    }
                }
            }

            // ── Check whether enough data is available to satisfy the read ───
            let vmin = termios.c_cc[VMIN] as usize;
            let vmin_eff = vmin.max(1);

            let mut cb = CONSOLE_BUF.lock();
            let ready = if canonical {
                // Canonical: a full line (terminated by NL or special char)
                // is required, or the buffer is at least as large as `buf`.
                cb.iter().any(|&b| b == b'\n' || b == 0x04) || cb.len() >= buf.len()
            } else {
                // Raw: VMIN bytes must be available.
                cb.len() >= vmin_eff || cb.len() >= buf.len()
            };

            if ready {
                while read_bytes < buf.len() {
                    if let Some(b) = cb.pop_front() {
                        buf[read_bytes] = b;
                        read_bytes += 1;
                        if canonical && (b == b'\n' || b == 0x04) {
                            // Line complete.
                            break;
                        }
                        if !canonical && read_bytes >= vmin_eff {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                return Ok(read_bytes);
            }
            drop(cb);

            unsafe { crate::sched::yield_now_current() };
        }
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> SysResult<usize> {
        Self::enforce_job_control_before_write()?;
        let rt = crate::runtime_base();
        for &b in buf {
            if b == b'\n' {
                rt.putchar(b'\r');
            }
            rt.putchar(b);
        }
        Ok(buf.len())
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 1,
            nlink: 1,
            rdev: VfsStat::makedev(5, 1),
            ..Default::default()
        })
    }

    fn is_tty(&self) -> bool {
        true
    }

    /// Device-specific control for the console terminal.
    ///
    /// Supported operations (set `kind = DeviceKind::Terminal`):
    ///
    /// | `op`                    | Direction | Description                    |
    /// |-------------------------|-----------|--------------------------------|
    /// | `TERMINAL_OP_TCGETS`    | out       | Copy termios → `out_ptr`       |
    /// | `TERMINAL_OP_TCSETS`    | in        | Copy `in_ptr` → termios        |
    /// | `TERMINAL_OP_TCSETSW`   | in        | Same as `TCSETS` (no drain)    |
    /// | `TERMINAL_OP_TCSETSF`   | in        | Same as `TCSETS` (no flush)    |
    fn device_call(&self, call: &abi::device::DeviceCall) -> SysResult<usize> {
        use abi::device::DeviceKind;
        use abi::termios::{
            TERMINAL_OP_TCGETPGRP, TERMINAL_OP_TCGETS, TERMINAL_OP_TCSETPGRP, TERMINAL_OP_TCSETS,
            TERMINAL_OP_TCSETSF, TERMINAL_OP_TCSETSW,
        };

        if call.kind != DeviceKind::Terminal {
            return Err(abi::errors::Errno::ENOSYS);
        }

        let termios_size = core::mem::size_of::<abi::termios::Termios>();
        let pgid_size = core::mem::size_of::<u32>();

        let caller = Self::current_caller();
        {
            let mut st = CONSOLE_TTY_STATE.lock();
            Self::maybe_acquire_controlling_tty(&mut st, caller);
        }

        match call.op {
            TERMINAL_OP_TCGETS => {
                // Write current termios to userspace out_ptr.
                if call.out_len < termios_size as u32 || call.out_ptr == 0 {
                    return Err(abi::errors::Errno::EINVAL);
                }
                let termios = CONSOLE_TTY_STATE.lock().termios;
                let bytes = unsafe {
                    core::slice::from_raw_parts(
                        &termios as *const abi::termios::Termios as *const u8,
                        termios_size,
                    )
                };
                unsafe {
                    crate::syscall::validate::copyout(call.out_ptr as usize, bytes)?;
                }
                Ok(0)
            }
            TERMINAL_OP_TCSETS | TERMINAL_OP_TCSETSW | TERMINAL_OP_TCSETSF => {
                // Read new termios from userspace in_ptr.
                if call.in_len < termios_size as u32 || call.in_ptr == 0 {
                    return Err(abi::errors::Errno::EINVAL);
                }
                let mut new_termios = abi::termios::Termios::default();
                let bytes = unsafe {
                    core::slice::from_raw_parts_mut(
                        &mut new_termios as *mut abi::termios::Termios as *mut u8,
                        termios_size,
                    )
                };
                unsafe {
                    crate::syscall::validate::copyin(bytes, call.in_ptr as usize)?;
                }
                CONSOLE_TTY_STATE.lock().termios = new_termios;
                Ok(0)
            }
            TERMINAL_OP_TCGETPGRP => {
                if call.out_len < pgid_size as u32 || call.out_ptr == 0 {
                    return Err(abi::errors::Errno::EINVAL);
                }

                let caller = caller.ok_or(abi::errors::Errno::ENOTTY)?;
                let fg_pgid = {
                    let st = CONSOLE_TTY_STATE.lock();
                    if st.controlling_sid != Some(caller.sid) {
                        return Err(abi::errors::Errno::ENOTTY);
                    }
                    st.foreground_pgid.ok_or(abi::errors::Errno::ENOTTY)?
                };

                unsafe {
                    crate::syscall::validate::copyout(
                        call.out_ptr as usize,
                        core::slice::from_raw_parts(&fg_pgid as *const u32 as *const u8, pgid_size),
                    )?;
                }
                Ok(0)
            }
            TERMINAL_OP_TCSETPGRP => {
                if call.in_len < pgid_size as u32 || call.in_ptr == 0 {
                    return Err(abi::errors::Errno::EINVAL);
                }

                let caller = caller.ok_or(abi::errors::Errno::ENOTTY)?;
                let mut new_pgid = 0u32;
                unsafe {
                    crate::syscall::validate::copyin(
                        core::slice::from_raw_parts_mut(
                            &mut new_pgid as *mut u32 as *mut u8,
                            pgid_size,
                        ),
                        call.in_ptr as usize,
                    )?;
                }
                if new_pgid == 0 {
                    return Err(abi::errors::Errno::EINVAL);
                }

                {
                    let st = CONSOLE_TTY_STATE.lock();
                    if st.controlling_sid != Some(caller.sid) {
                        return Err(abi::errors::Errno::ENOTTY);
                    }
                }

                if !crate::signal::process_group_exists_in_session(new_pgid, caller.sid) {
                    return Err(abi::errors::Errno::EPERM);
                }

                CONSOLE_TTY_STATE.lock().foreground_pgid = Some(new_pgid);
                Ok(0)
            }
            _ => Err(abi::errors::Errno::ENOSYS),
        }
    }
}

// ── /dev/null ────────────────────────────────────────────────────────────────

/// Character device node for `/dev/null`.
/// Reads return EOF immediately; writes silently succeed.
pub struct NullNode;

impl VfsNode for NullNode {
    fn read(&self, _offset: u64, _buf: &mut [u8]) -> SysResult<usize> {
        Ok(0) // EOF
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> SysResult<usize> {
        Ok(buf.len()) // silently discard
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 2,
            nlink: 1,
            rdev: VfsStat::makedev(1, 3),
            ..Default::default()
        })
    }
}

// ── /dev/zero ────────────────────────────────────────────────────────────────

/// Character device node for `/dev/zero`.
/// Reads fill the buffer with zero bytes; writes succeed silently.
pub struct ZeroNode;

impl VfsNode for ZeroNode {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        buf.fill(0);
        Ok(buf.len())
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> SysResult<usize> {
        Ok(buf.len())
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 3,
            nlink: 1,
            rdev: VfsStat::makedev(1, 5),
            ..Default::default()
        })
    }
}

// ── /dev/fb0 ─────────────────────────────────────────────────────────────────

#[repr(C)]
pub struct FbNode {
    fb: crate::FramebufferInfo,
    resource_id: u64,
}

impl FbNode {
    pub const fn new(fb: crate::FramebufferInfo, resource_id: u64) -> Self {
        Self { fb, resource_id }
    }
}

impl VfsNode for FbNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        use abi::display_driver_protocol::{FB_INFO_PAYLOAD_SIZE, FbInfoPayload};

        let payload = FbInfoPayload {
            device_handle: self.resource_id,
            width: self.fb.width,
            height: self.fb.height,
            stride: self.fb.pitch,
            bpp: self.fb.bpp as u32,
            format: self.fb.format as u32,
            _reserved: 0,
        };

        let slice = unsafe {
            core::slice::from_raw_parts(&payload as *const _ as *const u8, FB_INFO_PAYLOAD_SIZE)
        };

        let off = offset as usize;
        if off >= slice.len() {
            crate::kwarn!(
                "FbNode::read: EOF (offset={} >= slice.len={})",
                off,
                slice.len()
            );
            return Ok(0);
        }

        let avail = &slice[off..];
        let n = avail.len().min(buf.len());
        crate::kdebug!(
            "FbNode::read: off={} n={} buf_len={} total={}",
            off,
            n,
            buf.len(),
            slice.len()
        );
        buf[..n].copy_from_slice(&avail[..n]);
        Ok(n)
    }

    fn write(&self, offset: u64, buf: &[u8]) -> SysResult<usize> {
        let off = offset as usize;
        if off as u64 >= self.fb.byte_len {
            return Ok(0);
        }

        let n = buf
            .len()
            .min((self.fb.byte_len.saturating_sub(off as u64)) as usize);
        if n == 0 {
            return Ok(0);
        }

        unsafe {
            core::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                (self.fb.addr as usize + off) as *mut u8,
                n,
            );
        }
        Ok(n)
    }

    fn stat(&self) -> SysResult<VfsStat> {
        use abi::display_driver_protocol::FB_INFO_PAYLOAD_SIZE;
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: FB_INFO_PAYLOAD_SIZE as u64,
            ino: 4,
            nlink: 1,
            rdev: VfsStat::makedev(29, 0),
            ..Default::default()
        })
    }

    fn phys_region(&self) -> SysResult<(u64, usize)> {
        Ok((self.fb.addr, self.fb.byte_len as usize))
    }
}

// ── /dev/rtc ─────────────────────────────────────────────────────────────────

pub struct RtcNode;

impl VfsNode for RtcNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        // Return unix seconds as a 64-bit value or text.
        // For compatibility with sprout: it expects properties, but since it's now path-based,
        // we can return a simple string or binary. Let's return the string first.
        let mono_ns = crate::runtime_base().mono_ticks() as u128 * 1_000_000_000
            / crate::runtime_base().mono_freq_hz() as u128;
        let sys_ns = if crate::time::is_anchored() {
            crate::time::get_system_time_ns(mono_ns as u64)
        } else {
            0
        };
        let text = format!("{}\n", sys_ns / 1_000_000_000);
        let slice = text.as_bytes();

        let off = offset as usize;
        if off >= slice.len() {
            return Ok(0);
        }
        let avail = &slice[off..];
        let n = avail.len().min(buf.len());
        buf[..n].copy_from_slice(&avail[..n]);
        Ok(n)
    }

    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        Err(Errno::EROFS)
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o444,
            size: 0,
            ino: 5,
            nlink: 1,
            rdev: VfsStat::makedev(254, 0),
            ..Default::default()
        })
    }
}

// ── /dev/random ──────────────────────────────────────────────────────────────

/// Character device node for `/dev/random`.
///
/// Reads fill the buffer with cryptographically random bytes from the kernel
/// entropy pool.  Returns `EAGAIN` if the pool has not yet been seeded by a
/// hardware entropy source — callers that need blocking behaviour should
/// retry after a short sleep or use `/dev/urandom`.
/// Writes are ignored (Linux-compatible).
pub struct RandomNode;

impl VfsNode for RandomNode {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        crate::entropy::fill(buf)?;
        Ok(buf.len())
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> SysResult<usize> {
        // Writes add entropy (Linux-compatible behaviour).
        crate::entropy::add_sample(buf);
        Ok(buf.len())
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 6,
            nlink: 1,
            rdev: VfsStat::makedev(1, 8),
            ..Default::default()
        })
    }

    fn poll(&self) -> u16 {
        // Always ready for read once the pool is seeded.
        if crate::entropy::is_seeded() {
            abi::syscall::poll_flags::POLLIN | abi::syscall::poll_flags::POLLOUT
        } else {
            abi::syscall::poll_flags::POLLOUT
        }
    }
}

// ── /dev/urandom ─────────────────────────────────────────────────────────────

/// Character device node for `/dev/urandom`.
///
/// Reads always return random bytes even before the entropy pool is fully
/// seeded (non-blocking, like Linux `/dev/urandom`).  When the pool is not
/// yet seeded the output is deterministic but still mixed from the initial
/// pool state — suitable for bootstrapping purposes.
/// Writes are ignored (Linux-compatible).
pub struct UrandomNode;

impl VfsNode for UrandomNode {
    fn read(&self, _offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Always generate output regardless of seeded state.
        crate::entropy::fill_or_weak(buf);
        Ok(buf.len())
    }

    fn write(&self, _offset: u64, buf: &[u8]) -> SysResult<usize> {
        // Writes add entropy (Linux-compatible behaviour).
        crate::entropy::add_sample(buf);
        Ok(buf.len())
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o666,
            size: 0,
            ino: 7,
            nlink: 1,
            rdev: VfsStat::makedev(1, 9),
            ..Default::default()
        })
    }

    fn poll(&self) -> u16 {
        // Always ready for both read and write.
        abi::syscall::poll_flags::POLLIN | abi::syscall::poll_flags::POLLOUT
    }
}

// ── /dev/kmsg ────────────────────────────────────────────────────────────────

/// Character device node for `/dev/kmsg`.
///
/// Provides a read-only view of the kernel message buffer (ring buffer).
/// Currently handles a single snapshot of the buffer per read call for simplicity.
pub struct KmsgNode;

impl VfsNode for KmsgNode {
    fn read(&self, offset: u64, buf: &mut [u8]) -> SysResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // We use a temporary buffer to avoid holding the log lock for too long
        // and because copy_log_buffer currently returns the whole buffer.
        // For dmesg, a full snapshot is usually what's wanted.
        let mut temp = vec![0u8; crate::logging::get_log_buffer_len()];
        let n = crate::logging::copy_log_buffer(&mut temp);

        let off = offset as usize;
        if off >= n {
            return Ok(0);
        }

        let avail = &temp[off..n];
        let count = avail.len().min(buf.len());
        buf[..count].copy_from_slice(&avail[..count]);
        Ok(count)
    }

    fn write(&self, _offset: u64, _buf: &[u8]) -> SysResult<usize> {
        // Linux allows writing to /dev/kmsg to inject logs, but we'll stick to
        // read-only for now.
        Err(Errno::EPERM)
    }

    fn stat(&self) -> SysResult<VfsStat> {
        Ok(VfsStat {
            mode: VfsStat::S_IFCHR | 0o444,
            size: crate::logging::get_log_buffer_len() as u64,
            ino: 8,
            nlink: 1,
            rdev: VfsStat::makedev(1, 11),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn lookup(path: &str) -> SysResult<Arc<dyn VfsNode>> {
        DevFs::new().lookup(path)
    }

    #[test]
    fn test_null_read_returns_zero() {
        let node = lookup("null").unwrap();
        let mut buf = [0xFFu8; 8];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_null_write_succeeds() {
        let node = lookup("null").unwrap();
        let n = node.write(0, b"hello").unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn test_zero_read_fills_zeros() {
        let node = lookup("zero").unwrap();
        let mut buf = [0xFFu8; 4];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 4);
        assert_eq!(buf, [0u8; 4]);
    }

    #[test]
    fn test_zero_write_succeeds() {
        let node = lookup("zero").unwrap();
        let n = node.write(0, b"ignored").unwrap();
        assert_eq!(n, 7);
    }

    #[test]
    fn test_lookup_console_returns_node() {
        assert!(lookup("console").is_ok());
    }

    #[test]
    fn test_lookup_unknown_returns_enoent() {
        assert!(matches!(lookup("nonexistent"), Err(Errno::ENOENT)));
    }

    #[test]
    fn test_null_stat() {
        let node = lookup("null").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_chr());
    }

    #[test]
    fn test_zero_stat() {
        let node = lookup("zero").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_chr());
    }

    #[test]
    fn test_lookup_root_is_dir() {
        let node = lookup("").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_dir());
    }

    #[test]
    fn test_readdir_lists_builtin_nodes() {
        let node = lookup("").unwrap();
        let mut buf = [0u8; 64];
        let n = node.readdir(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("console"));
        assert!(s.contains("null"));
        assert!(s.contains("zero"));
    }

    #[test]
    fn test_register_and_lookup_dynamic_device() {
        struct TestDev;
        impl VfsNode for TestDev {
            fn read(&self, _: u64, _: &mut [u8]) -> SysResult<usize> {
                Ok(0)
            }
            fn write(&self, _: u64, buf: &[u8]) -> SysResult<usize> {
                Ok(buf.len())
            }
            fn stat(&self) -> SysResult<VfsStat> {
                Ok(VfsStat {
                    mode: VfsStat::S_IFCHR | 0o666,
                    size: 0,
                    ino: 999,
                    ..Default::default()
                })
            }
        }

        register("test_unique_dev_42", Arc::new(TestDev));
        let node = DevFs::new().lookup("test_unique_dev_42").unwrap();
        assert!(node.stat().unwrap().is_chr());
        // Clean up.
        unregister("test_unique_dev_42");
    }

    #[test]
    fn test_unregister_removes_device() {
        struct TestDev2;
        impl VfsNode for TestDev2 {
            fn read(&self, _: u64, _: &mut [u8]) -> SysResult<usize> {
                Ok(0)
            }
            fn write(&self, _: u64, buf: &[u8]) -> SysResult<usize> {
                Ok(buf.len())
            }
            fn stat(&self) -> SysResult<VfsStat> {
                Ok(VfsStat {
                    mode: VfsStat::S_IFCHR | 0o666,
                    size: 0,
                    ino: 998,
                    ..Default::default()
                })
            }
        }

        register("test_unique_dev_99", Arc::new(TestDev2));
        assert!(DevFs::new().lookup("test_unique_dev_99").is_ok());
        let removed = unregister("test_unique_dev_99");
        assert!(removed);
        assert!(matches!(
            DevFs::new().lookup("test_unique_dev_99"),
            Err(Errno::ENOENT)
        ));
    }

    // ── /dev/urandom tests ───────────────────────────────────────────────────

    #[test]
    fn test_lookup_urandom_returns_node() {
        assert!(lookup("urandom").is_ok());
    }

    #[test]
    fn test_urandom_stat_is_chr() {
        let node = lookup("urandom").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_chr());
    }

    #[test]
    fn test_urandom_read_fills_buffer() {
        let node = lookup("urandom").unwrap();
        let mut buf = [0u8; 16];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 16);
    }

    #[test]
    fn test_urandom_read_empty_buffer_returns_zero() {
        let node = lookup("urandom").unwrap();
        let mut buf = [];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_urandom_write_succeeds() {
        let node = lookup("urandom").unwrap();
        let n = node.write(0, b"some entropy").unwrap();
        assert_eq!(n, 12);
    }

    #[test]
    fn test_urandom_poll_always_readable() {
        let node = lookup("urandom").unwrap();
        let mask = node.poll();
        assert!(mask & abi::syscall::poll_flags::POLLIN != 0);
    }

    // ── /dev/random tests ────────────────────────────────────────────────────

    #[test]
    fn test_lookup_random_returns_node() {
        assert!(lookup("random").is_ok());
    }

    #[test]
    fn test_random_stat_is_chr() {
        let node = lookup("random").unwrap();
        let stat = node.stat().unwrap();
        assert!(stat.is_chr());
    }

    #[test]
    fn test_random_read_after_seeding() {
        // Seed the pool so /dev/random becomes readable.
        crate::entropy::add_sample(b"test_entropy_data_12345678");
        crate::entropy::mark_seeded();

        let node = lookup("random").unwrap();
        let mut buf = [0u8; 8];
        let n = node.read(0, &mut buf).unwrap();
        assert_eq!(n, 8);
    }

    #[test]
    fn test_random_write_succeeds() {
        let node = lookup("random").unwrap();
        let n = node.write(0, b"more entropy").unwrap();
        assert_eq!(n, 12);
    }

    #[test]
    fn test_readdir_lists_random_nodes() {
        let node = lookup("").unwrap();
        let mut buf = [0u8; 128];
        let n = node.readdir(0, &mut buf).unwrap();
        assert!(n > 0);
        let s = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(s.contains("random"));
        assert!(s.contains("urandom"));
    }

    // ── ConsoleNode termios tests ─────────────────────────────────────────────

    /// Serialise console-state tests that touch global statics.
    static CONSOLE_TEST_GUARD: spin::Mutex<()> = spin::Mutex::new(());

    #[test]
    fn test_console_is_tty() {
        let node = lookup("console").unwrap();
        assert!(node.is_tty());
    }

    #[test]
    fn test_console_default_termios_icanon() {
        let _g = CONSOLE_TEST_GUARD.lock();
        // Reset to a known state.
        ConsoleNode::set_termios(abi::termios::DEFAULT_TERMIOS);
        let t = ConsoleNode::get_termios();
        assert_ne!(t.c_lflag & abi::termios::ICANON, 0, "ICANON should be set");
        assert_ne!(t.c_lflag & abi::termios::ECHO, 0, "ECHO should be set");
        assert_ne!(t.c_lflag & abi::termios::ISIG, 0, "ISIG should be set");
        assert_ne!(t.c_iflag & abi::termios::ICRNL, 0, "ICRNL should be set");
    }

    #[test]
    fn test_console_set_raw_mode_clears_icanon() {
        let _g = CONSOLE_TEST_GUARD.lock();
        let mut raw = abi::termios::DEFAULT_TERMIOS;
        raw.c_lflag &= !(abi::termios::ICANON | abi::termios::ECHO | abi::termios::ISIG);
        raw.c_iflag &= !(abi::termios::ICRNL | abi::termios::IXON);
        ConsoleNode::set_termios(raw);

        let t = ConsoleNode::get_termios();
        assert_eq!(
            t.c_lflag & abi::termios::ICANON,
            0,
            "ICANON should be clear"
        );
        assert_eq!(t.c_lflag & abi::termios::ECHO, 0, "ECHO should be clear");
        assert_eq!(t.c_lflag & abi::termios::ISIG, 0, "ISIG should be clear");

        // Restore.
        ConsoleNode::set_termios(abi::termios::DEFAULT_TERMIOS);
    }

    #[test]
    fn test_console_read_returns_eintr_on_pending_interrupt() {
        use crate::sched::hooks::TAKE_PENDING_INTERRUPT_HOOK;
        use core::sync::atomic::{AtomicBool, Ordering};

        static INTERRUPT_PENDING: AtomicBool = AtomicBool::new(false);

        fn take_interrupt() -> bool {
            INTERRUPT_PENDING.swap(false, Ordering::SeqCst)
        }

        // Set the interrupt flag so the first loop iteration returns EINTR.
        INTERRUPT_PENDING.store(true, Ordering::SeqCst);
        unsafe { TAKE_PENDING_INTERRUPT_HOOK = Some(take_interrupt) };

        let node = ConsoleNode;
        let mut buf = [0u8; 4];
        let result = node.read(0, &mut buf);

        // Clear the hook to not affect other tests.
        unsafe { TAKE_PENDING_INTERRUPT_HOOK = None };

        assert_eq!(result, Err(abi::errors::Errno::EINTR));
    }

    #[test]
    fn test_console_device_call_unknown_kind_returns_enosys() {
        let node = ConsoleNode;
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::RtcCmos,
            op: 1,
            in_ptr: 0,
            in_len: 0,
            out_ptr: 0,
            out_len: 0,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Err(abi::errors::Errno::ENOSYS));
    }

    #[test]
    fn test_console_device_call_tcgets_null_ptr_returns_einval() {
        let node = ConsoleNode;
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::Terminal,
            op: abi::termios::TERMINAL_OP_TCGETS,
            in_ptr: 0,
            in_len: 0,
            out_ptr: 0, // null → EINVAL
            out_len: 0,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Err(abi::errors::Errno::EINVAL));
    }

    #[test]
    fn test_console_device_call_tcsets_null_ptr_returns_einval() {
        let node = ConsoleNode;
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::Terminal,
            op: abi::termios::TERMINAL_OP_TCSETS,
            in_ptr: 0, // null → EINVAL
            in_len: 0,
            out_ptr: 0,
            out_len: 0,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Err(abi::errors::Errno::EINVAL));
    }

    #[test]
    fn test_console_device_call_tcgets_writes_termios() {
        let _g = CONSOLE_TEST_GUARD.lock();
        ConsoleNode::set_termios(abi::termios::DEFAULT_TERMIOS);

        let node = ConsoleNode;
        let mut out = abi::termios::Termios::default();
        let size = core::mem::size_of::<abi::termios::Termios>();
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::Terminal,
            op: abi::termios::TERMINAL_OP_TCGETS,
            in_ptr: 0,
            in_len: 0,
            out_ptr: &mut out as *mut _ as u64,
            out_len: size as u32,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Ok(0));
        assert_eq!(out.c_lflag & abi::termios::ICANON, abi::termios::ICANON);
    }

    #[test]
    fn test_console_device_call_tcsets_updates_termios() {
        let _g = CONSOLE_TEST_GUARD.lock();
        ConsoleNode::set_termios(abi::termios::DEFAULT_TERMIOS);

        let node = ConsoleNode;
        let mut raw = abi::termios::DEFAULT_TERMIOS;
        raw.c_lflag &= !(abi::termios::ICANON | abi::termios::ECHO);

        let size = core::mem::size_of::<abi::termios::Termios>();
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::Terminal,
            op: abi::termios::TERMINAL_OP_TCSETS,
            in_ptr: &raw as *const _ as u64,
            in_len: size as u32,
            out_ptr: 0,
            out_len: 0,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Ok(0));

        let t = ConsoleNode::get_termios();
        assert_eq!(t.c_lflag & abi::termios::ICANON, 0);
        assert_eq!(t.c_lflag & abi::termios::ECHO, 0);

        // Restore.
        ConsoleNode::set_termios(abi::termios::DEFAULT_TERMIOS);
    }

    #[test]
    fn test_console_device_call_tcgetpgrp_requires_process_context() {
        let _g = CONSOLE_TEST_GUARD.lock();
        ConsoleNode::set_tty_owner_for_test(Some(1), Some(1));

        let node = ConsoleNode;
        let mut out_pgid = 0u32;
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::Terminal,
            op: abi::termios::TERMINAL_OP_TCGETPGRP,
            in_ptr: 0,
            in_len: 0,
            out_ptr: &mut out_pgid as *mut u32 as u64,
            out_len: core::mem::size_of::<u32>() as u32,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Err(abi::errors::Errno::ENOTTY));

        ConsoleNode::set_tty_owner_for_test(None, None);
    }

    #[test]
    fn test_console_device_call_tcsetpgrp_requires_process_context() {
        let _g = CONSOLE_TEST_GUARD.lock();
        ConsoleNode::set_tty_owner_for_test(Some(1), Some(1));

        let node = ConsoleNode;
        let new_pgid = 2u32;
        let call = abi::device::DeviceCall {
            kind: abi::device::DeviceKind::Terminal,
            op: abi::termios::TERMINAL_OP_TCSETPGRP,
            in_ptr: &new_pgid as *const u32 as u64,
            in_len: core::mem::size_of::<u32>() as u32,
            out_ptr: 0,
            out_len: 0,
        };
        let result = node.device_call(&call);
        assert_eq!(result, Err(abi::errors::Errno::ENOTTY));

        ConsoleNode::set_tty_owner_for_test(None, None);
    }

    // ── Ownership / rdev / nlink metadata tests ──────────────────────────────

    #[test]
    fn test_null_stat_has_rdev_and_nlink() {
        let st = NullNode.stat().unwrap();
        assert!(st.is_chr());
        assert_eq!(st.nlink, 1);
        assert_eq!(st.rdev, VfsStat::makedev(1, 3));
        assert_eq!(st.uid, 0);
        assert_eq!(st.gid, 0);
    }

    #[test]
    fn test_zero_stat_has_rdev_and_nlink() {
        let st = ZeroNode.stat().unwrap();
        assert!(st.is_chr());
        assert_eq!(st.nlink, 1);
        assert_eq!(st.rdev, VfsStat::makedev(1, 5));
    }

    #[test]
    fn test_console_stat_has_rdev_and_nlink() {
        let st = ConsoleNode.stat().unwrap();
        assert!(st.is_chr());
        assert_eq!(st.nlink, 1);
        assert_eq!(st.rdev, VfsStat::makedev(5, 1));
    }

    #[test]
    fn test_rtc_stat_has_rdev_and_nlink() {
        let st = RtcNode.stat().unwrap();
        assert!(st.is_chr());
        assert_eq!(st.nlink, 1);
        assert_eq!(st.rdev, VfsStat::makedev(254, 0));
    }

    #[test]
    fn test_random_stat_has_rdev_and_nlink() {
        let st = RandomNode.stat().unwrap();
        assert!(st.is_chr());
        assert_eq!(st.nlink, 1);
        assert_eq!(st.rdev, VfsStat::makedev(1, 8));
    }

    #[test]
    fn test_urandom_stat_has_rdev_and_nlink() {
        let st = UrandomNode.stat().unwrap();
        assert!(st.is_chr());
        assert_eq!(st.nlink, 1);
        assert_eq!(st.rdev, VfsStat::makedev(1, 9));
    }

    #[test]
    fn test_dev_dir_stat_has_nlink_two() {
        let st = DevDirNode.stat().unwrap();
        assert!(st.is_dir());
        assert!(
            st.nlink >= 2,
            "dev dir nlink should be >= 2, got {}",
            st.nlink
        );
    }
}
