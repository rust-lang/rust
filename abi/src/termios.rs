//! Terminal I/O settings — termios ABI types and constants.
//!
//! Mirrors the relevant subset of POSIX `<termios.h>` so that both the
//! kernel (terminal driver) and userspace (shell, readline) share the same
//! layout without an external libc dependency.
//!
//! # Wire format
//! All integers are **native-endian**; this is an in-kernel ABI that never
//! crosses a network boundary.
//!
//! # Relation to ioctls
//! The terminal ioctl commands (`TCGETS`, `TCSETS`) are encoded as
//! [`crate::device::DeviceKind::Terminal`] operations with the `op` field set
//! to [`TERMINAL_OP_TCGETS`] or [`TERMINAL_OP_TCSETS`].  The payload (the
//! `Termios` struct) is passed via the `in_ptr`/`out_ptr` fields of
//! [`crate::device::DeviceCall`].

// ── Termios struct ────────────────────────────────────────────────────────────

/// Number of special characters in `c_cc`.
pub const NCCS: usize = 32;

/// Terminal settings, mirroring POSIX `struct termios`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Termios {
    /// Input mode flags (see `iflag` constants).
    pub c_iflag: u32,
    /// Output mode flags (see `oflag` constants).
    pub c_oflag: u32,
    /// Control mode flags.
    pub c_cflag: u32,
    /// Local mode flags (see `lflag` constants).
    pub c_lflag: u32,
    /// Line discipline identifier (ignored; kept for ABI compatibility).
    pub c_line: u8,
    /// _pad to align c_cc to a 4-byte boundary.
    pub _pad: [u8; 3],
    /// Special characters array (indexed by `V*` constants).
    pub c_cc: [u8; NCCS],
}

// ── Input flags (c_iflag) ─────────────────────────────────────────────────────

/// Ignore break condition.
pub const IGNBRK: u32 = 0x0001;
/// Signal interrupt on break (if IGNBRK is not set).
pub const BRKINT: u32 = 0x0002;
/// Ignore framing and parity errors.
pub const IGNPAR: u32 = 0x0004;
/// Strip 8th bit off each input byte.
pub const ISTRIP: u32 = 0x0020;
/// Translate CR to NL on input.
pub const ICRNL: u32 = 0x0100;
/// Enable XON/XOFF flow control on output.
pub const IXON: u32 = 0x0400;
/// Enable XON/XOFF flow control on input.
pub const IXOFF: u32 = 0x1000;

// ── Output flags (c_oflag) ────────────────────────────────────────────────────

/// Enable implementation-defined output processing.
pub const OPOST: u32 = 0x0001;
/// Map NL to CR-NL on output.
pub const ONLCR: u32 = 0x0004;

// ── Local flags (c_lflag) ─────────────────────────────────────────────────────

/// Generate signals when special chars are received (Ctrl-C → SIGINT, etc.).
pub const ISIG: u32 = 0x0001;
/// Canonical mode — input is line-buffered and processed.
/// When clear, the terminal is in raw/non-canonical mode.
pub const ICANON: u32 = 0x0002;
/// Echo input characters back to the terminal.
pub const ECHO: u32 = 0x0008;
/// Echo ERASE character as BS-SP-BS (visual backspace).
pub const ECHOE: u32 = 0x0010;
/// Echo NL character even if ECHO is not set.
pub const ECHONL: u32 = 0x0040;

// ── Control characters (c_cc indices) ────────────────────────────────────────

/// Index of the interrupt character (Ctrl-C by default).
pub const VINTR: usize = 0;
/// Index of the quit character (Ctrl-\ by default).
pub const VQUIT: usize = 1;
/// Index of the erase character (DEL/Backspace by default).
pub const VERASE: usize = 2;
/// Index of the kill-line character (Ctrl-U by default).
pub const VKILL: usize = 3;
/// Index of the end-of-file character (Ctrl-D by default).
pub const VEOF: usize = 4;
/// Index of the inter-character timeout (in tenths of a second; raw mode).
pub const VTIME: usize = 5;
/// Index of the minimum number of characters for a read (raw mode).
pub const VMIN: usize = 6;
/// Index of the start (XON) character (Ctrl-Q by default).
pub const VSTART: usize = 8;
/// Index of the stop (XOFF) character (Ctrl-S by default).
pub const VSTOP: usize = 9;
/// Index of the suspend character (Ctrl-Z by default).
pub const VSUSP: usize = 10;

// ── Terminal ioctl operation codes ────────────────────────────────────────────

/// Get the current terminal settings.
/// The kernel writes the current `Termios` to the user buffer at `out_ptr`.
pub const TERMINAL_OP_TCGETS: u32 = 1;
/// Set terminal settings immediately.
/// The kernel reads a `Termios` from the user buffer at `in_ptr`.
pub const TERMINAL_OP_TCSETS: u32 = 2;
/// Set terminal settings after draining output.
/// The kernel reads a `Termios` from the user buffer at `in_ptr`.
pub const TERMINAL_OP_TCSETSW: u32 = 3;
/// Set terminal settings after flushing pending I/O.
/// The kernel reads a `Termios` from the user buffer at `in_ptr`.
pub const TERMINAL_OP_TCSETSF: u32 = 4;
/// Get foreground process group ID for the controlling terminal.
/// The kernel writes a `u32` pgid to the user buffer at `out_ptr`.
pub const TERMINAL_OP_TCGETPGRP: u32 = 5;
/// Set foreground process group ID for the controlling terminal.
/// The kernel reads a `u32` pgid from the user buffer at `in_ptr`.
pub const TERMINAL_OP_TCSETPGRP: u32 = 6;

// ── Default termios ───────────────────────────────────────────────────────────

/// Default terminal settings: canonical mode, echo enabled, CR→NL, ISIG.
///
/// This matches a typical initial state for an interactive terminal emulator.
pub const DEFAULT_TERMIOS: Termios = Termios {
    c_iflag: ICRNL | BRKINT,
    c_oflag: OPOST | ONLCR,
    c_cflag: 0x00BF, // CS8 | CREAD | CLOCAL
    c_lflag: ICANON | ECHO | ECHOE | ISIG,
    c_line: 0,
    _pad: [0; 3],
    //        [0]  [1]   [2]   [3]   [4]  [5]  [6]  [7]
    c_cc: [
        0x03, 0x1c, 0x7f, 0x15, 0x04, 0x00, 0x01, 0x00, //    [8]   [9]   [10] … [31]
        0x11, 0x13, 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn termios_size_is_stable() {
        // c_iflag(4) + c_oflag(4) + c_cflag(4) + c_lflag(4) + c_line(1) + _pad(3) + c_cc(32) = 52
        assert_eq!(core::mem::size_of::<Termios>(), 52);
    }

    #[test]
    fn default_termios_has_icanon_and_echo() {
        assert_ne!(DEFAULT_TERMIOS.c_lflag & ICANON, 0);
        assert_ne!(DEFAULT_TERMIOS.c_lflag & ECHO, 0);
        assert_ne!(DEFAULT_TERMIOS.c_lflag & ISIG, 0);
    }

    #[test]
    fn default_termios_vintr_is_ctrl_c() {
        assert_eq!(DEFAULT_TERMIOS.c_cc[VINTR], 0x03);
    }

    #[test]
    fn raw_mode_clears_icanon_and_echo() {
        let mut t = DEFAULT_TERMIOS;
        t.c_lflag &= !(ICANON | ECHO | ISIG);
        t.c_iflag &= !(ICRNL | IXON);
        assert_eq!(t.c_lflag & ICANON, 0);
        assert_eq!(t.c_lflag & ECHO, 0);
        assert_eq!(t.c_lflag & ISIG, 0);
    }

    #[test]
    fn default_termios_icrnl_set() {
        assert_ne!(DEFAULT_TERMIOS.c_iflag & ICRNL, 0);
    }
}
