//! Logging platform abstraction.
//!
//! Provides a stable interface for logging that abstracts over the underlying
//! syscall mechanism. This allows the logging implementation to evolve
//! independently of consumers.

use crate::syscall::log_write;
use core::fmt;

/// Log levels matching ABI definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Level {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

/// Write formatted output to the log at the specified level.
///
/// This is the low-level PAL primitive. Most code should use the
/// `error!`, `warn!`, `info!`, `debug!`, `trace!` macros instead.
pub fn write(level: Level, args: fmt::Arguments) {
    use core::fmt::Write;
    let mut cons = BufConsole::new(level as usize);
    let _ = cons.write_fmt(args);
    cons.flush();
}

/// Write formatted output to the log with source provenance.
///
/// The provenance string typically contains module path information.
pub fn write_with_provenance(level: Level, provenance: &str, args: fmt::Arguments) {
    use core::fmt::Write;
    let mut cons = BufConsole::new(level as usize);
    let _ = cons.write_str(provenance);
    let _ = cons.write_str(": ");
    let _ = cons.write_fmt(args);
    cons.flush();
}

/// Buffered console writer that accumulates output before syscall.
struct BufConsole {
    buf: [u8; 256],
    len: usize,
    level: usize,
}

impl BufConsole {
    fn new(level: usize) -> Self {
        BufConsole {
            buf: [0u8; 256],
            len: 0,
            level,
        }
    }

    fn flush(&mut self) {
        if self.len > 0 {
            if let Ok(s) = core::str::from_utf8(&self.buf[..self.len]) {
                let _ = log_write(s, self.level);
            }
            self.len = 0;
        }
    }
}

impl fmt::Write for BufConsole {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let mut remaining = bytes;

        while !remaining.is_empty() {
            let space = self.buf.len() - self.len;
            if space == 0 {
                self.flush();
                continue;
            }

            let chunk_len = core::cmp::min(space, remaining.len());
            self.buf[self.len..self.len + chunk_len].copy_from_slice(&remaining[..chunk_len]);
            self.len += chunk_len;
            remaining = &remaining[chunk_len..];
        }
        Ok(())
    }
}
