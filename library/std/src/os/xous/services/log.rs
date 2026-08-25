use core::sync::atomic::{Atomic, AtomicU32, Ordering};

use crate::os::xous::ffi::Connection;

/// Group a `usize` worth of bytes into a `usize` and return it, beginning from
/// `offset` * sizeof(usize) bytes from the start. For example,
/// `group_or_null([1,2,3,4,5,6,7,8], 1)` on a 32-bit system will return a
/// `usize` with 5678 packed into it.
fn group_or_null(data: &[u8], offset: usize) -> usize {
    let start = offset * size_of::<usize>();
    let mut out_array = [0u8; size_of::<usize>()];
    if start < data.len() {
        for (dest, src) in out_array.iter_mut().zip(&data[start..]) {
            *dest = *src;
        }
    }
    usize::from_le_bytes(out_array)
}

pub(crate) enum LogScalar<'a> {
    /// A panic occurred, and a panic log is forthcoming
    BeginPanic,

    /// Some number of bytes will be appended to the log message
    AppendPanicMessage(&'a [u8]),
}

impl<'a> Into<[usize; 5]> for LogScalar<'a> {
    fn into(self) -> [usize; 5] {
        match self {
            LogScalar::BeginPanic => [1000, 0, 0, 0, 0],
            LogScalar::AppendPanicMessage(c) =>
            // Text is grouped into 4x `usize` words. The id is 1100 plus
            // the number of characters in this message.
            // Ignore errors since we're already panicking.
            {
                [
                    1100 + c.len(),
                    group_or_null(&c, 0),
                    group_or_null(&c, 1),
                    group_or_null(&c, 2),
                    group_or_null(&c, 3),
                ]
            }
        }
    }
}

pub(crate) enum LogLend {
    StandardOutput = 1,
    StandardError = 2,
}

impl Into<usize> for LogLend {
    fn into(self) -> usize {
        self as usize
    }
}

/// Returns a `Connection` to the log server, which is used for printing messages to
/// the console and reporting panics.
///
/// If the log server has not yet started, this will block until the server is
/// running. It is safe to call this multiple times, because the address is
/// shared among all threads in a process.
pub(crate) fn log_server() -> Connection {
    static LOG_SERVER_CONNECTION: Atomic<u32> = AtomicU32::new(0);

    let cid = LOG_SERVER_CONNECTION.load(Ordering::Relaxed);
    if cid != 0 {
        return cid.into();
    }

    let cid = crate::os::xous::ffi::connect("xous-log-server ".try_into().unwrap()).unwrap();
    LOG_SERVER_CONNECTION.store(cid.into(), Ordering::Relaxed);
    cid
}
