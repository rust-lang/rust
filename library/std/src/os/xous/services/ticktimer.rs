use core::sync::atomic::{AtomicU32, Ordering};

use crate::os::xous::ffi::Connection;

pub(crate) enum TicktimerScalar {
    ElapsedMs,
    SleepMs(usize),
    LockMutex(usize /* cookie */),
    UnlockMutex(usize /* cookie */),
    WaitForCondition(usize /* cookie */, usize /* timeout (ms) */),
    NotifyCondition(usize /* cookie */, usize /* count */),
    FreeMutex(usize /* cookie */),
    FreeCondition(usize /* cookie */),
}

impl Into<[usize; 5]> for TicktimerScalar {
    fn into(self) -> [usize; 5] {
        match self {
            TicktimerScalar::ElapsedMs => [0, 0, 0, 0, 0],
            TicktimerScalar::SleepMs(msecs) => [1, msecs, 0, 0, 0],
            TicktimerScalar::LockMutex(cookie) => [6, cookie, 0, 0, 0],
            TicktimerScalar::UnlockMutex(cookie) => [7, cookie, 0, 0, 0],
            TicktimerScalar::WaitForCondition(cookie, timeout_ms) => [8, cookie, timeout_ms, 0, 0],
            TicktimerScalar::NotifyCondition(cookie, count) => [9, cookie, count, 0, 0],
            TicktimerScalar::FreeMutex(cookie) => [10, cookie, 0, 0, 0],
            TicktimerScalar::FreeCondition(cookie) => [11, cookie, 0, 0, 0],
        }
    }
}

/// Returns a `Connection` to the ticktimer server. This server is used for synchronization
/// primitives such as sleep, Mutex, and Condvar.
pub(crate) fn ticktimer_server() -> Connection {
    static TICKTIMER_SERVER_CONNECTION: AtomicU32 = AtomicU32::new(0);
    let cid = TICKTIMER_SERVER_CONNECTION.load(Ordering::Relaxed);
    if cid != 0 {
        return cid.into();
    }

    let cid = crate::os::xous::ffi::connect("ticktimer-server".try_into().unwrap()).unwrap();
    TICKTIMER_SERVER_CONNECTION.store(cid.into(), Ordering::Relaxed);
    cid
}
