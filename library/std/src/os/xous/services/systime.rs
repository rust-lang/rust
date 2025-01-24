use core::sync::atomic::{AtomicU32, Ordering};

use crate::os::xous::ffi::{Connection, connect};

pub(crate) enum SystimeScalar {
    GetUtcTimeMs,
}

impl Into<[usize; 5]> for SystimeScalar {
    fn into(self) -> [usize; 5] {
        match self {
            SystimeScalar::GetUtcTimeMs => [3, 0, 0, 0, 0],
        }
    }
}

/// Returns a `Connection` to the systime server. This server is used for reporting the
/// realtime clock.
pub(crate) fn systime_server() -> Connection {
    static SYSTIME_SERVER_CONNECTION: AtomicU32 = AtomicU32::new(0);
    let cid = SYSTIME_SERVER_CONNECTION.load(Ordering::Relaxed);
    if cid != 0 {
        return cid.into();
    }

    let cid = connect("timeserverpublic".try_into().unwrap()).unwrap();
    SYSTIME_SERVER_CONNECTION.store(cid.into(), Ordering::Relaxed);
    cid
}
