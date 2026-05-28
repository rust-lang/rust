use core::sync::atomic::{Atomic, AtomicU32, Ordering};

use crate::os::xous::ffi::Connection;
use crate::os::xous::services::connect;

pub(crate) enum NetBlockingScalar {
    StdGetTtlUdp(u16 /* fd */),                /* 36 */
    StdSetTtlUdp(u16 /* fd */, u32 /* ttl */), /* 37 */
    StdGetTtlTcp(u16 /* fd */),                /* 36 */
    StdSetTtlTcp(u16 /* fd */, u32 /* ttl */), /* 37 */
    StdGetNodelay(u16 /* fd */),               /* 38 */
    StdSetNodelay(u16 /* fd */, bool),         /* 39 */
    StdTcpClose(u16 /* fd */),                 /* 34 */
    StdUdpClose(u16 /* fd */),                 /* 41 */
    StdTcpStreamShutdown(u16 /* fd */, crate::net::Shutdown /* how */), /* 46 */
}

pub(crate) enum NetLendMut {
    StdTcpConnect,                                    /* 30 */
    StdTcpTx(u16 /* fd */),                           /* 31 */
    StdTcpPeek(u16 /* fd */, bool /* nonblocking */), /* 32 */
    StdTcpRx(u16 /* fd */, bool /* nonblocking */),   /* 33 */
    StdGetAddress(u16 /* fd */),                      /* 35 */
    StdUdpBind,                                       /* 40 */
    StdUdpRx(u16 /* fd */),                           /* 42 */
    StdUdpTx(u16 /* fd */),                           /* 43 */
    StdTcpListen,                                     /* 44 */
    StdTcpAccept(u16 /* fd */),                       /* 45 */
}

impl Into<usize> for NetLendMut {
    fn into(self) -> usize {
        match self {
            NetLendMut::StdTcpConnect => 30,
            NetLendMut::StdTcpTx(fd) => 31 | ((fd as usize) << 16),
            NetLendMut::StdTcpPeek(fd, blocking) => {
                32 | ((fd as usize) << 16) | if blocking { 0x8000 } else { 0 }
            }
            NetLendMut::StdTcpRx(fd, blocking) => {
                33 | ((fd as usize) << 16) | if blocking { 0x8000 } else { 0 }
            }
            NetLendMut::StdGetAddress(fd) => 35 | ((fd as usize) << 16),
            NetLendMut::StdUdpBind => 40,
            NetLendMut::StdUdpRx(fd) => 42 | ((fd as usize) << 16),
            NetLendMut::StdUdpTx(fd) => 43 | ((fd as usize) << 16),
            NetLendMut::StdTcpListen => 44,
            NetLendMut::StdTcpAccept(fd) => 45 | ((fd as usize) << 16),
        }
    }
}

impl<'a> Into<[usize; 5]> for NetBlockingScalar {
    fn into(self) -> [usize; 5] {
        match self {
            NetBlockingScalar::StdGetTtlTcp(fd) => [36 | ((fd as usize) << 16), 0, 0, 0, 0],
            NetBlockingScalar::StdGetTtlUdp(fd) => [36 | ((fd as usize) << 16), 0, 0, 0, 1],
            NetBlockingScalar::StdSetTtlTcp(fd, ttl) => {
                [37 | ((fd as usize) << 16), ttl as _, 0, 0, 0]
            }
            NetBlockingScalar::StdSetTtlUdp(fd, ttl) => {
                [37 | ((fd as usize) << 16), ttl as _, 0, 0, 1]
            }
            NetBlockingScalar::StdGetNodelay(fd) => [38 | ((fd as usize) << 16), 0, 0, 0, 0],
            NetBlockingScalar::StdSetNodelay(fd, enabled) => {
                [39 | ((fd as usize) << 16), if enabled { 1 } else { 0 }, 0, 0, 1]
            }
            NetBlockingScalar::StdTcpClose(fd) => [34 | ((fd as usize) << 16), 0, 0, 0, 0],
            NetBlockingScalar::StdUdpClose(fd) => [41 | ((fd as usize) << 16), 0, 0, 0, 0],
            NetBlockingScalar::StdTcpStreamShutdown(fd, how) => [
                46 | ((fd as usize) << 16),
                match how {
                    crate::net::Shutdown::Read => 1,
                    crate::net::Shutdown::Write => 2,
                    crate::net::Shutdown::Both => 3,
                },
                0,
                0,
                0,
            ],
        }
    }
}

/// Returns a `Connection` to the Network server. This server provides all
/// OS-level networking functions.
pub(crate) fn net_server() -> Connection {
    static NET_CONNECTION: Atomic<u32> = AtomicU32::new(0);
    let cid = NET_CONNECTION.load(Ordering::Relaxed);
    if cid != 0 {
        return cid.into();
    }

    let cid = connect("_Middleware Network Server_").unwrap();
    NET_CONNECTION.store(cid.into(), Ordering::Relaxed);
    cid
}
