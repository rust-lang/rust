use core::sync::atomic::{AtomicU32, Ordering};

use crate::os::xous::ffi::Connection;
use crate::os::xous::services::connect;

#[repr(usize)]
pub(crate) enum DnsLendMut {
    RawLookup = 6,
}

impl Into<usize> for DnsLendMut {
    fn into(self) -> usize {
        self as usize
    }
}

/// Returns a `Connection` to the DNS lookup server. This server is used for
/// querying domain name values.
pub(crate) fn dns_server() -> Connection {
    static DNS_CONNECTION: AtomicU32 = AtomicU32::new(0);
    let cid = DNS_CONNECTION.load(Ordering::Relaxed);
    if cid != 0 {
        return cid.into();
    }

    let cid = connect("_DNS Resolver Middleware_").unwrap();
    DNS_CONNECTION.store(cid.into(), Ordering::Relaxed);
    cid
}
