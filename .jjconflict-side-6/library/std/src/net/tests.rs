#![allow(warnings)] // not used on emscripten

use crate::env;
use crate::net::{Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV4, SocketAddrV6, ToSocketAddrs};
use crate::sync::atomic::{AtomicUsize, Ordering};

/// A localhost address whose port will be picked automatically by the OS.
pub const LOCALHOST_IP4: SocketAddr =
    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 0));
/// A localhost address whose port will be picked automatically by the OS.
pub const LOCALHOST_IP6: SocketAddr =
    SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1), 0, 0, 0));

pub fn sa4(a: Ipv4Addr, p: u16) -> SocketAddr {
    SocketAddr::V4(SocketAddrV4::new(a, p))
}

pub fn sa6(a: Ipv6Addr, p: u16) -> SocketAddr {
    SocketAddr::V6(SocketAddrV6::new(a, p, 0, 0))
}

pub fn tsa<A: ToSocketAddrs>(a: A) -> Result<Vec<SocketAddr>, String> {
    match a.to_socket_addrs() {
        Ok(a) => Ok(a.collect()),
        Err(e) => Err(e.to_string()),
    }
}

pub fn compare_ignore_zoneid(a: &SocketAddr, b: &SocketAddr) -> bool {
    match (a, b) {
        (SocketAddr::V6(a), SocketAddr::V6(b)) => {
            a.ip().segments() == b.ip().segments()
                && a.flowinfo() == b.flowinfo()
                && a.port() == b.port()
        }
        _ => a == b,
    }
}

/// A read-only anonymous mapping of `len` zero bytes.
///
/// The tests that need a buffer larger than `c_int::MAX` use this instead of a
/// `Vec`: the pages are demand-zero and never written, so they stay mapped to
/// the shared zero page and the mapping doesn't actually consume `len` bytes of
/// physical memory.
#[cfg(all(target_pointer_width = "64", unix))]
pub struct ZeroedMmap {
    ptr: *mut libc::c_void,
    len: usize,
}

#[cfg(all(target_pointer_width = "64", unix))]
impl ZeroedMmap {
    pub fn new(len: usize) -> ZeroedMmap {
        let ptr = unsafe {
            libc::mmap(
                crate::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap failed: {}", crate::io::Error::last_os_error());
        ZeroedMmap { ptr, len }
    }
}

#[cfg(all(target_pointer_width = "64", unix))]
impl crate::ops::Deref for ZeroedMmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: the mapping is live for `self.len` readable bytes until `Drop`.
        unsafe { crate::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

#[cfg(all(target_pointer_width = "64", unix))]
impl Drop for ZeroedMmap {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` come from the `mmap` call above and are unmapped once.
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

#[test]
fn hostname_smoketest() {
    // Just a smoke test to ensure it can be called.
    let name = crate::net::hostname();
    if cfg!(any(all(windows, not(target_vendor = "win7")), unix)) {
        // At least on Windows and Unix, this should succeed.
        // The `win7` Windows targets do not support it yet though.
        name.unwrap();
    }
}
