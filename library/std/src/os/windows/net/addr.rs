#![unstable(feature = "windows_unix_domain_sockets", issue = "56533")]

use crate::os::raw::{c_char, c_int};
use crate::path::Path;
use crate::sys::c::{self, SOCKADDR};
use crate::sys::cvt;
use crate::{io, mem};
pub fn sockaddr_un(path: &Path) -> io::Result<(c::sockaddr_un, c_int)> {
    let mut addr: c::sockaddr_un = unsafe { mem::zeroed() };
    addr.sun_family = c::AF_UNIX;
    // Winsock2 expects 'sun_path' to be a Win32 UTF-8 file system path
    let bytes = path
        .to_str()
        .map(|s| s.as_bytes())
        .ok_or(io::Error::new(io::ErrorKind::InvalidInput, "path contains invalid characters"))?;

    if bytes.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "paths may not contain interior null bytes",
        ));
    }

    if bytes.len() >= addr.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "path must be shorter than SUN_LEN",
        ));
    }
    for (dst, src) in addr.sun_path.iter_mut().zip(bytes.iter()) {
        *dst = *src as c_char;
    }
    // null byte for pathname addresses is already there because we zeroed the
    // struct

    let mut len = sun_path_offset(&addr) + bytes.len();
    match bytes.first() {
        Some(&0) | None => {}
        Some(_) => len += 1,
    }
    Ok((addr, len as _))
}
fn sun_path_offset(addr: &c::sockaddr_un) -> usize {
    // Work with an actual instance of the type since using a null pointer is UB
    let base = addr as *const _ as usize;
    let path = &addr.sun_path as *const _ as usize;
    path - base
}
#[allow(dead_code)]
pub struct SocketAddr {
    addr: c::sockaddr_un,
    len: c_int,
}
impl SocketAddr {
    pub fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(*mut SOCKADDR, *mut c_int) -> c_int,
    {
        unsafe {
            let mut addr: c::sockaddr_un = mem::zeroed();
            let mut len = mem::size_of::<c::sockaddr_un>() as c_int;
            cvt(f(&mut addr as *mut _ as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }
    fn from_parts(addr: c::sockaddr_un, mut len: c_int) -> io::Result<SocketAddr> {
        if len == 0 {
            // When there is a datagram from unnamed unix socket
            // linux returns zero bytes of address
            len = sun_path_offset(&addr) as c_int; // i.e. zero-length address
        } else if addr.sun_family != c::AF_UNIX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "file descriptor did not correspond to a Unix socket",
            ));
        }

        Ok(SocketAddr { addr, len })
    }
}
pub fn from_sockaddr_un(addr: c::sockaddr_un, len: c_int) -> io::Result<SocketAddr> {
    SocketAddr::from_parts(addr, len)
}
