use crate::io;
use crate::mem::{self, offset_of};
use crate::path::Path;
#[cfg(windows)]
use crate::sys::c::{AF_UNIX, SOCKADDR, SOCKADDR_UN};
use crate::sys::cvt;
#[cfg(windows)]
pub fn socketaddr_un(path: &Path) -> io::Result<(SOCKADDR_UN, usize)> {
    // path to bytes
    let bytes = path.as_os_str().as_encoded_bytes();
    let mut addr = SOCKADDR_UN { sun_family: AF_UNIX, sun_path: [0; 108] };
    for (i, &byte) in bytes.iter().take(108).enumerate() {
        addr.sun_path[i] = byte as i8;
    }
    let len = sun_path_offset() + bytes.len();
    Ok((addr, len))
}

fn sun_path_offset() -> usize {
    offset_of!(SOCKADDR_UN, sun_path)
}
pub struct SocketAddr {
    #[cfg(windows)]
    pub(super) addr: SOCKADDR_UN,
    pub(super) len: u32, // Use u32 here as same as libc::socklen_t
}
#[cfg(windows)]
impl SocketAddr {
    pub(super) fn new<F>(f: F) -> io::Result<SocketAddr>
    where
        F: FnOnce(*mut SOCKADDR, *mut u32) -> u32,
    {
        unsafe {
            let mut addr: SOCKADDR_UN = mem::zeroed();
            let mut len = mem::size_of::<SOCKADDR_UN>() as u32;
            cvt(f(&mut addr as *mut _ as *mut _, &mut len))?;
            SocketAddr::from_parts(addr, len)
        }
    }
    pub(super) fn from_parts(addr: SOCKADDR_UN, len: u32) -> io::Result<SocketAddr> {
        // maybe should check something here
        Ok(SocketAddr { addr, len })
    }
}
