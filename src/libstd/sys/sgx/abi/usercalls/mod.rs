use crate::cmp;
use crate::io::{Error as IoError, Result as IoResult, IoSlice, IoSliceMut};
use crate::time::Duration;

pub(crate) mod alloc;
#[macro_use]
pub(crate) mod raw;

use self::raw::*;

/// Usercall `read`. See the ABI documentation for more information.
///
/// This will do a single `read` usercall and scatter the read data among
/// `bufs`. To read to a single buffer, just pass a slice of length one.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn read(fd: Fd, bufs: &mut [IoSliceMut<'_>]) -> IoResult<usize> {
    unsafe {
        let total_len = bufs.iter().fold(0usize, |sum, buf| sum.saturating_add(buf.len()));
        let mut userbuf = alloc::User::<[u8]>::uninitialized(total_len);
        let ret_len = raw::read(fd, userbuf.as_mut_ptr(), userbuf.len()).from_sgx_result()?;
        let userbuf = &userbuf[..ret_len];
        let mut index = 0;
        for buf in bufs {
            let end = cmp::min(index + buf.len(), userbuf.len());
            if let Some(buflen) = end.checked_sub(index) {
                userbuf[index..end].copy_to_enclave(&mut buf[..buflen]);
                index += buf.len();
            } else {
                break
            }
        }
        Ok(userbuf.len())
    }
}

/// Usercall `read_alloc`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn read_alloc(fd: Fd) -> IoResult<Vec<u8>> {
    unsafe {
        let userbuf = ByteBuffer { data: crate::ptr::null_mut(), len: 0 };
        let mut userbuf = alloc::User::new_from_enclave(&userbuf);
        raw::read_alloc(fd, userbuf.as_raw_mut_ptr()).from_sgx_result()?;
        Ok(userbuf.copy_user_buffer())
    }
}

/// Usercall `write`. See the ABI documentation for more information.
///
/// This will do a single `write` usercall and gather the written data from
/// `bufs`. To write from a single buffer, just pass a slice of length one.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn write(fd: Fd, bufs: &[IoSlice<'_>]) -> IoResult<usize> {
    unsafe {
        let total_len = bufs.iter().fold(0usize, |sum, buf| sum.saturating_add(buf.len()));
        let mut userbuf = alloc::User::<[u8]>::uninitialized(total_len);
        let mut index = 0;
        for buf in bufs {
            let end = cmp::min(index + buf.len(), userbuf.len());
            if let Some(buflen) = end.checked_sub(index) {
                userbuf[index..end].copy_from_enclave(&buf[..buflen]);
                index += buf.len();
            } else {
                break
            }
        }
        raw::write(fd, userbuf.as_ptr(), userbuf.len()).from_sgx_result()
    }
}

/// Usercall `flush`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn flush(fd: Fd) -> IoResult<()> {
    unsafe { raw::flush(fd).from_sgx_result() }
}

/// Usercall `close`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn close(fd: Fd) {
    unsafe { raw::close(fd) }
}

fn string_from_bytebuffer(buf: &alloc::UserRef<ByteBuffer>, usercall: &str, arg: &str) -> String {
    String::from_utf8(buf.copy_user_buffer())
        .unwrap_or_else(|_| rtabort!("Usercall {}: expected {} to be valid UTF-8", usercall, arg))
}

/// Usercall `bind_stream`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn bind_stream(addr: &str) -> IoResult<(Fd, String)> {
    unsafe {
        let addr_user = alloc::User::new_from_enclave(addr.as_bytes());
        let mut local = alloc::User::<ByteBuffer>::uninitialized();
        let fd = raw::bind_stream(
            addr_user.as_ptr(),
            addr_user.len(),
            local.as_raw_mut_ptr()
        ).from_sgx_result()?;
        let local = string_from_bytebuffer(&local, "bind_stream", "local_addr");
        Ok((fd, local))
    }
}

/// Usercall `accept_stream`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn accept_stream(fd: Fd) -> IoResult<(Fd, String, String)> {
    unsafe {
        let mut bufs = alloc::User::<[ByteBuffer; 2]>::uninitialized();
        let mut buf_it = alloc::UserRef::iter_mut(&mut *bufs); // FIXME: can this be done
                                                               // without forcing coercion?
        let (local, peer) = (buf_it.next().unwrap(), buf_it.next().unwrap());
        let fd = raw::accept_stream(
            fd,
            local.as_raw_mut_ptr(),
            peer.as_raw_mut_ptr()
        ).from_sgx_result()?;
        let local = string_from_bytebuffer(&local, "accept_stream", "local_addr");
        let peer = string_from_bytebuffer(&peer, "accept_stream", "peer_addr");
        Ok((fd, local, peer))
    }
}

/// Usercall `connect_stream`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn connect_stream(addr: &str) -> IoResult<(Fd, String, String)> {
    unsafe {
        let addr_user = alloc::User::new_from_enclave(addr.as_bytes());
        let mut bufs = alloc::User::<[ByteBuffer; 2]>::uninitialized();
        let mut buf_it = alloc::UserRef::iter_mut(&mut *bufs); // FIXME: can this be done
                                                               // without forcing coercion?
        let (local, peer) = (buf_it.next().unwrap(), buf_it.next().unwrap());
        let fd = raw::connect_stream(
            addr_user.as_ptr(),
            addr_user.len(),
            local.as_raw_mut_ptr(),
            peer.as_raw_mut_ptr()
        ).from_sgx_result()?;
        let local = string_from_bytebuffer(&local, "connect_stream", "local_addr");
        let peer = string_from_bytebuffer(&peer, "connect_stream", "peer_addr");
        Ok((fd, local, peer))
    }
}

/// Usercall `launch_thread`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub unsafe fn launch_thread() -> IoResult<()> {
    raw::launch_thread().from_sgx_result()
}

/// Usercall `exit`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn exit(panic: bool) -> ! {
    unsafe { raw::exit(panic) }
}

/// Usercall `wait`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn wait(event_mask: u64, timeout: u64) -> IoResult<u64> {
    unsafe { raw::wait(event_mask, timeout).from_sgx_result() }
}

/// Usercall `send`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn send(event_set: u64, tcs: Option<Tcs>) -> IoResult<()> {
    unsafe { raw::send(event_set, tcs).from_sgx_result() }
}

/// Usercall `insecure_time`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn insecure_time() -> Duration {
    let t = unsafe { raw::insecure_time() };
    Duration::new(t / 1_000_000_000, (t % 1_000_000_000) as _)
}

/// Usercall `alloc`. See the ABI documentation for more information.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub fn alloc(size: usize, alignment: usize) -> IoResult<*mut u8> {
    unsafe { raw::alloc(size, alignment).from_sgx_result() }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
#[doc(inline)]
pub use self::raw::free;

fn check_os_error(err: Result) -> i32 {
    // FIXME: not sure how to make sure all variants of Error are covered
    if err == Error::NotFound as _ ||
       err == Error::PermissionDenied as _ ||
       err == Error::ConnectionRefused as _ ||
       err == Error::ConnectionReset as _ ||
       err == Error::ConnectionAborted as _ ||
       err == Error::NotConnected as _ ||
       err == Error::AddrInUse as _ ||
       err == Error::AddrNotAvailable as _ ||
       err == Error::BrokenPipe as _ ||
       err == Error::AlreadyExists as _ ||
       err == Error::WouldBlock as _ ||
       err == Error::InvalidInput as _ ||
       err == Error::InvalidData as _ ||
       err == Error::TimedOut as _ ||
       err == Error::WriteZero as _ ||
       err == Error::Interrupted as _ ||
       err == Error::Other as _ ||
       err == Error::UnexpectedEof as _ ||
       ((Error::UserRangeStart as _)..=(Error::UserRangeEnd as _)).contains(&err)
    {
        err
    } else {
        rtabort!("Usercall: returned invalid error value {}", err)
    }
}

trait FromSgxResult {
    type Return;

    fn from_sgx_result(self) -> IoResult<Self::Return>;
}

impl<T> FromSgxResult for (Result, T) {
    type Return = T;

    fn from_sgx_result(self) -> IoResult<Self::Return> {
        if self.0 == RESULT_SUCCESS {
            Ok(self.1)
        } else {
            Err(IoError::from_raw_os_error(check_os_error(self.0)))
        }
    }
}

impl FromSgxResult for Result {
    type Return = ();

    fn from_sgx_result(self) -> IoResult<Self::Return> {
        if self == RESULT_SUCCESS {
            Ok(())
        } else {
            Err(IoError::from_raw_os_error(check_os_error(self)))
        }
    }
}
