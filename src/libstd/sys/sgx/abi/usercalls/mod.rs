// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use fortanix_sgx_abi::*;

use io::{Error as IoError, Result as IoResult};

pub mod alloc;
#[macro_use]
mod raw;

pub(crate) fn copy_user_buffer(buf: &alloc::UserRef<ByteBuffer>) -> Vec<u8> {
    unsafe {
        let buf = buf.to_enclave();
        alloc::User::from_raw_parts(buf.data as _, buf.len).to_enclave()
    }
}

pub fn read(fd: Fd, buf: &mut [u8]) -> IoResult<usize> {
    unsafe {
        let mut userbuf = alloc::User::<[u8]>::uninitialized(buf.len());
        let len = raw::read(fd, userbuf.as_mut_ptr(), userbuf.len()).from_sgx_result()?;
        userbuf[..len].copy_to_enclave(&mut buf[..len]);
        Ok(len)
    }
}

pub fn read_alloc(fd: Fd) -> IoResult<Vec<u8>> {
    unsafe {
        let mut userbuf = alloc::User::<ByteBuffer>::uninitialized();
        raw::read_alloc(fd, userbuf.as_raw_mut_ptr()).from_sgx_result()?;
        Ok(copy_user_buffer(&userbuf))
    }
}

pub fn write(fd: Fd, buf: &[u8]) -> IoResult<usize> {
    unsafe {
        let userbuf = alloc::User::new_from_enclave(buf);
        raw::write(fd, userbuf.as_ptr(), userbuf.len()).from_sgx_result()
    }
}

pub fn flush(fd: Fd) -> IoResult<()> {
    unsafe { raw::flush(fd).from_sgx_result() }
}

pub fn close(fd: Fd) {
    unsafe { raw::close(fd) }
}

fn string_from_bytebuffer(buf: &alloc::UserRef<ByteBuffer>, usercall: &str, arg: &str) -> String {
    String::from_utf8(copy_user_buffer(buf))
        .unwrap_or_else(|_| panic!("Usercall {}: expected {} to be valid UTF-8", usercall, arg))
}

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

pub fn launch_thread() -> IoResult<()> {
    unsafe { raw::launch_thread().from_sgx_result() }
}

pub fn exit(panic: bool) -> ! {
    unsafe { raw::exit(panic) }
}

pub fn wait(event_mask: u64, timeout: u64) -> IoResult<u64> {
    unsafe { raw::wait(event_mask, timeout).from_sgx_result() }
}

pub fn send(event_set: u64, tcs: Option<Tcs>) -> IoResult<()> {
    unsafe { raw::send(event_set, tcs).from_sgx_result() }
}

pub fn alloc(size: usize, alignment: usize) -> IoResult<*mut u8> {
    unsafe { raw::alloc(size, alignment).from_sgx_result() }
}

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
        panic!("Usercall: returned invalid error value {}", err)
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
