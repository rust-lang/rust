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

mod alloc;
#[macro_use]
mod raw;

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
