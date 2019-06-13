//! System bindings for the wasm/web platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for wasm. Note that this wasm is *not* the emscripten
//! wasm, so we have no runtime here.
//!
//! This is all super highly experimental and not actually intended for
//! wide/production use yet, it's still all in the experimental category. This
//! will likely change over time.
//!
//! Currently all functions here are basically stubs that immediately return
//! errors. The hope is that with a portability lint we can turn actually just
//! remove all this and just omit parts of the standard library if we're
//! compiling for wasm. That way it's a compile time error for something that's
//! guaranteed to be a runtime error!

use crate::os::raw::c_char;
use crate::ptr;
use crate::sys::os_str::Buf;
use crate::sys_common::{AsInner, FromInner};
use crate::ffi::{OsString, OsStr};
use crate::time::Duration;

pub mod alloc;
pub mod args;
pub mod cmath;
pub mod env;
pub mod fs;
pub mod io;
pub mod memchr;
pub mod net;
pub mod os;
pub mod path;
pub mod pipe;
pub mod process;
pub mod stack_overflow;
pub mod thread;
pub mod time;
pub mod stdio;

pub use crate::sys_common::os_str_bytes as os_str;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "atomics")] {
        #[path = "condvar_atomics.rs"]
        pub mod condvar;
        #[path = "mutex_atomics.rs"]
        pub mod mutex;
        #[path = "rwlock_atomics.rs"]
        pub mod rwlock;
        #[path = "thread_local_atomics.rs"]
        pub mod thread_local;
    } else {
        pub mod condvar;
        pub mod mutex;
        pub mod rwlock;
        pub mod thread_local;
    }
}

#[cfg(not(test))]
pub fn init() {
}

pub fn unsupported<T>() -> crate::io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> crate::io::Error {
    crate::io::Error::new(crate::io::ErrorKind::Other,
                   "operation not supported on wasm yet")
}

pub fn decode_error_kind(_code: i32) -> crate::io::ErrorKind {
    crate::io::ErrorKind::Other
}

// This enum is used as the storage for a bunch of types which can't actually
// exist.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Void {}

pub unsafe fn strlen(mut s: *const c_char) -> usize {
    let mut n = 0;
    while *s != 0 {
        n += 1;
        s = s.offset(1);
    }
    return n
}

pub unsafe fn abort_internal() -> ! {
    ExitSysCall::perform(1)
}

// We don't have randomness yet, but I totally used a random number generator to
// generate these numbers.
//
// More seriously though this is just for DOS protection in hash maps. It's ok
// if we don't do that on wasm just yet.
pub fn hashmap_random_keys() -> (u64, u64) {
    (1, 2)
}

// Implement a minimal set of system calls to enable basic IO
pub enum SysCallIndex {
    Read = 0,
    Write = 1,
    Exit = 2,
    Args = 3,
    GetEnv = 4,
    SetEnv = 5,
    Time = 6,
}

#[repr(C)]
pub struct ReadSysCall {
    fd: usize,
    ptr: *mut u8,
    len: usize,
    result: usize,
}

impl ReadSysCall {
    pub fn perform(fd: usize, buffer: &mut [u8]) -> usize {
        let mut call_record = ReadSysCall {
            fd,
            len: buffer.len(),
            ptr: buffer.as_mut_ptr(),
            result: 0
        };
        if unsafe { syscall(SysCallIndex::Read, &mut call_record) } {
            call_record.result
        } else {
            0
        }
    }
}

#[repr(C)]
pub struct WriteSysCall {
    fd: usize,
    ptr: *const u8,
    len: usize,
}

impl WriteSysCall {
    pub fn perform(fd: usize, buffer: &[u8]) {
        let mut call_record = WriteSysCall {
            fd,
            len: buffer.len(),
            ptr: buffer.as_ptr()
        };
        unsafe { syscall(SysCallIndex::Write, &mut call_record); }
    }
}

#[repr(C)]
pub struct ExitSysCall {
    code: usize,
}

impl ExitSysCall {
    pub fn perform(code: usize) -> ! {
        let mut call_record = ExitSysCall {
            code
        };
        unsafe {
            syscall(SysCallIndex::Exit, &mut call_record);
            crate::intrinsics::abort();
        }
    }
}

fn receive_buffer<E, F: FnMut(&mut [u8]) -> Result<usize, E>>(estimate: usize, mut f: F)
    -> Result<Vec<u8>, E>
{
    let mut buffer = vec![0; estimate];
    loop {
        let result = f(&mut buffer)?;
        if result <= buffer.len() {
            buffer.truncate(result);
            break;
        }
        buffer.resize(result, 0);
    }
    Ok(buffer)
}

#[repr(C)]
pub struct ArgsSysCall {
    ptr: *mut u8,
    len: usize,
    result: usize
}

impl ArgsSysCall {
    pub fn perform() -> Vec<OsString> {
        receive_buffer(1024, |buffer| -> Result<usize, !> {
            let mut call_record = ArgsSysCall {
                len: buffer.len(),
                ptr: buffer.as_mut_ptr(),
                result: 0
            };
            if unsafe { syscall(SysCallIndex::Args, &mut call_record) } {
                Ok(call_record.result)
            } else {
                Ok(0)
            }
        })
            .unwrap()
            .split(|b| *b == 0)
            .map(|s| FromInner::from_inner(Buf { inner: s.to_owned() }))
            .collect()
    }
}

#[repr(C)]
pub struct GetEnvSysCall {
    key_ptr: *const u8,
    key_len: usize,
    value_ptr: *mut u8,
    value_len: usize,
    result: usize
}

impl GetEnvSysCall {
    pub fn perform(key: &OsStr) -> Option<OsString> {
        let key_buf = &AsInner::as_inner(key).inner;
        receive_buffer(64, |buffer| {
            let mut call_record = GetEnvSysCall {
                key_len: key_buf.len(),
                key_ptr: key_buf.as_ptr(),
                value_len: buffer.len(),
                value_ptr: buffer.as_mut_ptr(),
                result: !0usize
            };
            if unsafe { syscall(SysCallIndex::GetEnv, &mut call_record) } {
                if call_record.result == !0usize {
                    Err(())
                } else {
                    Ok(call_record.result)
                }
            } else {
                Err(())
            }
        }).ok().map(|s| {
            FromInner::from_inner(Buf { inner: s })
        })
    }
}

#[repr(C)]
pub struct SetEnvSysCall {
    key_ptr: *const u8,
    key_len: usize,
    value_ptr: *const u8,
    value_len: usize
}

impl SetEnvSysCall {
    pub fn perform(key: &OsStr, value: Option<&OsStr>) {
        let key_buf = &AsInner::as_inner(key).inner;
        let value_buf = value.map(|v| &AsInner::as_inner(v).inner);
        let mut call_record = SetEnvSysCall {
            key_len: key_buf.len(),
            key_ptr: key_buf.as_ptr(),
            value_len: value_buf.map(|v| v.len()).unwrap_or(!0usize),
            value_ptr: value_buf.map(|v| v.as_ptr()).unwrap_or(ptr::null())
        };
        unsafe { syscall(SysCallIndex::SetEnv, &mut call_record); }
    }
}

pub enum TimeClock {
    Monotonic = 0,
    System = 1,
}

#[repr(C)]
pub struct TimeSysCall {
    clock: usize,
    secs_hi: usize,
    secs_lo: usize,
    nanos: usize
}

impl TimeSysCall {
    pub fn perform(clock: TimeClock) -> Duration {
        let mut call_record = TimeSysCall {
            clock: clock as usize,
            secs_hi: 0,
            secs_lo: 0,
            nanos: 0
        };
        if unsafe { syscall(SysCallIndex::Time, &mut call_record) } {
            Duration::new(
                ((call_record.secs_hi as u64) << 32) | (call_record.secs_lo as u64),
                call_record.nanos as u32
            )
        } else {
            panic!("Time system call is not implemented by WebAssembly host");
        }
    }
}

unsafe fn syscall<T>(index: SysCallIndex, data: &mut T) -> bool {
    #[cfg(feature = "wasm_syscall")]
    extern {
        #[no_mangle]
        fn rust_wasm_syscall(index: usize, data: *mut Void) -> usize;
    }

    #[cfg(not(feature = "wasm_syscall"))]
    unsafe fn rust_wasm_syscall(_index: usize, _data: *mut Void) -> usize { 0 }

    rust_wasm_syscall(index as usize, data as *mut T as *mut Void) != 0
}
