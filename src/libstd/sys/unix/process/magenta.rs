// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use convert::TryInto;
use io;
use os::raw::c_char;
use u64;

use libc::{c_int, c_void};

pub type mx_handle_t = i32;
pub type mx_vaddr_t = usize;
pub type mx_rights_t = u32;
pub type mx_status_t = i32;

pub type mx_size_t = usize;

pub const MX_HANDLE_INVALID: mx_handle_t = 0;

pub type mx_time_t = u64;
pub const MX_TIME_INFINITE : mx_time_t = u64::MAX;

pub type mx_signals_t = u32;

pub const MX_OBJECT_SIGNAL_3         : mx_signals_t = 1 << 3;

pub const MX_TASK_TERMINATED        : mx_signals_t = MX_OBJECT_SIGNAL_3;

pub const MX_RIGHT_SAME_RIGHTS  : mx_rights_t = 1 << 31;

pub type mx_object_info_topic_t = u32;

pub const MX_INFO_PROCESS         : mx_object_info_topic_t = 3;

pub fn mx_cvt<T>(t: T) -> io::Result<T> where T: TryInto<mx_status_t>+Copy {
    if let Ok(status) = TryInto::try_into(t) {
        if status < 0 {
            Err(io::Error::from_raw_os_error(status))
        } else {
            Ok(t)
        }
    } else {
        Err(io::Error::last_os_error())
    }
}

// Safe wrapper around mx_handle_t
pub struct Handle {
    raw: mx_handle_t,
}

impl Handle {
    pub fn new(raw: mx_handle_t) -> Handle {
        Handle {
            raw: raw,
        }
    }

    pub fn raw(&self) -> mx_handle_t {
        self.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { mx_cvt(mx_handle_close(self.raw)).expect("Failed to close mx_handle_t"); }
    }
}

// Common MX_INFO header
#[derive(Default)]
#[repr(C)]
pub struct mx_info_header_t {
    pub topic: u32,              // identifies the info struct
    pub avail_topic_size: u16,   // “native” size of the struct
    pub topic_size: u16,         // size of the returned struct (<=topic_size)
    pub avail_count: u32,        // number of records the kernel has
    pub count: u32,              // number of records returned (limited by buffer size)
}

#[derive(Default)]
#[repr(C)]
pub struct mx_record_process_t {
    pub return_code: c_int,
}

// Returned for topic MX_INFO_PROCESS
#[derive(Default)]
#[repr(C)]
pub struct mx_info_process_t {
    pub hdr: mx_info_header_t,
    pub rec: mx_record_process_t,
}

extern {
    static __magenta_job_default: mx_handle_t;

    pub fn mx_task_kill(handle: mx_handle_t) -> mx_status_t;

    pub fn mx_handle_close(handle: mx_handle_t) -> mx_status_t;

    pub fn mx_handle_duplicate(handle: mx_handle_t, rights: mx_rights_t,
                               out: *const mx_handle_t) -> mx_handle_t;

    pub fn mx_handle_wait_one(handle: mx_handle_t, signals: mx_signals_t, timeout: mx_time_t,
                              pending: *mut mx_signals_t) -> mx_status_t;

    pub fn mx_object_get_info(handle: mx_handle_t, topic: u32, buffer: *mut c_void,
                              buffer_size: mx_size_t, actual_size: *mut mx_size_t,
                              avail: *mut mx_size_t) -> mx_status_t;
}

pub fn mx_job_default() -> mx_handle_t {
    unsafe { return __magenta_job_default; }
}

// From `enum special_handles` in system/ulib/launchpad/launchpad.c
// HND_LOADER_SVC = 0
// HND_EXEC_VMO = 1
pub const HND_SPECIAL_COUNT: usize = 2;

#[repr(C)]
pub struct launchpad_t {
    argc: u32,
    envc: u32,
    args: *const c_char,
    args_len: usize,
    env: *const c_char,
    env_len: usize,

    handles: *mut mx_handle_t,
    handles_info: *mut u32,
    handle_count: usize,
    handle_alloc: usize,

    entry: mx_vaddr_t,
    base: mx_vaddr_t,
    vdso_base: mx_vaddr_t,

    stack_size: usize,

    special_handles: [mx_handle_t; HND_SPECIAL_COUNT],
    loader_message: bool,
}

extern {
    pub fn launchpad_create(job: mx_handle_t, name: *const c_char,
                            lp: *mut *mut launchpad_t) -> mx_status_t;

    pub fn launchpad_start(lp: *mut launchpad_t) -> mx_status_t;

    pub fn launchpad_destroy(lp: *mut launchpad_t);

    pub fn launchpad_arguments(lp: *mut launchpad_t, argc: c_int,
                               argv: *const *const c_char) -> mx_status_t;

    pub fn launchpad_environ(lp: *mut launchpad_t, envp: *const *const c_char) -> mx_status_t;

    pub fn launchpad_clone_mxio_root(lp: *mut launchpad_t) -> mx_status_t;

    pub fn launchpad_clone_mxio_cwd(lp: *mut launchpad_t) -> mx_status_t;

    pub fn launchpad_clone_fd(lp: *mut launchpad_t, fd: c_int, target_fd: c_int) -> mx_status_t;

    pub fn launchpad_transfer_fd(lp: *mut launchpad_t, fd: c_int, target_fd: c_int) -> mx_status_t;

    pub fn launchpad_elf_load(lp: *mut launchpad_t, vmo: mx_handle_t) -> mx_status_t;

    pub fn launchpad_add_vdso_vmo(lp: *mut launchpad_t) -> mx_status_t;

    pub fn launchpad_load_vdso(lp: *mut launchpad_t, vmo: mx_handle_t) -> mx_status_t;

    pub fn launchpad_vmo_from_file(filename: *const c_char) -> mx_handle_t;
}

// Errors

#[allow(unused)] pub const ERR_INTERNAL: mx_status_t = -1;

// ERR_NOT_SUPPORTED: The operation is not implemented, supported,
// or enabled.
#[allow(unused)] pub const ERR_NOT_SUPPORTED: mx_status_t = -2;

// ERR_NO_RESOURCES: The system was not able to allocate some resource
// needed for the operation.
#[allow(unused)] pub const ERR_NO_RESOURCES: mx_status_t = -5;

// ERR_NO_MEMORY: The system was not able to allocate memory needed
// for the operation.
#[allow(unused)] pub const ERR_NO_MEMORY: mx_status_t = -4;

// ERR_CALL_FAILED: The second phase of mx_channel_call(; did not complete
// successfully.
#[allow(unused)] pub const ERR_CALL_FAILED: mx_status_t = -53;

// ======= Parameter errors =======
// ERR_INVALID_ARGS: an argument is invalid, ex. null pointer
#[allow(unused)] pub const ERR_INVALID_ARGS: mx_status_t = -10;

// ERR_WRONG_TYPE: The subject of the operation is the wrong type to
// perform the operation.
// Example: Attempting a message_read on a thread handle.
#[allow(unused)] pub const ERR_WRONG_TYPE: mx_status_t = -54;

// ERR_BAD_SYSCALL: The specified syscall number is invalid.
#[allow(unused)] pub const ERR_BAD_SYSCALL: mx_status_t = -11;

// ERR_BAD_HANDLE: A specified handle value does not refer to a handle.
#[allow(unused)] pub const ERR_BAD_HANDLE: mx_status_t = -12;

// ERR_OUT_OF_RANGE: An argument is outside the valid range for this
// operation.
#[allow(unused)] pub const ERR_OUT_OF_RANGE: mx_status_t = -13;

// ERR_BUFFER_TOO_SMALL: A caller provided buffer is too small for
// this operation.
#[allow(unused)] pub const ERR_BUFFER_TOO_SMALL: mx_status_t = -14;

// ======= Precondition or state errors =======
// ERR_BAD_STATE: operation failed because the current state of the
// object does not allow it, or a precondition of the operation is
// not satisfied
#[allow(unused)] pub const ERR_BAD_STATE: mx_status_t = -20;

// ERR_NOT_FOUND: The requested entity is not found.
#[allow(unused)] pub const ERR_NOT_FOUND: mx_status_t = -3;

// ERR_ALREADY_EXISTS: An object with the specified identifier
// already exists.
// Example: Attempting to create a file when a file already exists
// with that name.
#[allow(unused)] pub const ERR_ALREADY_EXISTS: mx_status_t = -15;

// ERR_ALREADY_BOUND: The operation failed because the named entity
// is already owned or controlled by another entity. The operation
// could succeed later if the current owner releases the entity.
#[allow(unused)] pub const ERR_ALREADY_BOUND: mx_status_t = -16;

// ERR_TIMED_OUT: The time limit for the operation elapsed before
// the operation completed.
#[allow(unused)] pub const ERR_TIMED_OUT: mx_status_t = -23;

// ERR_HANDLE_CLOSED: a handle being waited on was closed
#[allow(unused)] pub const ERR_HANDLE_CLOSED: mx_status_t = -24;

// ERR_REMOTE_CLOSED: The operation failed because the remote end
// of the subject of the operation was closed.
#[allow(unused)] pub const ERR_REMOTE_CLOSED: mx_status_t = -25;

// ERR_UNAVAILABLE: The subject of the operation is currently unable
// to perform the operation.
// Note: This is used when there's no direct way for the caller to
// observe when the subject will be able to perform the operation
// and should thus retry.
#[allow(unused)] pub const ERR_UNAVAILABLE: mx_status_t = -26;

// ERR_SHOULD_WAIT: The operation cannot be performed currently but
// potentially could succeed if the caller waits for a prerequisite
// to be satisfied, for example waiting for a handle to be readable
// or writable.
// Example: Attempting to read from a message pipe that has no
// messages waiting but has an open remote will return ERR_SHOULD_WAIT.
// Attempting to read from a message pipe that has no messages waiting
// and has a closed remote end will return ERR_REMOTE_CLOSED.
#[allow(unused)] pub const ERR_SHOULD_WAIT: mx_status_t = -27;

// ======= Permission check errors =======
// ERR_ACCESS_DENIED: The caller did not have permission to perform
// the specified operation.
#[allow(unused)] pub const ERR_ACCESS_DENIED: mx_status_t = -30;

// ======= Input-output errors =======
// ERR_IO: Otherwise unspecified error occurred during I/O.
#[allow(unused)] pub const ERR_IO: mx_status_t = -40;

// ERR_REFUSED: The entity the I/O operation is being performed on
// rejected the operation.
// Example: an I2C device NAK'ing a transaction or a disk controller
// rejecting an invalid command.
#[allow(unused)] pub const ERR_IO_REFUSED: mx_status_t = -41;

// ERR_IO_DATA_INTEGRITY: The data in the operation failed an integrity
// check and is possibly corrupted.
// Example: CRC or Parity error.
#[allow(unused)] pub const ERR_IO_DATA_INTEGRITY: mx_status_t = -42;

// ERR_IO_DATA_LOSS: The data in the operation is currently unavailable
// and may be permanently lost.
// Example: A disk block is irrecoverably damaged.
#[allow(unused)] pub const ERR_IO_DATA_LOSS: mx_status_t = -43;

// Filesystem specific errors
#[allow(unused)] pub const ERR_BAD_PATH: mx_status_t = -50;
#[allow(unused)] pub const ERR_NOT_DIR: mx_status_t = -51;
#[allow(unused)] pub const ERR_NOT_FILE: mx_status_t = -52;
