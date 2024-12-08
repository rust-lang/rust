#![allow(non_camel_case_types, unused)]

use libc::{c_int, c_void, size_t};

use crate::io;
use crate::mem::MaybeUninit;
use crate::os::raw::c_char;

pub type zx_handle_t = u32;
pub type zx_vaddr_t = usize;
pub type zx_rights_t = u32;
pub type zx_status_t = i32;

pub const ZX_HANDLE_INVALID: zx_handle_t = 0;

pub type zx_time_t = i64;
pub const ZX_TIME_INFINITE: zx_time_t = i64::MAX;

pub type zx_signals_t = u32;

pub const ZX_OBJECT_SIGNAL_3: zx_signals_t = 1 << 3;

pub const ZX_TASK_TERMINATED: zx_signals_t = ZX_OBJECT_SIGNAL_3;

pub const ZX_RIGHT_SAME_RIGHTS: zx_rights_t = 1 << 31;

// The upper four bits gives the minor version.
pub type zx_object_info_topic_t = u32;

pub const ZX_INFO_PROCESS: zx_object_info_topic_t = 3 | (1 << 28);

pub type zx_info_process_flags_t = u32;

pub fn zx_cvt<T>(t: T) -> io::Result<T>
where
    T: TryInto<zx_status_t> + Copy,
{
    if let Ok(status) = TryInto::try_into(t) {
        if status < 0 { Err(io::Error::from_raw_os_error(status)) } else { Ok(t) }
    } else {
        Err(io::Error::last_os_error())
    }
}

// Safe wrapper around zx_handle_t
pub struct Handle {
    raw: zx_handle_t,
}

impl Handle {
    pub fn new(raw: zx_handle_t) -> Handle {
        Handle { raw }
    }

    pub fn raw(&self) -> zx_handle_t {
        self.raw
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            zx_cvt(zx_handle_close(self.raw)).expect("Failed to close zx_handle_t");
        }
    }
}

// Returned for topic ZX_INFO_PROCESS
#[derive(Default)]
#[repr(C)]
pub struct zx_info_process_t {
    pub return_code: i64,
    pub start_time: zx_time_t,
    pub flags: zx_info_process_flags_t,
    pub reserved1: u32,
}

extern "C" {
    pub fn zx_job_default() -> zx_handle_t;

    pub fn zx_task_kill(handle: zx_handle_t) -> zx_status_t;

    pub fn zx_handle_close(handle: zx_handle_t) -> zx_status_t;

    pub fn zx_handle_duplicate(
        handle: zx_handle_t,
        rights: zx_rights_t,
        out: *const zx_handle_t,
    ) -> zx_handle_t;

    pub fn zx_object_wait_one(
        handle: zx_handle_t,
        signals: zx_signals_t,
        timeout: zx_time_t,
        pending: *mut zx_signals_t,
    ) -> zx_status_t;

    pub fn zx_object_get_info(
        handle: zx_handle_t,
        topic: u32,
        buffer: *mut c_void,
        buffer_size: size_t,
        actual_size: *mut size_t,
        avail: *mut size_t,
    ) -> zx_status_t;
}

#[derive(Default)]
#[repr(C)]
pub struct fdio_spawn_action_t {
    pub action: u32,
    pub reserved0: u32,
    pub local_fd: i32,
    pub target_fd: i32,
    pub reserved1: u64,
}

extern "C" {
    pub fn fdio_spawn_etc(
        job: zx_handle_t,
        flags: u32,
        path: *const c_char,
        argv: *const *const c_char,
        envp: *const *const c_char,
        action_count: size_t,
        actions: *const fdio_spawn_action_t,
        process: *mut zx_handle_t,
        err_msg: *mut c_char,
    ) -> zx_status_t;

    pub fn fdio_fd_clone(fd: c_int, out_handle: *mut zx_handle_t) -> zx_status_t;
    pub fn fdio_fd_create(handle: zx_handle_t, fd: *mut c_int) -> zx_status_t;
}

// fdio_spawn_etc flags

pub const FDIO_SPAWN_CLONE_JOB: u32 = 0x0001;
pub const FDIO_SPAWN_CLONE_LDSVC: u32 = 0x0002;
pub const FDIO_SPAWN_CLONE_NAMESPACE: u32 = 0x0004;
pub const FDIO_SPAWN_CLONE_STDIO: u32 = 0x0008;
pub const FDIO_SPAWN_CLONE_ENVIRON: u32 = 0x0010;
pub const FDIO_SPAWN_CLONE_UTC_CLOCK: u32 = 0x0020;
pub const FDIO_SPAWN_CLONE_ALL: u32 = 0xFFFF;

// fdio_spawn_etc actions

pub const FDIO_SPAWN_ACTION_CLONE_FD: u32 = 0x0001;
pub const FDIO_SPAWN_ACTION_TRANSFER_FD: u32 = 0x0002;

// Errors

#[allow(unused)]
pub const ERR_INTERNAL: zx_status_t = -1;

// ERR_NOT_SUPPORTED: The operation is not implemented, supported,
// or enabled.
#[allow(unused)]
pub const ERR_NOT_SUPPORTED: zx_status_t = -2;

// ERR_NO_RESOURCES: The system was not able to allocate some resource
// needed for the operation.
#[allow(unused)]
pub const ERR_NO_RESOURCES: zx_status_t = -3;

// ERR_NO_MEMORY: The system was not able to allocate memory needed
// for the operation.
#[allow(unused)]
pub const ERR_NO_MEMORY: zx_status_t = -4;

// ERR_CALL_FAILED: The second phase of zx_channel_call(; did not complete
// successfully.
#[allow(unused)]
pub const ERR_CALL_FAILED: zx_status_t = -5;

// ERR_INTERRUPTED_RETRY: The system call was interrupted, but should be
// retried.  This should not be seen outside of the VDSO.
#[allow(unused)]
pub const ERR_INTERRUPTED_RETRY: zx_status_t = -6;

// ======= Parameter errors =======
// ERR_INVALID_ARGS: an argument is invalid, ex. null pointer
#[allow(unused)]
pub const ERR_INVALID_ARGS: zx_status_t = -10;

// ERR_BAD_HANDLE: A specified handle value does not refer to a handle.
#[allow(unused)]
pub const ERR_BAD_HANDLE: zx_status_t = -11;

// ERR_WRONG_TYPE: The subject of the operation is the wrong type to
// perform the operation.
// Example: Attempting a message_read on a thread handle.
#[allow(unused)]
pub const ERR_WRONG_TYPE: zx_status_t = -12;

// ERR_BAD_SYSCALL: The specified syscall number is invalid.
#[allow(unused)]
pub const ERR_BAD_SYSCALL: zx_status_t = -13;

// ERR_OUT_OF_RANGE: An argument is outside the valid range for this
// operation.
#[allow(unused)]
pub const ERR_OUT_OF_RANGE: zx_status_t = -14;

// ERR_BUFFER_TOO_SMALL: A caller provided buffer is too small for
// this operation.
#[allow(unused)]
pub const ERR_BUFFER_TOO_SMALL: zx_status_t = -15;

// ======= Precondition or state errors =======
// ERR_BAD_STATE: operation failed because the current state of the
// object does not allow it, or a precondition of the operation is
// not satisfied
#[allow(unused)]
pub const ERR_BAD_STATE: zx_status_t = -20;

// ERR_TIMED_OUT: The time limit for the operation elapsed before
// the operation completed.
#[allow(unused)]
pub const ERR_TIMED_OUT: zx_status_t = -21;

// ERR_SHOULD_WAIT: The operation cannot be performed currently but
// potentially could succeed if the caller waits for a prerequisite
// to be satisfied, for example waiting for a handle to be readable
// or writable.
// Example: Attempting to read from a message pipe that has no
// messages waiting but has an open remote will return ERR_SHOULD_WAIT.
// Attempting to read from a message pipe that has no messages waiting
// and has a closed remote end will return ERR_REMOTE_CLOSED.
#[allow(unused)]
pub const ERR_SHOULD_WAIT: zx_status_t = -22;

// ERR_CANCELED: The in-progress operation (e.g., a wait) has been
// // canceled.
#[allow(unused)]
pub const ERR_CANCELED: zx_status_t = -23;

// ERR_PEER_CLOSED: The operation failed because the remote end
// of the subject of the operation was closed.
#[allow(unused)]
pub const ERR_PEER_CLOSED: zx_status_t = -24;

// ERR_NOT_FOUND: The requested entity is not found.
#[allow(unused)]
pub const ERR_NOT_FOUND: zx_status_t = -25;

// ERR_ALREADY_EXISTS: An object with the specified identifier
// already exists.
// Example: Attempting to create a file when a file already exists
// with that name.
#[allow(unused)]
pub const ERR_ALREADY_EXISTS: zx_status_t = -26;

// ERR_ALREADY_BOUND: The operation failed because the named entity
// is already owned or controlled by another entity. The operation
// could succeed later if the current owner releases the entity.
#[allow(unused)]
pub const ERR_ALREADY_BOUND: zx_status_t = -27;

// ERR_UNAVAILABLE: The subject of the operation is currently unable
// to perform the operation.
// Note: This is used when there's no direct way for the caller to
// observe when the subject will be able to perform the operation
// and should thus retry.
#[allow(unused)]
pub const ERR_UNAVAILABLE: zx_status_t = -28;

// ======= Permission check errors =======
// ERR_ACCESS_DENIED: The caller did not have permission to perform
// the specified operation.
#[allow(unused)]
pub const ERR_ACCESS_DENIED: zx_status_t = -30;

// ======= Input-output errors =======
// ERR_IO: Otherwise unspecified error occurred during I/O.
#[allow(unused)]
pub const ERR_IO: zx_status_t = -40;

// ERR_REFUSED: The entity the I/O operation is being performed on
// rejected the operation.
// Example: an I2C device NAK'ing a transaction or a disk controller
// rejecting an invalid command.
#[allow(unused)]
pub const ERR_IO_REFUSED: zx_status_t = -41;

// ERR_IO_DATA_INTEGRITY: The data in the operation failed an integrity
// check and is possibly corrupted.
// Example: CRC or Parity error.
#[allow(unused)]
pub const ERR_IO_DATA_INTEGRITY: zx_status_t = -42;

// ERR_IO_DATA_LOSS: The data in the operation is currently unavailable
// and may be permanently lost.
// Example: A disk block is irrecoverably damaged.
#[allow(unused)]
pub const ERR_IO_DATA_LOSS: zx_status_t = -43;

// Filesystem specific errors
#[allow(unused)]
pub const ERR_BAD_PATH: zx_status_t = -50;
#[allow(unused)]
pub const ERR_NOT_DIR: zx_status_t = -51;
#[allow(unused)]
pub const ERR_NOT_FILE: zx_status_t = -52;
// ERR_FILE_BIG: A file exceeds a filesystem-specific size limit.
#[allow(unused)]
pub const ERR_FILE_BIG: zx_status_t = -53;
// ERR_NO_SPACE: Filesystem or device space is exhausted.
#[allow(unused)]
pub const ERR_NO_SPACE: zx_status_t = -54;
