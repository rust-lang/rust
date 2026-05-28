#![expect(non_camel_case_types)]

use libc::size_t;

use crate::ffi::{c_char, c_int, c_void};
use crate::io;

//////////
// Time //
//////////

pub type zx_time_t = i64;

pub const ZX_TIME_INFINITE: zx_time_t = i64::MAX;

unsafe extern "C" {
    pub safe fn zx_clock_get_monotonic() -> zx_time_t;
}

/////////////
// Handles //
/////////////

pub type zx_handle_t = u32;

pub const ZX_HANDLE_INVALID: zx_handle_t = 0;

unsafe extern "C" {
    pub fn zx_handle_close(handle: zx_handle_t) -> zx_status_t;
}

/// A safe wrapper around `zx_handle_t`.
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

///////////
// Futex //
///////////

pub type zx_futex_t = crate::sync::atomic::Atomic<u32>;

unsafe extern "C" {
    pub fn zx_object_wait_one(
        handle: zx_handle_t,
        signals: zx_signals_t,
        timeout: zx_time_t,
        pending: *mut zx_signals_t,
    ) -> zx_status_t;

    pub fn zx_futex_wait(
        value_ptr: *const zx_futex_t,
        current_value: zx_futex_t,
        new_futex_owner: zx_handle_t,
        deadline: zx_time_t,
    ) -> zx_status_t;
    pub fn zx_futex_wake(value_ptr: *const zx_futex_t, wake_count: u32) -> zx_status_t;
    pub fn zx_futex_wake_single_owner(value_ptr: *const zx_futex_t) -> zx_status_t;
    pub safe fn zx_thread_self() -> zx_handle_t;
}

////////////////
// Properties //
////////////////

pub const ZX_PROP_NAME: u32 = 3;

unsafe extern "C" {
    pub fn zx_object_set_property(
        handle: zx_handle_t,
        property: u32,
        value: *const libc::c_void,
        value_size: libc::size_t,
    ) -> zx_status_t;
}

/////////////
// Signals //
/////////////

pub type zx_signals_t = u32;

pub const ZX_OBJECT_SIGNAL_3: zx_signals_t = 1 << 3;
pub const ZX_TASK_TERMINATED: zx_signals_t = ZX_OBJECT_SIGNAL_3;

/////////////////
// Object info //
/////////////////

// The upper four bits gives the minor version.
pub type zx_object_info_topic_t = u32;

pub const ZX_INFO_PROCESS: zx_object_info_topic_t = 3 | (1 << 28);

pub type zx_info_process_flags_t = u32;

// Returned for topic ZX_INFO_PROCESS
#[derive(Default)]
#[repr(C)]
pub struct zx_info_process_t {
    pub return_code: i64,
    pub start_time: zx_time_t,
    pub flags: zx_info_process_flags_t,
    pub reserved1: u32,
}

unsafe extern "C" {
    pub fn zx_object_get_info(
        handle: zx_handle_t,
        topic: u32,
        buffer: *mut c_void,
        buffer_size: size_t,
        actual_size: *mut size_t,
        avail: *mut size_t,
    ) -> zx_status_t;
}

///////////////
// Processes //
///////////////

#[derive(Default)]
#[repr(C)]
pub struct fdio_spawn_action_t {
    pub action: u32,
    pub reserved0: u32,
    pub local_fd: i32,
    pub target_fd: i32,
    pub reserved1: u64,
}

unsafe extern "C" {
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

    pub fn zx_task_kill(handle: zx_handle_t) -> zx_status_t;
}

// fdio_spawn_etc flags

pub const FDIO_SPAWN_CLONE_JOB: u32 = 0x0001;
pub const FDIO_SPAWN_CLONE_LDSVC: u32 = 0x0002;
pub const FDIO_SPAWN_CLONE_NAMESPACE: u32 = 0x0004;
pub const FDIO_SPAWN_CLONE_ENVIRON: u32 = 0x0010;
pub const FDIO_SPAWN_CLONE_UTC_CLOCK: u32 = 0x0020;

// fdio_spawn_etc actions

pub const FDIO_SPAWN_ACTION_TRANSFER_FD: u32 = 0x0002;

////////////
// Errors //
////////////

pub type zx_status_t = i32;

pub const ZX_OK: zx_status_t = 0;
pub const ZX_ERR_NOT_SUPPORTED: zx_status_t = -2;
pub const ZX_ERR_INVALID_ARGS: zx_status_t = -10;
pub const ZX_ERR_BAD_HANDLE: zx_status_t = -11;
pub const ZX_ERR_WRONG_TYPE: zx_status_t = -12;
pub const ZX_ERR_BAD_STATE: zx_status_t = -20;
pub const ZX_ERR_TIMED_OUT: zx_status_t = -21;

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
