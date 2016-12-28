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

pub const MX_HND_TYPE_JOB: u32 = 6;

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

// Handle Info entries associate a type and optional
// argument with each handle included in the process
// arguments message.
pub fn mx_hnd_info(hnd_type: u32, arg: u32) -> u32 {
    (hnd_type & 0xFFFF) | ((arg & 0xFFFF) << 16)
}

extern {
    pub fn mxio_get_startup_handle(id: u32) -> mx_handle_t;
}

// From `enum special_handles` in system/ulib/launchpad/launchpad.c
#[allow(unused)] pub const HND_LOADER_SVC: usize = 0;
// HND_EXEC_VMO = 1
#[allow(unused)] pub const HND_SPECIAL_COUNT: usize = 2;

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
