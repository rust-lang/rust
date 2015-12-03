// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use io::prelude::*;
use libc;
use sys_common::backtrace::{output, output_fileline};

pub fn print(w: &mut Write, idx: isize, addr: *mut libc::c_void,
             symaddr: *mut libc::c_void) -> io::Result<()> {
    use env;
    use ffi::CStr;
    use ptr;

    ////////////////////////////////////////////////////////////////////////
    // libbacktrace.h API
    ////////////////////////////////////////////////////////////////////////
    type backtrace_syminfo_callback =
        extern "C" fn(data: *mut libc::c_void,
                      pc: libc::uintptr_t,
                      symname: *const libc::c_char,
                      symval: libc::uintptr_t,
                      symsize: libc::uintptr_t);
    type backtrace_full_callback =
        extern "C" fn(data: *mut libc::c_void,
                      pc: libc::uintptr_t,
                      filename: *const libc::c_char,
                      lineno: libc::c_int,
                      function: *const libc::c_char) -> libc::c_int;
    type backtrace_error_callback =
        extern "C" fn(data: *mut libc::c_void,
                      msg: *const libc::c_char,
                      errnum: libc::c_int);
    enum backtrace_state {}
    #[link(name = "backtrace", kind = "static")]
    #[cfg(not(test))]
    extern {}

    extern {
        fn backtrace_create_state(filename: *const libc::c_char,
                                  threaded: libc::c_int,
                                  error: backtrace_error_callback,
                                  data: *mut libc::c_void)
                                  -> *mut backtrace_state;
        fn backtrace_syminfo(state: *mut backtrace_state,
                             addr: libc::uintptr_t,
                             cb: backtrace_syminfo_callback,
                             error: backtrace_error_callback,
                             data: *mut libc::c_void) -> libc::c_int;
        fn backtrace_pcinfo(state: *mut backtrace_state,
                            addr: libc::uintptr_t,
                            cb: backtrace_full_callback,
                            error: backtrace_error_callback,
                            data: *mut libc::c_void) -> libc::c_int;
    }

    ////////////////////////////////////////////////////////////////////////
    // helper callbacks
    ////////////////////////////////////////////////////////////////////////

    type FileLine = (*const libc::c_char, libc::c_int);

    extern fn error_cb(_data: *mut libc::c_void, _msg: *const libc::c_char,
                       _errnum: libc::c_int) {
        // do nothing for now
    }
    extern fn syminfo_cb(data: *mut libc::c_void,
                         _pc: libc::uintptr_t,
                         symname: *const libc::c_char,
                         _symval: libc::uintptr_t,
                         _symsize: libc::uintptr_t) {
        let slot = data as *mut *const libc::c_char;
        unsafe { *slot = symname; }
    }
    extern fn pcinfo_cb(data: *mut libc::c_void,
                        _pc: libc::uintptr_t,
                        filename: *const libc::c_char,
                        lineno: libc::c_int,
                        _function: *const libc::c_char) -> libc::c_int {
        if !filename.is_null() {
            let slot = data as *mut &mut [FileLine];
            let buffer = unsafe {ptr::read(slot)};

            // if the buffer is not full, add file:line to the buffer
            // and adjust the buffer for next possible calls to pcinfo_cb.
            if !buffer.is_empty() {
                buffer[0] = (filename, lineno);
                unsafe { ptr::write(slot, &mut buffer[1..]); }
            }
        }

        0
    }

    // The libbacktrace API supports creating a state, but it does not
    // support destroying a state. I personally take this to mean that a
    // state is meant to be created and then live forever.
    //
    // I would love to register an at_exit() handler which cleans up this
    // state, but libbacktrace provides no way to do so.
    //
    // With these constraints, this function has a statically cached state
    // that is calculated the first time this is requested. Remember that
    // backtracing all happens serially (one global lock).
    //
    // An additionally oddity in this function is that we initialize the
    // filename via self_exe_name() to pass to libbacktrace. It turns out
    // that on Linux libbacktrace seamlessly gets the filename of the
    // current executable, but this fails on freebsd. by always providing
    // it, we make sure that libbacktrace never has a reason to not look up
    // the symbols. The libbacktrace API also states that the filename must
    // be in "permanent memory", so we copy it to a static and then use the
    // static as the pointer.
    //
    // FIXME: We also call self_exe_name() on DragonFly BSD. I haven't
    //        tested if this is required or not.
    unsafe fn init_state() -> *mut backtrace_state {
        static mut STATE: *mut backtrace_state = ptr::null_mut();
        static mut LAST_FILENAME: [libc::c_char; 256] = [0; 256];
        if !STATE.is_null() { return STATE }
        let selfname = if cfg!(target_os = "freebsd") ||
            cfg!(target_os = "dragonfly") ||
            cfg!(target_os = "bitrig") ||
            cfg!(target_os = "openbsd") ||
            cfg!(target_os = "windows") {
                env::current_exe().ok()
            } else {
                None
            };
        let filename = match selfname.as_ref().and_then(|s| s.to_str()) {
            Some(path) => {
                let bytes = path.as_bytes();
                if bytes.len() < LAST_FILENAME.len() {
                    let i = bytes.iter();
                    for (slot, val) in LAST_FILENAME.iter_mut().zip(i) {
                        *slot = *val as libc::c_char;
                    }
                    LAST_FILENAME.as_ptr()
                } else {
                    ptr::null()
                }
            }
            None => ptr::null(),
        };
        STATE = backtrace_create_state(filename, 0, error_cb,
                                       ptr::null_mut());
        STATE
    }

    ////////////////////////////////////////////////////////////////////////
    // translation
    ////////////////////////////////////////////////////////////////////////

    // backtrace errors are currently swept under the rug, only I/O
    // errors are reported
    let state = unsafe { init_state() };
    if state.is_null() {
        return output(w, idx, addr, None)
    }
    let mut data = ptr::null();
    let data_addr = &mut data as *mut *const libc::c_char;
    let ret = unsafe {
        backtrace_syminfo(state, symaddr as libc::uintptr_t,
                          syminfo_cb, error_cb,
                          data_addr as *mut libc::c_void)
    };
    if ret == 0 || data.is_null() {
        try!(output(w, idx, addr, None));
    } else {
        try!(output(w, idx, addr, Some(unsafe { CStr::from_ptr(data).to_bytes() })));
    }

    // pcinfo may return an arbitrary number of file:line pairs,
    // in the order of stack trace (i.e. inlined calls first).
    // in order to avoid allocation, we stack-allocate a fixed size of entries.
    const FILELINE_SIZE: usize = 32;
    let mut fileline_buf = [(ptr::null(), -1); FILELINE_SIZE];
    let ret;
    let fileline_count;
    {
        let mut fileline_win: &mut [FileLine] = &mut fileline_buf;
        let fileline_addr = &mut fileline_win as *mut &mut [FileLine];
        ret = unsafe {
            backtrace_pcinfo(state, addr as libc::uintptr_t,
                             pcinfo_cb, error_cb,
                             fileline_addr as *mut libc::c_void)
        };
        fileline_count = FILELINE_SIZE - fileline_win.len();
    }
    if ret == 0 {
        for (i, &(file, line)) in fileline_buf[..fileline_count].iter().enumerate() {
            if file.is_null() { continue; } // just to be sure
            let file = unsafe { CStr::from_ptr(file).to_bytes() };
            try!(output_fileline(w, file, line, i == FILELINE_SIZE - 1));
        }
    }

    Ok(())
}
