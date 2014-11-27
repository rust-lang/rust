// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: move various extern bindings from here into liblibc or
// something similar

use libc;
use libc::{c_int, c_char, c_void};
use prelude::*;
use io::{IoResult, IoError};
use sys::fs::FileDesc;
use ptr;

use os::TMPBUF_SZ;

pub fn errno() -> uint {
    use libc::types::os::arch::extra::DWORD;

    #[link_name = "kernel32"]
    extern "system" {
        fn GetLastError() -> DWORD;
    }

    unsafe {
        GetLastError() as uint
    }
}

/// Get a detailed string description for the given error number
pub fn error_string(errnum: i32) -> String {
    use libc::types::os::arch::extra::DWORD;
    use libc::types::os::arch::extra::LPWSTR;
    use libc::types::os::arch::extra::LPVOID;
    use libc::types::os::arch::extra::WCHAR;

    #[link_name = "kernel32"]
    extern "system" {
        fn FormatMessageW(flags: DWORD,
                          lpSrc: LPVOID,
                          msgId: DWORD,
                          langId: DWORD,
                          buf: LPWSTR,
                          nsize: DWORD,
                          args: *const c_void)
                          -> DWORD;
    }

    static FORMAT_MESSAGE_FROM_SYSTEM: DWORD = 0x00001000;
    static FORMAT_MESSAGE_IGNORE_INSERTS: DWORD = 0x00000200;

    // This value is calculated from the macro
    // MAKELANGID(LANG_SYSTEM_DEFAULT, SUBLANG_SYS_DEFAULT)
    let langId = 0x0800 as DWORD;

    let mut buf = [0 as WCHAR, ..TMPBUF_SZ];

    unsafe {
        let res = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM |
                                 FORMAT_MESSAGE_IGNORE_INSERTS,
                                 ptr::null_mut(),
                                 errnum as DWORD,
                                 langId,
                                 buf.as_mut_ptr(),
                                 buf.len() as DWORD,
                                 ptr::null());
        if res == 0 {
            // Sometimes FormatMessageW can fail e.g. system doesn't like langId,
            let fm_err = errno();
            return format!("OS Error {} (FormatMessageW() returned error {})", errnum, fm_err);
        }

        let msg = String::from_utf16(::str::truncate_utf16_at_nul(&buf));
        match msg {
            Some(msg) => format!("OS Error {}: {}", errnum, msg),
            None => format!("OS Error {} (FormatMessageW() returned invalid UTF-16)", errnum),
        }
    }
}

pub unsafe fn pipe() -> IoResult<(FileDesc, FileDesc)> {
    // Windows pipes work subtly differently than unix pipes, and their
    // inheritance has to be handled in a different way that I do not
    // fully understand. Here we explicitly make the pipe non-inheritable,
    // which means to pass it to a subprocess they need to be duplicated
    // first, as in std::run.
    let mut fds = [0, ..2];
    match libc::pipe(fds.as_mut_ptr(), 1024 as ::libc::c_uint,
                     (libc::O_BINARY | libc::O_NOINHERIT) as c_int) {
        0 => {
            assert!(fds[0] != -1 && fds[0] != 0);
            assert!(fds[1] != -1 && fds[1] != 0);
            Ok((FileDesc::new(fds[0], true), FileDesc::new(fds[1], true)))
        }
        _ => Err(IoError::last_error()),
    }
}
