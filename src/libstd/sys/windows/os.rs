// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of `std::os` functionality for Windows

// FIXME: move various extern bindings from here into liblibc or
// something similar

use prelude::*;

use fmt;
use io::{IoResult, IoError};
use libc::{c_int, c_char, c_void};
use libc;
use os;
use path::BytesContainer;
use ptr;
use sync::atomic::{AtomicInt, INIT_ATOMIC_INT, SeqCst};
use sys::fs::FileDesc;
use slice;

use os::TMPBUF_SZ;
use libc::types::os::arch::extra::DWORD;

const BUF_BYTES : uint = 2048u;

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

pub fn fill_utf16_buf_and_decode(f: |*mut u16, DWORD| -> DWORD) -> Option<String> {
    unsafe {
        let mut n = TMPBUF_SZ as DWORD;
        let mut res = None;
        let mut done = false;
        while !done {
            let mut buf = Vec::from_elem(n as uint, 0u16);
            let k = f(buf.as_mut_ptr(), n);
            if k == (0 as DWORD) {
                done = true;
            } else if k == n &&
                      libc::GetLastError() ==
                      libc::ERROR_INSUFFICIENT_BUFFER as DWORD {
                n *= 2 as DWORD;
            } else if k >= n {
                n = k;
            } else {
                done = true;
            }
            if k != 0 && done {
                let sub = buf.slice(0, k as uint);
                // We want to explicitly catch the case when the
                // closure returned invalid UTF-16, rather than
                // set `res` to None and continue.
                let s = String::from_utf16(sub)
                    .expect("fill_utf16_buf_and_decode: closure created invalid UTF-16");
                res = Some(s)
            }
        }
        return res;
    }
}

pub fn getcwd() -> IoResult<Path> {
    use libc::DWORD;
    use libc::GetCurrentDirectoryW;
    use io::OtherIoError;

    let mut buf = [0 as u16, ..BUF_BYTES];
    unsafe {
        if libc::GetCurrentDirectoryW(buf.len() as DWORD, buf.as_mut_ptr()) == 0 as DWORD {
            return Err(IoError::last_error());
        }
    }

    match String::from_utf16(::str::truncate_utf16_at_nul(&buf)) {
        Some(ref cwd) => Ok(Path::new(cwd)),
        None => Err(IoError {
            kind: OtherIoError,
            desc: "GetCurrentDirectoryW returned invalid UTF-16",
            detail: None,
        }),
    }
}

pub unsafe fn get_env_pairs() -> Vec<Vec<u8>> {
    use libc::funcs::extra::kernel32::{
        GetEnvironmentStringsW,
        FreeEnvironmentStringsW
    };
    let ch = GetEnvironmentStringsW();
    if ch as uint == 0 {
        panic!("os::env() failure getting env string from OS: {}",
               os::last_os_error());
    }
    // Here, we lossily decode the string as UTF16.
    //
    // The docs suggest that the result should be in Unicode, but
    // Windows doesn't guarantee it's actually UTF16 -- it doesn't
    // validate the environment string passed to CreateProcess nor
    // SetEnvironmentVariable.  Yet, it's unlikely that returning a
    // raw u16 buffer would be of practical use since the result would
    // be inherently platform-dependent and introduce additional
    // complexity to this code.
    //
    // Using the non-Unicode version of GetEnvironmentStrings is even
    // worse since the result is in an OEM code page.  Characters that
    // can't be encoded in the code page would be turned into question
    // marks.
    let mut result = Vec::new();
    let mut i = 0;
    while *ch.offset(i) != 0 {
        let p = &*ch.offset(i);
        let mut len = 0;
        while *(p as *const _).offset(len) != 0 {
            len += 1;
        }
        let p = p as *const u16;
        let s = slice::from_raw_buf(&p, len as uint);
        result.push(String::from_utf16_lossy(s).into_bytes());
        i += len as int + 1;
    }
    FreeEnvironmentStringsW(ch);
    result
}

pub fn split_paths(unparsed: &[u8]) -> Vec<Path> {
    // On Windows, the PATH environment variable is semicolon separated.  Double
    // quotes are used as a way of introducing literal semicolons (since
    // c:\some;dir is a valid Windows path). Double quotes are not themselves
    // permitted in path names, so there is no way to escape a double quote.
    // Quoted regions can appear in arbitrary locations, so
    //
    //   c:\foo;c:\som"e;di"r;c:\bar
    //
    // Should parse as [c:\foo, c:\some;dir, c:\bar].
    //
    // (The above is based on testing; there is no clear reference available
    // for the grammar.)

    let mut parsed = Vec::new();
    let mut in_progress = Vec::new();
    let mut in_quote = false;

    for b in unparsed.iter() {
        match *b {
            b';' if !in_quote => {
                parsed.push(Path::new(in_progress.as_slice()));
                in_progress.truncate(0)
            }
            b'"' => {
                in_quote = !in_quote;
            }
            _  => {
                in_progress.push(*b);
            }
        }
    }
    parsed.push(Path::new(in_progress));
    parsed
}

pub fn join_paths<T: BytesContainer>(paths: &[T]) -> Result<Vec<u8>, &'static str> {
    let mut joined = Vec::new();
    let sep = b';';

    for (i, path) in paths.iter().map(|p| p.container_as_bytes()).enumerate() {
        if i > 0 { joined.push(sep) }
        if path.contains(&b'"') {
            return Err("path segment contains `\"`");
        } else if path.contains(&sep) {
            joined.push(b'"');
            joined.push_all(path);
            joined.push(b'"');
        } else {
            joined.push_all(path);
        }
    }

    Ok(joined)
}

pub fn load_self() -> Option<Vec<u8>> {
    unsafe {
        fill_utf16_buf_and_decode(|buf, sz| {
            libc::GetModuleFileNameW(0u as libc::DWORD, buf, sz)
        }).map(|s| s.into_string().into_bytes())
    }
}

pub fn chdir(p: &Path) -> IoResult<()> {
    let mut p = p.as_str().unwrap().utf16_units().collect::<Vec<u16>>();
    p.push(0);

    unsafe {
        match libc::SetCurrentDirectoryW(p.as_ptr()) != (0 as libc::BOOL) {
            true => Ok(()),
            false => Err(IoError::last_error()),
        }
    }
}

pub fn page_size() -> uint {
    use mem;
    unsafe {
        let mut info = mem::zeroed();
        libc::GetSystemInfo(&mut info);

        return info.dwPageSize as uint;
    }
}
