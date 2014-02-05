// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_camel_case_types)];

use c_str::CString;
use container::Container;
use io::{Writer, IoResult};
use iter::{range, Iterator};
use libc;
use option::{Some, None};
use result::{Ok, Err};
use unstable::raw::{Repr, Slice};
use vec::ImmutableVector;

static MAX_BACKTRACE: uint = 128;

pub fn log_enabled() -> bool {
    log_enabled!(::logging::DEBUG)
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
pub fn write(w: &mut Writer) -> IoResult<()> {
    use ptr::RawPtr;
    use ptr;

    extern {
        fn backtrace(array: *mut *mut libc::c_void,
                     size: libc::c_int) -> libc::c_int;
        fn backtrace_symbols(array: *mut *libc::c_void,
                             size: libc::c_int) -> **libc::c_char;
    }

    let array = [0u, ..MAX_BACKTRACE];
    let Slice { data, len } = array.repr();
    let len = unsafe {
        backtrace(data as *mut *mut libc::c_void, len as libc::c_int)
    };
    if len == 0 { return Ok(()) }

    let arr = unsafe {
        backtrace_symbols(data as *mut *libc::c_void, len)
    };
    if arr.is_null() { return Ok(()) }

    if_ok!(write!(w, "stack backtrace:\n"));
    for i in range(0, len) {
        let c_str = unsafe { CString::new(*ptr::offset(arr, i as int), false) };
        if_ok!(w.write_str("  "));
        let bytes = c_str.as_bytes();
        if_ok!(w.write(bytes.slice_to(bytes.len() - 1)));
        if_ok!(w.write_str("\n"));
    }

    Ok(())
}

// Windows uses functions from DbgHelp.dll which looks like it's not installed
// on mingw by default. Right now we don't have a great way of conditionally
// linking to something based on is presence, so for now we just ignore
// windows.
//
// The commented out code below is an untested implementation (but compiles) as
// I could never find DbgHelp.dll...
#[cfg(target_os = "win32")]
fn write(_: &mut Writer) -> IoResult<()> { Ok(()) }
/*
#[cfg(target_os = "win32")]
pub fn write(w: &mut Writer) -> IoResult<()> {
    use mem;
    use unstable::intrinsics;

    struct SYMBOL_INFO {
        SizeOfStruct: libc::c_ulong,
        TypeIndex: libc::c_ulong,
        Reserved: [u64, ..2],
        Index: libc::c_ulong,
        Size: libc::c_ulong,
        ModBase: u64,
        Flags: libc::c_ulong,
        Value: u64,
        Address: u64,
        Register: libc::c_ulong,
        Scope: libc::c_ulong,
        Tag: libc::c_ulong,
        NameLen: libc::c_ulong,
        MaxNameLen: libc::c_ulong,
        // note that windows has this as 1, but it basically just means that the
        // name is inline at the end of the struct. For us, we just bump the
        // struct size up to 256.
        Name: [libc::c_char, ..256],
    }

    #[link(name = "DbgHelp")]
    extern "system" {
        fn CaptureStackBackTrace(
            FramesToSkip: libc::c_ulong,
            FramesToCapture: libc::c_ulong,
            BackTrace: *mut *libc::c_void,
            BackTraceHash: *libc::c_ulong,
        ) -> libc::c_short;

        fn SymFromAddr(
            hProcess: libc::HANDLE,
            Address: i64,
            Displacement: *i64,
            Symbol: *mut SYMBOL_INFO,
        ) -> libc::BOOL;

        fn GetCurrentProcess() -> libc::HANDLE;
        fn SymInitialize(
            hProcess: libc::HANDLE,
            UserSearchPath: *libc::c_void,
            fInvadeProcess: libc::BOOL,
        ) -> libc::BOOL;
    }

    let process = unsafe { GetCurrentProcess() };
    let ret = unsafe { SymInitialize(process, 0 as *libc::c_void, libc::TRUE) };
    if ret != libc::TRUE { return Ok(()) }

    let array = [0u, ..MAX_BACKTRACE];
    let Slice { data, len } = array.repr();
    let len = unsafe {
        CaptureStackBackTrace(0, len as libc::c_ulong,
                              data as *mut *libc::c_void, 0 as *libc::c_ulong)
    };
    if len == 0 { return Ok(()) }

    if_ok!(write!(w, "stack backtrace:\n"));
    let mut info: SYMBOL_INFO = unsafe { intrinsics::init() };
    for i in range(0, len) {
        info.MaxNameLen = (info.Name.len() - 1) as libc::c_ulong;
        info.SizeOfStruct = (mem::size_of::<SYMBOL_INFO>() -
                             info.Name.len() + 1) as libc::c_ulong;

        let ret = unsafe {
            SymFromAddr(process, array[i] as i64, 0 as *i64, &mut info)
        };

        if ret == libc::TRUE {
            let cstr = unsafe { CString::new(info.Name.as_ptr(), false) };
            if_ok!(match cstr.as_str() {
                Some(s) => writeln!(w, "{}: {} - {:#10x}", i, s, info.Address),
                None => writeln!(w, "{}: <not-utf8> - {:#10x}", i, info.Address)
            })
        } else {
            if_ok!(w.write_str("  <SymFromAddr failed>\n"));
        }
    }

    Ok(())
}
*/


// Apparently freebsd has its backtrace functionality through an external
// libexecinfo library (not on the system by default). For now, just ignore
// writing backtraces on freebsd
#[cfg(target_os = "freebsd")]
fn write(_: &mut Writer) -> IoResult<()> { Ok(()) }

// Android, like freebsd, I believe also uses libexecinfo as a separate library.
// For now this is just a stub.
#[cfg(target_os = "android")]
fn write(_: &mut Writer) -> IoResult<()> { Ok(()) }
