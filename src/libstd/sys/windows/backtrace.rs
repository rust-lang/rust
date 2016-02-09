// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! As always, windows has something very different than unix, we mainly want
//! to avoid having to depend too much on libunwind for windows.
//!
//! If you google around, you'll find a fair bit of references to built-in
//! functions to get backtraces on windows. It turns out that most of these are
//! in an external library called dbghelp. I was unable to find this library
//! via `-ldbghelp`, but it is apparently normal to do the `dlopen` equivalent
//! of it.
//!
//! You'll also find that there's a function called CaptureStackBackTrace
//! mentioned frequently (which is also easy to use), but sadly I didn't have a
//! copy of that function in my mingw install (maybe it was broken?). Instead,
//! this takes the route of using StackWalk64 in order to walk the stack.

#![allow(dead_code, deprecated)]

use io::prelude::*;

use dynamic_lib::DynamicLibrary;
use ffi::CStr;
use io;
use libc::{c_ulong, c_int, c_char, c_void};
use mem;
use path::Path;
use ptr;
use sync::StaticMutex;
use sys::c;
use sys_common::backtrace::{output, output_fileline};

macro_rules! sym {
    ($lib:expr, $e:expr, $t:ident) => (unsafe {
        let lib = $lib;
        match lib.symbol($e) {
            Ok(f) => $crate::mem::transmute::<*mut u8, $t>(f),
            Err(..) => return Ok(())
        }
    })
}

type SymFromAddrFn =
    extern "system" fn(c::HANDLE, u64, *mut u64,
                       *mut c::SYMBOL_INFO) -> c::BOOL;
type SymGetLineFromAddr64Fn =
    extern "system" fn(c::HANDLE, u64, *mut u32,
                       *mut c::IMAGEHLP_LINE64) -> c::BOOL;
type SymInitializeFn =
    extern "system" fn(c::HANDLE, *mut c_void,
                       c::BOOL) -> c::BOOL;
type SymCleanupFn =
    extern "system" fn(c::HANDLE) -> c::BOOL;

type StackWalk64Fn =
    extern "system" fn(c::DWORD, c::HANDLE, c::HANDLE,
                       *mut c::STACKFRAME64, *mut c::CONTEXT,
                       *mut c_void, *mut c_void,
                       *mut c_void, *mut c_void) -> c::BOOL;

#[cfg(target_arch = "x86")]
pub fn init_frame(frame: &mut c::STACKFRAME64,
                  ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Eip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Esp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Ebp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_I386
}

#[cfg(target_arch = "x86_64")]
pub fn init_frame(frame: &mut c::STACKFRAME64,
                  ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Rip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_AMD64
}

struct Cleanup {
    handle: c::HANDLE,
    SymCleanup: SymCleanupFn,
}

impl Drop for Cleanup {
    fn drop(&mut self) { (self.SymCleanup)(self.handle); }
}

pub fn write(w: &mut Write) -> io::Result<()> {
    // According to windows documentation, all dbghelp functions are
    // single-threaded.
    static LOCK: StaticMutex = StaticMutex::new();
    let _g = LOCK.lock();

    // Open up dbghelp.dll, we don't link to it explicitly because it can't
    // always be found. Additionally, it's nice having fewer dependencies.
    let path = Path::new("dbghelp.dll");
    let dbghelp = match DynamicLibrary::open(Some(&path)) {
        Ok(lib) => lib,
        Err(..) => return Ok(()),
    };

    // Fetch the symbols necessary from dbghelp.dll
    let SymInitialize = sym!(&dbghelp, "SymInitialize", SymInitializeFn);
    let SymCleanup = sym!(&dbghelp, "SymCleanup", SymCleanupFn);
    let StackWalk64 = sym!(&dbghelp, "StackWalk64", StackWalk64Fn);

    // Allocate necessary structures for doing the stack walk
    let process = unsafe { c::GetCurrentProcess() };
    let thread = unsafe { c::GetCurrentThread() };
    let mut context: c::CONTEXT = unsafe { mem::zeroed() };
    unsafe { c::RtlCaptureContext(&mut context); }
    let mut frame: c::STACKFRAME64 = unsafe { mem::zeroed() };
    let image = init_frame(&mut frame, &context);

    // Initialize this process's symbols
    let ret = SymInitialize(process, ptr::null_mut(), c::TRUE);
    if ret != c::TRUE { return Ok(()) }
    let _c = Cleanup { handle: process, SymCleanup: SymCleanup };

    // And now that we're done with all the setup, do the stack walking!
    // Start from -1 to avoid printing this stack frame, which will
    // always be exactly the same.
    let mut i = -1;
    try!(write!(w, "stack backtrace:\n"));
    while StackWalk64(image, process, thread, &mut frame, &mut context,
                      ptr::null_mut(),
                      ptr::null_mut(),
                      ptr::null_mut(),
                      ptr::null_mut()) == c::TRUE {
        let addr = frame.AddrPC.Offset;
        if addr == frame.AddrReturn.Offset || addr == 0 ||
           frame.AddrReturn.Offset == 0 { break }

        i += 1;

        if i >= 0 {
            try!(print(w, i, addr-1, &dbghelp, process));
        }
    }

    Ok(())
}

fn print(w: &mut Write, i: isize, addr: u64, dbghelp: &DynamicLibrary,
         process: c::HANDLE) -> io::Result<()> {
    let SymFromAddr = sym!(dbghelp, "SymFromAddr", SymFromAddrFn);
    let SymGetLineFromAddr64 = sym!(dbghelp, "SymGetLineFromAddr64", SymGetLineFromAddr64Fn);

    let mut info: c::SYMBOL_INFO = unsafe { mem::zeroed() };
    info.MaxNameLen = c::MAX_SYM_NAME as c_ulong;
    // the struct size in C.  the value is different to
    // `size_of::<SYMBOL_INFO>() - MAX_SYM_NAME + 1` (== 81)
    // due to struct alignment.
    info.SizeOfStruct = 88;

    let mut displacement = 0u64;
    let ret = SymFromAddr(process, addr, &mut displacement, &mut info);

    let name = if ret == c::TRUE {
        let ptr = info.Name.as_ptr() as *const c_char;
        Some(unsafe { CStr::from_ptr(ptr).to_bytes() })
    } else {
        None
    };

    try!(output(w, i, addr as usize as *mut c_void, name));

    // Now find out the filename and line number
    let mut line: c::IMAGEHLP_LINE64 = unsafe { mem::zeroed() };
    line.SizeOfStruct = ::mem::size_of::<c::IMAGEHLP_LINE64>() as u32;

    let mut displacement = 0u32;
    let ret = SymGetLineFromAddr64(process, addr, &mut displacement, &mut line);
    if ret == c::TRUE {
        output_fileline(w,
                        unsafe { CStr::from_ptr(line.Filename).to_bytes() },
                        line.LineNumber as c_int,
                        false)
    } else {
        Ok(())
    }
}
