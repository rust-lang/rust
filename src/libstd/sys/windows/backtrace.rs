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

#![allow(deprecated)] // dynamic_lib

use io::prelude::*;

use io;
use libc::c_void;
use mem;
use ptr;
use path::PathBuf;
use fs::{OpenOptions, File};
use sys::ext::fs::OpenOptionsExt;
use sys::c;
use sys::dynamic_lib::DynamicLibrary;
use sys::mutex::Mutex;
use sys::handle::Handle;

macro_rules! sym {
    ($lib:expr, $e:expr, $t:ident) => (
        match $lib.symbol($e) {
            Ok(f) => $crate::mem::transmute::<usize, $t>(f),
            Err(..) => return Ok(())
        }
    )
}

#[cfg(target_env = "msvc")]
#[path = "printing/msvc.rs"]
mod printing;

#[cfg(target_env = "gnu")]
#[path = "printing/gnu.rs"]
mod printing;

type SymInitializeFn =
    unsafe extern "system" fn(c::HANDLE, *mut c_void,
                              c::BOOL) -> c::BOOL;
type SymCleanupFn =
    unsafe extern "system" fn(c::HANDLE) -> c::BOOL;

type StackWalk64Fn =
    unsafe extern "system" fn(c::DWORD, c::HANDLE, c::HANDLE,
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
    fn drop(&mut self) {
        unsafe { (self.SymCleanup)(self.handle); }
    }
}

pub fn write(w: &mut Write) -> io::Result<()> {
    // According to windows documentation, all dbghelp functions are
    // single-threaded.
    static LOCK: Mutex = Mutex::new();
    unsafe {
        LOCK.lock();
        let res = _write(w);
        LOCK.unlock();
        return res
    }
}

unsafe fn _write(w: &mut Write) -> io::Result<()> {
    let dbghelp = match DynamicLibrary::open("dbghelp.dll") {
        Ok(lib) => lib,
        Err(..) => return Ok(()),
    };

    // Fetch the symbols necessary from dbghelp.dll
    let SymInitialize = sym!(dbghelp, "SymInitialize", SymInitializeFn);
    let SymCleanup = sym!(dbghelp, "SymCleanup", SymCleanupFn);
    let StackWalk64 = sym!(dbghelp, "StackWalk64", StackWalk64Fn);

    // Allocate necessary structures for doing the stack walk
    let process = c::GetCurrentProcess();
    let thread = c::GetCurrentThread();
    let mut context: c::CONTEXT = mem::zeroed();
    c::RtlCaptureContext(&mut context);
    let mut frame: c::STACKFRAME64 = mem::zeroed();
    let image = init_frame(&mut frame, &context);

    // Initialize this process's symbols
    let ret = SymInitialize(process, ptr::null_mut(), c::TRUE);
    if ret != c::TRUE { return Ok(()) }
    let _c = Cleanup { handle: process, SymCleanup: SymCleanup };

    // And now that we're done with all the setup, do the stack walking!
    // Start from -1 to avoid printing this stack frame, which will
    // always be exactly the same.
    let mut i = -1;
    write!(w, "stack backtrace:\n")?;
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
            printing::print(w, i, addr - 1, process, &dbghelp)?;
        }
    }

    Ok(())
}

fn query_full_process_image_name() -> io::Result<PathBuf> {
    unsafe {
        let process_handle = Handle::new(c::OpenProcess(c::PROCESS_QUERY_INFORMATION,
                                                        c::FALSE,
                                                        c::GetCurrentProcessId()));
        super::fill_utf16_buf(|buf, mut sz| {
            if c::QueryFullProcessImageNameW(process_handle.raw(), 0, buf, &mut sz) == 0 {
                0
            } else {
                sz
            }
        }, super::os2path)
    }
}

fn lock_and_get_executable_filename() -> io::Result<(PathBuf, File)> {
    // We query the current image name, open the file without FILE_SHARE_DELETE so it
    // can't be moved and then get the current image name again. If the names are the
    // same than we have successfully locked the file
    let image_name1 = query_full_process_image_name()?;
    let file = OpenOptions::new()
                .read(true)
                .share_mode(c::FILE_SHARE_READ | c::FILE_SHARE_WRITE)
                .open(&image_name1)?;
    let image_name2 = query_full_process_image_name()?;

    if image_name1 != image_name2 {
        return Err(io::Error::new(io::ErrorKind::Other,
                                  "executable moved while trying to lock it"));
    }

    Ok((image_name1, file))
}

// Get the executable filename for libbacktrace
// This returns the path in the ANSI code page and a File which should remain open
// for as long as the path should remain valid
pub fn get_executable_filename() -> io::Result<(Vec<i8>, File)> {
    let (executable, file) = lock_and_get_executable_filename()?;
    let u16_executable = super::to_u16s(executable.into_os_string())?;
    Ok((super::wide_char_to_multi_byte(c::CP_ACP, c::WC_NO_BEST_FIT_CHARS,
                                       &u16_executable, true)?, file))
}
