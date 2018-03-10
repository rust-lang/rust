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

use io;
use libc::c_void;
use mem;
use ptr;
use sys::c;
use sys::dynamic_lib::DynamicLibrary;
use sys_common::backtrace::Frame;

macro_rules! sym {
    ($lib:expr, $e:expr, $t:ident) => (
        $lib.symbol($e).map(|f| unsafe {
            $crate::mem::transmute::<usize, $t>(f)
        })
    )
}

mod printing;

#[cfg(target_env = "gnu")]
#[path = "backtrace_gnu.rs"]
pub mod gnu;

pub use self::printing::{resolve_symname, foreach_symbol_fileline};

pub fn unwind_backtrace(frames: &mut [Frame])
    -> io::Result<(usize, BacktraceContext)>
{
    let dbghelp = DynamicLibrary::open("dbghelp.dll")?;

    // Fetch the symbols necessary from dbghelp.dll
    let SymInitialize = sym!(dbghelp, "SymInitialize", SymInitializeFn)?;
    let SymCleanup = sym!(dbghelp, "SymCleanup", SymCleanupFn)?;
    let StackWalkEx = sym!(dbghelp, "StackWalkEx", StackWalkExFn)?;

    // Allocate necessary structures for doing the stack walk
    let process = unsafe { c::GetCurrentProcess() };
    let thread = unsafe { c::GetCurrentThread() };
    let mut context: c::CONTEXT = unsafe { mem::zeroed() };
    unsafe { c::RtlCaptureContext(&mut context) };
    let mut frame: c::STACKFRAME_EX = unsafe { mem::zeroed() };
    frame.StackFrameSize = mem::size_of_val(&frame) as c::DWORD;
    let image = init_frame(&mut frame, &context);

    let backtrace_context = BacktraceContext {
        handle: process,
        SymCleanup,
        dbghelp,
    };

    // Initialize this process's symbols
    let ret = unsafe { SymInitialize(process, ptr::null_mut(), c::TRUE) };
    if ret != c::TRUE {
        return Ok((0, backtrace_context))
    }

    // And now that we're done with all the setup, do the stack walking!
    let mut i = 0;
    unsafe {
        while i < frames.len() &&
              StackWalkEx(image, process, thread, &mut frame, &mut context,
                          ptr::null_mut(),
                          ptr::null_mut(),
                          ptr::null_mut(),
                          ptr::null_mut(),
                          0) == c::TRUE
        {
            let addr = (frame.AddrPC.Offset - 1) as *const u8;

            frames[i] = Frame {
                symbol_addr: addr,
                exact_position: addr,
                inline_context: frame.InlineFrameContext,
            };
            i += 1;
        }
    }

    Ok((i, backtrace_context))
}

type SymInitializeFn =
    unsafe extern "system" fn(c::HANDLE, *mut c_void,
                              c::BOOL) -> c::BOOL;
type SymCleanupFn =
    unsafe extern "system" fn(c::HANDLE) -> c::BOOL;

type StackWalkExFn =
    unsafe extern "system" fn(c::DWORD, c::HANDLE, c::HANDLE,
                              *mut c::STACKFRAME_EX, *mut c::CONTEXT,
                              *mut c_void, *mut c_void,
                              *mut c_void, *mut c_void, c::DWORD) -> c::BOOL;

#[cfg(target_arch = "x86")]
fn init_frame(frame: &mut c::STACKFRAME_EX,
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
fn init_frame(frame: &mut c::STACKFRAME_EX,
              ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Rip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_AMD64
}

pub struct BacktraceContext {
    handle: c::HANDLE,
    SymCleanup: SymCleanupFn,
    // Only used in printing for msvc and not gnu
    #[allow(dead_code)]
    dbghelp: DynamicLibrary,
}

impl Drop for BacktraceContext {
    fn drop(&mut self) {
        unsafe { (self.SymCleanup)(self.handle); }
    }
}
