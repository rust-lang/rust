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

pub use self::printing::{foreach_symbol_fileline, resolve_symname};
use self::printing::{load_printing_fns_ex, load_printing_fns_64};

pub fn unwind_backtrace(frames: &mut [Frame]) -> io::Result<(usize, BacktraceContext)> {
    let dbghelp = DynamicLibrary::open("dbghelp.dll")?;

    // Fetch the symbols necessary from dbghelp.dll
    let SymInitialize = sym!(dbghelp, "SymInitialize", SymInitializeFn)?;
    let SymCleanup = sym!(dbghelp, "SymCleanup", SymCleanupFn)?;


    // StackWalkEx might not be present and we'll fall back to StackWalk64
    let sw_var = match sym!(dbghelp, "StackWalkEx", StackWalkExFn) {
         Ok(StackWalkEx) =>
            StackWalkVariant::StackWalkEx(
                StackWalkEx,
                load_printing_fns_ex(&dbghelp)?,
            ),
        Err(e) => match sym!(dbghelp, "StackWalk64", StackWalk64Fn) {
            Ok(StackWalk64) =>
                StackWalkVariant::StackWalk64(
                    StackWalk64,
                    load_printing_fns_64(&dbghelp)?,
                ),
            Err(..) => return Err(e),
        },
    };

    // Allocate necessary structures for doing the stack walk
    let process = unsafe { c::GetCurrentProcess() };

    let backtrace_context = BacktraceContext {
        handle: process,
        SymCleanup,
        StackWalkVariant: sw_var,
        dbghelp,
    };

    // Initialize this process's symbols
    let ret = unsafe { SymInitialize(process, ptr::null_mut(), c::TRUE) };
    if ret != c::TRUE {
        return Ok((0, backtrace_context));
    }

    // And now that we're done with all the setup, do the stack walking!
    match backtrace_context.StackWalkVariant {
        StackWalkVariant::StackWalkEx(f, _) => set_frames_ex(f, frames, backtrace_context, process),
        StackWalkVariant::StackWalk64(f, _) => set_frames_64(f, frames, backtrace_context, process),
    }
}

fn set_frames_ex(
    StackWalkEx: StackWalkExFn,
    frames: &mut [Frame],
    backtrace_context: BacktraceContext,
    process: c::HANDLE,
) -> io::Result<(usize, BacktraceContext)> {
    let thread = unsafe { c::GetCurrentProcess() };
    let mut context: c::CONTEXT = unsafe { mem::zeroed() };
    unsafe { c::RtlCaptureContext(&mut context) };

    let mut frame: c::STACKFRAME_EX = unsafe { mem::zeroed() };
    frame.StackFrameSize = mem::size_of_val(&frame) as c::DWORD;
    let image = init_frame_ex(&mut frame, &context);

    let mut i = 0;
    unsafe {
        while i < frames.len()
            && StackWalkEx(
                image,
                process,
                thread,
                &mut frame,
                &mut context,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ) == c::TRUE
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

fn set_frames_64(
    StackWalk64: StackWalk64Fn,
    frames: &mut [Frame],
    backtrace_context: BacktraceContext,
    process: c::HANDLE,
) -> io::Result<(usize, BacktraceContext)> {
    let thread = unsafe { c::GetCurrentProcess() };
    let mut context: c::CONTEXT = unsafe { mem::zeroed() };
    unsafe { c::RtlCaptureContext(&mut context) };

    let mut frame: c::STACKFRAME64 = unsafe { mem::zeroed() };
    let image = init_frame_64(&mut frame, &context);

    // Start from -1 to avoid printing this stack frame, which will
    // always be exactly the same.
    let mut i = 0;
    unsafe {
        while i < frames.len()
            && StackWalk64(
                image,
                process,
                thread,
                &mut frame,
                &mut context,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            ) == c::TRUE
        {
            let addr = frame.AddrPC.Offset;
            if addr == frame.AddrReturn.Offset || addr == 0 || frame.AddrReturn.Offset == 0
            {
                break;
            }

            frames[i] = Frame {
                symbol_addr: (addr - 1) as *const u8,
                exact_position: (addr - 1) as *const u8,
                inline_context: 0,
            };
            i += 1;
        }
    }

    Ok((i, backtrace_context))
}

type SymInitializeFn = unsafe extern "system" fn(c::HANDLE, *mut c_void, c::BOOL) -> c::BOOL;
type SymCleanupFn = unsafe extern "system" fn(c::HANDLE) -> c::BOOL;

type StackWalkExFn = unsafe extern "system" fn(
    c::DWORD,
    c::HANDLE,
    c::HANDLE,
    *mut c::STACKFRAME_EX,
    *mut c::CONTEXT,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    c::DWORD,
) -> c::BOOL;

type StackWalk64Fn = unsafe extern "system" fn(
    c::DWORD,
    c::HANDLE,
    c::HANDLE,
    *mut c::STACKFRAME64,
    *mut c::CONTEXT,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut c_void,
) -> c::BOOL;

#[cfg(target_arch = "x86")]
fn init_frame_ex(frame: &mut c::STACKFRAME_EX, ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Eip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Esp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Ebp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_I386
}

#[cfg(target_arch = "x86_64")]
fn init_frame_ex(frame: &mut c::STACKFRAME_EX, ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Rip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_AMD64
}

#[cfg(target_arch = "x86")]
fn init_frame_64(frame: &mut c::STACKFRAME64, ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Eip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Esp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Ebp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_I386
}

#[cfg(target_arch = "x86_64")]
fn init_frame_64(frame: &mut c::STACKFRAME64, ctx: &c::CONTEXT) -> c::DWORD {
    frame.AddrPC.Offset = ctx.Rip as u64;
    frame.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp as u64;
    frame.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp as u64;
    frame.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
    c::IMAGE_FILE_MACHINE_AMD64
}

enum StackWalkVariant {
    StackWalkEx(StackWalkExFn, printing::PrintingFnsEx),
    StackWalk64(StackWalk64Fn, printing::PrintingFns64),
}


pub struct BacktraceContext {
    handle: c::HANDLE,
    SymCleanup: SymCleanupFn,
    // Only used in printing for msvc and not gnu
    // The gnu version is effectively a ZST dummy.
    #[allow(dead_code)]
    StackWalkVariant: StackWalkVariant,
    // keeping DynamycLibrary loaded until its functions no longer needed
    #[allow(dead_code)]
    dbghelp: DynamicLibrary,
}

impl Drop for BacktraceContext {
    fn drop(&mut self) {
        unsafe {
            (self.SymCleanup)(self.handle);
        }
    }
}
