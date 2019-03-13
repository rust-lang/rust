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

use crate::io;
use crate::mem;
use crate::ptr;
use crate::sys::c;
use crate::sys::dynamic_lib::DynamicLibrary;
use crate::sys_common::backtrace::Frame;

use libc::c_void;

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
use self::printing::{load_printing_fns_64, load_printing_fns_ex};

pub fn unwind_backtrace(frames: &mut [Frame]) -> io::Result<(usize, BacktraceContext)> {
    let dbghelp = DynamicLibrary::open("dbghelp.dll")?;

    // Fetch the symbols necessary from dbghelp.dll
    let SymInitialize = sym!(dbghelp, "SymInitialize", SymInitializeFn)?;
    let SymCleanup = sym!(dbghelp, "SymCleanup", SymCleanupFn)?;

    // StackWalkEx might not be present and we'll fall back to StackWalk64
    let sw_var = match sym!(dbghelp, "StackWalkEx", StackWalkExFn) {
        Ok(StackWalkEx) => {
            StackWalkVariant::StackWalkEx(StackWalkEx, load_printing_fns_ex(&dbghelp)?)
        }
        Err(e) => match sym!(dbghelp, "StackWalk64", StackWalk64Fn) {
            Ok(StackWalk64) => {
                StackWalkVariant::StackWalk64(StackWalk64, load_printing_fns_64(&dbghelp)?)
            }
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
        StackWalkVariant::StackWalkEx(StackWalkEx, _) => {
            set_frames(StackWalkEx, frames).map(|i| (i, backtrace_context))
        }

        StackWalkVariant::StackWalk64(StackWalk64, _) => {
            set_frames(StackWalk64, frames).map(|i| (i, backtrace_context))
        }
    }
}

fn set_frames<W: StackWalker>(StackWalk: W, frames: &mut [Frame]) -> io::Result<usize> {
    let process = unsafe { c::GetCurrentProcess() };
    let thread = unsafe { c::GetCurrentProcess() };
    let mut context: c::CONTEXT = unsafe { mem::zeroed() };
    unsafe { c::RtlCaptureContext(&mut context) };
    let mut frame = W::Item::new();
    let image = frame.init(&context);

    let mut i = 0;
    while i < frames.len()
        && StackWalk.walk(image, process, thread, &mut frame, &mut context) == c::TRUE
    {
        let addr = frame.get_addr();
        frames[i] = Frame {
            symbol_addr: addr,
            exact_position: addr,
            inline_context: frame.get_inline_context(),
        };

        i += 1
    }
    Ok(i)
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

trait StackWalker {
    type Item: StackFrame;

    fn walk(
        &self,
        _: c::DWORD,
        _: c::HANDLE,
        _: c::HANDLE,
        _: &mut Self::Item,
        _: &mut c::CONTEXT
    ) -> c::BOOL;
}

impl StackWalker for StackWalkExFn {
    type Item = c::STACKFRAME_EX;
    fn walk(
        &self,
        image: c::DWORD,
        process: c::HANDLE,
        thread: c::HANDLE,
        frame: &mut Self::Item,
        context: &mut c::CONTEXT,
    ) -> c::BOOL {
        unsafe {
            self(
                image,
                process,
                thread,
                frame,
                context,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            )
        }
    }
}

impl StackWalker for StackWalk64Fn {
    type Item = c::STACKFRAME64;
    fn walk(
        &self,
        image: c::DWORD,
        process: c::HANDLE,
        thread: c::HANDLE,
        frame: &mut Self::Item,
        context: &mut c::CONTEXT,
    ) -> c::BOOL {
        unsafe {
            self(
                image,
                process,
                thread,
                frame,
                context,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
    }
}

trait StackFrame {
    fn new() -> Self;
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD;
    fn get_addr(&self) -> *const u8;
    fn get_inline_context(&self) -> u32;
}

impl StackFrame for c::STACKFRAME_EX {
    fn new() -> c::STACKFRAME_EX {
        unsafe { mem::zeroed() }
    }

    #[cfg(target_arch = "x86")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Eip as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Esp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.Ebp as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_I386
    }

    #[cfg(target_arch = "x86_64")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Rip as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Rsp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.Rbp as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_AMD64
    }

    #[cfg(target_arch = "arm")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Pc as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Sp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.R11 as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_ARMNT
    }

    #[cfg(target_arch = "aarch64")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Pc as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Sp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.Fp as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_ARM64
    }

    fn get_addr(&self) -> *const u8 {
        (self.AddrPC.Offset - 1) as *const u8
    }

    fn get_inline_context(&self) -> u32 {
        self.InlineFrameContext
    }
}

impl StackFrame for c::STACKFRAME64 {
    fn new() -> c::STACKFRAME64 {
        unsafe { mem::zeroed() }
    }

    #[cfg(target_arch = "x86")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Eip as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Esp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.Ebp as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_I386
    }

    #[cfg(target_arch = "x86_64")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Rip as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Rsp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.Rbp as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_AMD64
    }

    #[cfg(target_arch = "arm")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Pc as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Sp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.R11 as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_ARMNT
    }

    #[cfg(target_arch = "aarch64")]
    fn init(&mut self, ctx: &c::CONTEXT) -> c::DWORD {
        self.AddrPC.Offset = ctx.Pc as u64;
        self.AddrPC.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrStack.Offset = ctx.Sp as u64;
        self.AddrStack.Mode = c::ADDRESS_MODE::AddrModeFlat;
        self.AddrFrame.Offset = ctx.Fp as u64;
        self.AddrFrame.Mode = c::ADDRESS_MODE::AddrModeFlat;
        c::IMAGE_FILE_MACHINE_ARM64
    }

    fn get_addr(&self) -> *const u8 {
        (self.AddrPC.Offset - 1) as *const u8
    }

    fn get_inline_context(&self) -> u32 {
        0
    }
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
