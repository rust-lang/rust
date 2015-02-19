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

#![allow(dead_code)]

use dynamic_lib::DynamicLibrary;
use ffi::CStr;
use intrinsics;
use old_io::{IoResult, Writer};
use libc;
use mem;
use ops::Drop;
use option::Option::{Some};
use old_path::Path;
use ptr;
use result::Result::{Ok, Err};
use slice::SliceExt;
use str::{self, StrExt};
use sync::{StaticMutex, MUTEX_INIT};

use sys_common::backtrace::*;

#[allow(non_snake_case)]
extern "system" {
    fn GetCurrentProcess() -> libc::HANDLE;
    fn GetCurrentThread() -> libc::HANDLE;
    fn RtlCaptureContext(ctx: *mut arch::CONTEXT);
}

type SymFromAddrFn =
    extern "system" fn(libc::HANDLE, u64, *mut u64,
                       *mut SYMBOL_INFO) -> libc::BOOL;
type SymInitializeFn =
    extern "system" fn(libc::HANDLE, *mut libc::c_void,
                       libc::BOOL) -> libc::BOOL;
type SymCleanupFn =
    extern "system" fn(libc::HANDLE) -> libc::BOOL;

type StackWalk64Fn =
    extern "system" fn(libc::DWORD, libc::HANDLE, libc::HANDLE,
                       *mut STACKFRAME64, *mut arch::CONTEXT,
                       *mut libc::c_void, *mut libc::c_void,
                       *mut libc::c_void, *mut libc::c_void) -> libc::BOOL;

const MAX_SYM_NAME: uint = 2000;
const IMAGE_FILE_MACHINE_I386: libc::DWORD = 0x014c;
const IMAGE_FILE_MACHINE_IA64: libc::DWORD = 0x0200;
const IMAGE_FILE_MACHINE_AMD64: libc::DWORD = 0x8664;

#[repr(C)]
struct SYMBOL_INFO {
    SizeOfStruct: libc::c_ulong,
    TypeIndex: libc::c_ulong,
    Reserved: [u64; 2],
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
    // note that windows has this as 1, but it basically just means that
    // the name is inline at the end of the struct. For us, we just bump
    // the struct size up to MAX_SYM_NAME.
    Name: [libc::c_char; MAX_SYM_NAME],
}


#[repr(C)]
enum ADDRESS_MODE {
    AddrMode1616,
    AddrMode1632,
    AddrModeReal,
    AddrModeFlat,
}

struct ADDRESS64 {
    Offset: u64,
    Segment: u16,
    Mode: ADDRESS_MODE,
}

struct STACKFRAME64 {
    AddrPC: ADDRESS64,
    AddrReturn: ADDRESS64,
    AddrFrame: ADDRESS64,
    AddrStack: ADDRESS64,
    AddrBStore: ADDRESS64,
    FuncTableEntry: *mut libc::c_void,
    Params: [u64; 4],
    Far: libc::BOOL,
    Virtual: libc::BOOL,
    Reserved: [u64; 3],
    KdHelp: KDHELP64,
}

struct KDHELP64 {
    Thread: u64,
    ThCallbackStack: libc::DWORD,
    ThCallbackBStore: libc::DWORD,
    NextCallback: libc::DWORD,
    FramePointer: libc::DWORD,
    KiCallUserMode: u64,
    KeUserCallbackDispatcher: u64,
    SystemRangeStart: u64,
    KiUserExceptionDispatcher: u64,
    StackBase: u64,
    StackLimit: u64,
    Reserved: [u64; 5],
}

#[cfg(target_arch = "x86")]
mod arch {
    use libc;

    const MAXIMUM_SUPPORTED_EXTENSION: uint = 512;

    #[repr(C)]
    pub struct CONTEXT {
        ContextFlags: libc::DWORD,
        Dr0: libc::DWORD,
        Dr1: libc::DWORD,
        Dr2: libc::DWORD,
        Dr3: libc::DWORD,
        Dr6: libc::DWORD,
        Dr7: libc::DWORD,
        FloatSave: FLOATING_SAVE_AREA,
        SegGs: libc::DWORD,
        SegFs: libc::DWORD,
        SegEs: libc::DWORD,
        SegDs: libc::DWORD,
        Edi: libc::DWORD,
        Esi: libc::DWORD,
        Ebx: libc::DWORD,
        Edx: libc::DWORD,
        Ecx: libc::DWORD,
        Eax: libc::DWORD,
        Ebp: libc::DWORD,
        Eip: libc::DWORD,
        SegCs: libc::DWORD,
        EFlags: libc::DWORD,
        Esp: libc::DWORD,
        SegSs: libc::DWORD,
        ExtendedRegisters: [u8; MAXIMUM_SUPPORTED_EXTENSION],
    }

    #[repr(C)]
    pub struct FLOATING_SAVE_AREA {
        ControlWord: libc::DWORD,
        StatusWord: libc::DWORD,
        TagWord: libc::DWORD,
        ErrorOffset: libc::DWORD,
        ErrorSelector: libc::DWORD,
        DataOffset: libc::DWORD,
        DataSelector: libc::DWORD,
        RegisterArea: [u8; 80],
        Cr0NpxState: libc::DWORD,
    }

    pub fn init_frame(frame: &mut super::STACKFRAME64,
                      ctx: &CONTEXT) -> libc::DWORD {
        frame.AddrPC.Offset = ctx.Eip as u64;
        frame.AddrPC.Mode = super::ADDRESS_MODE::AddrModeFlat;
        frame.AddrStack.Offset = ctx.Esp as u64;
        frame.AddrStack.Mode = super::ADDRESS_MODE::AddrModeFlat;
        frame.AddrFrame.Offset = ctx.Ebp as u64;
        frame.AddrFrame.Mode = super::ADDRESS_MODE::AddrModeFlat;
        super::IMAGE_FILE_MACHINE_I386
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use libc::{c_longlong, c_ulonglong};
    use libc::types::os::arch::extra::{WORD, DWORD, DWORDLONG};
    use simd;

    #[repr(C)]
    pub struct CONTEXT {
        _align_hack: [simd::u64x2; 0], // FIXME align on 16-byte
        P1Home: DWORDLONG,
        P2Home: DWORDLONG,
        P3Home: DWORDLONG,
        P4Home: DWORDLONG,
        P5Home: DWORDLONG,
        P6Home: DWORDLONG,

        ContextFlags: DWORD,
        MxCsr: DWORD,

        SegCs: WORD,
        SegDs: WORD,
        SegEs: WORD,
        SegFs: WORD,
        SegGs: WORD,
        SegSs: WORD,
        EFlags: DWORD,

        Dr0: DWORDLONG,
        Dr1: DWORDLONG,
        Dr2: DWORDLONG,
        Dr3: DWORDLONG,
        Dr6: DWORDLONG,
        Dr7: DWORDLONG,

        Rax: DWORDLONG,
        Rcx: DWORDLONG,
        Rdx: DWORDLONG,
        Rbx: DWORDLONG,
        Rsp: DWORDLONG,
        Rbp: DWORDLONG,
        Rsi: DWORDLONG,
        Rdi: DWORDLONG,
        R8:  DWORDLONG,
        R9:  DWORDLONG,
        R10: DWORDLONG,
        R11: DWORDLONG,
        R12: DWORDLONG,
        R13: DWORDLONG,
        R14: DWORDLONG,
        R15: DWORDLONG,

        Rip: DWORDLONG,

        FltSave: FLOATING_SAVE_AREA,

        VectorRegister: [M128A; 26],
        VectorControl: DWORDLONG,

        DebugControl: DWORDLONG,
        LastBranchToRip: DWORDLONG,
        LastBranchFromRip: DWORDLONG,
        LastExceptionToRip: DWORDLONG,
        LastExceptionFromRip: DWORDLONG,
    }

    #[repr(C)]
    pub struct M128A {
        _align_hack: [simd::u64x2; 0], // FIXME align on 16-byte
        Low:  c_ulonglong,
        High: c_longlong
    }

    #[repr(C)]
    pub struct FLOATING_SAVE_AREA {
        _align_hack: [simd::u64x2; 0], // FIXME align on 16-byte
        _Dummy: [u8; 512] // FIXME: Fill this out
    }

    pub fn init_frame(frame: &mut super::STACKFRAME64,
                      ctx: &CONTEXT) -> DWORD {
        frame.AddrPC.Offset = ctx.Rip as u64;
        frame.AddrPC.Mode = super::ADDRESS_MODE::AddrModeFlat;
        frame.AddrStack.Offset = ctx.Rsp as u64;
        frame.AddrStack.Mode = super::ADDRESS_MODE::AddrModeFlat;
        frame.AddrFrame.Offset = ctx.Rbp as u64;
        frame.AddrFrame.Mode = super::ADDRESS_MODE::AddrModeFlat;
        super::IMAGE_FILE_MACHINE_AMD64
    }
}

#[repr(C)]
struct Cleanup {
    handle: libc::HANDLE,
    SymCleanup: SymCleanupFn,
}

impl Drop for Cleanup {
    fn drop(&mut self) { (self.SymCleanup)(self.handle); }
}

pub fn write(w: &mut Writer) -> IoResult<()> {
    // According to windows documentation, all dbghelp functions are
    // single-threaded.
    static LOCK: StaticMutex = MUTEX_INIT;
    let _g = LOCK.lock();

    // Open up dbghelp.dll, we don't link to it explicitly because it can't
    // always be found. Additionally, it's nice having fewer dependencies.
    let path = Path::new("dbghelp.dll");
    let lib = match DynamicLibrary::open(Some(&path)) {
        Ok(lib) => lib,
        Err(..) => return Ok(()),
    };

    macro_rules! sym{ ($e:expr, $t:ident) => (unsafe {
        match lib.symbol($e) {
            Ok(f) => mem::transmute::<*mut u8, $t>(f),
            Err(..) => return Ok(())
        }
    }) }

    // Fetch the symbols necessary from dbghelp.dll
    let SymFromAddr = sym!("SymFromAddr", SymFromAddrFn);
    let SymInitialize = sym!("SymInitialize", SymInitializeFn);
    let SymCleanup = sym!("SymCleanup", SymCleanupFn);
    let StackWalk64 = sym!("StackWalk64", StackWalk64Fn);

    // Allocate necessary structures for doing the stack walk
    let process = unsafe { GetCurrentProcess() };
    let thread = unsafe { GetCurrentThread() };
    let mut context: arch::CONTEXT = unsafe { intrinsics::init() };
    unsafe { RtlCaptureContext(&mut context); }
    let mut frame: STACKFRAME64 = unsafe { intrinsics::init() };
    let image = arch::init_frame(&mut frame, &context);

    // Initialize this process's symbols
    let ret = SymInitialize(process, ptr::null_mut(), libc::TRUE);
    if ret != libc::TRUE { return Ok(()) }
    let _c = Cleanup { handle: process, SymCleanup: SymCleanup };

    // And now that we're done with all the setup, do the stack walking!
    let mut i = 0;
    try!(write!(w, "stack backtrace:\n"));
    while StackWalk64(image, process, thread, &mut frame, &mut context,
                      ptr::null_mut(),
                      ptr::null_mut(),
                      ptr::null_mut(),
                      ptr::null_mut()) == libc::TRUE{
        let addr = frame.AddrPC.Offset;
        if addr == frame.AddrReturn.Offset || addr == 0 ||
           frame.AddrReturn.Offset == 0 { break }

        i += 1;
        try!(write!(w, "  {:2}: {:#2$x}", i, addr, HEX_WIDTH));
        let mut info: SYMBOL_INFO = unsafe { intrinsics::init() };
        info.MaxNameLen = MAX_SYM_NAME as libc::c_ulong;
        // the struct size in C.  the value is different to
        // `size_of::<SYMBOL_INFO>() - MAX_SYM_NAME + 1` (== 81)
        // due to struct alignment.
        info.SizeOfStruct = 88;

        let mut displacement = 0u64;
        let ret = SymFromAddr(process, addr as u64, &mut displacement,
                              &mut info);

        if ret == libc::TRUE {
            try!(write!(w, " - "));
            let ptr = info.Name.as_ptr() as *const libc::c_char;
            let bytes = unsafe { CStr::from_ptr(ptr).to_bytes() };
            match str::from_utf8(bytes) {
                Ok(s) => try!(demangle(w, s)),
                Err(..) => try!(w.write_all(&bytes[..bytes.len()-1])),
            }
        }
        try!(w.write_all(&['\n' as u8]));
    }

    Ok(())
}
