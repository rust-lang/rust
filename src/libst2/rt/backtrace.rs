// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple backtrace functionality (to print on panic)

#![allow(non_camel_case_types)]

use io::{IoResult, Writer};
use iter::Iterator;
use option::{Some, None};
use os;
use result::{Ok, Err};
use str::{StrPrelude, from_str};
use sync::atomic;
use unicode::char::UnicodeChar;

pub use self::imp::write;

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn log_enabled() -> bool { unimplemented!() }

#[cfg(target_word_size = "64")] const HEX_WIDTH: uint = 18;
#[cfg(target_word_size = "32")] const HEX_WIDTH: uint = 10;

// All rust symbols are in theory lists of "::"-separated identifiers. Some
// assemblers, however, can't handle these characters in symbol names. To get
// around this, we use C++-style mangling. The mangling method is:
//
// 1. Prefix the symbol with "_ZN"
// 2. For each element of the path, emit the length plus the element
// 3. End the path with "E"
//
// For example, "_ZN4testE" => "test" and "_ZN3foo3bar" => "foo::bar".
//
// We're the ones printing our backtraces, so we can't rely on anything else to
// demangle our symbols. It's *much* nicer to look at demangled symbols, so
// this function is implemented to give us nice pretty output.
//
// Note that this demangler isn't quite as fancy as it could be. We have lots
// of other information in our symbols like hashes, version, type information,
// etc. Additionally, this doesn't handle glue symbols at all.
fn demangle(writer: &mut Writer, s: &str) -> IoResult<()> { unimplemented!() }

/// Backtrace support built on libgcc with some extra OS-specific support
///
/// Some methods of getting a backtrace:
///
/// * The backtrace() functions on unix. It turns out this doesn't work very
///   well for green threads on OSX, and the address to symbol portion of it
///   suffers problems that are described below.
///
/// * Using libunwind. This is more difficult than it sounds because libunwind
///   isn't installed everywhere by default. It's also a bit of a hefty library,
///   so possibly not the best option. When testing, libunwind was excellent at
///   getting both accurate backtraces and accurate symbols across platforms.
///   This route was not chosen in favor of the next option, however.
///
/// * We're already using libgcc_s for exceptions in rust (triggering task
///   unwinding and running destructors on the stack), and it turns out that it
///   conveniently comes with a function that also gives us a backtrace. All of
///   these functions look like _Unwind_*, but it's not quite the full
///   repertoire of the libunwind API. Due to it already being in use, this was
///   the chosen route of getting a backtrace.
///
/// After choosing libgcc_s for backtraces, the sad part is that it will only
/// give us a stack trace of instruction pointers. Thankfully these instruction
/// pointers are accurate (they work for green and native threads), but it's
/// then up to us again to figure out how to translate these addresses to
/// symbols. As with before, we have a few options. Before, that, a little bit
/// of an interlude about symbols. This is my very limited knowledge about
/// symbol tables, and this information is likely slightly wrong, but the
/// general idea should be correct.
///
/// When talking about symbols, it's helpful to know a few things about where
/// symbols are located. Some symbols are located in the dynamic symbol table
/// of the executable which in theory means that they're available for dynamic
/// linking and lookup. Other symbols end up only in the local symbol table of
/// the file. This loosely corresponds to pub and priv functions in Rust.
///
/// Armed with this knowledge, we know that our solution for address to symbol
/// translation will need to consult both the local and dynamic symbol tables.
/// With that in mind, here's our options of translating an address to
/// a symbol.
///
/// * Use dladdr(). The original backtrace()-based idea actually uses dladdr()
///   behind the scenes to translate, and this is why backtrace() was not used.
///   Conveniently, this method works fantastically on OSX. It appears dladdr()
///   uses magic to consult the local symbol table, or we're putting everything
///   in the dynamic symbol table anyway. Regardless, for OSX, this is the
///   method used for translation. It's provided by the system and easy to do.o
///
///   Sadly, all other systems have a dladdr() implementation that does not
///   consult the local symbol table. This means that most functions are blank
///   because they don't have symbols. This means that we need another solution.
///
/// * Use unw_get_proc_name(). This is part of the libunwind api (not the
///   libgcc_s version of the libunwind api), but involves taking a dependency
///   to libunwind. We may pursue this route in the future if we bundle
///   libunwind, but libunwind was unwieldy enough that it was not chosen at
///   this time to provide this functionality.
///
/// * Shell out to a utility like `readelf`. Crazy though it may sound, it's a
///   semi-reasonable solution. The stdlib already knows how to spawn processes,
///   so in theory it could invoke readelf, parse the output, and consult the
///   local/dynamic symbol tables from there. This ended up not getting chosen
///   due to the craziness of the idea plus the advent of the next option.
///
/// * Use `libbacktrace`. It turns out that this is a small library bundled in
///   the gcc repository which provides backtrace and symbol translation
///   functionality. All we really need from it is the backtrace functionality,
///   and we only really need this on everything that's not OSX, so this is the
///   chosen route for now.
///
/// In summary, the current situation uses libgcc_s to get a trace of stack
/// pointers, and we use dladdr() or libbacktrace to translate these addresses
/// to symbols. This is a bit of a hokey implementation as-is, but it works for
/// all unix platforms we support right now, so it at least gets the job done.
#[cfg(unix)]
mod imp {
    use c_str::CString;
    use io::{IoResult, Writer};
    use libc;
    use mem;
    use option::{Some, None, Option};
    use result::{Ok, Err};
    use rustrt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};

    /// As always - iOS on arm uses SjLj exceptions and
    /// _Unwind_Backtrace is even not available there. Still,
    /// backtraces could be extracted using a backtrace function,
    /// which thanks god is public
    ///
    /// As mentioned in a huge comment block above, backtrace doesn't
    /// play well with green threads, so while it is extremely nice
    /// and simple to use it should be used only on iOS devices as the
    /// only viable option.
    #[cfg(all(target_os = "ios", target_arch = "arm"))]
    #[inline(never)]
    pub fn write(w: &mut Writer) -> IoResult<()> { unimplemented!() }

    #[cfg(not(all(target_os = "ios", target_arch = "arm")))]
    #[inline(never)] // if we know this is a function call, we can skip it when
                     // tracing
    pub fn write(w: &mut Writer) -> IoResult<()> { unimplemented!() }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn print(w: &mut Writer, idx: int, addr: *mut libc::c_void) -> IoResult<()> { unimplemented!() }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn print(w: &mut Writer, idx: int, addr: *mut libc::c_void) -> IoResult<()> { unimplemented!() }

    // Finally, after all that work above, we can emit a symbol.
    fn output(w: &mut Writer, idx: int, addr: *mut libc::c_void,
              s: Option<CString>) -> IoResult<()> { unimplemented!() }

    /// Unwind library interface used for backtraces
    ///
    /// Note that the native libraries come from librustrt, not this
    /// module.
    /// Note that dead code is allowed as here are just bindings
    /// iOS doesn't use all of them it but adding more
    /// platform-specific configs pollutes the code too much
    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[allow(dead_code)]
    mod uw {
        pub use self::_Unwind_Reason_Code::*;

        use libc;

        #[repr(C)]
        pub enum _Unwind_Reason_Code {
            _URC_NO_REASON = 0,
            _URC_FOREIGN_EXCEPTION_CAUGHT = 1,
            _URC_FATAL_PHASE2_ERROR = 2,
            _URC_FATAL_PHASE1_ERROR = 3,
            _URC_NORMAL_STOP = 4,
            _URC_END_OF_STACK = 5,
            _URC_HANDLER_FOUND = 6,
            _URC_INSTALL_CONTEXT = 7,
            _URC_CONTINUE_UNWIND = 8,
            _URC_FAILURE = 9, // used only by ARM EABI
        }

        pub enum _Unwind_Context {}

        pub type _Unwind_Trace_Fn =
                extern fn(ctx: *mut _Unwind_Context,
                          arg: *mut libc::c_void) -> _Unwind_Reason_Code;

        extern {
            // No native _Unwind_Backtrace on iOS
            #[cfg(not(all(target_os = "ios", target_arch = "arm")))]
            pub fn _Unwind_Backtrace(trace: _Unwind_Trace_Fn,
                                     trace_argument: *mut libc::c_void)
                        -> _Unwind_Reason_Code;

            #[cfg(all(not(target_os = "android"),
                      not(all(target_os = "linux", target_arch = "arm"))))]
            pub fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> libc::uintptr_t;

            #[cfg(all(not(target_os = "android"),
                      not(all(target_os = "linux", target_arch = "arm"))))]
            pub fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void)
                -> *mut libc::c_void;
        }

        // On android, the function _Unwind_GetIP is a macro, and this is the
        // expansion of the macro. This is all copy/pasted directly from the
        // header file with the definition of _Unwind_GetIP.
        #[cfg(any(target_os = "android",
                  all(target_os = "linux", target_arch = "arm")))]
        pub unsafe fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> libc::uintptr_t { unimplemented!() }

        // This function also doesn't exist on Android or ARM/Linux, so make it
        // a no-op
        #[cfg(any(target_os = "android",
                  all(target_os = "linux", target_arch = "arm")))]
        pub unsafe fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void)
            -> *mut libc::c_void
        { unimplemented!() }
    }
}

/// As always, windows has something very different than unix, we mainly want
/// to avoid having to depend too much on libunwind for windows.
///
/// If you google around, you'll find a fair bit of references to built-in
/// functions to get backtraces on windows. It turns out that most of these are
/// in an external library called dbghelp. I was unable to find this library
/// via `-ldbghelp`, but it is apparently normal to do the `dlopen` equivalent
/// of it.
///
/// You'll also find that there's a function called CaptureStackBackTrace
/// mentioned frequently (which is also easy to use), but sadly I didn't have a
/// copy of that function in my mingw install (maybe it was broken?). Instead,
/// this takes the route of using StackWalk64 in order to walk the stack.
#[cfg(windows)]
#[allow(dead_code, non_snake_case)]
mod imp {
    use c_str::CString;
    use intrinsics;
    use io::{IoResult, Writer};
    use libc;
    use mem;
    use ops::Drop;
    use option::{Some, None};
    use path::Path;
    use result::{Ok, Err};
    use rustrt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
    use slice::SlicePrelude;
    use str::StrPrelude;
    use dynamic_lib::DynamicLibrary;

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
        // note that windows has this as 1, but it basically just means that
        // the name is inline at the end of the struct. For us, we just bump
        // the struct size up to MAX_SYM_NAME.
        Name: [libc::c_char, ..MAX_SYM_NAME],
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
        Params: [u64, ..4],
        Far: libc::BOOL,
        Virtual: libc::BOOL,
        Reserved: [u64, ..3],
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
        Reserved: [u64, ..5],
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
            ExtendedRegisters: [u8, ..MAXIMUM_SUPPORTED_EXTENSION],
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
            RegisterArea: [u8, ..80],
            Cr0NpxState: libc::DWORD,
        }

        pub fn init_frame(frame: &mut super::STACKFRAME64,
                          ctx: &CONTEXT) -> libc::DWORD { unimplemented!() }
    }

    #[cfg(target_arch = "x86_64")]
    mod arch {
        use libc::{c_longlong, c_ulonglong};
        use libc::types::os::arch::extra::{WORD, DWORD, DWORDLONG};
        use simd;

        #[repr(C)]
        pub struct CONTEXT {
            _align_hack: [simd::u64x2, ..0], // FIXME align on 16-byte
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

            VectorRegister: [M128A, .. 26],
            VectorControl: DWORDLONG,

            DebugControl: DWORDLONG,
            LastBranchToRip: DWORDLONG,
            LastBranchFromRip: DWORDLONG,
            LastExceptionToRip: DWORDLONG,
            LastExceptionFromRip: DWORDLONG,
        }

        #[repr(C)]
        pub struct M128A {
            _align_hack: [simd::u64x2, ..0], // FIXME align on 16-byte
            Low:  c_ulonglong,
            High: c_longlong
        }

        #[repr(C)]
        pub struct FLOATING_SAVE_AREA {
            _align_hack: [simd::u64x2, ..0], // FIXME align on 16-byte
            _Dummy: [u8, ..512] // FIXME: Fill this out
        }

        pub fn init_frame(frame: &mut super::STACKFRAME64,
                          ctx: &CONTEXT) -> DWORD { unimplemented!() }
    }

    #[repr(C)]
    struct Cleanup {
        handle: libc::HANDLE,
        SymCleanup: SymCleanupFn,
    }

    impl Drop for Cleanup {
        fn drop(&mut self) { unimplemented!() }
    }

    pub fn write(w: &mut Writer) -> IoResult<()> { unimplemented!() }
}
