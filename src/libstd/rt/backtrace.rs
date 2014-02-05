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

use char::Char;
use container::Container;
use from_str::from_str;
use io::{IoResult, Writer};
use iter::Iterator;
use option::{Some, None};
use result::{Ok, Err};
use str::StrSlice;

pub use self::imp::write;

// This function is defined in this module so that the way to enable logging of
// backtraces has the word 'backtrace' in it: std::rt::backtrace.
pub fn log_enabled() -> bool {
    log_enabled!(::logging::DEBUG)
}

#[cfg(target_word_size = "64")] static HEX_WIDTH: uint = 18;
#[cfg(target_word_size = "32")] static HEX_WIDTH: uint = 10;

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
fn demangle(writer: &mut Writer, s: &str) -> IoResult<()> {
    // First validate the symbol. If it doesn't look like anything we're
    // expecting, we just print it literally. Note that we must handle non-rust
    // symbols because we could have any function in the backtrace.
    let mut valid = true;
    if s.len() > 4 && s.starts_with("_ZN") && s.ends_with("E") {
        let mut chars = s.slice(3, s.len() - 1).chars();
        while valid {
            let mut i = 0;
            for c in chars {
                if c.is_digit() {
                    i = i * 10 + c as uint - '0' as uint;
                } else {
                    break
                }
            }
            if i == 0 {
                valid = chars.next().is_none();
                break
            } else if chars.by_ref().take(i - 1).len() != i - 1 {
                valid = false;
            }
        }
    } else {
        valid = false;
    }

    // Alright, let's do this.
    if !valid {
        if_ok!(writer.write_str(s));
    } else {
        let mut s = s.slice_from(3);
        let mut first = true;
        while s.len() > 1 {
            if !first {
                if_ok!(writer.write_str("::"));
            } else {
                first = false;
            }
            let mut rest = s;
            while rest.char_at(0).is_digit() {
                rest = rest.slice_from(1);
            }
            let i: uint = from_str(s.slice_to(s.len() - rest.len())).unwrap();
            if_ok!(writer.write_str(rest.slice_to(i)));
            s = rest.slice_from(i);
        }
    }

    Ok(())
}

/// Backtrace support built on libunwind
///
/// It turns out that libunwind gives us an awesome function which will
/// basically tell us all IPs in the current backtrace. We can then use an
/// OS-specific method to query what the symbol is for this address, and then
/// print everything as a backtrace.
///
/// Sadly we don't have filenames/line numbers at this point, but hey, at least
/// this is better than nothing!
#[cfg(unix)]
mod imp {
    use c_str::CString;
    use cast;
    use io::{IoResult, IoError, Writer};
    use libc;
    use option::{Some, None, Option};
    use result::{Ok, Err};
    use unstable::intrinsics;
    use uw = rt::libunwind;

    #[inline(never)] // if we know this is a function call, we can skip it when
                     // tracing
    pub fn write(w: &mut Writer) -> IoResult<()> {
        if_ok!(writeln!(w, "stack backtrace:"));

        extern fn trace_fn(ctx: *uw::_Unwind_Context,
                           arg: *libc::c_void) -> uw::_Unwind_Reason_Code {
            let cx: &mut Context = unsafe { cast::transmute(arg) };
            let ip = unsafe { uw::_Unwind_GetIP(ctx) as *libc::c_void };

            cx.idx += 1;
            // Don't print out the first few frames (they're not user frames)
            if cx.idx <= 0 { return uw::_URC_NO_REASON }
            // Once we hit an error, stop trying to print more frames
            if cx.last_error.is_some() { return uw::_URC_FAILURE }

            match cx.print(ip) {
                Ok(()) => {}
                Err(e) => { cx.last_error = Some(e); }
            }
            return uw::_URC_NO_REASON
        }

        // We know for a fact that this function, `begin_unwind_inner`, and the
        // begin_unwind wrapper entering the `inner` variant are all
        // #[inline(never)], so skip the first two frames.
        let mut cx = Context { writer: w, last_error: None, idx: -3 };
        match unsafe {
            uw::_Unwind_Backtrace(trace_fn,
                                  &mut cx as *mut Context as *libc::c_void)
        } {
            uw::_URC_NO_REASON => {
                match cx.last_error {
                    Some(err) => Err(err),
                    None => Ok(())
                }
            }
            _ => Ok(()),
        }
    }

    struct Context<'a> {
        idx: int,
        writer: &'a mut Writer,
        last_error: Option<IoError>,
    }

    impl<'a> Context<'a> {
        fn print(&mut self, addr: *libc::c_void) -> IoResult<()> {
            struct Dl_info {
                dli_fname: *libc::c_char,
                dli_fbase: *libc::c_void,
                dli_sname: *libc::c_char,
                dli_saddr: *libc::c_void,
            }
            extern {
                fn dladdr(addr: *libc::c_void,
                          info: *mut Dl_info) -> libc::c_int;
            }

            let mut info: Dl_info = unsafe { intrinsics::init() };
            if unsafe { dladdr(addr, &mut info) == 0 } {
                if_ok!(writeln!(self.writer, "  {:2}: {:2$}", self.idx, addr,
                                super::HEX_WIDTH));
            } else {
                let symname = unsafe { CString::new(info.dli_sname, false) };
                if_ok!(write!(self.writer, "  {:2}: {:2$} - ", self.idx, addr,
                              super::HEX_WIDTH));
                match symname.as_str() {
                    Some(s) => if_ok!(super::demangle(self.writer, s)),
                    None => if_ok!(write!(self.writer, "<unknown>")),
                }
                if_ok!(self.writer.write(['\n' as u8]));
            }
            Ok(())
        }
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
#[allow(dead_code)]
mod imp {
    use c_str::CString;
    use container::Container;
    use io::{IoResult, Writer};
    use iter::Iterator;
    use libc;
    use mem;
    use ops::Drop;
    use option::{Some, None};
    use path::Path;
    use result::{Ok, Err};
    use str::StrSlice;
    use unstable::dynamic_lib::DynamicLibrary;
    use unstable::intrinsics;
    use unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
    use vec::ImmutableVector;

    extern "system" {
        fn GetCurrentProcess() -> libc::HANDLE;
        fn GetCurrentThread() -> libc::HANDLE;
        fn RtlCaptureContext(ctx: *mut arch::CONTEXT);
    }

    type SymFromAddrFn =
        extern "system" fn(libc::HANDLE, u64, *mut u64,
                           *mut SYMBOL_INFO) -> libc::BOOL;
    type SymInitializeFn =
        extern "system" fn(libc::HANDLE, *libc::c_void,
                           libc::BOOL) -> libc::BOOL;
    type SymCleanupFn =
        extern "system" fn(libc::HANDLE) -> libc::BOOL;

    type StackWalk64Fn =
        extern "system" fn(libc::DWORD, libc::HANDLE, libc::HANDLE,
                           *mut STACKFRAME64, *mut arch::CONTEXT,
                           *libc::c_void, *libc::c_void,
                           *libc::c_void, *libc::c_void) -> libc::BOOL;

    static MAX_SYM_NAME: uint = 2000;
    static IMAGE_FILE_MACHINE_I386: libc::DWORD = 0x014c;
    static IMAGE_FILE_MACHINE_IA64: libc::DWORD = 0x0200;
    static IMAGE_FILE_MACHINE_AMD64: libc::DWORD = 0x8664;

    #[packed]
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
        FuncTableEntry: *libc::c_void,
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

        static MAXIMUM_SUPPORTED_EXTENSION: uint = 512;

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
                          ctx: &CONTEXT) -> libc::DWORD {
            frame.AddrPC.Offset = ctx.Eip as u64;
            frame.AddrPC.Mode = super::AddrModeFlat;
            frame.AddrStack.Offset = ctx.Esp as u64;
            frame.AddrStack.Mode = super::AddrModeFlat;
            frame.AddrFrame.Offset = ctx.Ebp as u64;
            frame.AddrFrame.Mode = super::AddrModeFlat;
            super::IMAGE_FILE_MACHINE_I386
        }
    }

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
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        let _g = unsafe { LOCK.lock() };

        // Open up dbghelp.dll, we don't link to it explicitly because it can't
        // always be found. Additionally, it's nice having fewer dependencies.
        let path = Path::new("dbghelp.dll");
        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            Err(..) => return Ok(()),
        };

        macro_rules! sym( ($e:expr, $t:ident) => (
            match unsafe { lib.symbol::<$t>($e) } {
                Ok(f) => f,
                Err(..) => return Ok(())
            }
        ) )

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
        let ret = SymInitialize(process, 0 as *libc::c_void, libc::TRUE);
        if ret != libc::TRUE { return Ok(()) }
        let _c = Cleanup { handle: process, SymCleanup: SymCleanup };

        // And now that we're done with all the setup, do the stack walking!
        let mut i = 0;
        if_ok!(write!(w, "stack backtrace:\n"));
        while StackWalk64(image, process, thread, &mut frame, &mut context,
                          0 as *libc::c_void, 0 as *libc::c_void,
                          0 as *libc::c_void, 0 as *libc::c_void) == libc::TRUE{
            let addr = frame.AddrPC.Offset;
            if addr == frame.AddrReturn.Offset || addr == 0 ||
               frame.AddrReturn.Offset == 0 { break }

            i += 1;
            if_ok!(write!(w, "  {:2}: {:#2$x}", i, addr, super::HEX_WIDTH));
            let mut info: SYMBOL_INFO = unsafe { intrinsics::init() };
            info.MaxNameLen = MAX_SYM_NAME as libc::c_ulong;
            info.SizeOfStruct = (mem::size_of::<SYMBOL_INFO>() -
                                 info.Name.len() + 1) as libc::c_ulong;

            let mut displacement = 0u64;
            let ret = SymFromAddr(process, addr as u64, &mut displacement,
                                  &mut info);

            if ret == libc::TRUE {
                if_ok!(write!(w, " - "));
                let cstr = unsafe { CString::new(info.Name.as_ptr(), false) };
                let bytes = cstr.as_bytes();
                match cstr.as_str() {
                    Some(s) => if_ok!(super::demangle(w, s)),
                    None => if_ok!(w.write(bytes.slice_to(bytes.len() - 1))),
                }
            }
            if_ok!(w.write(['\n' as u8]));
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use io::MemWriter;
    use str;

    #[test]
    fn demangle() {
        macro_rules! t( ($a:expr, $b:expr) => ({
            let mut m = MemWriter::new();
            super::demangle(&mut m, $a);
            assert_eq!(str::from_utf8_owned(m.unwrap()).unwrap(), $b.to_owned());
        }) )

        t!("test", "test");
        t!("_ZN4testE", "test");
        t!("_ZN4test", "_ZN4test");
        t!("_ZN4test1a2bcE", "test::a::bc");
    }
}
