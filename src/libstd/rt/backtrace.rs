// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple backtrace functionality (to print on failure)

#![allow(non_camel_case_types)]

use collections::Collection;
use from_str::from_str;
use io::{IoResult, Writer};
use iter::Iterator;
use option::{Some, None};
use os;
use result::{Ok, Err};
use str::StrSlice;
use sync::atomic;
use unicode::char::UnicodeChar;

pub use self::imp::write;

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn log_enabled() -> bool {
    static mut ENABLED: atomic::AtomicInt = atomic::INIT_ATOMIC_INT;
    unsafe {
        match ENABLED.load(atomic::SeqCst) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }

    let val = match os::getenv("RUST_BACKTRACE") {
        Some(..) => 2,
        None => 1,
    };
    unsafe { ENABLED.store(val, atomic::SeqCst); }
    val == 2
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
            } else if chars.by_ref().take(i - 1).count() != i - 1 {
                valid = false;
            }
        }
    } else {
        valid = false;
    }

    // Alright, let's do this.
    if !valid {
        try!(writer.write_str(s));
    } else {
        let mut s = s.slice_from(3);
        let mut first = true;
        while s.len() > 1 {
            if !first {
                try!(writer.write_str("::"));
            } else {
                first = false;
            }
            let mut rest = s;
            while rest.char_at(0).is_digit() {
                rest = rest.slice_from(1);
            }
            let i: uint = from_str(s.slice_to(s.len() - rest.len())).unwrap();
            s = rest.slice_from(i);
            rest = rest.slice_to(i);
            while rest.len() > 0 {
                if rest.starts_with("$") {
                    macro_rules! demangle(
                        ($($pat:expr => $demangled:expr),*) => ({
                            $(if rest.starts_with($pat) {
                                try!(writer.write_str($demangled));
                                rest = rest.slice_from($pat.len());
                              } else)*
                            {
                                try!(writer.write_str(rest));
                                break;
                            }

                        })
                    )
                    // see src/librustc/back/link.rs for these mappings
                    demangle! (
                        "$SP$" => "@",
                        "$UP$" => "Box",
                        "$RP$" => "*",
                        "$BP$" => "&",
                        "$LT$" => "<",
                        "$GT$" => ">",
                        "$LP$" => "(",
                        "$RP$" => ")",
                        "$C$"  => ",",

                        // in theory we can demangle any Unicode code point, but
                        // for simplicity we just catch the common ones.
                        "$x20" => " ",
                        "$x27" => "'",
                        "$x5b" => "[",
                        "$x5d" => "]"
                    )
                } else {
                    let idx = match rest.find('$') {
                        None => rest.len(),
                        Some(i) => i,
                    };
                    try!(writer.write_str(rest.slice_to(idx)));
                    rest = rest.slice_from(idx);
                }
            }
        }
    }

    Ok(())
}

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
    use rt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};

    /// As always - iOS on arm uses SjLj exceptions and
    /// _Unwind_Backtrace is even not available there. Still,
    /// backtraces could be extracted using a backtrace function,
    /// which thanks god is public
    ///
    /// As mentioned in a huge comment block above, backtrace doesn't
    /// play well with green threads, so while it is extremely nice
    /// and simple to use it should be used only on iOS devices as the
    /// only viable option.
    #[cfg(target_os = "ios", target_arch = "arm")]
    #[inline(never)]
    pub fn write(w: &mut Writer) -> IoResult<()> {
        use iter::{Iterator, range};
        use result;
        use slice::{MutableSlice};

        extern {
            fn backtrace(buf: *mut *mut libc::c_void,
                         sz: libc::c_int) -> libc::c_int;
        }

        // while it doesn't requires lock for work as everything is
        // local, it still displays much nicer backtraces when a
        // couple of tasks fail simultaneously
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        let _g = unsafe { LOCK.lock() };

        try!(writeln!(w, "stack backtrace:"));
        // 100 lines should be enough
        static SIZE: uint = 100;
        let mut buf: [*mut libc::c_void, ..SIZE] = unsafe {mem::zeroed()};
        let cnt = unsafe { backtrace(buf.as_mut_ptr(), SIZE as libc::c_int) as uint};

        // skipping the first one as it is write itself
        result::fold_(range(1, cnt).map(|i| {
            print(w, i as int, buf[i])
        }))
    }

    #[cfg(not(target_os = "ios", target_arch = "arm"))]
    #[inline(never)] // if we know this is a function call, we can skip it when
                     // tracing
    pub fn write(w: &mut Writer) -> IoResult<()> {
        use io::IoError;

        struct Context<'a> {
            idx: int,
            writer: &'a mut Writer+'a,
            last_error: Option<IoError>,
        }

        // When using libbacktrace, we use some necessary global state, so we
        // need to prevent more than one thread from entering this block. This
        // is semi-reasonable in terms of printing anyway, and we know that all
        // I/O done here is blocking I/O, not green I/O, so we don't have to
        // worry about this being a native vs green mutex.
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        let _g = unsafe { LOCK.lock() };

        try!(writeln!(w, "stack backtrace:"));

        let mut cx = Context { writer: w, last_error: None, idx: 0 };
        return match unsafe {
            uw::_Unwind_Backtrace(trace_fn,
                                  &mut cx as *mut Context as *mut libc::c_void)
        } {
            uw::_URC_NO_REASON => {
                match cx.last_error {
                    Some(err) => Err(err),
                    None => Ok(())
                }
            }
            _ => Ok(()),
        };

        extern fn trace_fn(ctx: *mut uw::_Unwind_Context,
                           arg: *mut libc::c_void) -> uw::_Unwind_Reason_Code {
            let cx: &mut Context = unsafe { mem::transmute(arg) };
            let ip = unsafe { uw::_Unwind_GetIP(ctx) as *mut libc::c_void };
            // dladdr() on osx gets whiny when we use FindEnclosingFunction, and
            // it appears to work fine without it, so we only use
            // FindEnclosingFunction on non-osx platforms. In doing so, we get a
            // slightly more accurate stack trace in the process.
            //
            // This is often because failure involves the last instruction of a
            // function being "call std::rt::begin_unwind", with no ret
            // instructions after it. This means that the return instruction
            // pointer points *outside* of the calling function, and by
            // unwinding it we go back to the original function.
            let ip = if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
                ip
            } else {
                unsafe { uw::_Unwind_FindEnclosingFunction(ip) }
            };

            // Don't print out the first few frames (they're not user frames)
            cx.idx += 1;
            if cx.idx <= 0 { return uw::_URC_NO_REASON }
            // Don't print ginormous backtraces
            if cx.idx > 100 {
                match write!(cx.writer, " ... <frames omitted>\n") {
                    Ok(()) => {}
                    Err(e) => { cx.last_error = Some(e); }
                }
                return uw::_URC_FAILURE
            }

            // Once we hit an error, stop trying to print more frames
            if cx.last_error.is_some() { return uw::_URC_FAILURE }

            match print(cx.writer, cx.idx, ip) {
                Ok(()) => {}
                Err(e) => { cx.last_error = Some(e); }
            }

            // keep going
            return uw::_URC_NO_REASON
        }
    }

    #[cfg(target_os = "macos")]
    #[cfg(target_os = "ios")]
    fn print(w: &mut Writer, idx: int, addr: *mut libc::c_void) -> IoResult<()> {
        use intrinsics;
        #[repr(C)]
        struct Dl_info {
            dli_fname: *const libc::c_char,
            dli_fbase: *mut libc::c_void,
            dli_sname: *const libc::c_char,
            dli_saddr: *mut libc::c_void,
        }
        extern {
            fn dladdr(addr: *const libc::c_void,
                      info: *mut Dl_info) -> libc::c_int;
        }

        let mut info: Dl_info = unsafe { intrinsics::init() };
        if unsafe { dladdr(addr as *const libc::c_void, &mut info) == 0 } {
            output(w, idx,addr, None)
        } else {
            output(w, idx, addr, Some(unsafe {
                CString::new(info.dli_sname, false)
            }))
        }
    }

    #[cfg(not(target_os = "macos"), not(target_os = "ios"))]
    fn print(w: &mut Writer, idx: int, addr: *mut libc::c_void) -> IoResult<()> {
        use collections::Collection;
        use iter::Iterator;
        use os;
        use path::GenericPath;
        use ptr::RawPtr;
        use ptr;
        use slice::{ImmutableSlice, MutableSlice};

        ////////////////////////////////////////////////////////////////////////
        // libbacktrace.h API
        ////////////////////////////////////////////////////////////////////////
        type backtrace_syminfo_callback =
            extern "C" fn(data: *mut libc::c_void,
                          pc: libc::uintptr_t,
                          symname: *const libc::c_char,
                          symval: libc::uintptr_t,
                          symsize: libc::uintptr_t);
        type backtrace_error_callback =
            extern "C" fn(data: *mut libc::c_void,
                          msg: *const libc::c_char,
                          errnum: libc::c_int);
        enum backtrace_state {}
        #[link(name = "backtrace", kind = "static")]
        #[cfg(not(test))]
        extern {}

        extern {
            fn backtrace_create_state(filename: *const libc::c_char,
                                      threaded: libc::c_int,
                                      error: backtrace_error_callback,
                                      data: *mut libc::c_void)
                                            -> *mut backtrace_state;
            fn backtrace_syminfo(state: *mut backtrace_state,
                                 addr: libc::uintptr_t,
                                 cb: backtrace_syminfo_callback,
                                 error: backtrace_error_callback,
                                 data: *mut libc::c_void) -> libc::c_int;
        }

        ////////////////////////////////////////////////////////////////////////
        // helper callbacks
        ////////////////////////////////////////////////////////////////////////

        extern fn error_cb(_data: *mut libc::c_void, _msg: *const libc::c_char,
                           _errnum: libc::c_int) {
            // do nothing for now
        }
        extern fn syminfo_cb(data: *mut libc::c_void,
                             _pc: libc::uintptr_t,
                             symname: *const libc::c_char,
                             _symval: libc::uintptr_t,
                             _symsize: libc::uintptr_t) {
            let slot = data as *mut *const libc::c_char;
            unsafe { *slot = symname; }
        }

        // The libbacktrace API supports creating a state, but it does not
        // support destroying a state. I personally take this to mean that a
        // state is meant to be created and then live forever.
        //
        // I would love to register an at_exit() handler which cleans up this
        // state, but libbacktrace provides no way to do so.
        //
        // With these constraints, this function has a statically cached state
        // that is calculated the first time this is requested. Remember that
        // backtracing all happens serially (one global lock).
        //
        // An additionally oddity in this function is that we initialize the
        // filename via self_exe_name() to pass to libbacktrace. It turns out
        // that on Linux libbacktrace seamlessly gets the filename of the
        // current executable, but this fails on freebsd. by always providing
        // it, we make sure that libbacktrace never has a reason to not look up
        // the symbols. The libbacktrace API also states that the filename must
        // be in "permanent memory", so we copy it to a static and then use the
        // static as the pointer.
        //
        // FIXME: We also call self_exe_name() on DragonFly BSD. I haven't
        //        tested if this is required or not.
        unsafe fn init_state() -> *mut backtrace_state {
            static mut STATE: *mut backtrace_state = 0 as *mut backtrace_state;
            static mut LAST_FILENAME: [libc::c_char, ..256] = [0, ..256];
            if !STATE.is_null() { return STATE }
            let selfname = if cfg!(target_os = "freebsd") ||
                              cfg!(target_os = "dragonfly") {
                os::self_exe_name()
            } else {
                None
            };
            let filename = match selfname {
                Some(path) => {
                    let bytes = path.as_vec();
                    if bytes.len() < LAST_FILENAME.len() {
                        let i = bytes.iter();
                        for (slot, val) in LAST_FILENAME.iter_mut().zip(i) {
                            *slot = *val as libc::c_char;
                        }
                        LAST_FILENAME.as_ptr()
                    } else {
                        ptr::null()
                    }
                }
                None => ptr::null(),
            };
            STATE = backtrace_create_state(filename, 0, error_cb,
                                           ptr::null_mut());
            return STATE
        }

        ////////////////////////////////////////////////////////////////////////
        // translation
        ////////////////////////////////////////////////////////////////////////

        // backtrace errors are currently swept under the rug, only I/O
        // errors are reported
        let state = unsafe { init_state() };
        if state.is_null() {
            return output(w, idx, addr, None)
        }
        let mut data = 0 as *const libc::c_char;
        let data_addr = &mut data as *mut *const libc::c_char;
        let ret = unsafe {
            backtrace_syminfo(state, addr as libc::uintptr_t,
                              syminfo_cb, error_cb,
                              data_addr as *mut libc::c_void)
        };
        if ret == 0 || data.is_null() {
            output(w, idx, addr, None)
        } else {
            output(w, idx, addr, Some(unsafe { CString::new(data, false) }))
        }
    }

    // Finally, after all that work above, we can emit a symbol.
    fn output(w: &mut Writer, idx: int, addr: *mut libc::c_void,
              s: Option<CString>) -> IoResult<()> {
        try!(write!(w, "  {:2}: {:2$} - ", idx, addr, super::HEX_WIDTH));
        match s.as_ref().and_then(|c| c.as_str()) {
            Some(string) => try!(super::demangle(w, string)),
            None => try!(write!(w, "<unknown>")),
        }
        w.write(['\n' as u8])
    }

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
            #[cfg(not(target_os = "ios", target_arch = "arm"))]
            pub fn _Unwind_Backtrace(trace: _Unwind_Trace_Fn,
                                     trace_argument: *mut libc::c_void)
                        -> _Unwind_Reason_Code;

            #[cfg(not(target_os = "android"),
                  not(target_os = "linux", target_arch = "arm"))]
            pub fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> libc::uintptr_t;
            #[cfg(not(target_os = "android"),
                  not(target_os = "linux", target_arch = "arm"))]
            pub fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void)
                -> *mut libc::c_void;
        }

        // On android, the function _Unwind_GetIP is a macro, and this is the
        // expansion of the macro. This is all copy/pasted directly from the
        // header file with the definition of _Unwind_GetIP.
        #[cfg(target_os = "android")]
        #[cfg(target_os = "linux", target_arch = "arm")]
        pub unsafe fn _Unwind_GetIP(ctx: *mut _Unwind_Context) -> libc::uintptr_t {
            #[repr(C)]
            enum _Unwind_VRS_Result {
                _UVRSR_OK = 0,
                _UVRSR_NOT_IMPLEMENTED = 1,
                _UVRSR_FAILED = 2,
            }
            #[repr(C)]
            enum _Unwind_VRS_RegClass {
                _UVRSC_CORE = 0,
                _UVRSC_VFP = 1,
                _UVRSC_FPA = 2,
                _UVRSC_WMMXD = 3,
                _UVRSC_WMMXC = 4,
            }
            #[repr(C)]
            enum _Unwind_VRS_DataRepresentation {
                _UVRSD_UINT32 = 0,
                _UVRSD_VFPX = 1,
                _UVRSD_FPAX = 2,
                _UVRSD_UINT64 = 3,
                _UVRSD_FLOAT = 4,
                _UVRSD_DOUBLE = 5,
            }

            type _Unwind_Word = libc::c_uint;
            extern {
                fn _Unwind_VRS_Get(ctx: *mut _Unwind_Context,
                                   klass: _Unwind_VRS_RegClass,
                                   word: _Unwind_Word,
                                   repr: _Unwind_VRS_DataRepresentation,
                                   data: *mut libc::c_void)
                    -> _Unwind_VRS_Result;
            }

            let mut val: _Unwind_Word = 0;
            let ptr = &mut val as *mut _Unwind_Word;
            let _ = _Unwind_VRS_Get(ctx, _UVRSC_CORE, 15, _UVRSD_UINT32,
                                    ptr as *mut libc::c_void);
            (val & !1) as libc::uintptr_t
        }

        // This function also doesn't exist on Android or ARM/Linux, so make it
        // a no-op
        #[cfg(target_os = "android")]
        #[cfg(target_os = "linux", target_arch = "arm")]
        pub unsafe fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void)
            -> *mut libc::c_void
        {
            pc
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
#[allow(dead_code, non_snake_case)]
mod imp {
    use c_str::CString;
    use core_collections::Collection;
    use intrinsics;
    use io::{IoResult, Writer};
    use libc;
    use mem;
    use ops::Drop;
    use option::{Some, None};
    use path::Path;
    use result::{Ok, Err};
    use rt::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};
    use slice::ImmutableSlice;
    use str::StrSlice;
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

    static MAX_SYM_NAME: uint = 2000;
    static IMAGE_FILE_MACHINE_I386: libc::DWORD = 0x014c;
    static IMAGE_FILE_MACHINE_IA64: libc::DWORD = 0x0200;
    static IMAGE_FILE_MACHINE_AMD64: libc::DWORD = 0x8664;

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

        static MAXIMUM_SUPPORTED_EXTENSION: uint = 512;

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
                          ctx: &CONTEXT) -> DWORD {
            frame.AddrPC.Offset = ctx.Rip as u64;
            frame.AddrPC.Mode = super::AddrModeFlat;
            frame.AddrStack.Offset = ctx.Rsp as u64;
            frame.AddrStack.Mode = super::AddrModeFlat;
            frame.AddrFrame.Offset = ctx.Rbp as u64;
            frame.AddrFrame.Mode = super::AddrModeFlat;
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
        static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
        let _g = unsafe { LOCK.lock() };

        // Open up dbghelp.dll, we don't link to it explicitly because it can't
        // always be found. Additionally, it's nice having fewer dependencies.
        let path = Path::new("dbghelp.dll");
        let lib = match DynamicLibrary::open(Some(&path)) {
            Ok(lib) => lib,
            Err(..) => return Ok(()),
        };

        macro_rules! sym( ($e:expr, $t:ident) => (unsafe {
            match lib.symbol($e) {
                Ok(f) => mem::transmute::<*mut u8, $t>(f),
                Err(..) => return Ok(())
            }
        }) )

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
        let ret = SymInitialize(process, 0 as *mut libc::c_void, libc::TRUE);
        if ret != libc::TRUE { return Ok(()) }
        let _c = Cleanup { handle: process, SymCleanup: SymCleanup };

        // And now that we're done with all the setup, do the stack walking!
        let mut i = 0i;
        try!(write!(w, "stack backtrace:\n"));
        while StackWalk64(image, process, thread, &mut frame, &mut context,
                          0 as *mut libc::c_void,
                          0 as *mut libc::c_void,
                          0 as *mut libc::c_void,
                          0 as *mut libc::c_void) == libc::TRUE{
            let addr = frame.AddrPC.Offset;
            if addr == frame.AddrReturn.Offset || addr == 0 ||
               frame.AddrReturn.Offset == 0 { break }

            i += 1;
            try!(write!(w, "  {:2}: {:#2$x}", i, addr, super::HEX_WIDTH));
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
                let cstr = unsafe { CString::new(info.Name.as_ptr(), false) };
                let bytes = cstr.as_bytes();
                match cstr.as_str() {
                    Some(s) => try!(super::demangle(w, s)),
                    None => try!(w.write(bytes.slice_to(bytes.len() - 1))),
                }
            }
            try!(w.write(['\n' as u8]));
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use io::MemWriter;

    macro_rules! t( ($a:expr, $b:expr) => ({
        let mut m = MemWriter::new();
        super::demangle(&mut m, $a).unwrap();
        assert_eq!(String::from_utf8(m.unwrap()).unwrap(), $b.to_string());
    }) )

    #[test]
    fn demangle() {
        t!("test", "test");
        t!("_ZN4testE", "test");
        t!("_ZN4test", "_ZN4test");
        t!("_ZN4test1a2bcE", "test::a::bc");
    }

    #[test]
    fn demangle_dollars() {
        t!("_ZN4$UP$E", "Box");
        t!("_ZN8$UP$testE", "Boxtest");
        t!("_ZN8$UP$test4foobE", "Boxtest::foob");
        t!("_ZN8$x20test4foobE", " test::foob");
    }

    #[test]
    fn demangle_many_dollars() {
        t!("_ZN12test$x20test4foobE", "test test::foob");
        t!("_ZN12test$UP$test4foobE", "testBoxtest::foob");
    }
}
