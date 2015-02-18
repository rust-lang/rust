// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

use prelude::v1::*;

use ffi::CStr;
use old_io::IoResult;
use libc;
use mem;
use str;
use sync::{StaticMutex, MUTEX_INIT};

use sys_common::backtrace::*;

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
pub fn write(w: &mut Writer) -> IoResult<()> {
    use result;

    extern {
        fn backtrace(buf: *mut *mut libc::c_void,
                     sz: libc::c_int) -> libc::c_int;
    }

    // while it doesn't requires lock for work as everything is
    // local, it still displays much nicer backtraces when a
    // couple of tasks panic simultaneously
    static LOCK: StaticMutex = MUTEX_INIT;
    let _g = unsafe { LOCK.lock() };

    try!(writeln!(w, "stack backtrace:"));
    // 100 lines should be enough
    const SIZE: uint = 100;
    let mut buf: [*mut libc::c_void; SIZE] = unsafe {mem::zeroed()};
    let cnt = unsafe { backtrace(buf.as_mut_ptr(), SIZE as libc::c_int) as uint};

    // skipping the first one as it is write itself
    let iter = (1..cnt).map(|i| {
        print(w, i as int, buf[i])
    });
    result::fold(iter, (), |_, _| ())
}

#[cfg(not(all(target_os = "ios", target_arch = "arm")))]
#[inline(never)] // if we know this is a function call, we can skip it when
                 // tracing
pub fn write(w: &mut Writer) -> IoResult<()> {
    use old_io::IoError;

    struct Context<'a> {
        idx: int,
        writer: &'a mut (Writer+'a),
        last_error: Option<IoError>,
    }

    // When using libbacktrace, we use some necessary global state, so we
    // need to prevent more than one thread from entering this block. This
    // is semi-reasonable in terms of printing anyway, and we know that all
    // I/O done here is blocking I/O, not green I/O, so we don't have to
    // worry about this being a native vs green mutex.
    static LOCK: StaticMutex = MUTEX_INIT;
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
        // This is often because panic involves the last instruction of a
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

#[cfg(any(target_os = "macos", target_os = "ios"))]
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
    if unsafe { dladdr(addr, &mut info) == 0 } {
        output(w, idx,addr, None)
    } else {
        output(w, idx, addr, Some(unsafe {
            CStr::from_ptr(info.dli_sname).to_bytes()
        }))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn print(w: &mut Writer, idx: int, addr: *mut libc::c_void) -> IoResult<()> {
    use env;
    use ptr;

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
        static mut LAST_FILENAME: [libc::c_char; 256] = [0; 256];
        if !STATE.is_null() { return STATE }
        let selfname = if cfg!(target_os = "freebsd") ||
                          cfg!(target_os = "dragonfly") ||
                          cfg!(target_os = "openbsd") {
            env::current_exe().ok()
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
    let mut data = ptr::null();
    let data_addr = &mut data as *mut *const libc::c_char;
    let ret = unsafe {
        backtrace_syminfo(state, addr as libc::uintptr_t,
                          syminfo_cb, error_cb,
                          data_addr as *mut libc::c_void)
    };
    if ret == 0 || data.is_null() {
        output(w, idx, addr, None)
    } else {
        output(w, idx, addr, Some(unsafe { CStr::from_ptr(data).to_bytes() }))
    }
}

// Finally, after all that work above, we can emit a symbol.
fn output(w: &mut Writer, idx: int, addr: *mut libc::c_void,
          s: Option<&[u8]>) -> IoResult<()> {
    try!(write!(w, "  {:2}: {:2$?} - ", idx, addr, HEX_WIDTH));
    match s.and_then(|s| str::from_utf8(s).ok()) {
        Some(string) => try!(demangle(w, string)),
        None => try!(write!(w, "<unknown>")),
    }
    w.write_all(&['\n' as u8])
}

/// Unwind library interface used for backtraces
///
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

        #[cfg(all(not(all(target_os = "android", target_arch = "arm")),
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
    #[cfg(any(all(target_os = "android", target_arch = "arm"),
              all(target_os = "linux", target_arch = "arm")))]
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
        let _ = _Unwind_VRS_Get(ctx, _Unwind_VRS_RegClass::_UVRSC_CORE, 15,
                                _Unwind_VRS_DataRepresentation::_UVRSD_UINT32,
                                ptr as *mut libc::c_void);
        (val & !1) as libc::uintptr_t
    }

    // This function also doesn't exist on Android or ARM/Linux, so make it
    // a no-op
    #[cfg(any(target_os = "android",
              all(target_os = "linux", target_arch = "arm")))]
    pub unsafe fn _Unwind_FindEnclosingFunction(pc: *mut libc::c_void)
        -> *mut libc::c_void
    {
        pc
    }
}
