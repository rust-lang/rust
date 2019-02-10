use crate::error::Error;
use crate::ffi::CStr;
use crate::fmt;
use crate::intrinsics;
use crate::io;
use crate::sys_common::backtrace::Frame;

use unwind as uw;

pub struct BacktraceContext;

struct Context<'a> {
    idx: usize,
    frames: &'a mut [Frame],
}

#[derive(Debug)]
struct UnwindError(uw::_Unwind_Reason_Code);

impl Error for UnwindError {
    fn description(&self) -> &'static str {
        "unexpected return value while unwinding"
    }
}

impl fmt::Display for UnwindError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {:?}", self.description(), self.0)
    }
}

#[inline(never)] // if we know this is a function call, we can skip it when
                 // tracing
pub fn unwind_backtrace(frames: &mut [Frame]) -> io::Result<(usize, BacktraceContext)> {
    let mut cx = Context { idx: 0, frames };
    let result_unwind =
        unsafe { uw::_Unwind_Backtrace(trace_fn, &mut cx as *mut Context as *mut libc::c_void) };
    // See libunwind:src/unwind/Backtrace.c for the return values.
    // No, there is no doc.
    match result_unwind {
        // These return codes seem to be benign and need to be ignored for backtraces
        // to show up properly on all tested platforms.
        uw::_URC_END_OF_STACK | uw::_URC_FATAL_PHASE1_ERROR | uw::_URC_FAILURE => {
            Ok((cx.idx, BacktraceContext))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::Other,
            UnwindError(result_unwind),
        )),
    }
}

extern "C" fn trace_fn(
    ctx: *mut uw::_Unwind_Context,
    arg: *mut libc::c_void,
) -> uw::_Unwind_Reason_Code {
    let cx = unsafe { &mut *(arg as *mut Context) };
    if cx.idx >= cx.frames.len() {
        return uw::_URC_NORMAL_STOP;
    }

    let mut ip_before_insn = 0;
    let mut ip = unsafe { uw::_Unwind_GetIPInfo(ctx, &mut ip_before_insn) as *mut libc::c_void };
    if !ip.is_null() && ip_before_insn == 0 {
        // this is a non-signaling frame, so `ip` refers to the address
        // after the calling instruction. account for that.
        ip = (ip as usize - 1) as *mut _;
    }

    let symaddr = unsafe { uw::_Unwind_FindEnclosingFunction(ip) };
    cx.frames[cx.idx] = Frame {
        symbol_addr: symaddr as *mut u8,
        exact_position: ip as *mut u8,
        inline_context: 0,
    };
    cx.idx += 1;

    uw::_URC_NO_REASON
}

pub fn foreach_symbol_fileline<F>(_: Frame, _: F, _: &BacktraceContext) -> io::Result<bool>
where
    F: FnMut(&[u8], u32) -> io::Result<()>,
{
    // No way to obtain this information on CloudABI.
    Ok(false)
}

pub fn resolve_symname<F>(frame: Frame, callback: F, _: &BacktraceContext) -> io::Result<()>
where
    F: FnOnce(Option<&str>) -> io::Result<()>,
{
    unsafe {
        let mut info: Dl_info = intrinsics::init();
        let symname =
            if dladdr(frame.exact_position as *mut _, &mut info) == 0 || info.dli_sname.is_null() {
                None
            } else {
                CStr::from_ptr(info.dli_sname).to_str().ok()
            };
        callback(symname)
    }
}

#[repr(C)]
struct Dl_info {
    dli_fname: *const libc::c_char,
    dli_fbase: *mut libc::c_void,
    dli_sname: *const libc::c_char,
    dli_saddr: *mut libc::c_void,
}

extern "C" {
    fn dladdr(addr: *const libc::c_void, info: *mut Dl_info) -> libc::c_int;
}
