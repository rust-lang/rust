use crate::ffi::{CStr, c_void};
use crate::mem::MaybeUninit;
use crate::ptr;

/// Prints the current backtrace, even in a signal handlers.
///
/// Since `backtrace` requires locking and memory allocation, it cannot be used
/// from inside a signal handler. Instead, this uses `libunwind` and `dladdr`,
/// even though both of them are not guaranteed to be async-signal-safe, strictly
/// speaking. However, at least LLVM's libunwind (used by macOS) has a [test] for
/// unwinding in signal handlers, and `dladdr` is used by `backtrace_symbols_fd`
/// in glibc, which it [documents] as async-signal-safe. In practice, this hack
/// works well enough on GNU/Linux and macOS (and perhaps some other platforms,
/// but we haven't enabled those yet).
///
/// [test]: https://github.com/llvm/llvm-project/blob/a6385a3fc8a88f092d07672210a1e773481c2919/libunwind/test/signal_unwind.pass.cpp
/// [documents]: https://www.gnu.org/software/libc/manual/html_node/Backtraces.html#index-backtrace_005fsymbols_005ffd
pub fn print() {
    extern "C" fn frame(
        ctx: *mut unwind::_Unwind_Context,
        arg: *mut c_void,
    ) -> unwind::_Unwind_Reason_Code {
        let count = unsafe { &mut *(arg as *mut usize) };
        let depth = *count;
        *count += 1;
        if depth >= 128 {
            return unwind::_URC_NORMAL_STOP;
        }

        let ip = unsafe { unwind::_Unwind_GetIP(ctx) };
        let mut info = MaybeUninit::uninit();
        let r = unsafe { libc::dladdr(ip.cast(), info.as_mut_ptr()) };
        if r != 0 {
            let info = unsafe { info.assume_init() };
            if !info.dli_sname.is_null() {
                let name = unsafe { CStr::from_ptr(info.dli_sname) };
                if let Ok(name) = name.to_str() {
                    rtprintpanic!("{depth}: {}\n", rustc_demangle::demangle(name));
                    return unwind::_URC_NO_REASON;
                }
            }
        }

        rtprintpanic!("{depth}: {ip:p}\n");
        unwind::_URC_NO_REASON
    }

    let mut count = 0usize;
    unsafe { unwind::_Unwind_Backtrace(frame, ptr::from_mut(&mut count).cast()) };
    if count >= 128 {
        rtprintpanic!("[... some frames omitted ...]\n");
    }
}
