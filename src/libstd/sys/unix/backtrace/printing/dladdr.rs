use crate::io;
use crate::intrinsics;
use crate::ffi::CStr;
use crate::sys::backtrace::BacktraceContext;
use crate::sys_common::backtrace::Frame;

pub fn resolve_symname<F>(frame: Frame,
                          callback: F,
                          _: &BacktraceContext) -> io::Result<()>
    where F: FnOnce(Option<&str>) -> io::Result<()>
{
    unsafe {
        let mut info: Dl_info = intrinsics::init();
        let symname = if dladdr(frame.exact_position as *mut _, &mut info) == 0 ||
                         info.dli_sname.is_null() {
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

extern {
    fn dladdr(addr: *const libc::c_void, info: *mut Dl_info) -> libc::c_int;
}
