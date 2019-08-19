mod dladdr;

use crate::sys::backtrace::BacktraceContext;
use crate::sys_common::backtrace::Frame;
use crate::io;

#[cfg(target_os = "emscripten")]
pub use self::dladdr::resolve_symname;

#[cfg(target_os = "emscripten")]
pub fn foreach_symbol_fileline<F>(_: Frame, _: F, _: &BacktraceContext) -> io::Result<bool>
where
    F: FnMut(&[u8], u32) -> io::Result<()>
{
    Ok(false)
}

#[cfg(not(target_os = "emscripten"))]
pub use crate::sys_common::gnu::libbacktrace::foreach_symbol_fileline;

#[cfg(not(target_os = "emscripten"))]
pub fn resolve_symname<F>(frame: Frame, callback: F, bc: &BacktraceContext) -> io::Result<()>
where
    F: FnOnce(Option<&str>) -> io::Result<()>
{
    crate::sys_common::gnu::libbacktrace::resolve_symname(frame, |symname| {
        if symname.is_some() {
            callback(symname)
        } else {
            dladdr::resolve_symname(frame, callback, bc)
        }
    }, bc)
}
