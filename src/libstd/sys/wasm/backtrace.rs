use crate::io;
use crate::sys::unsupported;
use crate::sys_common::backtrace::Frame;

pub struct BacktraceContext;

pub fn unwind_backtrace(_frames: &mut [Frame])
    -> io::Result<(usize, BacktraceContext)>
{
    unsupported()
}

pub fn resolve_symname<F>(_frame: Frame,
                          _callback: F,
                          _: &BacktraceContext) -> io::Result<()>
    where F: FnOnce(Option<&str>) -> io::Result<()>
{
    unsupported()
}

pub fn foreach_symbol_fileline<F>(_: Frame,
                                  _: F,
                                  _: &BacktraceContext) -> io::Result<bool>
    where F: FnMut(&[u8], u32) -> io::Result<()>
{
    unsupported()
}
