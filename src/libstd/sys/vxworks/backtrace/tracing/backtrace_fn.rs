/// As always - iOS on arm uses SjLj exceptions and
/// _Unwind_Backtrace is even not available there. Still,
/// backtraces could be extracted using a backtrace function,
/// which thanks god is public
///
/// As mentioned in a huge comment block in `super::super`, backtrace
/// doesn't play well with green threads, so while it is extremely nice and
/// simple to use it should be used only on iOS devices as the only viable
/// option.

use crate::io;
use crate::ptr;
use crate::sys::backtrace::BacktraceContext;
use crate::sys_common::backtrace::Frame;

#[inline(never)] // if we know this is a function call, we can skip it when
                 // tracing
pub fn unwind_backtrace(frames: &mut [Frame])
    -> io::Result<(usize, BacktraceContext)>
{
    const FRAME_LEN: usize = 100;
    assert!(FRAME_LEN >= frames.len());
    let mut raw_frames = [ptr::null_mut(); FRAME_LEN];
    let nb_frames = unsafe {
        backtrace(raw_frames.as_mut_ptr(), raw_frames.len() as libc::c_int)
    } as usize;
    for (from, to) in raw_frames.iter().zip(frames.iter_mut()).take(nb_frames) {
        *to = Frame {
            exact_position: *from as *mut u8,
            symbol_addr: *from as *mut u8,
            inline_context: 0,
        };
    }
    Ok((nb_frames as usize, BacktraceContext))
}

extern {
    fn backtrace(buf: *mut *mut libc::c_void, sz: libc::c_int) -> libc::c_int;
}
