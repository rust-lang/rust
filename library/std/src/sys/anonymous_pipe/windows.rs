use crate::os::windows::io::FromRawHandle;
use crate::sys::c;
use crate::sys::handle::Handle;
use crate::{io, ptr};

pub type AnonPipe = Handle;

pub fn pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let mut read_pipe = c::INVALID_HANDLE_VALUE;
    let mut write_pipe = c::INVALID_HANDLE_VALUE;

    let ret = unsafe { c::CreatePipe(&mut read_pipe, &mut write_pipe, ptr::null_mut(), 0) };

    if ret == 0 {
        Err(io::Error::last_os_error())
    } else {
        unsafe { Ok((Handle::from_raw_handle(read_pipe), Handle::from_raw_handle(write_pipe))) }
    }
}
