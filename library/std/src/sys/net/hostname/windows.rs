use crate::ffi::OsString;
use crate::io::Result;
use crate::mem::MaybeUninit;
use crate::os::windows::ffi::OsStringExt;
use crate::sys::pal::c;
use crate::sys::pal::winsock::{self, cvt};

pub fn hostname() -> Result<OsString> {
    winsock::startup();

    // The documentation of GetHostNameW says that a buffer size of 256 is
    // always enough.
    let mut buffer = [const { MaybeUninit::<u16>::uninit() }; 256];
    // SAFETY: these parameters specify a valid, writable region of memory.
    cvt(unsafe { c::GetHostNameW(buffer.as_mut_ptr().cast(), buffer.len() as i32) })?;
    // Use `lstrlenW` here as it does not require the bytes after the nul
    // terminator to be initialized.
    // SAFETY: if `GetHostNameW` returns successfully, the name is nul-terminated.
    let len = unsafe { c::lstrlenW(buffer.as_ptr().cast()) };
    // SAFETY: the length of the name is `len`, hence `len` bytes have been
    //         initialized by `GetHostNameW`.
    let name = unsafe { buffer[..len as usize].assume_init_ref() };
    Ok(OsString::from_wide(name))
}
